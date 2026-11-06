"""The leg simulator: executes drive commands on a track and keeps the race clock.

Stall-like states (recoverable, but they cost time and need a counter-intuitive fix):
  COMMITTED   |hitch| above the vehicle's critical angle: reversing only folds the
              trailer further; the driver must pull forward to straighten.
  JACKKNIFED  |hitch| at the mechanical limit: reverse motion that would fold it
              further is blocked outright (+ penalty the first time it happens).
  PINNED      a rail was touched: the vehicle stopped at the contact point; motion
              that pushes into the rail again is blocked (no movement, time lost).
Plus: each change between forward and reverse costs `shuttle_s`, and every command
costs a fixed reaction time, so dithering adds up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .ground import DEFAULT_RMS_DEG_PER_M, GroundField
from .geometry import CORNER_NAMES, EDGE_NAMES, poly_edges, segments_intersect, trailer_corners, vehicle_corners
from .kinematics import critical_hitch_angle, hitch_angle, step, trailer_axle, wrap
from .track import Track
from .vehicles import VehicleSpec, get_vehicle

# bump when the physics change so that runs on different physics are not compared
SIM_VERSION = 3  # 2: uneven-ground trailer disturbance; 3: driver-chosen speed, steering slew per second, acceleration, look-around
# (continuous control, Leg.control_step, reuses the v3 physics; runs record it as config.control)

RAIL_NAMES = {0: "rail", 1: "rail", 2: "start wall", 3: "dock wall", 4: "kerb", 5: "side-road wall", 6: "side-road end wall", 7: "road end"}
TERMINAL = ("FINISHED", "TIME_UP", "DNF")


@dataclass
class LegConfig:
    time_budget: float | None = None  # seconds of sim time; None -> derived from par
    max_commands: int = 80
    command_overhead_s: float = 1.0
    touch_penalty_s: float = 5.0
    jackknife_penalty_s: float = 10.0
    max_touches: int = 8
    think_s_per_1k_tokens: float = 0.0  # optional: thinking costs sim time
    trace_dt: float = 0.1  # trace sampling interval (sim seconds)
    look_overhead_s: float = 0.3  # glancing somewhere else costs a moment (plus the head turn)
    max_looks: int = 200
    ground_rms_deg_per_m: float = DEFAULT_RMS_DEG_PER_M  # uneven-ground nudges on the trailer (0 = ideal)
    control_dt: float = 0.05  # continuous control: integration sub-step (sim seconds)
    stop_speed: float = 0.05  # |v| below this counts as standing still (stop metrics)

    def budget_for(self, track: Track) -> float:
        if self.time_budget is not None:
            return float(self.time_budget)
        if track.par and track.par.get("time"):
            par = float(track.par["time"])
            return float(round(max(3.0 * par, par + 120.0)))
        spec = get_vehicle(track.vehicle_key)
        return float(round(4.0 * track.length / spec.speed_rev + 120.0))


@dataclass
class DriveResult:
    command: int
    steer_deg_requested: float
    distance_requested: float
    steer_deg_applied: float
    distance_applied: float
    traveled: float
    stop_reason: str  # completed | contact | blocked | jackknife | time_up | finished | rejected
    contact: dict | None
    time_cost: float
    status: str
    notes: list[str] = field(default_factory=list)
    speed_mps: float = 0.0

    def to_dict(self):
        return asdict(self)


class Leg:
    def __init__(self, track: Track, spec: VehicleSpec | None = None, config: LegConfig | None = None):
        self.track = track
        self.spec = spec or get_vehicle(track.vehicle_key)
        self.cfg = config or LegConfig()
        self.budget = self.cfg.budget_for(track)
        self.beta_crit = critical_hitch_angle(self.spec)
        self.state = np.array(track.start_state, dtype=float)
        self.ground = GroundField(f"{track.name}|{track.seed}", self.cfg.ground_rms_deg_per_m)
        self.steer = 0.0
        self.v = 0.0  # signed speed (m/s) while moving, for the trace
        self.gaze_deg = 180.0  # where the driver looks, relative to the vehicle heading (0 ahead, +left, 180 back)
        self.looks = 0
        # driver's-seat perception (perception.attach): a head with limits that does not wrap through the back
        self.head_limits: tuple[float, float] | None = None
        self.rig = None
        self._head_moved_t = -1e9
        self.t = 0.0
        self.commands = 0
        self.touches = 0
        self.shuttles = 0
        self.jackknifes = 0
        self.blocked = 0
        self.last_dir = 0
        self.pinned = False
        self.jackknifed = False
        self.status = "READY"
        self.outcome: str | None = None
        self.outcome_reason = ""
        self.odometer = 0.0
        self.reverse_m = 0.0
        self.forward_m = 0.0
        self.stall_time = {"COMMITTED": 0.0, "JACKKNIFED": 0.0, "PINNED": 0.0}
        self.max_abs_hitch = 0.0
        # continuous control state
        self.continuous = False
        self.hold_s = 0.0  # the vehicle cannot move until this runs out (gear change, penalty)
        self.target_steer = 0.0  # rad, the held steering target
        self.target_speed = 0.0  # m/s signed, the held speed target
        self.gaze_target = self.gaze_deg
        self.moving_time = 0.0
        self.stops = 0  # times the vehicle came to a standstill after moving
        self._was_moving = False
        self._since_pin = 0.0
        self._blocking = False  # inside one "pushing against a rail / jackknife" episode
        self._block_odo = 0.0  # odometer when that episode began (it ends after 5 cm of real movement)
        self._s_hint = None
        self.phase = 0  # index into the track's phases (a plain course has one)
        self.progress_s = self._trailer_s()
        self.start_s = self.progress_s
        self.best_s = self.progress_s
        self.trace: dict[str, list] = {k: [] for k in ("t", "x", "y", "th1", "th2", "steer", "v")}
        self.events: list[dict] = []
        self._last_trace_t = -1e9
        self._record(force=True)
        self._event("start", budget=self.budget)

    # ------------------------------------------------------------------ helpers
    @property
    def done(self) -> bool:
        return self.outcome is not None

    @property
    def hitch(self) -> float:
        return hitch_angle(self.state)

    @property
    def time_left(self) -> float:
        return max(0.0, self.budget - self.t)

    def _trailer_s(self) -> float:
        """Progress along the manoeuvre (for a plain course: the trailer axle's arc length along the route)."""
        s, self._s_hint = self.track.progress_of(self.state, self.spec, self.phase, self._s_hint)
        return s

    def _goal_reached(self) -> bool:
        """True when the last phase is complete; an earlier phase that completes moves the leg to the next one."""
        tr = self.track
        while tr.phase_done(self.state, self.spec, self.phase):
            if self.phase + 1 >= tr.n_phases:
                return True
            self.phase += 1
            self._s_hint = None
            info = tr.phase_info(self.phase)
            self._event("phase", phase=self.phase + 1, of=tr.n_phases, label=info["name"])
            self.progress_s = self._trailer_s()
            self.best_s = max(self.best_s, self.progress_s)
        return False

    def _record(self, force=False):
        if force or self.t - self._last_trace_t >= self.cfg.trace_dt - 1e-9:
            x, y, t1, t2 = self.state
            tr = self.trace
            tr["t"].append(round(self.t, 3)); tr["x"].append(round(float(x), 3)); tr["y"].append(round(float(y), 3))
            tr["th1"].append(round(float(t1), 4)); tr["th2"].append(round(float(t2), 4)); tr["steer"].append(round(self.steer, 4))
            tr["v"].append(round(self.v, 2))
            if "gaze" in tr:
                tr["gaze"].append(round(self.gaze_deg, 1))
            self._last_trace_t = self.t

    def _event(self, kind: str, **kw):
        e = {"t": round(self.t, 3), "type": kind}
        e.update(kw)
        self.events.append(e)

    def _collision(self, state) -> dict | None:
        vc = vehicle_corners(state, self.spec)
        tc = trailer_corners(state, self.spec)
        allc = np.concatenate([vc, tc])
        segs, labels = self.track.segments_near(allc.min(0), allc.max(0), pad=0.05)
        if len(segs) == 0:
            return None
        for body, corners in (("trailer", tc), ("vehicle", vc)):
            hit = segments_intersect(poly_edges(corners), segs)
            if hit.any():
                ei, si = np.argwhere(hit)[0]
                seg = segs[si]
                # nearest corner of the offending edge to the rail = the part that touched
                a, b = corners[ei], corners[(ei + 1) % 4]
                da = _point_seg_dist(a, seg); db = _point_seg_dist(b, seg)
                corner = CORNER_NAMES[ei] if da <= db else CORNER_NAMES[(ei + 1) % 4]
                point = (a if da <= db else b)
                return {"body": body, "edge": EDGE_NAMES[ei], "corner": corner,
                        "rail": RAIL_NAMES[int(labels[si])], "x": round(float(point[0]), 3), "y": round(float(point[1]), 3)}
        return None

    def _set_status(self, moving_dir: int):
        if self.outcome:
            self.status = self.outcome
            return
        ab = abs(self.hitch)
        if self.jackknifed:
            st = "JACKKNIFED"
        elif self.pinned:
            st = "PINNED"
        elif ab > self.beta_crit and self.beta_crit < self.spec.max_hitch - 1e-6:
            st = "COMMITTED"
        elif moving_dir < 0:
            st = "REVERSING"
        elif moving_dir > 0:
            st = "PULLING_FORWARD"
        else:
            st = "STOPPED"
        if st != self.status:
            self._event("status", status=st)
        self.status = st

    def _advance_time(self, dt: float, label: str | None = None):
        """Spend sim time while standing (overheads, penalties, shuttles)."""
        if dt <= 0:
            return
        self.t += dt
        if self.status in self.stall_time:
            self.stall_time[self.status] += dt
        self._record()

    def spend(self, dt: float, kind: str):
        """Charge sim time for something other than driving (e.g. starting a program)."""
        if self.done or dt <= 0:
            return
        self._event(kind, seconds=round(dt, 3))
        self._advance_time(dt)
        self._check_budget()

    def charge_thinking(self, output_tokens: int):
        if self.cfg.think_s_per_1k_tokens > 0 and not self.done:
            dt = self.cfg.think_s_per_1k_tokens * output_tokens / 1000.0
            self._advance_time(dt)
            self._event("think", seconds=round(dt, 3), tokens=output_tokens)
            self._check_budget()

    def _check_budget(self) -> bool:
        if self.outcome is None and self.t >= self.budget - 1e-9:
            self.t = min(self.t, self.budget) if self.t > self.budget else self.t
            self._finish("TIME_UP", "the leg clock ran out")
            return True
        return False

    def _finish(self, outcome: str, reason: str):
        self.outcome = outcome
        self.outcome_reason = reason
        self.status = outcome
        self._record(force=True)
        self._event("finish", outcome=outcome, reason=reason)

    # ------------------------------------------------------------------ the API
    def look(self, yaw_deg: float) -> dict:
        """Turn the driver's gaze (0 = straight ahead, +left, 180 = straight back). Costs a little time."""
        if self.done:
            return {"ok": False, "note": "the leg is already over", "gaze_deg": self.gaze_deg, "time_cost": 0.0}
        if self.looks >= self.cfg.max_looks:
            return {"ok": False, "note": f"look limit ({self.cfg.max_looks}) reached", "gaze_deg": self.gaze_deg, "time_cost": 0.0}
        try:
            yaw = float(yaw_deg)
        except (TypeError, ValueError):
            yaw = self.gaze_deg
        if not math.isfinite(yaw):
            yaw = self.gaze_deg
        if self.head_limits is not None:  # a real head: clamp to its range, turn the long way if need be
            yaw = max(self.head_limits[0], min(self.head_limits[1], yaw))
            turn = abs(yaw - self.gaze_deg)
        else:
            yaw = (yaw + 180.0) % 360.0 - 180.0  # (-180, 180]
            if yaw == -180.0:
                yaw = 180.0
            turn = abs((yaw - self.gaze_deg + 180.0) % 360.0 - 180.0)
        t0 = self.t
        self.looks += 1
        self.gaze_deg = self.gaze_target = yaw
        self._event("look", yaw_deg=round(yaw, 1))
        self._advance_time(self.cfg.look_overhead_s + turn / self.spec.head_rate_deg_per_s)
        self._check_budget()
        return {"ok": True, "gaze_deg": round(yaw, 1), "time_cost": round(self.t - t0, 3)}

    def drive(self, steer_deg: float, distance_m: float, speed_mps: float | None = None) -> DriveResult:
        """Execute one drive command. Positive steer = wheels left; negative distance = reverse.
        speed_mps: optional cruising speed (default: the vehicle's comfortable speed; capped at its maximum)."""
        spec = self.spec
        if self.done:
            return DriveResult(self.commands, steer_deg, distance_m, 0, 0, 0, "rejected", None, 0.0, self.status,
                               ["the leg is already over"])
        try:
            steer_deg = float(steer_deg); distance_m = float(distance_m)
        except (TypeError, ValueError):
            steer_deg, distance_m = 0.0, 0.0
        if not (math.isfinite(steer_deg) and math.isfinite(distance_m)):
            steer_deg, distance_m = 0.0, 0.0
        notes = []
        self.commands += 1
        t0 = self.t
        target = math.radians(max(-spec.max_steer_deg, min(spec.max_steer_deg, steer_deg)))
        if abs(steer_deg) > spec.max_steer_deg + 1e-9:
            notes.append(f"steering clamped to ±{spec.max_steer_deg:g}°")
        dist = max(-spec.max_cmd_distance, min(spec.max_cmd_distance, distance_m))
        if abs(distance_m) > spec.max_cmd_distance + 1e-9:
            notes.append(f"distance clamped to ±{spec.max_cmd_distance:g} m")
        direction = 0 if abs(dist) < 1e-9 else (1 if dist > 0 else -1)
        speed = spec.default_speed(direction or -1)
        if speed_mps is not None:
            try:
                req = float(speed_mps)
            except (TypeError, ValueError):
                req = float("nan")
            if math.isfinite(req) and req > 0:
                vmax = spec.max_speed(direction or -1)
                speed = min(max(req, 0.1), vmax)
                if req > vmax + 1e-9:
                    notes.append(f"speed capped at {vmax:g} m/s")
        self._event("command", n=self.commands, steer_deg=round(steer_deg, 2), distance=round(distance_m, 3), speed=round(speed, 2))
        self._advance_time(self.cfg.command_overhead_s)
        stop_reason, contact, traveled = "completed", None, 0.0

        if direction == 0:
            # dry steering while standing still
            dt = abs(target - self.steer) / math.radians(spec.steer_rate_deg_per_s)
            self.steer = target
            self._advance_time(dt)
        else:
            if self.last_dir != 0 and direction != self.last_dir:
                self.shuttles += 1
                self._event("shuttle", to="forward" if direction > 0 else "reverse")
                self._advance_time(spec.shuttle_s)
            n = max(1, int(math.ceil(abs(dist) / spec.sim_ds)))
            d = dist / n
            # the wheels turn at a fixed rate per second, so the faster you drive the less they turn per metre
            rate = math.radians(spec.steer_rate_deg_per_s) * abs(d) / speed
            # speeding up and braking: trapezoidal profile (triangular if the move is short)
            D = abs(dist)
            t_move = D / speed + speed / spec.accel if D >= speed * speed / spec.accel else 2.0 * math.sqrt(D / spec.accel)
            dt_step = t_move / n
            # bumps jolt the trailer harder at speed
            jolt = max(0.6, speed / spec.default_speed(direction))
            self.v = direction * speed
            moved_any = False
            for _ in range(n):
                if self.t >= self.budget:
                    stop_reason = "time_up"
                    break
                ns = self.steer + max(-rate, min(rate, target - self.steer))
                new_state = step(self.state, ns, d, spec)
                if self.ground.rms:
                    ax, ay = trailer_axle(new_state, spec)
                    new_state[3] += self.ground(ax, ay) * abs(d) * jolt
                b_old, b_new = abs(self.hitch), abs(wrap(new_state[2] - new_state[3]))
                if b_new >= spec.max_hitch and b_new > b_old:
                    stop_reason = "jackknife"
                    break
                hit = self._collision(new_state)
                if hit:
                    contact = hit
                    stop_reason = "contact"
                    break
                self.state, self.steer = new_state, ns
                traveled += abs(d)
                moved_any = True
                self.t += dt_step
                self.moving_time += dt_step
                if self.status in self.stall_time:
                    self.stall_time[self.status] += dt_step
                self.max_abs_hitch = max(self.max_abs_hitch, b_new)
                # leaving a stall state by moving
                if self.pinned and traveled > 0.05:
                    self.pinned = False
                if self.jackknifed and b_new < spec.max_hitch - math.radians(1.0):
                    self.jackknifed = False
                self._record()
            if moved_any:
                self.stops += 1  # a drive command always ends standing still
                self.last_dir = direction
                self.odometer += traveled
                if direction < 0:
                    self.reverse_m += traveled
                else:
                    self.forward_m += traveled
            if stop_reason == "contact":
                if self.pinned and traveled < 0.05:
                    stop_reason = "blocked"
                    self.blocked += 1
                    notes.append("blocked: you are already against the rail in that direction")
                else:
                    self.touches += 1
                    self.pinned = True
                    self._event("contact", **contact)
                    self._advance_time(self.cfg.touch_penalty_s)
            elif stop_reason == "jackknife":
                if not self.jackknifed:
                    self.jackknifes += 1
                    self.jackknifed = True
                    self._event("jackknife", hitch_deg=round(math.degrees(self.hitch), 1))
                    self._advance_time(self.cfg.jackknife_penalty_s)
                elif traveled < 0.05:
                    stop_reason = "blocked"
                    self.blocked += 1
                    notes.append("blocked: the trailer is jackknifed; reversing cannot fold it further")
        self.v = 0.0
        self.progress_s = self._trailer_s()
        self.best_s = max(self.best_s, self.progress_s)
        self._set_status(direction if traveled > 0 else 0)
        self._record(force=True)

        # terminal checks: parked > touches > budget > commands
        if self._goal_reached():
            stop_reason = "finished" if stop_reason == "completed" else stop_reason
            self._finish("FINISHED", self.track.finish_reason)
        elif self.touches > self.cfg.max_touches:
            self._finish("DNF", f"more than {self.cfg.max_touches} rail touches")
        elif self._check_budget():
            pass
        elif self.commands >= self.cfg.max_commands:
            self._finish("DNF", f"command limit ({self.cfg.max_commands}) reached")

        return DriveResult(
            command=self.commands, steer_deg_requested=steer_deg, distance_requested=distance_m,
            steer_deg_applied=round(math.degrees(target), 2), distance_applied=round(dist, 3),
            traveled=round(traveled, 3), stop_reason=stop_reason, contact=contact,
            time_cost=round(self.t - t0, 3), status=self.status, notes=notes, speed_mps=round(speed, 2) if direction else 0.0,
        )

    # ------------------------------------------------------------------ continuous control
    def set_controls(self, steer_deg, speed_mps) -> list[str]:
        """Continuous mode: set the held steering target (deg, +left) and signed speed target (m/s, - reverse, 0 = brake).
        Returns notes about clamping. The controls stay in force until changed."""
        spec, notes = self.spec, []
        try:
            sd = float(steer_deg)
        except (TypeError, ValueError):
            sd = float("nan")
        try:
            v = float(speed_mps)
        except (TypeError, ValueError):
            v = float("nan")
        if not math.isfinite(sd):
            notes.append("steer_deg was not a number; keeping the previous steering target")
            sd = math.degrees(self.target_steer)
        if not math.isfinite(v):
            notes.append("speed_mps was not a number; braking to a stop")
            v = 0.0
        if abs(sd) > spec.max_steer_deg + 1e-9:
            notes.append(f"steering clamped to ±{spec.max_steer_deg:g}°")
            sd = max(-spec.max_steer_deg, min(spec.max_steer_deg, sd))
        lo, hi = -spec.speed_rev_max, spec.speed_fwd_max
        if v < lo - 1e-9 or v > hi + 1e-9:
            notes.append(f"speed capped at {hi if v > 0 else lo:g} m/s")
            v = max(lo, min(hi, v))
        self.target_steer, self.target_speed = math.radians(sd), v
        return notes

    def set_gaze(self, yaw_deg) -> None:
        """Continuous mode: where to look (the head turns there at the driver's head rate; no standing time)."""
        try:
            yaw = float(yaw_deg)
        except (TypeError, ValueError):
            return
        if math.isfinite(yaw):
            if self.head_limits is not None:
                self.gaze_target = max(self.head_limits[0], min(self.head_limits[1], yaw))
                return
            yaw = (yaw + 180.0) % 360.0 - 180.0
            self.gaze_target = 180.0 if yaw == -180.0 else yaw

    def count_command(self) -> bool:
        """Continuous mode: one decision by the driver. False (and DNF) once the command budget is exceeded."""
        if self.done:
            return False
        self.commands += 1
        if self.commands > self.cfg.max_commands:
            self._finish("DNF", f"command limit ({self.cfg.max_commands}) reached")
            return False
        return True

    def control_step(self, dt: float) -> None:
        """Advance the leg by `dt` sim seconds under the held controls. The vehicle keeps its speed between calls:
        it only stops when asked to (speed 0), on a gear change, or when it hits a rail / the jackknife limit."""
        if self.done or dt <= 0:
            return
        self.continuous = True
        spec, cfg = self.spec, self.cfg
        rate = math.radians(spec.steer_rate_deg_per_s)
        remaining = dt
        while remaining > 1e-9 and not self.done:
            h = min(remaining, cfg.control_dt)
            remaining -= h
            # head turn
            if self.head_limits is not None:  # no wrap: the head sweeps through straight ahead
                dg = self.gaze_target - self.gaze_deg
                if dg:
                    turn = max(-spec.head_rate_deg_per_s * h, min(spec.head_rate_deg_per_s * h, dg))
                    self.gaze_deg = self.gaze_target if abs(dg - turn) < 1e-6 else self.gaze_deg + turn
                    self._head_moved_t = self.t + h
                dg = 0.0
            else:
                dg = (self.gaze_target - self.gaze_deg + 180.0) % 360.0 - 180.0
            if dg:
                turn = max(-spec.head_rate_deg_per_s * h, min(spec.head_rate_deg_per_s * h, dg))
                self.gaze_deg = (self.gaze_deg + turn + 180.0) % 360.0 - 180.0
                if abs(dg - turn) < 1e-6:
                    self.gaze_deg = self.gaze_target
            want = self.target_speed if abs(self.target_speed) >= 1e-3 else 0.0
            wdir = 0 if want == 0 else (1 if want > 0 else -1)
            ns = self.steer + max(-rate * h, min(rate * h, self.target_steer - self.steer))
            v0 = self.v
            if self.hold_s > 0:  # standing: gear change or penalty (the wheels can still be turned)
                self.hold_s = max(0.0, self.hold_s - h)
                self.v, v1 = 0.0, 0.0
            elif v0 == 0.0 and wdir and self.last_dir and wdir != self.last_dir:
                self.shuttles += 1
                self._event("shuttle", to="forward" if wdir > 0 else "reverse")
                self.last_dir = wdir
                self.hold_s = max(0.0, spec.shuttle_s - h)
                v1 = 0.0
            else:
                if v0 != 0.0 and (wdir == 0 or (v0 > 0) != (wdir > 0)):
                    v1 = (max(0.0, v0 - spec.accel * h) if v0 > 0 else min(0.0, v0 + spec.accel * h))  # brake first
                else:
                    v1 = v0 + max(-spec.accel * h, min(spec.accel * h, want - v0))
                if abs(v1) < 1e-6:
                    v1 = 0.0
            dist = 0.5 * (v0 + v1) * h
            self.v = v1
            if abs(dist) > 1e-9:
                self._move(dist, self.steer, ns, max(abs(v0), abs(v1)))
            else:
                self.steer = ns
            self.t += h
            moving = abs(self.v) >= cfg.stop_speed
            if moving or abs(dist) > 1e-9:
                self.moving_time += h
            if self._was_moving and self.v == 0.0:
                self.stops += 1
            self._was_moving = self.v != 0.0
            if self.status in self.stall_time:
                self.stall_time[self.status] += h
            self.progress_s = self._trailer_s()
            self.best_s = max(self.best_s, self.progress_s)
            self._set_status(0 if self.v == 0 else (1 if self.v > 0 else -1))
            self._record()
            if self._goal_reached():
                self.v = 0.0
                self._finish("FINISHED", self.track.finish_reason)
            elif self.touches > cfg.max_touches:
                self.v = 0.0
                self._finish("DNF", f"more than {cfg.max_touches} rail touches")
            elif self._check_budget():
                self.v = 0.0

    def _move(self, dist: float, steer0: float, steer1: float, speed: float):
        """Integrate one continuous sub-step of signed distance `dist` (steering ramps steer0 -> steer1)."""
        spec, cfg = self.spec, self.cfg
        direction = 1 if dist > 0 else -1
        n = max(1, int(math.ceil(abs(dist) / spec.sim_ds)))
        d = dist / n
        jolt = max(0.6, speed / spec.default_speed(direction))
        traveled = 0.0
        for i in range(n):
            ns = steer0 + (steer1 - steer0) * (i + 1) / n
            new_state = step(self.state, ns, d, spec)
            if self.ground.rms:
                ax, ay = trailer_axle(new_state, spec)
                new_state[3] += self.ground(ax, ay) * abs(d) * jolt
            b_old, b_new = abs(self.hitch), abs(wrap(new_state[2] - new_state[3]))
            if b_new >= spec.max_hitch and b_new > b_old:
                self.v = 0.0
                self._block_odo = self.odometer + traveled
                if not self.jackknifed:
                    self.jackknifes += 1
                    self.jackknifed = True
                    self._blocking = True
                    self._event("jackknife", hitch_deg=round(math.degrees(self.hitch), 1))
                    self.hold_s = cfg.jackknife_penalty_s
                elif not self._blocking:
                    self._blocking = True
                    self.blocked += 1
                    self._event("blocked", why="jackknifed")
                break
            hit = self._collision(new_state)
            if hit:
                self.v = 0.0
                self._block_odo = self.odometer + traveled
                if self.pinned and self._since_pin < 0.05:
                    if not self._blocking:
                        self._blocking = True
                        self.blocked += 1
                        self._event("blocked", why="rail")
                else:
                    self.touches += 1
                    self.pinned = True
                    self._since_pin = 0.0
                    self._blocking = True
                    self._event("contact", **hit)
                    self.hold_s = cfg.touch_penalty_s
                break
            self.state = new_state
            traveled += abs(d)
            if self._blocking and self.odometer + traveled - self._block_odo > 0.05:
                self._blocking = False
            self.max_abs_hitch = max(self.max_abs_hitch, b_new)
            if self.pinned:
                self._since_pin += abs(d)
                if self._since_pin > 0.05:
                    self.pinned = False
            if self.jackknifed and b_new < spec.max_hitch - math.radians(1.0):
                self.jackknifed = False
        # the wheels turn at their rate per second whether or not the rig could move (the steering does not change
        # the footprint), so a rig pushing against a rail can still steer its way out
        self.steer = steer1
        if traveled:
            self.last_dir = direction
            self.odometer += traveled
            if direction < 0:
                self.reverse_m += traveled
            else:
                self.forward_m += traveled

    def abort(self, reason: str):
        if not self.done:
            self._finish("DNF", reason)

    # ------------------------------------------------------------------ results
    @property
    def short_m(self) -> float:
        if self.outcome == "FINISHED":
            return 0.0
        return max(0.0, self.track.goal_s - self.progress_s)

    def metrics(self) -> dict:
        tl = self.track.goal_s - self.start_s
        done_s = self.progress_s - self.start_s
        return {
            "outcome": self.outcome or "RUNNING", "reason": self.outcome_reason,
            "time": round(self.t, 2), "budget": self.budget, "commands": self.commands,
            "touches": self.touches, "shuttles": self.shuttles, "jackknifes": self.jackknifes,
            "blocked": self.blocked, "odometer": round(self.odometer, 2),
            "reverse_m": round(self.reverse_m, 2), "forward_m": round(self.forward_m, 2),
            "progress": 1.0 if self.outcome == "FINISHED" else (round(min(0.999, max(0.0, done_s / tl)), 4) if tl > 0 else 0.0),
            "short_m": round(self.short_m, 2),
            "max_abs_hitch_deg": round(math.degrees(self.max_abs_hitch), 1),
            "stall_time": {k: round(v, 2) for k, v in self.stall_time.items()},
            "stops": self.stops, "moving_time": round(self.moving_time, 2),
            "stopped_time": round(max(0.0, self.t - self.moving_time), 2),
            "mean_speed": round(self.odometer / self.t, 3) if self.t > 0 else 0.0,
            "par_time": (self.track.par or {}).get("time"),
            **({"phase": self.phase + 1, "phases": self.track.n_phases} if self.track.n_phases > 1 else {}),
        }


def _point_seg_dist(p, seg) -> float:
    a, b = seg[0], seg[1]
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))
