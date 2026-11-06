"""Continuous control: the vehicle keeps rolling between the driver's decisions.

The Engine owns "what the vehicle is doing right now": either held controls (steer + signed speed, set by the
continuous `drive` tool) or an installed model-written controller (controller.py) that is called every tick.
It advances the leg tick by tick and collects reasons to wake the driver (the model) up: contact, jackknife,
the trailer going past the critical angle (tools only), a controller error, `wake()` from the controller, the
end of the leg.

Two clocks:
  paused    the sim stands still while the model thinks; a new decision takes effect after `reaction_s`
            (the old controls keep acting meanwhile).
  realtime  the sim keeps running (paced to the wall clock, x `latency_scale`) while the model thinks, so a slow
            answer is spent driving on the previous controls / controller.
"""
from __future__ import annotations

import math
import threading
import time

from .controller import ControllerError, ControllerProcess
from .observe import observe
from .sim import Leg

WAKE_EVENTS = ("contact", "jackknife")


class Engine:
    def __init__(self, leg: Leg, obs_level: str = "local", hz: float = 10.0, wake_on_committed: bool = True,
                 tick_timeout_s: float = 0.5, sandbox: bool = True, perception: str | None = None):
        self.leg = leg
        self.obs_level = obs_level
        self.dt = 1.0 / hz
        self.wake_on_committed = wake_on_committed
        self.tick_timeout_s = tick_timeout_s
        self.sandbox = sandbox
        self.controller: ControllerProcess | None = None
        self.controller_n = 0
        self.wakes: list[tuple[str, str]] = []  # (kind, detail), pending
        self.output: list[tuple[float, str]] = []  # controller print() output with sim time
        self.ticks = 0
        self._ev_mark = len(leg.events)
        leg.continuous = True
        if obs_level == "driver":
            from .perception import attach

            attach(leg, level=perception)

    # ------------------------------------------------------------------ controller management
    def install(self, code: str) -> str | None:
        """Start a new controller; on success it replaces the running one. Returns an error message or None."""
        try:
            new = ControllerProcess(code, tick_timeout_s=self.tick_timeout_s, sandbox=self.sandbox)
        except ControllerError as e:
            if e.out:
                self.output.append((self.leg.t, e.out))
            return str(e)
        self.remove()
        self.controller = new
        self.controller_n += 1
        if new.startup_out:
            self.output.append((self.leg.t, new.startup_out))
        self.leg._event("controller", n=self.controller_n)
        return None

    def remove(self):
        if self.controller is not None:
            self.controller.close()
            self.controller = None

    def close(self):
        self.remove()

    def controller_obs(self) -> dict:
        o = observe(self.leg, self.obs_level)
        o["t"] = round(self.leg.t, 3)
        o["dt"] = self.dt
        return o

    # ------------------------------------------------------------------ stepping
    def tick(self):
        leg = self.leg
        if leg.done:
            return
        if self.controller is not None:
            try:
                (steer, speed, look), out, wake = self.controller.tick(self.controller_obs())
            except ControllerError as e:
                self.controller = None
                leg.set_controls(math.degrees(leg.target_steer), 0.0)  # brake to a stop
                if e.out:
                    self.output.append((leg.t, e.out))
                for w in getattr(e, "wake", []) or []:
                    self.wakes.append(("wake", w))
                self.wakes.append(("controller error", str(e)))
                leg._event("controller_error", message=str(e)[-300:])
            else:
                leg.set_controls(steer, speed)
                if look is not None:
                    leg.set_gaze(look)
                if out:
                    self.output.append((leg.t, out))
                for w in wake:
                    self.wakes.append(("wake", w))
                    leg._event("wake", message=w[:120])
        leg.control_step(self.dt)
        self.ticks += 1
        self._scan()

    def _scan(self):
        leg = self.leg
        for e in leg.events[self._ev_mark:]:
            k = e["type"]
            if k == "contact":
                self.wakes.append(("contact", f"the {e['body']}'s {e['corner']} corner touched the {e['rail']}"))
            elif k == "jackknife":
                self.wakes.append(("jackknife", f"the trailer hit the jackknife limit ({e['hitch_deg']:+.0f}°)"))
            elif k == "blocked" and self.controller is None:
                self.wakes.append(("blocked", "pushing against the " + ("rail" if e.get("why") == "rail" else "jackknife limit")))
            elif k == "status" and e["status"] == "COMMITTED" and self.wake_on_committed:
                self.wakes.append(("committed", "the hitch angle went past the point where reversing can still straighten it"))
            elif k == "finish":
                self.wakes.append(("leg over", f"{e['outcome']}: {e['reason']}"))
        self._ev_mark = len(leg.events)

    def run(self, seconds: float, interrupt: bool = True) -> float:
        """Advance up to `seconds` of sim time (paused clock: as fast as possible). Stops early when the leg ends,
        or, with interrupt, as soon as there is a reason to wake the driver. Returns the sim time advanced."""
        leg = self.leg
        t0 = leg.t
        n = int(round(seconds / self.dt))
        for _ in range(n):
            if leg.done or (interrupt and self.wakes):
                break
            self.tick()
        return leg.t - t0

    def run_while(self, fn, scale: float = 1.0, offset_s: float = 0.0):
        """Realtime clock: call fn() (the model request) while the sim keeps driving in a background thread, paced
        to the wall clock (sim seconds = max(0, wall - offset_s) * scale). Afterwards the sim is caught up exactly.
        Returns (fn's result, wall seconds, sim seconds driven meanwhile)."""
        leg = self.leg
        stop = threading.Event()
        t_wall0, done = time.time(), [0.0]
        failure: list[BaseException] = []

        def target(now):
            return max(0.0, now - t_wall0 - offset_s) * scale

        def bg():
            try:
                while not stop.is_set() and not leg.done:
                    tg = target(time.time())
                    while done[0] + self.dt <= tg + 1e-9 and not leg.done and not stop.is_set():
                        self.tick()
                        done[0] += self.dt
                    stop.wait(0.02)
            except BaseException as e:  # noqa: BLE001 - surfaced in the caller's thread
                failure.append(e)

        th = threading.Thread(target=bg, daemon=True)
        th.start()
        try:
            result = fn()
        finally:
            stop.set()
            th.join()
        if failure:
            raise failure[0]
        wall = time.time() - t_wall0
        tg = target(t_wall0 + wall)
        while done[0] + self.dt <= tg + 1e-9 and not leg.done:
            self.tick()
            done[0] += self.dt
        return result, wall, done[0]

    # ------------------------------------------------------------------ reporting
    def mark(self) -> dict:
        leg = self.leg
        return {"t": leg.t, "odo": leg.odometer, "ev": len(leg.events), "out": len(self.output), "stops": leg.stops,
                "touches": leg.touches}

    def take_wakes(self) -> list[tuple[str, str]]:
        w, self.wakes = self.wakes, []
        # one line per kind (a controller can call wake() every tick)
        seen, out = set(), []
        for k, d in w:
            if (k, d) not in seen:
                seen.add((k, d))
                out.append((k, d))
        return out[:6]

    def describe(self, m: dict, label: str = "SINCE YOUR LAST DECISION") -> list[str]:
        """What happened since mark m, in a few lines."""
        leg = self.leg
        dt, dist = leg.t - m["t"], leg.odometer - m["odo"]
        lines = []
        if dt > 1e-6:
            stops = leg.stops - m["stops"]
            lines.append(f"{label} ({dt:.1f} s): drove {dist:.1f} m"
                         + (f", came to a stop {stops} time(s)" if stops else ", without stopping" if dist > 0.05 else ""))
        evs = []
        for e in leg.events[m["ev"]:]:
            k = e["type"]
            if k == "contact":
                evs.append(f"t={e['t']:.1f} CONTACT: the {e['body']}'s {e['corner']} corner touched the {e['rail']}")
            elif k == "jackknife":
                evs.append(f"t={e['t']:.1f} JACKKNIFE at {e['hitch_deg']:+.0f}°")
            elif k == "shuttle":
                evs.append(f"t={e['t']:.1f} gear change to {e['to']}")
            elif k == "status" and e["status"] in ("COMMITTED", "JACKKNIFED", "PINNED"):
                evs.append(f"t={e['t']:.1f} status {e['status']}")
            elif k == "controller_error":
                evs.append(f"t={e['t']:.1f} CONTROLLER ERROR (vehicle braking to a stop)")
            elif k == "wake":
                evs.append(f"t={e['t']:.1f} controller called wake({e['message']!r})")
        if len(evs) > 10:
            evs = evs[:4] + [f"... {len(evs) - 8} more ..."] + evs[-4:]
        if evs:
            lines.append("EVENTS: " + "; ".join(evs))
        outs = self.output[m["out"]:]
        if outs:
            text = "".join(o for _, o in outs).rstrip()
            if len(text) > 1500:
                text = text[:500] + "\n...[output truncated]...\n" + text[-900:]
            lines.append("CONTROLLER OUTPUT:\n" + text)
        return lines

    def motion_line(self) -> str:
        leg = self.leg
        v = leg.v
        how = "standing still" if v == 0 else (f"reversing at {-v:.2f} m/s" if v < 0 else f"pulling forward at {v:.2f} m/s")
        hold = f" (gear change / penalty: {leg.hold_s:.1f} s before it can move)" if leg.hold_s > 0 else ""
        who = (f"controller #{self.controller_n} is driving" if self.controller is not None
               else f"held controls: steer {math.degrees(leg.target_steer):+.1f}°, speed {leg.target_speed:+.2f} m/s")
        return f"MOTION: {how}{hold}; {who}."
