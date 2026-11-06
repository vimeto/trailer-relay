"""A privileged classical driver: sampling-based MPC on the exact kinematic model.

It sees the full state and the track's clearance raster, and drives through the
same `Leg.drive` API as every other driver. It serves three purposes:
  1. certifies that a generated track is drivable (the seeder rejects/widens otherwise),
  2. sets the par time used to size each leg's clock,
  3. is an upper-bound baseline in the results table.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import perimeter_samples
from .kinematics import critical_hitch_angle, step_batch, trailer_axle, wrap
from .sim import Leg


class Autopilot:
    label = "Autopilot (privileged MPC)"
    path_ref = None  # "trailer" | "vehicle": track the cross-track error of that point along each rollout

    def __init__(self, step_frac: float = 0.12, horizon_frac: float = 0.65, n1: int = 15, n2: int = 7):
        self.step_frac, self.horizon_frac, self.n1, self.n2 = step_frac, horizon_frac, n1, n2

    # ----------------------------------------------------------------- set-up
    def _setup(self, leg: Leg):
        sp = leg.spec
        self.spec, self.track, self.grid = sp, leg.track, leg.track.grid()
        C = sp.combo_length
        self.step = max(0.3, self.step_frac * C)
        self.h2 = max(self.horizon_frac * C, 2.2 * sp.trailer_length)
        self.ds = max(sp.sim_ds * 2, C / 90)
        self.margin = max(0.06, 0.04 * sp.max_width)
        spacing = max(0.15, C / 40)
        self.v_pts = perimeter_samples(sp.body_front, sp.body_rear, sp.body_width / 2, spacing)
        self.t_pts = perimeter_samples(sp.trailer_front, sp.trailer_rear, sp.trailer_width / 2, spacing)
        self.blim = 0.85 * min(critical_hitch_angle(sp), sp.max_hitch)

    def _clearance(self, x, y, t1, t2):
        sp = self.spec
        c1, s1 = np.cos(t1)[:, None], np.sin(t1)[:, None]
        P = self.v_pts
        vx = x[:, None] + P[:, 0] * c1 - P[:, 1] * s1
        vy = y[:, None] + P[:, 0] * s1 + P[:, 1] * c1
        hx = x - sp.hitch_offset * np.cos(t1)
        hy = y - sp.hitch_offset * np.sin(t1)
        ax = hx - sp.trailer_length * np.cos(t2)
        ay = hy - sp.trailer_length * np.sin(t2)
        c2, s2 = np.cos(t2)[:, None], np.sin(t2)[:, None]
        Q = self.t_pts
        tx = ax[:, None] + Q[:, 0] * c2 - Q[:, 1] * s2
        ty = ay[:, None] + Q[:, 0] * s2 + Q[:, 1] * c2
        cv = self.grid.clearance(vx, vy).min(axis=1)
        ct = self.grid.clearance(tx, ty).min(axis=1)
        return np.minimum(cv, ct), ax, ay

    def _rollout(self, state, steer0, targets: np.ndarray, d: float, split: int | None = None):
        """Simulate K steering sequences. Returns end states, min clearance during the first
        `split` steps (hard constraint) and during the rest (soft), and max |hitch|."""
        K, n = targets.shape
        split = n if split is None else split
        x = np.full(K, state[0]); y = np.full(K, state[1])
        t1 = np.full(K, state[2]); t2 = np.full(K, state[3])
        steer = np.full(K, steer0)
        rate = math.radians(self.spec.steer_rate_deg_per_m) * abs(d)
        minc1 = np.full(K, np.inf); minc2 = np.full(K, np.inf)
        maxb = np.abs(wrap(t1 - t2))
        ck = max(1, split // 8)
        ck2 = max(1, (n - split) // 10)
        ax = ay = None
        pen = np.zeros(K)  # worst cross-track error along the rollout (route-following drivers only)
        ref = self.path_ref
        for i in range(n):
            steer = steer + np.clip(targets[:, i] - steer, -rate, rate)
            x, y, t1, t2 = step_batch(x, y, t1, t2, steer, d, self.spec)
            maxb = np.maximum(maxb, np.abs(wrap(t1 - t2)))
            if i < split:
                if i % ck == ck - 1 or i == split - 1:
                    c, ax, ay = self._clearance(x, y, t1, t2)
                    minc1 = np.minimum(minc1, c)
                    if ref:
                        pen = np.maximum(pen, self.grid.offroute(*((ax, ay) if ref == "trailer" else (x, y))))
            elif (i - split) % ck2 == ck2 - 1 or i == n - 1:
                c, ax, ay = self._clearance(x, y, t1, t2)
                minc2 = np.minimum(minc2, c)
                if ref:
                    pen = np.maximum(pen, self.grid.offroute(*((ax, ay) if ref == "trailer" else (x, y))))
        if ax is None:
            _, ax, ay = self._clearance(x, y, t1, t2)
        self.last_offroute = pen
        return x, y, t1, t2, minc1, minc2, maxb, ax, ay

    # ----------------------------------------------------------------- planning
    def plan(self, leg: Leg) -> tuple[float, float]:
        sp = self.spec
        s_now, goal = self._progress(leg)
        remaining = goal - s_now
        if abs(leg.hitch) > self.blim or leg.status in ("JACKKNIFED", "COMMITTED"):
            return self._recover(leg)
        if remaining < 0.03:
            return self._recover(leg)  # at the goal but not parked: straighten and retry
        d1 = min(self.step, remaining + 0.02)
        rest = max(0.0, min(self.h2, remaining - d1))
        n1s = max(1, int(math.ceil(d1 / self.ds)))
        dstep = d1 / n1s
        n2s = int(math.ceil(rest / dstep)) if rest > 0.2 else 0
        smax = sp.max_steer
        a = np.linspace(-smax, smax, self.n1)
        if n2s >= 2:
            b = np.linspace(-smax, smax, self.n2)
            A, B, Cc = (g.ravel() for g in np.meshgrid(a, b, b, indexing="ij"))
            h = n2s // 2
            targets = np.concatenate([np.repeat(A[:, None], n1s, 1), np.repeat(B[:, None], h, 1),
                                      np.repeat(Cc[:, None], n2s - h, 1)], 1)
        else:
            A = a
            targets = np.repeat(A[:, None], n1s, 1)
        x, y, t1, t2, minc1, minc2, maxb, ax, ay = self._rollout(leg.state, leg.steer, targets, -dstep, split=n1s)
        s_end = self.grid.arclength(ax, ay)
        prog = np.minimum(s_end, goal) - s_now
        H = self.grid.route_heading(ax, ay)
        eh = wrap(t2 + math.pi - H)
        W = sp.max_width
        near = remaining < 0.6 * sp.combo_length
        minc = np.minimum(minc1, minc2)
        cost = (-prog / self.step
                - 2.0 * np.minimum(minc, 0.6 * W) / W
                + 8.0 * np.maximum(0.0, self.margin - minc2) / W
                + (6.0 if near else 1.5) * eh ** 2
                + 4.0 * np.maximum(0, maxb - 0.5 * self.blim) ** 2
                + 0.05 * (A / smax) ** 2
                + self._extra_cost())
        feasible = (minc1 > self.margin) & (maxb < self.blim)
        if not feasible.any():
            return self._recover(leg)
        cost = np.where(feasible, cost, np.inf)
        i = int(np.argmin(cost))
        return math.degrees(A[i]), -d1

    def _extra_cost(self):
        return 0.0

    def _progress(self, leg: Leg) -> tuple[float, float]:
        """(where the reference point is along the planning route, where it has to get to)."""
        return leg.progress_s, self.track.goal_s

    def _recover(self, leg: Leg) -> tuple[float, float]:
        """Pull forward to straighten the trailer and regain clearance."""
        sp = self.spec
        smax = sp.max_steer
        a = np.linspace(-smax, smax, 21)
        best = None
        for frac in (0.4, 0.25, 0.12):
            dist = frac * sp.combo_length
            n = max(2, int(math.ceil(dist / self.ds)))
            targets = np.repeat(a[:, None], n, 1)
            x, y, t1, t2, minc, _, maxb, ax, ay = self._rollout(leg.state, leg.steer, targets, dist / n)
            H = self.grid.route_heading(ax, ay)
            eh = wrap(t2 + math.pi - H)
            b = np.abs(wrap(t1 - t2))
            cost = 3.0 * b + 1.5 * np.abs(eh) - 1.0 * np.minimum(minc, 0.5 * sp.max_width) / sp.max_width
            ok = minc > self.margin
            if ok.any():
                i = int(np.argmin(np.where(ok, cost, np.inf)))
                best = (math.degrees(a[i]), dist)
                break
        return best or (0.0, 0.2 * sp.combo_length)

    def run_leg(self, leg: Leg, max_iters: int = 400) -> dict:
        if leg.track.task == "turnaround" and not isinstance(self, TurnAutopilot):
            return TurnAutopilot(self.step_frac, self.horizon_frac, self.n1, self.n2).run_leg(leg, max_iters)
        self._setup(leg)
        it = 0
        while not leg.done and it < max_iters:
            steer, dist = self.plan(leg)
            leg.drive(steer, dist)
            it += 1
        if not leg.done:
            leg.abort("autopilot iteration cap")
        return {"transcript": [], "usage": {}}


class TurnAutopilot(Autopilot):
    """Two-phase driver for turnaround courses: the reverse MPC on route 1 (trailer axle), then a forward MPC on
    route 2 (vehicle rear axle). Each phase plans on its own clearance/route raster."""

    k_route = 4.0

    def plan(self, leg: Leg) -> tuple[float, float]:
        self.grid = self.track.grid(leg.phase)
        self.path_ref = "trailer" if leg.phase == 0 else "vehicle"
        if leg.phase == 0:
            return super().plan(leg)
        return self._plan_forward(leg)

    def _extra_cost(self):
        """Open junctions have no corridor to keep the plan near the guide route: penalise the worst cross-track
        error along the rollout (not just at its end, or a long horizon cuts the corner mid-way)."""
        return self.k_route * (self.last_offroute / self.spec.max_width) ** 2

    def _progress(self, leg: Leg) -> tuple[float, float]:
        _, _, off = self.track.route(leg.phase)
        return leg.progress_s - off, self.track.phase_goal(leg.phase)

    def _plan_forward(self, leg: Leg) -> tuple[float, float]:
        sp = self.spec
        s_now, goal = self._progress(leg)
        remaining = goal - s_now
        if remaining < 0.03:
            return self._back_up(leg)  # overshot the reference pose without being done: back up and come again
        d1 = min(self.step, remaining + 0.02)
        rest = max(0.0, min(self.h2, remaining - d1))
        n1s = max(1, int(math.ceil(d1 / self.ds)))
        dstep = d1 / n1s
        n2s = int(math.ceil(rest / dstep)) if rest > 0.2 else 0
        smax = sp.max_steer
        a = np.linspace(-smax, smax, self.n1)
        if n2s >= 2:
            b = np.linspace(-smax, smax, self.n2)
            A, B, Cc = (g.ravel() for g in np.meshgrid(a, b, b, indexing="ij"))
            h = n2s // 2
            targets = np.concatenate([np.repeat(A[:, None], n1s, 1), np.repeat(B[:, None], h, 1),
                                      np.repeat(Cc[:, None], n2s - h, 1)], 1)
        else:
            A = a
            targets = np.repeat(A[:, None], n1s, 1)
        x, y, t1, t2, minc1, minc2, maxb, _, _ = self._rollout(leg.state, leg.steer, targets, dstep, split=n1s)
        prog = np.minimum(self.grid.arclength(x, y), goal) - s_now
        eh = wrap(t1 - self.grid.route_heading(x, y))
        W = sp.max_width
        near = remaining < 0.8 * sp.combo_length
        minc = np.minimum(minc1, minc2)
        cost = (-prog / self.step
                - 2.0 * np.minimum(minc, 0.6 * W) / W
                + 8.0 * np.maximum(0.0, self.margin - minc2) / W
                + (6.0 if near else 1.0) * eh ** 2
                + (3.0 if near else 0.5) * wrap(t1 - t2) ** 2
                + 0.05 * (A / smax) ** 2
                + self._extra_cost())
        feasible = minc1 > self.margin
        if not feasible.any():
            return self._back_up(leg)
        i = int(np.argmin(np.where(feasible, cost, np.inf)))
        return math.degrees(A[i]), d1

    def _back_up(self, leg: Leg) -> tuple[float, float]:
        """Reverse a little to regain clearance, keeping the hitch small."""
        sp = self.spec
        a = np.linspace(-sp.max_steer, sp.max_steer, 21)
        for frac in (0.3, 0.18, 0.1):
            dist = frac * sp.combo_length
            n = max(2, int(math.ceil(dist / self.ds)))
            x, y, t1, t2, minc, _, maxb, _, _ = self._rollout(leg.state, leg.steer, np.repeat(a[:, None], n, 1), -dist / n)
            eh = wrap(t1 - self.grid.route_heading(x, y))
            cost = 3.0 * maxb + 1.0 * np.abs(eh) - 1.5 * np.minimum(minc, 0.6 * sp.max_width) / sp.max_width
            ok = (minc > self.margin) & (maxb < self.blim)
            if ok.any():
                i = int(np.argmin(np.where(ok, cost, np.inf)))
                return math.degrees(a[i]), -dist
        return 0.0, -0.1 * sp.combo_length
