"""Intiaanikäännös: turning a vehicle + trailer around by using a side road.

The rig starts on a two-lane road, facing away from where it has to end up, with its trailer just past a
side road. It has to
  phase 1 (reverse): back the whole rig into the side road (vehicle and trailer, trailer roughly straight), then
  phase 2 (forward): drive out of the side road, turning back the way it came, and stop with the whole rig in the
                     exit zone at the far end of the road, facing back.

Geometry is built in a local frame (main road along +x, side road towards +y), mirrored to a random side and
rotated to a random heading. The kerbs are one closed wall polygon; both junction corners are rounded with
radius rc, and the rounding centres are the turning centres of the two guide routes:
  route 1 (phase 1, trailer axle): west along y = wm - wb, arc of radius wb + rc into the side road, up its centre;
  route 2 (phase 2, vehicle rear axle): down the side-road centre, arc of radius wb + rc onto y = wm - wb, west.
The routes only guide the progress measure, the route arrow and the autopilot; any path that gets there counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import points_in_rect, rect_corners, trailer_corners, vehicle_corners
from .kinematics import trailer_axle, wrap
from .track import Track
from .vehicles import VehicleSpec, get_vehicle

TURN_DIFFICULTY = {
    #            trailer-route radius (x comfortable)  side-road clearance (x W)  main-road clearance (x W)
    "easy": dict(rf=1.45, clr_b=0.60, clr_m=1.10),
    "medium": dict(rf=1.20, clr_b=0.45, clr_m=0.90),
    "hard": dict(rf=1.05, clr_b=0.33, clr_m=0.70),
}
PHASE1_TOL_DEG = 30.0
PHASE2_TOL_DEG = 20.0
WALL_LABEL = {"kerb": 4, "side": 5, "end": 6, "road_end": 7}


# ----------------------------------------------------------------------------- routes / polylines
def _arc(c, r, a0, a1, ds):
    n = max(2, int(math.ceil(abs(a1 - a0) * r / ds)))
    a = np.linspace(a0, a1, n + 1)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def _line(p, q, ds):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(1, int(math.ceil(np.linalg.norm(q - p) / ds)))
    t = np.linspace(0, 1, n + 1)[:, None]
    return p + t * (q - p)


def _chain(*parts):
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p[1:] if np.allclose(out[-1][-1], p[0], atol=1e-6) else p)
    return np.concatenate(out)


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _headings(P):
    d = np.diff(P, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([h, h[-1:]])


def project_polyline(P, S, p, hint_s=None, window: float = 15.0) -> float:
    lo, hi = 0, len(P) - 1
    if hint_s is not None:
        lo = max(0, int(np.searchsorted(S, hint_s - window)) - 1)
        hi = min(len(P) - 1, int(np.searchsorted(S, hint_s + window)) + 1)
    A, B = P[lo:hi], P[lo + 1:hi + 1]
    if len(A) == 0:
        return float(S[lo])
    AB = B - A
    t = np.clip(np.einsum("ij,ij->i", p - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12), 0, 1)
    d = np.linalg.norm(A + AB * t[:, None] - p, axis=1)
    i = int(np.argmin(d))
    return float(S[lo + i] + t[i] * np.linalg.norm(AB[i]))


# ----------------------------------------------------------------------------- layout (local frame)
def turn_params(spec: VehicleSpec, difficulty: str = "medium", widen: float = 1.0) -> dict:
    from .seeder import comfortable_radius, swept_band

    D = TURN_DIFFICULTY[difficulty]
    C, W = spec.combo_length, spec.max_width
    wb = W / 2 + D["clr_b"] * W * widen
    # widening (after a failed certification) loosens the turn as well as the clearances: long trailers need both
    r1 = max(comfortable_radius(spec) * D["rf"] * widen, wb + 0.3 * C / 4, spec.wheelbase / math.tan(spec.max_steer) * 1.15)
    rc = max(0.25 * W, r1 - wb)
    r_out = swept_band(spec, r1)[1]
    clr_m = D["clr_m"] * W * widen
    # the far kerb has to clear the outward swing of the front: steady-state band + transient overshoot (~ trailer length)
    wm = max(W / 2 + clr_m, 0.5 * (r_out + clr_m + 0.15 * spec.trailer_length - rc), wb + 0.25 * W)
    Db = max(1.35 * C, rc + 0.5 * C)  # the whole rig has to fit in the side road
    Le = 1.4 * C
    xb = Le + 0.6 * C + wb + rc
    # start: trailer rear past the junction by enough to set the trailer up for the swing (long trailers need more)
    xa0 = xb + wb + rc + 0.25 * C + 0.6 * spec.trailer_length + spec.trailer_rear
    xr0 = xa0 + spec.trailer_length + spec.hitch_offset
    Lm = xr0 + spec.body_front + 0.3 * C
    return {k: round(float(v), 4) for k, v in dict(wb=wb, r1=r1, rc=rc, wm=wm, Db=Db, Le=Le, xb=xb, xa0=xa0, xr0=xr0, Lm=Lm, C=C).items()}


def _local_layout(spec: VehicleSpec, p: dict, ds: float = 0.25) -> dict:
    wm, wb, rc, Db, Le, xb, Lm, C = (p[k] for k in ("wm", "wb", "rc", "Db", "Le", "xb", "Lm", "C"))
    yr = wm - wb  # the lane both routes use on the main road
    cE = (xb + wb + rc, wm + rc)  # east corner rounding centre = phase-1 turning centre
    cW = (xb - wb - rc, wm + rc)  # west corner rounding centre = phase-2 turning centre
    ytop = wm + Db
    # wall polygon (closed), with a label per segment
    pieces = [
        (_line((0, -wm), (Lm, -wm), 1.0), "kerb"),
        (_line((Lm, -wm), (Lm, wm), 1.0), "road_end"),
        (_line((Lm, wm), (cE[0], wm), 1.0), "kerb"),
        (_arc(cE, rc, -math.pi / 2, -math.pi, ds * 2), "kerb"),
        (_line((xb + wb, wm + rc), (xb + wb, ytop), 1.0), "side"),
        (_line((xb + wb, ytop), (xb - wb, ytop), 1.0), "end"),
        (_line((xb - wb, ytop), (xb - wb, wm + rc), 1.0), "side"),
        (_arc(cW, rc, 0.0, -math.pi / 2, ds * 2), "kerb"),
        (_line((cW[0], wm), (0, wm), 1.0), "kerb"),
        (_line((0, wm), (0, -wm), 1.0), "road_end"),
    ]
    pts, labels = [pieces[0][0][:1]], []
    for P, lab in pieces:
        pts.append(P[1:])
        labels += [WALL_LABEL[lab]] * (len(P) - 1)
    walls = np.concatenate(pts)
    r12 = wb + rc
    x_route = p["xa0"] + 0.3 * C  # route 1 starts a little east of the trailer axle's start
    route1 = _chain(_line((x_route, yr), (cE[0], yr), ds), _arc(cE, r12, -math.pi / 2, -math.pi, ds),
                    _line((xb, cE[1]), (xb, ytop - 0.1), ds))
    route2 = _chain(_line((xb, ytop - 0.1), (xb, cW[1]), ds), _arc(cW, r12, 0.0, -math.pi / 2, ds), _line((cW[0], yr), (0.0, yr), ds))
    side_zone = dict(center=(xb, wm + Db / 2), heading=math.pi / 2, half_len=Db / 2, half_w=wb)
    exit_zone = dict(center=(Le / 2, 0.0), heading=math.pi, half_len=Le / 2, half_w=wm)
    # reference poses: the rig straight and centred in the side road (vehicle facing out); then centred in the exit zone
    y_front = wm + Db / 2 - C / 2
    ref2 = np.array([xb, y_front + spec.body_front])
    goal1 = ref2 + np.array([0.0, spec.hitch_offset + spec.trailer_length])
    x_g2 = (Le + spec.body_front - spec.hitch_offset - spec.trailer_length - spec.trailer_rear) / 2
    goal2 = np.array([x_g2, yr])
    start = np.array([p["xr0"], yr, 0.0, 0.0])
    return dict(walls=walls, wall_labels=np.array(labels), routes=[route1, route2], zones=[side_zone, exit_zone],
                goal1=goal1, ref2=ref2, goal2=goal2, start=start, headings=[-math.pi / 2, math.pi])


def _transform(layout: dict, side: int, rot: float) -> dict:
    c, s = math.cos(rot), math.sin(rot)
    M = np.array([[c, -s], [s, c]]) @ np.diag([1.0, side])

    def P(a):
        return np.atleast_2d(np.asarray(a, float)) @ M.T

    def H(h):
        return wrap(side * h + rot)

    out = dict(layout)
    out["walls"] = P(layout["walls"])
    out["routes"] = [P(r) for r in layout["routes"]]
    for k in ("goal1", "ref2", "goal2"):
        out[k] = P(layout[k])[0]
    st = layout["start"]
    out["start"] = np.array([*P(st[:2])[0], H(st[2]), H(st[3])])
    out["zones"] = [dict(center=P(z["center"])[0], heading=H(z["heading"]), half_len=z["half_len"], half_w=z["half_w"]) for z in layout["zones"]]
    out["headings"] = [H(h) for h in layout["headings"]]
    return out


# ----------------------------------------------------------------------------- the track
@dataclass
class TurnTrack(Track):
    params: dict = field(default_factory=dict)
    side: int = 1
    rot: float = 0.0
    geo: dict = field(default_factory=dict, repr=False)

    task = "turnaround"
    finish_reason = "turned around: the whole rig is stopped in the exit zone, facing back the way it came"

    @classmethod
    def build(cls, name, spec: VehicleSpec, difficulty, seed, params: dict, side: int, rot: float, meta=None) -> "TurnTrack":
        geo = _transform(_local_layout(spec, params), side, rot)
        r1 = geo["routes"][0]
        tr = cls(name=name, vehicle_key=spec.key, template="turn", difficulty=difficulty, seed=seed, primitives=[],
                 centerline=r1, headings=_headings(r1), halfwidth=np.full(len(r1), params["wm"]), s=_arclen(r1),
                 start_state=geo["start"], dock_length=params["Le"], heading_tol_deg=PHASE2_TOL_DEG, meta=dict(meta or {}),
                 params=params, side=side, rot=rot, geo=geo)
        tr._finish_geo()
        return tr

    def _finish_geo(self):
        g = self.geo
        g["route_s"] = [_arclen(r) for r in g["routes"]]
        g["route_h"] = [_headings(r) for r in g["routes"]]
        S1, S2 = g["route_s"]
        g["s1_goal"] = project_polyline(g["routes"][0], S1, g["goal1"])
        g["s2_ref"] = project_polyline(g["routes"][1], S2, g["ref2"])
        g["s2_goal"] = project_polyline(g["routes"][1], S2, g["goal2"])
        g["offsets"] = [0.0, g["s1_goal"] - g["s2_ref"]]

    # ---- geometry overrides ------------------------------------------------
    @property
    def walls(self):
        return [np.vstack([self.geo["walls"]])]

    @property
    def length(self) -> float:
        return self.goal_s

    @property
    def goal_s(self) -> float:
        g = self.geo
        return g["s1_goal"] + g["s2_goal"] - g["s2_ref"]

    def _half(self):
        W = self.geo["walls"]
        k = len(W) // 2
        return W[: k + 1], W[k:]

    @property
    def left_rail(self) -> np.ndarray:  # two halves of the closed wall, for consumers that expect two rails
        return self._half()[0]

    @property
    def right_rail(self) -> np.ndarray:
        return self._half()[1]

    def zone_corners(self, i: int) -> np.ndarray:
        z = self.geo["zones"][i]
        return rect_corners(z["center"][0], z["center"][1], z["heading"], z["half_len"], z["half_len"], z["half_w"])

    @property
    def dock(self) -> dict:
        z = self.geo["zones"][1]
        return {"center": np.asarray(z["center"]), "heading": z["heading"], "length": 2 * z["half_len"], "halfwidth": z["half_w"]}

    @property
    def dock_corners(self) -> np.ndarray:
        return self.zone_corners(1)

    def rail_segments(self):
        if getattr(self, "_segs", None) is None:
            W = self.geo["walls"]
            self._segs = np.stack([W[:-1], W[1:]], 1)
            self._seg_labels = np.asarray(self.geo["wall_labels"], int)
            lo = np.minimum(self._segs[:, 0], self._segs[:, 1])
            hi = np.maximum(self._segs[:, 0], self._segs[:, 1])
            self._seg_bbox = np.concatenate([lo, hi], axis=1)
        return self._segs, self._seg_labels

    def bbox(self):
        W = self.geo["walls"]
        return W.min(0), W.max(0)

    def is_parked(self, trailer_corners, trailer_heading) -> bool:  # not used for the phase logic
        return False

    # ---- phases -------------------------------------------------------------
    @property
    def n_phases(self) -> int:
        return 2

    def phase_info(self, phase: int) -> dict:
        if phase == 0:
            return {"name": "reverse the whole rig into the side road", "direction": -1, "ref": "trailer",
                    "goal": "whole rig (vehicle and trailer) inside the side road, trailer within %g deg of in line with it" % PHASE1_TOL_DEG}
        return {"name": "drive out forward and stop in the exit zone, facing back the way you came", "direction": 1, "ref": "vehicle",
                "goal": "whole rig (vehicle and trailer) inside the exit zone, vehicle within %g deg of facing back down the road" % PHASE2_TOL_DEG}

    def route(self, phase: int):
        g = self.geo
        return g["routes"][phase], g["route_s"][phase], g["offsets"][phase]

    def phase_goal(self, phase: int) -> float:
        """Route-local arclength of the phase's reference pose."""
        g = self.geo
        return g["s1_goal"] if phase == 0 else g["s2_goal"]

    def progress_of(self, state, spec, phase, hint_s=None):
        P, S, off = self.route(phase)
        p = trailer_axle(state, spec) if phase == 0 else np.asarray(state[:2], float)
        s = project_polyline(P, S, p, hint_s)
        return off + s, s

    def phase_done(self, state, spec, phase) -> bool:
        z = self.geo["zones"][phase]
        pts = np.concatenate([vehicle_corners(state, spec), trailer_corners(state, spec)])
        h = state[3] if phase == 0 else state[2]
        inside = points_in_rect(pts, z["center"], z["heading"], z["half_len"], z["half_w"], tol=0.02).all()
        tol = PHASE1_TOL_DEG if phase == 0 else PHASE2_TOL_DEG
        return bool(inside and abs(wrap(h - self.geo["headings"][phase])) <= math.radians(tol))

    def target_corners(self, phase: int) -> np.ndarray:
        return self.zone_corners(min(phase, 1))

    # ---- clearance grids (one per phase: the clearance is shared, the route look-ups are not) ----
    def grid(self, phase: int = 0, max_cells: int = 700):
        gs = self.__dict__.setdefault("_grids", {})
        if phase not in gs:
            base = gs.get(0) if phase else None
            P, S, _ = self.route(phase)
            gs[phase] = PolyGrid(self, P, S, base=base, max_cells=max_cells)
        return gs[phase]

    # ---- serialisation ------------------------------------------------------
    def to_dict(self, include_geometry: bool = True) -> dict:
        d = {"name": self.name, "vehicle": self.vehicle_key, "template": "turn", "task": self.task,
             "difficulty": self.difficulty, "seed": self.seed, "primitives": [],
             "start_state": [float(v) for v in self.start_state], "dock_length": self.dock_length,
             "heading_tol_deg": self.heading_tol_deg, "length": self.length, "goal_s": self.goal_s, "par": self.par,
             "meta": self.meta, "params": self.params, "side": self.side, "rot": self.rot,
             "phases": [{**self.phase_info(i), "goal_s": round(self.geo["offsets"][i] + self.phase_goal(i), 3)} for i in range(2)]}
        if include_geometry:
            r = lambda a: np.round(np.asarray(a, float), 3).tolist()  # noqa: E731
            g = self.geo
            d["geometry"] = {
                "walls": [r(g["walls"])], "wall_labels": [int(x) for x in g["wall_labels"]],
                "zones": [{"name": n, "kind": k, "polygon": r(self.zone_corners(i))}
                          for i, (n, k) in enumerate((("side road", "phase1"), ("exit zone", "phase2")))],
                "routes": [r(x) for x in g["routes"]],
                # compatibility with viewers that draw a railed corridor
                "centerline": r(self.centerline), "headings": r(self.headings), "halfwidth": r(self.halfwidth), "s": r(self.s),
                "left_rail": r(self.left_rail), "right_rail": r(self.right_rail), "dock_corners": r(self.dock_corners),
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TurnTrack":
        spec = get_vehicle(d["vehicle"])
        tr = cls.build(d["name"], spec, d["difficulty"], d["seed"], d["params"], d["side"], d["rot"], meta=d.get("meta", {}))
        tr.par = d.get("par")
        return tr


class PolyGrid:
    """Raster of signed clearance to a closed wall polygon (+ inside), plus arclength / heading along one route."""

    def __init__(self, track: TurnTrack, P: np.ndarray, S: np.ndarray, base: "PolyGrid | None" = None, max_cells: int = 700):
        if base is not None:
            self.origin, self.res, self.shape, self.clear = base.origin, base.res, base.shape, base.clear
        else:
            lo, hi = track.bbox()
            lo, hi = lo - 2.0, hi + 2.0
            self.res = res = max(0.04, float(np.max(hi - lo)) / max_cells)
            self.shape = (int(math.ceil((hi[0] - lo[0]) / res)) + 1, int(math.ceil((hi[1] - lo[1]) / res)) + 1)
            self.origin = lo
        nx, ny = self.shape
        xs = self.origin[0] + self.res * np.arange(nx)
        ys = self.origin[1] + self.res * np.arange(ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([gx.ravel(), gy.ravel()], 1)
        if base is None:
            self.clear = self._signed_distance(G, track.geo["walls"]).reshape(nx, ny)
        # route look-ups on a ~0.5 m polyline
        step = max(1, int(round(0.5 / max(np.diff(S).mean(), 1e-6))))
        idx = np.unique(np.concatenate([np.arange(0, len(S), step), [len(S) - 1]]))
        Pc, Sc = P[idx], S[idx]
        A, AB = Pc[:-1], np.diff(Pc, axis=0)
        L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)
        H = np.arctan2(AB[:, 1], AB[:, 0])
        sarr = np.empty(len(G), np.float32)
        harr = np.empty(len(G), np.float32)
        darr = np.empty(len(G), np.float32)
        chunk = max(256, int(3e6 // max(len(A), 1)))
        for i0 in range(0, len(G), chunk):
            g = G[i0:i0 + chunk]
            d = g[:, None, :] - A[None]
            t = np.clip(np.einsum("pij,ij->pi", d, AB) / L2, 0, 1)
            q = A[None] + t[..., None] * AB[None]
            dist = np.linalg.norm(g[:, None, :] - q, axis=2)
            j = np.argmin(dist, axis=1)
            r = np.arange(len(g))
            darr[i0:i0 + chunk] = dist[r, j]
            sarr[i0:i0 + chunk] = Sc[j] + t[r, j] * np.sqrt(L2[j])
            harr[i0:i0 + chunk] = H[j]
        self.s = sarr.reshape(nx, ny)
        self.heading = harr.reshape(nx, ny)
        self.dist = darr.reshape(nx, ny)

    @staticmethod
    def _signed_distance(G, W):
        A, B = W[:-1], W[1:]
        AB = B - A
        L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)
        out = np.empty(len(G), np.float32)
        chunk = max(256, int(3e6 // max(len(A), 1)))
        for i0 in range(0, len(G), chunk):
            g = G[i0:i0 + chunk]
            d = g[:, None, :] - A[None]
            t = np.clip(np.einsum("pij,ij->pi", d, AB) / L2, 0, 1)
            dist = np.linalg.norm(d - t[..., None] * AB[None], axis=2).min(1)
            # even-odd ray cast towards +x
            ya, yb = A[None, :, 1], B[None, :, 1]
            py = g[:, None, 1]
            cross = (ya > py) != (yb > py)
            xi = A[None, :, 0] + (py - ya) / np.where(yb - ya == 0, 1e-12, yb - ya) * AB[None, :, 0]
            inside = (cross & (xi > g[:, None, 0])).sum(1) % 2 == 1
            out[i0:i0 + chunk] = np.where(inside, dist, -dist)
        return out

    def _idx(self, x, y):
        i = np.rint((x - self.origin[0]) / self.res).astype(int)
        j = np.rint((y - self.origin[1]) / self.res).astype(int)
        out = (i < 0) | (j < 0) | (i >= self.shape[0]) | (j >= self.shape[1])
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1), out

    def clearance(self, x, y):
        i, j, out = self._idx(x, y)
        return np.where(out, -10.0, self.clear[i, j])

    def arclength(self, x, y):
        i, j, _ = self._idx(x, y)
        return self.s[i, j]

    def route_heading(self, x, y):
        i, j, _ = self._idx(x, y)
        return self.heading[i, j]

    def offroute(self, x, y):
        i, j, _ = self._idx(x, y)
        return self.dist[i, j]


# ----------------------------------------------------------------------------- generation
def generate_turn_track(vehicle: str, difficulty: str = "medium", seed: int = 0, certify: bool = True,
                        max_attempts: int = 6, name: str | None = None) -> TurnTrack:
    import random

    from .seeder import _sub_seed, certify_track

    spec = get_vehicle(vehicle)
    rng = random.Random(_sub_seed(vehicle, "turn", difficulty, seed))
    rot = rng.uniform(-math.pi, math.pi)
    side = rng.choice((-1, 1))
    widen, last = 1.0, None
    for attempt in range(max_attempts):
        params = turn_params(spec, difficulty, widen)
        tr = TurnTrack.build(name or f"{vehicle}-turn-{difficulty}-{seed}", spec, difficulty, seed, params, side, rot,
                             meta={"widen": round(widen, 3), "attempt": attempt})
        if not certify:
            return tr
        par = certify_track(tr)
        last = tr
        if par["outcome"] == "FINISHED" and par["touches"] == 0:
            tr.par = par
            tr.meta["certified"] = True
            return tr
        widen *= 1.15
    assert last is not None
    last.meta["certified"] = False
    last.par = None
    return last
