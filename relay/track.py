"""Railed reversing courses.

A track is a corridor around a centreline. The route runs from the start cap
(where the vehicle's nose is) to the dock wall (the end of the handoff bay).
The vehicle starts facing *against* the route, so the whole course is driven in
reverse with the trailer leading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import points_in_rect
from .kinematics import wrap
from .vehicles import VehicleSpec


@dataclass
class Primitive:
    kind: str  # "straight" | "arc"
    length: float = 0.0  # straight length
    radius: float = 0.0  # arc radius
    angle: float = 0.0  # arc angle (rad), + = turn left along the route
    halfwidth: float = 2.0

    def to_dict(self):
        return {"kind": self.kind, "length": self.length, "radius": self.radius, "angle": self.angle, "halfwidth": self.halfwidth}


def build_centerline(primitives: list[Primitive], ds: float = 0.25, smooth_m: float = 3.0):
    """Sample a centreline: returns points (N,2), headings (N,), halfwidths (N,), arclength (N,)."""
    x, y, h = 0.0, 0.0, 0.0
    pts, hs, hws = [(x, y)], [h], [primitives[0].halfwidth]
    for p in primitives:
        if p.kind == "straight":
            n = max(1, int(math.ceil(p.length / ds)))
            step = p.length / n
            for _ in range(n):
                x += step * math.cos(h)
                y += step * math.sin(h)
                pts.append((x, y)); hs.append(h); hws.append(p.halfwidth)
        elif p.kind == "arc":
            arc_len = abs(p.angle) * p.radius
            n = max(2, int(math.ceil(arc_len / ds)))
            dh = p.angle / n
            sgn = 1.0 if p.angle > 0 else -1.0
            # centre of curvature on the left (+) or right (-) of the heading
            cx = x - sgn * p.radius * math.sin(h)
            cy = y + sgn * p.radius * math.cos(h)
            for _ in range(n):
                h += dh
                x = cx + sgn * p.radius * math.sin(h)
                y = cy - sgn * p.radius * math.cos(h)
                pts.append((x, y)); hs.append(h); hws.append(p.halfwidth)
        else:
            raise ValueError(p.kind)
    pts = np.array(pts)
    hs = np.array(hs)
    hws = np.array(hws, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # smooth width transitions with a box filter (keep the ends exact)
    k = max(1, int(round(smooth_m / ds)))
    if k > 1:
        pad = np.pad(hws, (k, k), mode="edge")
        kern = np.ones(2 * k + 1) / (2 * k + 1)
        hws = np.convolve(pad, kern, mode="valid")
    return pts, hs, hws, s


@dataclass
class Track:
    name: str
    vehicle_key: str
    template: str
    difficulty: str
    seed: int
    primitives: list[Primitive]
    centerline: np.ndarray
    headings: np.ndarray
    halfwidth: np.ndarray
    s: np.ndarray
    start_state: np.ndarray
    dock_length: float
    heading_tol_deg: float = 12.0
    par: dict | None = None
    meta: dict = field(default_factory=dict)
    # asymmetric corridors: extra room on one side (lay-bys, a yard beside the bay); None = symmetric (halfwidth)
    hw_left: np.ndarray | None = None
    hw_right: np.ndarray | None = None

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_primitives(cls, name, spec: VehicleSpec, template, difficulty, seed, primitives,
                        dock_slack: float, start_margin: float = 0.8, rotate: float = 0.0,
                        heading_tol_deg: float = 12.0, meta=None, smooth_m: float = 3.0) -> "Track":
        pts, hs, hws, s = build_centerline(primitives, smooth_m=smooth_m)
        if rotate:
            c, sn = math.cos(rotate), math.sin(rotate)
            pts = pts @ np.array([[c, sn], [-sn, c]])
            hs = hs + rotate
        # vehicle faces against the route; rear axle a margin + body_front into the course
        s_r = start_margin + spec.body_front
        p0, h0 = pts[0], hs[0]
        start = np.array([p0[0] + s_r * math.cos(h0), p0[1] + s_r * math.sin(h0), wrap(h0 + math.pi), wrap(h0 + math.pi)])
        dock_len = spec.trailer_front + spec.trailer_rear + dock_slack
        meta = dict(meta or {})
        meta.update({"rotate": rotate, "smooth_m": smooth_m})
        return cls(name=name, vehicle_key=spec.key, template=template, difficulty=difficulty, seed=seed,
                   primitives=primitives, centerline=pts, headings=hs, halfwidth=hws, s=s,
                   start_state=start, dock_length=dock_len, heading_tol_deg=heading_tol_deg, meta=meta)

    # ---- derived geometry ---------------------------------------------------
    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def normals(self) -> np.ndarray:
        return np.stack([-np.sin(self.headings), np.cos(self.headings)], axis=1)

    @property
    def hw_l(self) -> np.ndarray:
        return self.halfwidth if self.hw_left is None else self.hw_left

    @property
    def hw_r(self) -> np.ndarray:
        return self.halfwidth if self.hw_right is None else self.hw_right

    @property
    def left_rail(self) -> np.ndarray:  # left of the route direction
        return self.centerline + self.hw_l[:, None] * self.normals

    @property
    def right_rail(self) -> np.ndarray:
        return self.centerline - self.hw_r[:, None] * self.normals

    def set_asymmetry(self, extra_left: np.ndarray, extra_right: np.ndarray) -> None:
        """Widen one side or the other along the route. The route (and the bay, centred on it) stay where they are,
        so the middle of the rails is no longer where you need to be."""
        self.hw_left = self.halfwidth + np.asarray(extra_left, float)
        self.hw_right = self.halfwidth + np.asarray(extra_right, float)
        for k in ("_segs", "_grid", "_rail_samples", "_perc_walls"):
            self.__dict__.pop(k, None)

    @property
    def dock(self) -> dict:
        h = float(self.headings[-1])
        end = self.centerline[-1]
        c = end - 0.5 * self.dock_length * np.array([math.cos(h), math.sin(h)])
        return {"center": c, "heading": h, "length": self.dock_length, "halfwidth": float(self.halfwidth[-1])}

    @property
    def dock_corners(self) -> np.ndarray:
        d = self.dock
        from .geometry import rect_corners

        return rect_corners(d["center"][0], d["center"][1], d["heading"], d["length"] / 2, d["length"] / 2, d["halfwidth"])

    @property
    def goal_s(self) -> float:
        """Arc length at which the trailer axle sits when the trailer body is centred in the bay."""
        from .vehicles import get_vehicle

        sp = get_vehicle(self.vehicle_key)
        return self.length - 0.5 * self.dock_length + 0.5 * (sp.trailer_front - sp.trailer_rear)

    def rail_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """All rail segments (K,2,2) and a label per segment.

        Labels: 0 = route-left rail, 1 = route-right rail, 2 = start wall, 3 = dock wall.
        """
        if getattr(self, "_segs", None) is None:
            L, R = self.left_rail, self.right_rail
            segs = [np.stack([L[:-1], L[1:]], 1), np.stack([R[:-1], R[1:]], 1),
                    np.array([[R[0], L[0]]]), np.array([[R[-1], L[-1]]])]
            labels = [np.zeros(len(L) - 1, int), np.ones(len(R) - 1, int), np.array([2]), np.array([3])]
            self._segs = np.concatenate(segs)
            self._seg_labels = np.concatenate(labels)
            lo = np.minimum(self._segs[:, 0], self._segs[:, 1])
            hi = np.maximum(self._segs[:, 0], self._segs[:, 1])
            self._seg_bbox = np.concatenate([lo, hi], axis=1)
        return self._segs, self._seg_labels

    def segments_near(self, lo: np.ndarray, hi: np.ndarray, pad: float = 0.0):
        segs, labels = self.rail_segments()
        b = self._seg_bbox
        m = (b[:, 0] <= hi[0] + pad) & (b[:, 2] >= lo[0] - pad) & (b[:, 1] <= hi[1] + pad) & (b[:, 3] >= lo[1] - pad)
        return segs[m], labels[m]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        allp = np.concatenate([self.left_rail, self.right_rail])
        return allp.min(0), allp.max(0)

    # ---- progress along the route ------------------------------------------
    def project(self, p: np.ndarray, hint_s: float | None = None, window: float = 15.0) -> tuple[float, float]:
        """Arc length and signed lateral offset (+ = route-left) of point p."""
        P = self.centerline
        if hint_s is None:
            lo, hi = 0, len(P) - 1
        else:
            lo = int(np.searchsorted(self.s, hint_s - window))
            hi = int(np.searchsorted(self.s, hint_s + window))
            lo, hi = max(0, lo - 1), min(len(P) - 1, hi + 1)
        A, B = P[lo:hi], P[lo + 1:hi + 1]
        if len(A) == 0:
            return float(self.s[lo]), 0.0
        AB = B - A
        t = np.clip(np.einsum("ij,ij->i", p - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12), 0, 1)
        Q = A + AB * t[:, None]
        d = np.linalg.norm(Q - p, axis=1)
        i = int(np.argmin(d))
        s = float(self.s[lo + i] + t[i] * np.linalg.norm(AB[i]))
        n = np.array([-AB[i, 1], AB[i, 0]]) / max(np.linalg.norm(AB[i]), 1e-12)
        return s, float(np.dot(p - Q[i], n))

    def is_parked(self, trailer_corners: np.ndarray, trailer_heading: float) -> bool:
        d = self.dock
        inside = points_in_rect(trailer_corners, d["center"], d["heading"], d["length"] / 2, d["halfwidth"], tol=0.02)
        aligned = abs(wrap(trailer_heading + math.pi - d["heading"])) <= math.radians(self.heading_tol_deg)
        return bool(inside.all() and aligned)

    # ---- phases (a plain course has one: reverse the trailer into the bay) ------
    # Multi-phase courses (turnaround.py) override these; the simulator and the observation only use this API.
    task = "relay"
    walls = None  # list of wall polylines, for courses that are not a single railed corridor

    @property
    def n_phases(self) -> int:
        return 1

    def phase_info(self, phase: int) -> dict:
        return {"name": "reverse into the bay", "direction": -1, "ref": "trailer"}

    def progress_of(self, state, spec, phase: int, hint_s: float | None = None) -> tuple[float, float]:
        """(distance along the whole manoeuvre, comparable with goal_s; projection hint for the next call)."""
        from .kinematics import trailer_axle

        s, _ = self.project(trailer_axle(state, spec), hint_s=hint_s)
        return s, s

    def phase_done(self, state, spec, phase: int) -> bool:
        from .geometry import trailer_corners

        return self.is_parked(trailer_corners(state, spec), state[3])

    def target_corners(self, phase: int) -> np.ndarray:
        return self.dock_corners

    def route(self, phase: int):
        """(points, arclength, offset): the route the arrow points along in this phase; progress = offset + s."""
        return self.centerline, self.s, 0.0

    finish_reason = "trailer parked in the handoff bay"

    # ---- clearance grid (for the autopilot / analysis) ----------------------
    def grid(self, max_cells: int = 700):
        if getattr(self, "_grid", None) is None:
            self._grid = ClearanceGrid(self, max_cells=max_cells)
        return self._grid

    # ---- serialisation ------------------------------------------------------
    def to_dict(self, include_geometry: bool = True) -> dict:
        d = {
            "name": self.name, "vehicle": self.vehicle_key, "template": self.template,
            "difficulty": self.difficulty, "seed": self.seed,
            "primitives": [p.to_dict() for p in self.primitives],
            "start_state": [float(v) for v in self.start_state],
            "dock_length": self.dock_length, "heading_tol_deg": self.heading_tol_deg,
            "length": self.length, "goal_s": self.goal_s, "par": self.par, "meta": self.meta,
        }
        if include_geometry:
            r = lambda a: np.round(a, 3).tolist()  # noqa: E731
            d["geometry"] = {
                "centerline": r(self.centerline), "headings": r(self.headings), "halfwidth": r(self.halfwidth),
                "s": r(self.s), "left_rail": r(self.left_rail), "right_rail": r(self.right_rail),
                "dock_corners": r(self.dock_corners),
            }
            if self.hw_left is not None:
                d["geometry"]["hw_left"], d["geometry"]["hw_right"] = r(self.hw_left), r(self.hw_right)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        if d.get("template") == "turn":
            from .turnaround import TurnTrack

            return TurnTrack.from_dict(d)
        prims = [Primitive(**p) for p in d["primitives"]]
        g = d.get("geometry")
        if g:
            pts, hs, hws, s = (np.array(g[k], dtype=float) for k in ("centerline", "headings", "halfwidth", "s"))
        else:
            meta = d.get("meta", {})
            pts, hs, hws, s = build_centerline(prims, smooth_m=meta.get("smooth_m", 3.0))
            rot = meta.get("rotate", 0.0)
            if rot:
                c, sn = math.cos(rot), math.sin(rot)
                pts = pts @ np.array([[c, sn], [-sn, c]])
                hs = hs + rot
        tr = cls(name=d["name"], vehicle_key=d["vehicle"], template=d["template"], difficulty=d["difficulty"],
                 seed=d["seed"], primitives=prims, centerline=pts, headings=hs, halfwidth=hws, s=s,
                 start_state=np.array(d["start_state"], dtype=float), dock_length=d["dock_length"],
                 heading_tol_deg=d.get("heading_tol_deg", 12.0), par=d.get("par"), meta=d.get("meta", {}))
        if g and "hw_left" in g:
            tr.hw_left, tr.hw_right = np.array(g["hw_left"], float), np.array(g["hw_right"], float)
        return tr


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class ClearanceGrid:
    """Raster of signed clearance to the rails, route arc length and route heading.

    Used by the autopilot for fast vectorised collision look-ups. The exact
    collision test in the simulator uses segment intersection instead.
    """

    def __init__(self, track: Track, max_cells: int = 700):
        lo, hi = track.bbox()
        lo = lo - 3.0
        hi = hi + 3.0
        res = max(0.04, float(np.max(hi - lo)) / max_cells)
        nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
        ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
        self.origin, self.res, self.shape = lo, res, (nx, ny)
        # coarse centreline for the distance transform
        step = max(1, int(round(0.5 / max(np.diff(track.s).mean(), 1e-6))))
        idx = np.unique(np.concatenate([np.arange(0, len(track.s), step), [len(track.s) - 1]]))
        P, S, H = track.centerline[idx], track.s[idx], track.headings[idx]
        WL, WR = track.hw_l[idx], track.hw_r[idx]
        A, B = P[:-1], P[1:]
        AB = B - A
        L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)
        xs = lo[0] + res * np.arange(nx)
        ys = lo[1] + res * np.arange(ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([gx.ravel(), gy.ravel()], 1)
        clear = np.empty(len(G), np.float32)
        sarr = np.empty(len(G), np.float32)
        harr = np.empty(len(G), np.float32)
        nseg = len(A)
        chunk = max(256, int(4e6 // max(nseg, 1)))
        for i0 in range(0, len(G), chunk):
            g = G[i0:i0 + chunk]
            d = g[:, None, :] - A[None]
            t = np.einsum("pij,ij->pi", d, AB) / L2
            tc = np.clip(t, 0, 1)
            q = A[None] + tc[..., None] * AB[None]
            dist = np.linalg.norm(g[:, None, :] - q, axis=2)
            j = np.argmin(dist, axis=1)
            r = np.arange(len(g))
            tj, tcj = t[r, j], tc[r, j]
            left = _cross2(AB[j], g - A[j]) >= 0
            wl = WL[j] + (WL[j + 1] - WL[j]) * tcj
            wr = WR[j] + (WR[j + 1] - WR[j]) * tcj
            c = np.where(left, wl, wr) - dist[r, j]
            beyond = ((j == 0) & (tj < 0)) | ((j == nseg - 1) & (tj > 1))
            c = np.where(beyond, -np.abs(c) - 1.0, c)
            clear[i0:i0 + chunk] = c
            sarr[i0:i0 + chunk] = S[j] + (S[j + 1] - S[j]) * tcj
            harr[i0:i0 + chunk] = H[j] + (H[j + 1] - H[j]) * tcj
        self.clear = clear.reshape(nx, ny)
        self.s = sarr.reshape(nx, ny)
        self.heading = harr.reshape(nx, ny)

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
