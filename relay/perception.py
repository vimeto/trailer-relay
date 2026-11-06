"""Driver's-seat perception (the "driver" observation level): what a human driver could actually make out.

The other observation levels hand the driver exact rail points in the trailer's own frame. Here the driver has
a head, two mirrors and (car, truck) a reversing camera on the trailer's rear, and every number is a perception:

  HEAD    The driver sits in the vehicle (car / truck: on the LEFT, as in Finland; tractor / mower: centred) and
          turns the head (`look`) at the vehicle's head rate between vehicle-specific limits: you cannot look
          through the back of your own head, so going from over-the-left-shoulder to over-the-right-shoulder sweeps
          through straight ahead. While the head is turning (and for SACCADE_S after it settles) nothing is seen.
  DIRECT  a 120-degree cone around the gaze, out to the vehicle's view range, from the driver's eye. The trailer
          body hides what is behind it; a truck's cab hides everything further back than CAB_BLOCK_DEG.
  MIRRORS looking AT a mirror (gaze within GLANCE_TOL_DEG of it) adds the mirror's view: a narrow cone from the
          mirror backwards along that side of the rig (convex right mirror: wider, distances worse).
  CAMERA  looking at the dashboard screen adds the reversing camera: a wide, short-range view straight back from
          the trailer's rear (it moves with the trailer).
  PERIPHERY the edge of vision: beyond the direct cone, out to +-PERIPHERY_HALF_DEG around the gaze, but only close by
          (the level's periphery range, a few metres) and very roughly - enough to notice that the front corner is
          about to touch a rail, not to steer by.
  TARGET  whether there is a line of sight from the eye to the centre of the current target (bay / zone) - the
          barriers and the trailer body block it - and if so roughly where it is, whatever you are looking at.

Every seen point is reported as [bearing_deg, distance_m]: the bearing in the VEHICLE frame (0 = straight ahead,
+ = left, +-180 = straight back) as seen from that view's origin (eye, mirror or camera; also in the vehicle frame).
Distances get worse with range, like human depth judgement: a few cm at 2 m, about half a metre at 10 m. The error
is mostly a slowly drifting bias per view (a filter cannot average it away), plus a little per-glance jitter, and
the number is rounded more coarsely the further away it is. The rig itself (hitch, trailer pose relative to the
vehicle) is known exactly: the driver knows their own vehicle and can feel the hitch angle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import zlib

import numpy as np

from .geometry import segments_intersect, trailer_corners
from .kinematics import hitch_point, trailer_rear_center
from .vehicles import VehicleSpec

PERCEPTION_VERSION = 2  # 1: head, mirrors, camera, depth noise, target LOS; 2: + peripheral vision, difficulty levels
GLANCE_TOL_DEG = 7.0  # gaze within this of a mirror / the screen = looking at it
SETTLE_DEG = 1.0  # head counts as still when this close to where it is going
SACCADE_S = 0.1  # vision is suppressed while the head turns and for this long after
BIAS_TAU_S = 3.0  # correlation time of the per-view distance / bearing bias
BIAS_DT = 0.1
SOURCES = ("direct", "left_mirror", "right_mirror", "camera", "target", "periphery")
PERIPHERY_HALF_DEG = 100.0  # human horizontal field: about +-100 degrees around the gaze


@dataclass(frozen=True)
class Level:
    """Difficulty dial for driver's-seat perception."""
    noise: float  # multiplier on every distance and bearing error
    camera_range_mult: float  # x the vehicle's sensor range
    periphery_range: float  # m; 0 = no peripheral vision (version 1)


LEVELS = {
    "easy": Level(noise=0.35, camera_range_mult=2.0, periphery_range=5.0),
    "medium": Level(noise=0.6, camera_range_mult=1.5, periphery_range=4.0),
    "hard": Level(noise=1.0, camera_range_mult=1.0, periphery_range=3.0),
    "v1": Level(noise=1.0, camera_range_mult=1.0, periphery_range=0.0),  # what the first driver runs had
}
DEFAULT_LEVEL = "medium"  # calibrated 2026-11-06: Sol parks about half the legs


@dataclass(frozen=True)
class Mirror:
    name: str  # left_mirror | right_mirror
    pos: tuple  # (x, y) in the vehicle frame (x forward from the rear axle, y left)
    axis_deg: float  # where the mirror looks, vehicle frame (about +-170 = backwards along that side)
    fov_deg: float
    noise: float  # distance-error multiplier (convex mirrors: things look further away)


@dataclass(frozen=True)
class Rig:
    eye: tuple  # (x, y) vehicle frame
    gaze_min: float  # head limits (deg, vehicle frame)
    gaze_max: float
    fov_deg: float
    range: float
    cab_block_deg: float | None  # direct view blocked beyond this |bearing| (truck sleeper cab)
    mirrors: tuple = ()
    screen: tuple | None = None  # (x, y) of the dashboard screen (None: no reversing camera)
    camera_fov_deg: float = 130.0
    camera_range: float = 0.0
    mirror_gaze: dict = field(default_factory=dict)  # mirror name -> gaze direction from the eye
    screen_gaze: float | None = None
    level: str = DEFAULT_LEVEL
    noise: float = 1.0
    periphery_deg: float = PERIPHERY_HALF_DEG
    periphery_range: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mirrors"] = [asdict(m) for m in self.mirrors]
        return d


def _dir(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def rig_for(spec: VehicleSpec, level: str = DEFAULT_LEVEL) -> Rig:
    lv = LEVELS[level]
    bf, w = spec.body_front, spec.body_width
    if spec.key == "car":  # left-hand drive; rear window, but the trailer fills it
        eye, lim, block = (0.45 * bf, 0.22 * w), (-160.0, 160.0), None
        mx = bf - 1.3
        mirrors = (Mirror("left_mirror", (mx, w / 2 + 0.12), 180 - 11 + 2, 22.0, 1.4),
                   Mirror("right_mirror", (mx, -(w / 2 + 0.12)), -(180 - 14 + 2), 28.0, 1.8))
        screen, cam_range = (bf - 1.0, 0.0), spec.sensor_range
    elif spec.key == "truck":  # sleeper cab: no view backwards except in the mirrors; wide-angle mirrors
        eye, lim, block = (bf - 1.6, w / 2 - 0.67), (-120.0, 120.0), 100.0
        mx = bf - 0.9
        mirrors = (Mirror("left_mirror", (mx, w / 2 + 0.25), 180 - 18 + 2, 36.0, 1.5),
                   Mirror("right_mirror", (mx, -(w / 2 + 0.25)), -(180 - 18 + 2), 36.0, 1.8))
        screen, cam_range = (bf - 0.9, 0.0), spec.sensor_range
    elif spec.key == "tractor":  # centre seat, glass cab, mirrors, no camera on a farm trailer
        eye, lim, block = (0.25 * bf, 0.0), (-170.0, 170.0), None
        mx = bf - 1.6
        mirrors = (Mirror("left_mirror", (mx, w / 2 + 0.2), 180 - 15 + 2, 30.0, 1.5),
                   Mirror("right_mirror", (mx, -(w / 2 + 0.2)), -(180 - 15 + 2), 30.0, 1.5))
        screen, cam_range = None, 0.0
    else:  # ride-on mower (and anything else): open seat, twist round, no mirrors, no camera
        eye, lim, block = (0.1, 0.0), (-170.0, 170.0), None
        mirrors, screen, cam_range = (), None, 0.0
    return Rig(eye=eye, gaze_min=lim[0], gaze_max=lim[1], fov_deg=spec.view_fov_deg, range=spec.view_range, cab_block_deg=block,
               mirrors=mirrors, screen=screen, camera_range=round(cam_range * lv.camera_range_mult, 2),
               mirror_gaze={m.name: round(_dir(eye, m.pos), 1) for m in mirrors},
               screen_gaze=None if screen is None else round(_dir(eye, screen), 1),
               level=level, noise=lv.noise, periphery_range=lv.periphery_range)


# ---------------------------------------------------------------------------- state attached to a Leg
class PerceptionState:
    """Per-leg noise state: an Ornstein-Uhlenbeck bias per (view, quantity), advanced on a fixed time grid, plus
    per-glance jitter that is a pure function of (seed, time step, view), so re-observing is idempotent."""

    def __init__(self, seed: int):
        self.seed = int(seed) & 0x7FFFFFFF
        self.rng = np.random.default_rng(self.seed)
        self.z = self.rng.standard_normal((len(SOURCES), 2))
        self.n = 0

    def bias(self, t: float) -> np.ndarray:
        n = int(round(t / BIAS_DT))
        if n > self.n:
            a = math.exp(-BIAS_DT * (n - self.n) / BIAS_TAU_S)
            self.z = a * self.z + math.sqrt(max(0.0, 1 - a * a)) * self.rng.standard_normal(self.z.shape)
            self.n = n
        return self.z

    def jitter(self, t: float, source: str, n: int) -> np.ndarray:
        g = np.random.default_rng([self.seed, int(round(t / BIAS_DT)), SOURCES.index(source)])
        return g.standard_normal((n, 2))


def attach(leg, seed: int | None = None, level: str | None = None) -> Rig:
    """Switch a leg to driver's-seat perception: head limits, start looking ahead, record the gaze in the trace."""
    if getattr(leg, "rig", None) is not None:
        return leg.rig
    rig = rig_for(leg.spec, level or DEFAULT_LEVEL)
    leg.rig = rig
    leg.head_limits = (rig.gaze_min, rig.gaze_max)
    leg.gaze_deg = leg.gaze_target = 0.0
    leg._head_moved_t = -1e9
    if seed is None:
        seed = zlib.crc32(f"{leg.track.name}|{leg.track.seed}".encode())
    leg.perc = PerceptionState(seed)
    n = len(leg.trace["t"])
    leg.trace["gaze"] = [0.0] * n
    return rig


# ---------------------------------------------------------------------------- geometry helpers
def wall_polylines(track) -> list[np.ndarray]:
    if track.walls is not None:
        return [np.asarray(w, float) for w in track.walls]
    L, R = track.left_rail, track.right_rail
    return [L, R, np.array([R[0], L[0]]), np.array([R[-1], L[-1]])]


def _resample(poly: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / spacing) + 1)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], 1)


def wall_points(track, spec: VehicleSpec) -> np.ndarray:
    """Barrier points every ~0.25-1 m along every wall (cached on the track)."""
    cache = track.__dict__.setdefault("_perc_walls", {})
    if spec.key not in cache:
        sp = float(np.clip(spec.combo_length / 20.0, 0.25, 1.0))
        cache[spec.key] = np.concatenate([_resample(p, sp) for p in wall_polylines(track)])
    return cache[spec.key]


def _blocked(origin: np.ndarray, pts: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Does the sight line origin -> point cross any of segs (K,2,2)?"""
    if len(pts) == 0 or len(segs) == 0:
        return np.zeros(len(pts), bool)
    A = np.stack([np.broadcast_to(origin, pts.shape), pts], 1)
    return segments_intersect(A, segs).any(axis=1)


def _quant(d: np.ndarray) -> np.ndarray:
    step = np.where(d < 3, 0.05, np.where(d < 6, 0.1, np.where(d < 12, 0.25, 0.5)))
    return np.round(d / step) * step


def _sigma_rel(d: np.ndarray, kind: str) -> np.ndarray:
    if kind == "camera":
        return np.minimum(0.15, 0.02 + 0.012 * d)
    if kind == "periphery":
        return np.minimum(0.3, 0.08 + 0.03 * d)
    return np.minimum(0.12, 0.012 + 0.0045 * d)


class _Frame:
    """World <-> vehicle frame for the current state."""

    def __init__(self, state):
        self.p = np.asarray(state[:2], float)
        self.h = float(state[2])
        c, s = math.cos(self.h), math.sin(self.h)
        self.R = np.array([[c, -s], [s, c]])

    def to_world(self, xy) -> np.ndarray:
        return self.p + self.R @ np.asarray(xy, float)

    def to_vehicle(self, P) -> np.ndarray:
        return (np.atleast_2d(P) - self.p) @ self.R


def _wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------- one view
def _view(leg, fr: _Frame, kind: str, origin_v, axis_deg: float, fov: float, rng: float, noise: float,
          occluders: np.ndarray, walls_w: np.ndarray, zone_w: np.ndarray, wall_segs: np.ndarray, bias: np.ndarray,
          cab_block: float | None = None, min_ecc: float = 0.0) -> dict:
    o_w = fr.to_world(origin_v)
    out = {"origin": [round(float(origin_v[0]), 2), round(float(origin_v[1]), 2)], "axis_deg": round(float(_wrap_deg(axis_deg)), 1),
           "fov_deg": fov, "range": round(rng, 1)}
    src = SOURCES.index(kind)
    for key, P, ground in (("walls", walls_w, False), ("target_corners", zone_w, True)):
        if len(P) == 0:
            out[key] = []
            continue
        d = P - o_w
        dist = np.hypot(d[:, 0], d[:, 1])
        bear = _wrap_deg(np.degrees(np.arctan2(d[:, 1], d[:, 0]) - fr.h))  # vehicle frame
        ecc = np.abs(_wrap_deg(bear - axis_deg))
        m = (dist <= rng) & (dist > 0.05) & (ecc <= fov / 2) & (ecc > min_ecc)
        if cab_block is not None:
            m &= np.abs(bear) <= cab_block
        idx = np.nonzero(m)[0]
        if len(idx) and len(occluders):
            idx = idx[~_blocked(o_w, P[idx], occluders)]
        if ground and len(idx):  # painted markings: hidden behind barriers (test a point just short of the corner)
            u = (P[idx] - o_w) / np.maximum(dist[idx], 1e-9)[:, None]
            near = P[idx] - 0.15 * u
            idx = idx[~_blocked(o_w, near, wall_segs)]
        if not ground and len(idx):  # far away you make out fewer separate points; the periphery is coarser still
            k = 1 + (dist[idx] / max(4.0, 0.35 * rng)).astype(int)
            if kind == "direct":
                k = k * np.where(ecc[idx] > 35, 2, 1)
            elif kind == "periphery":
                k = k * 2
            idx = idx[(idx % k) == 0]
        if not len(idx):
            out[key] = []
            continue
        jit = leg.perc.jitter(leg.t, kind, len(P))[idx]
        dd = dist[idx]
        sig = _sigma_rel(dd, kind) * noise * leg.rig.noise
        dd = dd * np.exp(sig * (0.8 * bias[src, 0] + 0.6 * jit[:, 0]))
        if kind == "direct":
            sb = 0.3 + 3.0 * (ecc[idx] / 60.0) ** 2
        elif kind == "camera":
            sb = 0.8 + 2.5 * (ecc[idx] / 65.0) ** 2
        elif kind == "periphery":
            sb = 4.0 * np.ones(len(idx))
        else:
            sb = 0.5 * noise * np.ones(len(idx))
        bb = _wrap_deg(bear[idx] + sb * leg.rig.noise * (0.5 * bias[src, 1] + jit[:, 1]))
        order = np.argsort(bb) if ground else np.arange(len(idx))
        qd = np.round(dd / 0.25) * 0.25 if kind == "periphery" else _quant(dd)
        nd = 0 if kind == "periphery" else 1
        out[key] = [[round(float(b), nd), round(float(q), 2)] for b, q in zip(bb[order], qd[order])]
    return out


def perceive(leg) -> dict:
    """The perception part of the observation (merged into observe() for level "driver")."""
    rig = attach(leg)
    sp, st = leg.spec, leg.state
    fr = _Frame(st)
    bias = leg.perc.bias(leg.t)
    settled = abs(leg.gaze_target - leg.gaze_deg) <= SETTLE_DEG and leg.t - leg._head_moved_t >= SACCADE_S - 1e-9
    gaze = float(leg.gaze_deg)
    looking_at = None
    for m in rig.mirrors:
        if abs(gaze - rig.mirror_gaze[m.name]) <= GLANCE_TOL_DEG:
            looking_at = m.name
    if rig.screen_gaze is not None and abs(gaze - rig.screen_gaze) <= GLANCE_TOL_DEG:
        looking_at = "camera"
    hp = fr.to_vehicle(hitch_point(st, sp))[0]
    tr_rear = fr.to_vehicle(trailer_rear_center(st, sp))[0]
    trailer_rel = math.degrees(st[3] - st[2])
    trailer_rel = float(_wrap_deg(trailer_rel))
    out = {
        "head": {"gaze_deg": round(gaze, 1), "target_deg": round(float(leg.gaze_target), 1), "min_deg": rig.gaze_min,
                 "max_deg": rig.gaze_max, "rate_deg_s": sp.head_rate_deg_per_s, "settled": bool(settled), "looking_at": looking_at},
        "rig": {"eye": list(rig.eye), "hitch": [round(float(hp[0]), 3), round(float(hp[1]), 3)],
                "trailer_rear": [round(float(tr_rear[0]), 3), round(float(tr_rear[1]), 3)],
                "trailer_heading_deg": round(trailer_rel, 2), "trailer_half_width": sp.trailer_width / 2,
                "vehicle_front": sp.body_front, "vehicle_half_width": sp.body_width / 2,
                "mirrors": dict(rig.mirror_gaze), "screen_gaze_deg": rig.screen_gaze},
        "views": {},
    }
    track = leg.track
    segs = track.rail_segments()[0]
    tcorn = trailer_corners(st, sp)
    t_edges = np.stack([tcorn, np.roll(tcorn, -1, axis=0)], 1)
    zone_w = np.asarray(track.target_corners(leg.phase), float)
    eye_w = fr.to_world(rig.eye)
    # target line of sight (independent of the gaze)
    tc = zone_w.mean(0)
    los = not (_blocked(eye_w, tc[None], segs)[0] or _blocked(eye_w, tc[None], t_edges)[0])
    if los:
        d = tc - eye_w
        dist = float(np.hypot(*d))
        bear = float(_wrap_deg(math.degrees(math.atan2(d[1], d[0]) - fr.h)))
        jit = leg.perc.jitter(leg.t, "target", 1)[0]
        dist *= math.exp(min(0.15, 0.03 + 0.006 * dist) * (0.8 * bias[4, 0] + 0.6 * jit[0]))
        bear += 1.5 * (0.5 * bias[4, 1] + jit[1])
        step = 0.5 if dist < 15 else 1.0
        out["target"] = {"in_sight": True, "bearing_deg": round(float(_wrap_deg(bear)), 0), "dist_m": round(dist / step) * step}
    else:
        out["target"] = {"in_sight": False}
    if not settled:
        return out
    W = wall_points(track, sp)
    out["views"]["direct"] = _view(leg, fr, "direct", rig.eye, gaze, rig.fov_deg, rig.range, 1.0, t_edges, W, zone_w, segs, bias,
                                   cab_block=rig.cab_block_deg)
    if rig.periphery_range > 0:
        out["views"]["periphery"] = _view(leg, fr, "periphery", rig.eye, gaze, 2 * rig.periphery_deg, rig.periphery_range, 1.0,
                                          t_edges, W, np.zeros((0, 2)), segs, bias, cab_block=rig.cab_block_deg,
                                          min_ecc=rig.fov_deg / 2)
    for m in rig.mirrors:
        if looking_at == m.name:
            out["views"][m.name] = _view(leg, fr, m.name, m.pos, m.axis_deg, m.fov_deg, rig.range, m.noise, t_edges, W, zone_w, segs, bias)
    if looking_at == "camera":
        cam_axis = 180.0 + trailer_rel
        out["views"]["camera"] = _view(leg, fr, "camera", tr_rear, cam_axis, rig.camera_fov_deg, rig.camera_range, 1.0,
                                       np.zeros((0, 2, 2)), W, zone_w, segs, bias)
    return out


# ---------------------------------------------------------------------------- text
VIEW_TITLES = {"direct": "DIRECT VIEW (from your eye)", "left_mirror": "LEFT MIRROR", "right_mirror": "RIGHT MIRROR",
               "camera": "REVERSING CAMERA (on the trailer's rear)", "periphery": "CORNER OF YOUR EYE (outside the direct view; very rough, close by only)"}


def _pts(p) -> str:
    return " ".join(f"{b:+.0f}°/{d:g}m" for b, d in p) if p else "-"


def render(obs: dict, turn: bool) -> list[str]:
    h, rg, tg = obs["head"], obs["rig"], obs["target"]
    at = {"left_mirror": "the LEFT MIRROR", "right_mirror": "the RIGHT MIRROR", "camera": "the reversing-camera SCREEN"}.get(h["looking_at"])
    lines = [f"HEAD: gaze {h['gaze_deg']:+.0f}° (0 ahead, + left; you can turn {h['min_deg']:+.0f}° to {h['max_deg']:+.0f}°)"
             + (f", looking at {at}" if at else "") + ("" if h["settled"] else f", STILL TURNING towards {h['target_deg']:+.0f}° (nothing seen while turning)")]
    lines.append(f"RIG (vehicle frame: x m ahead of your rear axle, y m to your left): hitch ({rg['hitch'][0]:+.2f}, {rg['hitch'][1]:+.2f}); "
                 f"trailer rear centre ({rg['trailer_rear'][0]:+.2f}, {rg['trailer_rear'][1]:+.2f}); trailer points {rg['trailer_heading_deg']:+.1f}° "
                 f"off your heading")
    zone = ("target zone" if turn else "bay") + " corners"
    if tg["in_sight"]:
        lines.append(f"TARGET: in sight, bearing {tg['bearing_deg']:+.0f}° from your eye (vehicle frame), roughly {tg['dist_m']:g} m away")
    else:
        lines.append("TARGET: no line of sight (barriers or your trailer are in the way)")
    if h["settled"]:
        lines.append("SEEN (each point = bearing in the vehicle frame / distance from that view's origin; far distances are rough):")
        for name, v in obs["views"].items():
            lines.append(f"  {VIEW_TITLES[name]} origin ({v['origin'][0]:+.2f}, {v['origin'][1]:+.2f}), looking {v['axis_deg']:+.0f}° "
                         f"±{v['fov_deg'] / 2:g}°, up to {v['range']:g} m:")
            lines.append(f"    {'kerbs' if turn else 'rails'}: {_pts(v['walls'])}")
            if v["target_corners"]:
                lines.append(f"    {zone} (painted): {_pts(v['target_corners'])}")
    return lines
