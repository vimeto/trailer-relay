"""The driver's report: what the driver is told after every command.

The driver never sees the global state. It gets what a driver could perceive from
the seat and mirrors:
  * the result of the last command (moved, stopped by contact, blocked, ...),
  * clock / budget / counters and a status word,
  * steering and hitch angle (with the mechanical limit),
  * proximity sensors (distance to the nearest rail along fixed rays),
  * a LOCAL MAP: rail points within a radius, in a frame attached to the trailer,
  * a route arrow and the odometer-style progress along the course.

Observation levels (an ablation axis):
  "sensors": no local map
  "local":   default
  "sketch":  local + a small ASCII top-down sketch in the same egocentric frame
  "full":    additionally global poses + the whole course outline (debug/upper bound)
  "cone":    VISION ONLY: no parking sensors, no route arrow; the local map holds only what lies inside the
             driver's 120-degree view cone (from the driver's seat, out to view_range). The driver turns the
             cone with look(yaw_deg) and has to remember what it saw.
  "cone-sketch": cone + the ASCII sketch with unseen cells shown as '?'
  "driver":  DRIVER'S SEAT (perception.py): a head with limits, mirrors, a reversing camera, range-dependent distance
             errors, a line-of-sight target cue; no sensors, no route arrow, no exact map
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import points_in_rect, ray_distance
from .kinematics import trailer_axle, trailer_rear_center
from .sim import DriveResult, Leg

LEVELS = ("sensors", "local", "sketch", "full", "cone", "cone-sketch", "driver")
CONE_LEVELS = ("cone", "cone-sketch")


def _resample(poly: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / spacing) + 1)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], 1)


def map_spacing(leg: Leg) -> float:
    return float(np.clip(leg.spec.combo_length / 10.0, 0.3, 2.0))


def _rail_samples(leg: Leg):
    tr = leg.track
    if getattr(tr, "_rail_samples", None) is None:
        sp = map_spacing(leg)
        tr._rail_samples = (_resample(tr.right_rail, sp), _resample(tr.left_rail, sp))  # driver-left, driver-right
    return tr._rail_samples


def side_word(v: float) -> str:
    return "LEFT" if v > 0 else "RIGHT"


def _sensor_layout(leg: Leg):
    sp = leg.spec
    th, vh = sp.trailer_width / 2, sp.body_width / 2
    tmid = (sp.trailer_front - sp.trailer_rear) / 2
    vmid = (sp.body_front - sp.body_rear) / 2
    r2 = 1 / math.sqrt(2)
    trailer = [("rear-left", (-sp.trailer_rear, th), (-r2, r2)), ("rear", (-sp.trailer_rear, 0), (-1, 0)),
               ("rear-right", (-sp.trailer_rear, -th), (-r2, -r2)), ("left", (tmid, th), (0, 1)), ("right", (tmid, -th), (0, -1))]
    vehicle = [("front-left", (sp.body_front, vh), (r2, r2)), ("front", (sp.body_front, 0), (1, 0)),
               ("front-right", (sp.body_front, -vh), (r2, -r2)), ("left", (vmid, vh), (0, 1)), ("right", (vmid, -vh), (0, -1))]
    return trailer, vehicle


def _to_world(origin, heading, p, d):
    c, s = math.cos(heading), math.sin(heading)
    wp = np.array([origin[0] + p[0] * c - p[1] * s, origin[1] + p[0] * s + p[1] * c])
    wd = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
    return wp, wd


def sensors(leg: Leg) -> dict:
    sp = leg.spec
    st = leg.state
    rng = sp.sensor_range
    segs, _ = leg.track.rail_segments()
    lo = st[:2] - (sp.combo_length + rng)
    hi = st[:2] + (sp.combo_length + rng)
    near, _ = leg.track.segments_near(lo, hi)
    tlay, vlay = _sensor_layout(leg)
    ax = trailer_axle(st, sp)
    out = {"trailer": {}, "vehicle": {}}
    for name, p, d in tlay:
        wp, wd = _to_world(ax, st[3], p, d)
        out["trailer"][name] = round(ray_distance(wp, wd, near, rng), 2)
    for name, p, d in vlay:
        wp, wd = _to_world(st[:2], st[2], p, d)
        out["vehicle"][name] = round(ray_distance(wp, wd, near, rng), 2)
    out["range"] = rng
    return out


def rear_frame(leg: Leg):
    """Origin = trailer rear centre; u = metres behind the trailer; v = metres to the driver's left."""
    o = trailer_rear_center(leg.state, leg.spec)
    t2 = leg.state[3]
    eu = np.array([-math.cos(t2), -math.sin(t2)])
    ev = np.array([-math.sin(t2), math.cos(t2)])
    return o, eu, ev


def to_rear(points: np.ndarray, frame) -> np.ndarray:
    o, eu, ev = frame
    d = np.atleast_2d(points) - o
    return np.stack([d @ eu, d @ ev], 1)


def eye_point(leg: Leg) -> np.ndarray:
    """The driver's seat: a little ahead of the rear axle."""
    sp, st = leg.spec, leg.state
    return st[:2] + 0.4 * sp.body_front * np.array([math.cos(st[2]), math.sin(st[2])])


def gaze_heading(leg: Leg) -> float:
    return float(leg.state[2] + math.radians(leg.gaze_deg))


def visible(leg: Leg, pts: np.ndarray) -> np.ndarray:
    """Which world points lie inside the driver's view cone."""
    pts = np.atleast_2d(pts)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    d = pts - eye_point(leg)
    dist = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0]) - gaze_heading(leg)
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    return (dist <= leg.spec.view_range) & (np.abs(ang) <= math.radians(leg.spec.view_fov_deg) / 2)


def local_map(leg: Leg, cone: bool = False) -> dict:
    sp = leg.spec
    frame = rear_frame(leg)
    o = frame[0]
    R = sp.view_radius
    veh = leg.state[:2]
    dec = 2 if sp.combo_length < 5 else 1
    out = {}
    if leg.track.task == "turnaround":
        return _local_map_turn(leg, frame, R, dec, cone)
    for key, pts in zip(("left_rail", "right_rail"), _rail_samples(leg)):
        if cone:
            m = visible(leg, pts)
        else:
            m = (np.linalg.norm(pts - o, axis=1) <= R) | (np.linalg.norm(pts - veh, axis=1) <= 0.5 * R)
        sel = pts[m]
        uv = to_rear(sel, frame) if len(sel) else np.zeros((0, 2))
        out[key] = [[round(float(a), dec), round(float(b), dec)] for a, b in uv]
    dc = leg.track.dock_corners
    if cone:
        seen = visible(leg, dc)
        if seen.any():
            out["bay_corners"] = [[round(float(a), dec), round(float(b), dec)] for a, b in to_rear(dc[seen], frame)]
            if seen[0] and seen[1]:
                out["dock_wall"] = [[round(float(a), dec), round(float(b), dec)] for a, b in to_rear(dc[[0, 1]], frame)]
        out["radius"] = sp.view_range
        out["cone"] = True
        return out
    if (np.linalg.norm(dc - o, axis=1) <= R + leg.track.dock_length).any():
        uv = to_rear(dc, frame)
        out["bay_corners"] = [[round(float(a), dec), round(float(b), dec)] for a, b in uv]
        wall = to_rear(dc[[0, 1]], frame)  # leading edge along the route = the dock wall (route end)
        out["dock_wall"] = [[round(float(a), dec), round(float(b), dec)] for a, b in wall]
    out["radius"] = R
    return out


def _wall_samples(leg: Leg) -> np.ndarray:
    tr = leg.track
    if getattr(tr, "_wall_samples", None) is None:
        tr._wall_samples = _resample(tr.geo["walls"], map_spacing(leg))
    return tr._wall_samples


def _local_map_turn(leg: Leg, frame, R: float, dec: int, cone: bool) -> dict:
    """Turnaround courses have one closed kerb line, not two rails: kerb points are split by which side of
    the trailer's axis they lie on. bay_corners = the current phase's target zone."""
    o = frame[0]
    veh = leg.state[:2]
    pts = _wall_samples(leg)
    fwd = leg.track.phase_info(leg.phase)["direction"] > 0
    if cone:
        m = visible(leg, pts)
    else:  # driving forward the driver watches ahead of the vehicle as much as behind the trailer
        m = (np.linalg.norm(pts - o, axis=1) <= R) | (np.linalg.norm(pts - veh, axis=1) <= (R if fwd else 0.5 * R))
    uv = to_rear(pts[m], frame) if m.any() else np.zeros((0, 2))
    rnd = lambda a: [[round(float(x), dec), round(float(y), dec)] for x, y in a]  # noqa: E731
    out = {"left_rail": rnd(uv[uv[:, 1] > 0]), "right_rail": rnd(uv[uv[:, 1] <= 0])}
    tc = leg.track.target_corners(leg.phase)
    seen = visible(leg, tc) if cone else np.ones(len(tc), bool)
    if seen.any():
        out["bay_corners"] = rnd(to_rear(tc[seen], frame))
    out["radius"] = leg.spec.view_range if cone else R
    if cone:
        out["cone"] = True
    return out


def sketch(leg: Leg, max_rows: int = 28, max_cols: int = 25, cone: bool = False) -> dict:
    """ASCII top-down sketch in the rear frame. Top = further behind the trailer (where you are reversing to);
    the left column is the driver's LEFT. '#' rail, '.' drivable, ' ' outside the course, 'T' trailer, 'V' vehicle,
    'B' handoff bay, '+' trailer rear centre."""
    sp = leg.spec
    frame = rear_frame(leg)
    o, eu, ev = frame
    R = sp.view_radius
    u_lo, u_hi = -(sp.combo_length + 0.5), R
    cell = max(0.2, (u_hi - u_lo) / max_rows, 2 * (0.55 * R) / max_cols)
    nrow = int(math.ceil((u_hi - u_lo) / cell))
    ncol = max_cols | 1
    v_half = cell * ncol / 2
    us = u_hi - (np.arange(nrow) + 0.5) * cell
    vs = v_half - (np.arange(ncol) + 0.5) * cell
    U, V = np.meshgrid(us, vs, indexing="ij")
    W = o + U[..., None] * eu + V[..., None] * ev  # world coordinates of cell centres
    pts = W.reshape(-1, 2)
    grid = np.full(nrow * ncol, " ", dtype="<U1")
    inside = leg.track.grid().clearance(pts[:, 0], pts[:, 1]) > 0
    grid[inside] = "."
    st = leg.state
    dc = leg.track.target_corners(leg.phase)
    c = dc.mean(0)
    hdg = math.atan2(*(dc[1] - dc[2])[::-1])
    grid[points_in_rect(pts, c, hdg, np.linalg.norm(dc[1] - dc[2]) / 2, np.linalg.norm(dc[0] - dc[1]) / 2)] = "B"
    ax = trailer_axle(st, sp)
    tc = ax + (sp.trailer_front - sp.trailer_rear) / 2 * np.array([math.cos(st[3]), math.sin(st[3])])
    grid[points_in_rect(pts, tc, st[3], (sp.trailer_front + sp.trailer_rear) / 2, sp.trailer_width / 2 + 0.25 * cell)] = "T"
    vc = st[:2] + (sp.body_front - sp.body_rear) / 2 * np.array([math.cos(st[2]), math.sin(st[2])])
    grid[points_in_rect(pts, vc, st[2], (sp.body_front + sp.body_rear) / 2, sp.body_width / 2 + 0.25 * cell)] = "V"
    # rails, incl. start and dock walls
    segs, _ = leg.track.rail_segments()
    lo, hi = pts.min(0) - cell, pts.max(0) + cell
    near, _ = leg.track.segments_near(lo, hi)
    for a, b in near:
        n = max(2, int(np.linalg.norm(b - a) / (cell / 3)) + 1)
        q = to_rear(a + np.linspace(0, 1, n)[:, None] * (b - a), frame)
        r = np.floor((u_hi - q[:, 0]) / cell).astype(int)
        k = np.floor((v_half - q[:, 1]) / cell).astype(int)
        ok = (r >= 0) & (r < nrow) & (k >= 0) & (k < ncol)
        grid[r[ok] * ncol + k[ok]] = "#"
    if cone:
        # the driver knows where its own rig is, but the course only where it is looking
        unseen = ~visible(leg, pts) & (grid != "T") & (grid != "V")
        grid[unseen] = "?"
    r0, k0 = int((u_hi - 0) / cell), int(v_half / cell)
    if 0 <= r0 < nrow:
        grid[r0 * ncol + k0] = "+"
    rows = ["".join(grid[i * ncol:(i + 1) * ncol]) for i in range(nrow)]
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return {"cell": round(cell, 2), "rows": rows}


def route_arrow(leg: Leg) -> dict:
    """Bearing of the course centreline ~0.6 C beyond the trailer, relative to straight back."""
    tr, sp = leg.track, leg.spec
    if tr.task == "turnaround":
        return _route_arrow_turn(leg)
    s_look = min(tr.length, leg.progress_s + sp.trailer_rear + 0.6 * sp.combo_length)
    p = np.array([np.interp(s_look, tr.s, tr.centerline[:, 0]), np.interp(s_look, tr.s, tr.centerline[:, 1])])
    uv = to_rear(p, rear_frame(leg))[0]
    bearing = math.degrees(math.atan2(uv[1], uv[0]))
    return {"distance": round(float(np.hypot(*uv)), 1), "bearing_deg": round(bearing, 0)}


def _route_arrow_turn(leg: Leg) -> dict:
    """Phase-aware arrow. Reversing (phase 1): the route ~0.6 C beyond the trailer's rear, bearing from straight back.
    Driving forward (phase 2): the route ~0.6 C ahead of the vehicle's front, bearing from straight ahead."""
    tr, sp, st = leg.track, leg.spec, leg.state
    P, S, off = tr.route(leg.phase)
    s_local = leg.progress_s - off
    fwd = tr.phase_info(leg.phase)["direction"] > 0
    s_look = min(S[-1], s_local + (sp.body_front if fwd else sp.trailer_rear) + 0.6 * sp.combo_length)
    p = np.array([np.interp(s_look, S, P[:, 0]), np.interp(s_look, S, P[:, 1])])
    frame = rear_frame(leg)
    if fwd:
        h = st[2]
        base = st[:2] + sp.body_front * np.array([math.cos(h), math.sin(h)])
        d = p - base
        a, b = d[0] * math.cos(h) + d[1] * math.sin(h), -d[0] * math.sin(h) + d[1] * math.cos(h)
    else:
        a, b = to_rear(p, frame)[0]
    uv = to_rear(p, frame)[0]
    return {"distance": round(float(math.hypot(a, b)), 1), "bearing_deg": round(math.degrees(math.atan2(b, a)), 0),
            "from": "vehicle front, 0 = straight ahead" if fwd else "trailer rear, 0 = straight back",
            "direction": "forward" if fwd else "reverse", "point_uv": [round(float(uv[0]), 1), round(float(uv[1]), 1)]}


def observe(leg: Leg, level: str = "local") -> dict:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    sp = leg.spec
    obs = {
        "status": leg.status,
        "clock": {"t": round(leg.t, 1), "budget": leg.budget, "left": round(leg.time_left, 1)},
        "counters": {"commands": leg.commands, "max_commands": leg.cfg.max_commands, "touches": leg.touches,
                     "max_touches": leg.cfg.max_touches, "gear_changes": leg.shuttles},
        "steer_deg": round(math.degrees(leg.steer), 1),
        "hitch_deg": round(math.degrees(leg.hitch), 1),
        "max_hitch_deg": sp.max_hitch_deg,
        "progress": {"covered": round(max(0.0, leg.progress_s - leg.start_s), 1), "total": round(leg.track.goal_s - leg.start_s, 1),
                     "remaining": round(max(0.0, leg.track.goal_s - leg.progress_s), 1)},
    }
    if leg.track.n_phases > 1:
        info = leg.track.phase_info(leg.phase)
        obs["phase"] = {"index": leg.phase + 1, "of": leg.track.n_phases, "task": info["name"],
                        "direction": "forward" if info["direction"] > 0 else "reverse", "goal": info["goal"]}
    if leg.continuous:
        obs["motion"] = {"v_mps": round(leg.v, 3), "target_steer_deg": round(math.degrees(leg.target_steer), 1),
                         "target_speed_mps": round(leg.target_speed, 2), "hold_s": round(leg.hold_s, 2)}
    if level == "driver":
        from .perception import perceive

        obs["speed"] = {"default_rev": leg.spec.speed_rev, "max_rev": leg.spec.speed_rev_max,
                        "default_fwd": leg.spec.speed_fwd, "max_fwd": leg.spec.speed_fwd_max}
        obs.update(perceive(leg))
        return obs
    cone = level in CONE_LEVELS
    if cone:
        obs["view"] = {"gaze_deg": round(leg.gaze_deg, 1), "fov_deg": leg.spec.view_fov_deg, "range": leg.spec.view_range}
    else:
        obs["sensors"] = sensors(leg)
        obs["route"] = route_arrow(leg)
    obs["speed"] = {"default_rev": leg.spec.speed_rev, "max_rev": leg.spec.speed_rev_max,
                    "default_fwd": leg.spec.speed_fwd, "max_fwd": leg.spec.speed_fwd_max}
    if level in ("local", "sketch", "full") or cone:
        obs["local_map"] = local_map(leg, cone=cone)
    if level in ("sketch", "cone-sketch"):
        obs["sketch"] = sketch(leg, cone=cone)
    if level == "full":
        obs["global"] = {
            "state": [round(float(v), 3) for v in leg.state],
            "trailer_axle": [round(float(v), 3) for v in trailer_axle(leg.state, sp)],
            "centerline": np.round(leg.track.route(leg.phase)[0][:: max(1, int(2.0 / 0.25))], 1).tolist(),
            "dock_corners": np.round(leg.track.target_corners(leg.phase), 2).tolist(),
        }
    return obs


# ----------------------------------------------------------------------- text
def _fmt_dist(d: float, rng: float) -> str:
    return f">{rng:g}" if d >= rng - 1e-6 else f"{d:.1f}"


def _fmt_pts(pts) -> str:
    return " ".join(f"({a:g},{b:g})" for a, b in pts) if pts else "(none in view)"


def gaze_words(g: float) -> str:
    a = abs(g)
    if a <= 15:
        return "straight ahead"
    if a >= 165:
        return "straight back"
    side = "left" if g > 0 else "right"
    if a < 75:
        return f"ahead and to your {side}"
    if a <= 105:
        return f"to your {side}"
    return f"back over your {side} shoulder"


def describe_result(r: DriveResult) -> str:
    verb = "reversed" if r.distance_applied < 0 else ("pulled forward" if r.distance_applied > 0 else "steered while stationary")
    if r.stop_reason == "rejected":
        return "RESULT: command rejected — " + "; ".join(r.notes)
    if r.distance_applied == 0:
        head = f"RESULT: {verb}; wheels now at {r.steer_deg_applied:+.1f}°."
    else:
        head = f"RESULT: {verb} {r.traveled:.2f} m of {abs(r.distance_applied):.2f} m requested at {r.speed_mps:g} m/s"
        why = {
            "completed": "(completed).",
            "finished": "(completed).",
            "contact": "— STOPPED: contact!",
            "blocked": "— BLOCKED, no movement.",
            "jackknife": "— STOPPED: trailer hit the jackknife limit!",
            "time_up": "— the clock ran out.",
        }[r.stop_reason]
        head += " " + why
    if r.contact:
        c = r.contact
        head += f" The {c['body']}'s {c['corner']} corner touched the {c['rail']} (+penalty)."
    if r.notes:
        head += " Note: " + "; ".join(r.notes) + "."
    head += f" That took {r.time_cost:.1f} s."
    return head


def render_report(obs: dict, result: DriveResult | None = None, leg_over: str | None = None) -> str:
    lines = []
    if result is not None:
        lines.append(describe_result(result))
    c, k = obs["clock"], obs["counters"]
    lines.append(f"STATUS: {obs['status']}")
    lines.append(f"CLOCK: {c['t']:.1f} s used of {c['budget']:.0f} s ({c['left']:.1f} s left) | commands {k['commands']}/{k['max_commands']} "
                 f"| rail touches {k['touches']} (max {k['max_touches']}) | gear changes {k['gear_changes']}")
    h = obs["hitch_deg"]
    hs = "straight" if abs(h) < 0.5 else f"trailer rear swung to your {side_word(h)}"
    lines.append(f"STEERING: {obs['steer_deg']:+.1f}° ({'left' if obs['steer_deg'] > 0 else 'right' if obs['steer_deg'] < 0 else 'centred'}) | "
                 f"HITCH ANGLE: {h:+.1f}° ({hs}; mechanical limit ±{obs['max_hitch_deg']:g}°)")
    p = obs["progress"]
    turn = "phase" in obs
    if turn:
        ph = obs["phase"]
        lines.append(f"PHASE {ph['index']}/{ph['of']} ({ph['direction'].upper()}): {ph['task']}. Done when: {ph['goal']}.")
        lines.append(f"PROGRESS: {p['covered']:.1f} m of the {p['total']:.1f} m manoeuvre covered; {p['remaining']:.1f} m to go (both phases).")
    else:
        lines.append(f"PROGRESS: trailer has reversed {p['covered']:.1f} m of the {p['total']:.1f} m course; {p['remaining']:.1f} m to go (to the bay centre).")
    if "route" in obs:
        r = obs["route"]
        b = r["bearing_deg"]
        if r.get("direction") == "forward":
            bword = "straight ahead" if abs(b) < 3 else f"{abs(b):.0f}° to your {side_word(b)} of straight ahead"
            lines.append(f"ROUTE: the way out, {r['distance']:.0f} m ahead of your vehicle's front, lies {bword}.")
        else:
            bword = "straight back" if abs(b) < 3 else f"{abs(b):.0f}° to your {side_word(b)} of straight back"
            lines.append(f"ROUTE: the course centre {r['distance']:.0f} m beyond the trailer lies {bword}.")
    if "sensors" in obs:
        s = obs["sensors"]
        rng = s["range"]
        tlist = " | ".join(f"{n} {_fmt_dist(v, rng)}" for n, v in s["trailer"].items())
        vlist = " | ".join(f"{n} {_fmt_dist(v, rng)}" for n, v in s["vehicle"].items())
        lines.append(f"PROXIMITY (m to nearest {'kerb' if turn else 'rail'}): trailer: {tlist}")
        lines.append(f"                               vehicle: {vlist}")
    if "head" in obs:
        from .perception import render as render_perception

        lines.extend(render_perception(obs, turn))
    if "view" in obs:
        vw = obs["view"]
        lines.append(f"VIEW: you are looking {gaze_words(vw['gaze_deg'])} (gaze {vw['gaze_deg']:+.0f}°; you see ±{vw['fov_deg'] / 2:g}° around it, "
                     f"up to {vw['range']:g} m). Only what is inside this cone is listed below; use look(yaw_deg) to look elsewhere.")
    if "local_map" in obs:
        m = obs["local_map"]
        what = "WHAT YOU SEE" if m.get("cone") else f"LOCAL MAP within {m['radius']:g} m"
        lines.append(f"{what} (u = metres behind the trailer's rear, v = metres to your left; trailer rear centre = (0,0)):")
        rail = "kerb" if turn else "rail"
        lines.append(f"  {rail} on your LEFT:  {_fmt_pts(m['left_rail'])}")
        lines.append(f"  {rail} on your RIGHT: {_fmt_pts(m['right_rail'])}")
        if turn:
            zone = "SIDE ROAD (phase-1 target)" if obs["phase"]["index"] == 1 else "EXIT ZONE (phase-2 target)"
            lines.append(f"  {zone} corners: {_fmt_pts(m['bay_corners'])}" if "bay_corners" in m else f"  {zone}: not in view")
        elif "bay_corners" in m:
            wall = f"; dock wall from {_fmt_pts(m['dock_wall'])}" if "dock_wall" in m else (
                " (only part of the bay is in view)" if m.get("cone") else "")
            lines.append(f"  HANDOFF BAY corners: {_fmt_pts(m['bay_corners'])}{wall}")
        else:
            lines.append("  handoff bay: not in view" + ("" if m.get("cone") else " yet"))
    if "sketch" in obs:
        sk = obs["sketch"]
        lines.append(f"SKETCH (top-down, 1 char = {sk['cell']:g} m; TOP = further behind the trailer, where you are reversing to; "
                     f"LEFT column = your left; # {'kerb' if turn else 'rail'}, . drivable, T trailer, V vehicle, "
                     f"{'B target zone' if turn else 'B bay'}, + trailer rear centre"
                     + (", ? not in view" if obs.get("view") else "") + "):")
        lines.extend("  |" + row.rstrip() for row in sk["rows"])
    if "global" in obs:
        g = obs["global"]
        lines.append(f"GLOBAL: state (x,y,heading,trailer heading) = {g['state']}; trailer axle {g['trailer_axle']}")
        lines.append(f"  {'this phase’s guide route' if turn else 'course centreline (start->bay)'}: {g['centerline']}")
        lines.append(f"  {'target zone' if turn else 'bay'} corners: {g['dock_corners']}")
    if leg_over:
        lines.append(f"LEG OVER: {leg_over}")
    return "\n".join(lines)
