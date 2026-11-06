"""Procedural, seeded course generation with drivability certification.

Every course is built from primitives scaled to the vehicle (combo length C,
comfortable reversing radius R) and then certified by the autopilot. If the
autopilot cannot finish cleanly the corridor is widened and re-certified, so
every published track is known to be drivable, and its par time is recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from .kinematics import critical_hitch_angle
from .track import Primitive, Track
from .vehicles import VEHICLES, VehicleSpec, get_vehicle

TEMPLATES = ("straight", "lturn", "sbend", "alley", "hairpin", "chicane")  # railed corridors; "turn" -> turnaround.py
DIFFICULTY = {
    #           radius factor   per-side clearance (x max width)  dock slack (m per C)
    "easy": dict(rf=(2.0, 2.6), clr=0.80, slack=0.16),
    "medium": dict(rf=(1.5, 2.0), clr=0.58, slack=0.11),
    "hard": dict(rf=(1.2, 1.5), clr=0.42, slack=0.07),
}


def comfortable_radius(spec: VehicleSpec) -> float:
    b = 0.55 * min(critical_hitch_angle(spec), spec.max_hitch, math.radians(60))
    return max(spec.trailer_length, spec.wheelbase) / math.tan(b) + 0.25 * spec.max_width


def swept_band(spec: VehicleSpec, R: float) -> tuple[float, float]:
    """Radial band [r_in, r_out] swept by the combination in a steady turn whose trailer
    axle follows radius R (off-tracking + front overhang swing)."""
    w1, w2 = spec.body_width / 2, spec.trailer_width / 2
    Rh = math.hypot(R, spec.trailer_length)
    R1 = math.sqrt(max(Rh ** 2 - spec.hitch_offset ** 2, 0.25))
    r_in = min(R - w2, R1 - w1)
    r_out = max(math.hypot(R + w2, spec.trailer_front), math.hypot(R1 + w1, spec.body_front), math.hypot(R1 + w1, spec.body_rear))
    return r_in, r_out


def curve_halfwidth(spec: VehicleSpec, R: float, clearance: float) -> float:
    r_in, r_out = swept_band(spec, R)
    asym = abs(0.5 * (r_in + r_out) - R)
    return 0.5 * (r_out - r_in) + clearance + 0.3 * asym


def _sub_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16)


def make_primitives(spec: VehicleSpec, template: str, difficulty: str, rng: random.Random, widen: float = 1.0):
    D = DIFFICULTY[difficulty]
    C = spec.combo_length
    W = spec.max_width
    clr = D["clr"] * W * widen
    hw = W / 2 + clr
    R = max(comfortable_radius(spec) * rng.uniform(*D["rf"]), hw + 1.0)
    hwc = lambda r: max(hw, curve_halfwidth(spec, r, clr))  # noqa: E731  corridor flares on curves
    start_len = spec.body_front + spec.hitch_offset + spec.trailer_length + spec.trailer_rear + 0.8 + 0.25 * C
    side = rng.choice((-1, 1))
    S = lambda L, w=hw: Primitive("straight", length=L, halfwidth=w)  # noqa: E731
    A = lambda r, ang, w=None: Primitive("arc", radius=r, angle=ang, halfwidth=w if w is not None else hwc(r))  # noqa: E731
    if template == "straight":
        prims = [S(start_len + rng.uniform(1.0, 1.6) * C)]
    elif template == "lturn":
        prims = [S(start_len + rng.uniform(0.3, 0.7) * C), A(R, side * math.pi / 2), S(rng.uniform(1.0, 1.4) * C)]
    elif template == "sbend":
        th = math.radians(rng.uniform(35, 60))
        prims = [S(start_len + rng.uniform(0.2, 0.5) * C), A(R, side * th), S(rng.uniform(0.2, 0.5) * C),
                 A(R, -side * th), S(rng.uniform(1.0, 1.3) * C)]
    elif template == "alley":
        # wide yard, tight 90-degree swing into a narrow bay: the classic alley dock
        hw_yard = max(hw * 1.7, hwc(R * 0.9))
        hw_bay = W / 2 + 0.75 * clr
        prims = [S(start_len + rng.uniform(0.3, 0.6) * C, hw_yard), A(max(R * 0.9, hw_yard + 1.0), side * math.pi / 2, hw_yard),
                 S(0.35 * C, hw_bay), S(rng.uniform(0.8, 1.1) * C, hw_bay)]
    elif template == "hairpin":
        ang = math.radians(rng.uniform(150, 180))
        prims = [S(start_len + rng.uniform(0.2, 0.4) * C), A(max(R * 1.3, hw + 1.5), side * ang), S(rng.uniform(1.0, 1.3) * C)]
    elif template == "chicane":
        th = math.radians(rng.uniform(25, 40))
        r = R * 1.1
        prims = [S(start_len + 0.2 * C), A(r, side * th), A(r, -side * 2 * th), A(r, side * th), S(rng.uniform(1.0, 1.2) * C)]
    else:
        raise ValueError(f"unknown template {template!r}")
    return prims


def _bump(s: np.ndarray, a: float, b: float, ramp: float) -> np.ndarray:
    """1 on [a, b], cosine ramps of length `ramp` on both sides, 0 elsewhere."""
    up = np.clip((s - (a - ramp)) / max(ramp, 1e-9), 0, 1)
    down = np.clip(((b + ramp) - s) / max(ramp, 1e-9), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * np.minimum(up, down))


def asymmetry_profile(track: Track, spec: VehicleSpec, rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extra width per side along the route: a YARD beside the bay (the bay is painted against one rail of a wider
    apron, so the rails' middle is ~1 trailer width off the bay) and one or two LAY-BYS earlier on.
    The route and the bay do not move; the extra room only makes "stay between the rails" the wrong target."""
    s, L, C, W = track.s, track.length, spec.combo_length, spec.max_width
    left, right = np.zeros_like(s), np.zeros_like(s)
    yard_side = rng.choice((-1, 1))
    yard_w = rng.uniform(1.6, 2.4) * W
    kappa = np.gradient(track.headings) / np.maximum(np.gradient(s), 1e-9)
    bent = np.nonzero(np.abs(kappa) > 1e-3)[0]
    after_bend = float(s[bent[-1]]) + 0.35 * C if len(bent) else 0.0
    yard = yard_w * _bump(s, max(L - track.dock_length - 0.5 * C, min(after_bend, L - track.dock_length)), L + 1.0, 0.35 * C)
    (left if yard_side > 0 else right)[:] += yard
    hw = track.halfwidth
    lo, hi = 0.25 * L, max(0.3 * L, L - track.dock_length - 1.2 * C)
    bays = []
    for _ in range(rng.choice((1, 2))):
        for _try in range(30):  # lay-bys go on straight stretches
            c = rng.uniform(lo, hi)
            half = rng.uniform(0.2, 0.4) * C
            span = (s > c - half - 0.3 * C) & (s < c + half + 0.3 * C)
            if span.any() and np.abs(kappa[span]).max() < 0.25 / max(hw.max() + W, 1e-9):
                break
        else:
            continue
        side = rng.choice((-1, 1))
        w = rng.uniform(0.6, 1.0) * W
        (left if side > 0 else right)[:] += w * _bump(s, c - half, c + half, 0.3 * C)
        bays.append({"s": round(c, 1), "len": round(2 * half, 1), "side": side, "extra": round(w, 2)})
    # never widen the inside of a bend towards its centre of curvature (the rail would fold over itself)
    with np.errstate(divide="ignore"):
        room = np.where(np.abs(kappa) > 1e-6, 0.8 / np.abs(kappa) - hw, np.inf)
    left = np.where(kappa > 0, np.minimum(left, np.maximum(room, 0)), left)
    right = np.where(kappa < 0, np.minimum(right, np.maximum(room, 0)), right)
    k = max(1, int(round(0.15 * C / max(np.diff(s).mean(), 1e-9))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    left = np.convolve(np.pad(left, k, mode="edge"), ker, mode="valid")
    right = np.convolve(np.pad(right, k, mode="edge"), ker, mode="valid")
    return left, right, {"yard_side": yard_side, "yard_extra": round(yard_w, 2), "laybys": bays}


def generate_track(vehicle: str, template: str, difficulty: str = "medium", seed: int = 0,
                   certify: bool = True, max_attempts: int = 6, name: str | None = None, asym: bool = False) -> Track:
    if template == "turn":  # intiaanikäännös: its own layout and certification (relay/turnaround.py)
        from .turnaround import generate_turn_track

        return generate_turn_track(vehicle, difficulty, seed, certify=certify, max_attempts=max_attempts, name=name)
    spec = get_vehicle(vehicle)
    rng = random.Random(_sub_seed(vehicle, template, difficulty, seed))
    rotate = rng.uniform(-math.pi, math.pi)
    shape_state = rng.getstate()
    widen = 1.0
    last = None
    for attempt in range(max_attempts):
        rng.setstate(shape_state)  # same shape each attempt; only the width changes
        prims = make_primitives(spec, template, difficulty, rng, widen=widen)
        slack = DIFFICULTY[difficulty]["slack"] * spec.combo_length * (1 + 0.25 * attempt)
        tr = Track.from_primitives(name or f"{vehicle}-{template}-{difficulty}-{seed}{'-asym' if asym else ''}", spec, template,
                                   difficulty, seed, prims, dock_slack=slack, rotate=rotate, smooth_m=0.3 * spec.combo_length,
                                   meta={"widen": round(widen, 3), "attempt": attempt, "smooth_m": 0.3 * spec.combo_length})
        if asym:
            el, er, info = asymmetry_profile(tr, spec, random.Random(_sub_seed(vehicle, template, difficulty, seed, "asym")))
            tr.set_asymmetry(el, er)
            tr.meta["asym"] = info
        if not certify:
            return tr
        par = certify_track(tr)
        last = tr
        if par["outcome"] == "FINISHED" and par["touches"] == 0:
            tr.par = par
            tr.meta["certified"] = True
            return tr
        widen *= 1.18
    assert last is not None
    last.meta["certified"] = False
    last.par = None
    return last


def certify_track(track: Track) -> dict:
    from .autopilot import Autopilot
    from .sim import Leg, LegConfig

    leg = Leg(track, config=LegConfig(time_budget=1e5, max_commands=400, max_touches=0))
    Autopilot().run_leg(leg)
    m = leg.metrics()
    return {"outcome": m["outcome"], "time": m["time"], "commands": m["commands"], "touches": m["touches"],
            "shuttles": m["shuttles"], "driver": "autopilot"}


# --------------------------------------------------------------------------- relays
RELAYS = {
    # the default four-leg relay: one leg per vehicle class
    "standard": [("car", "lturn", "medium"), ("tractor", "sbend", "medium"), ("mower", "alley", "medium"), ("truck", "lturn", "medium")],
    "gauntlet": [("car", "chicane", "hard"), ("tractor", "hairpin", "hard"), ("mower", "sbend", "hard"), ("truck", "alley", "hard")],
    # easier rung: same four vehicles, gentler courses (separates weaker models)
    "warmup": [("car", "straight", "easy"), ("tractor", "lturn", "easy"), ("mower", "straight", "easy"), ("truck", "straight", "easy")],
    "mini": [("car", "straight", "easy")],
    # intiaanikäännös: reverse the whole rig into a side road, then drive out the way you came (two phases)
    "turnaround": [("car", "turn", "medium"), ("tractor", "turn", "medium"), ("mower", "turn", "medium"), ("truck", "turn", "medium")],
    # asymmetric corridors: the standard legs, with a yard beside the bay and lay-bys (the rails' middle is not the way)
    "asym": [("car", "lturn", "medium", {"asym": True}), ("tractor", "sbend", "medium", {"asym": True}),
             ("mower", "alley", "medium", {"asym": True}), ("truck", "lturn", "medium", {"asym": True})],
}


def generate_relay(name: str = "standard", seed: int = 0, legs=None) -> dict:
    legs = legs or RELAYS[name]
    tracks = [generate_track(v, t, d, seed=_sub_seed(name, seed, i) % 100000, **(kw[0] if kw else {}))
              for i, (v, t, d, *kw) in enumerate(legs)]
    return {"name": name, "seed": seed, "id": f"{name}-s{seed}", "tracks": tracks}


def relay_to_dict(relay: dict) -> dict:
    return {"name": relay["name"], "seed": relay["seed"], "id": relay["id"], "tracks": [t.to_dict() for t in relay["tracks"]]}


def relay_from_dict(d: dict) -> dict:
    return {"name": d["name"], "seed": d["seed"], "id": d["id"], "tracks": [Track.from_dict(t) for t in d["tracks"]]}


def load_or_generate_relay(name: str, seed: int, cache_dir: str | Path = "tracks") -> dict:
    """Cached relays are certified (and their par times measured) under one physics version; a cache from
    older physics is archived to <cache_dir>/_v<old>/ and the relay is regenerated."""
    from .sim import SIM_VERSION

    p = Path(cache_dir) / f"{name}-s{seed}.json"
    if p.exists():
        d = json.loads(p.read_text())
        old = d.get("sim_version", 2)
        if old == SIM_VERSION:
            return relay_from_dict(d)
        arch = p.parent / f"_v{old}"
        arch.mkdir(exist_ok=True)
        p.replace(arch / p.name)
    relay = generate_relay(name, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**relay_to_dict(relay), "sim_version": SIM_VERSION}))
    return relay


__all__ = ["generate_track", "generate_relay", "certify_track", "TEMPLATES", "RELAYS", "VEHICLES"]
