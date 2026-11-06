"""The results digest for the viewer's Results view: one compact file (runs/results.json) with every leg of every
continuous-control run, classified by FAILURE MODE, plus the learning studies.

    python -m relay results        (also written at the end of `run` and `learn`)

Per leg:
  outcome / reason / time / par / progress / touches / shuttles / jackknifes
  mode     why it failed: 'rail-hit loop' (touch limit), 'out of time', 'command limit', 'gave up' (stopped calling
           tools), 'jackknife', 'error'; None if parked
  culprit  which part hit the rails most: own front corner | own rear corner | trailer rear | trailer front
  contacts [{t, part, frac (where along the course, 0 = start, 1 = bay), bend (on a bend?), gaze (what the driver was
           looking at then: camera | mirror | ahead | side | back)}]
  gaze     driver's seat only: share of the leg spent looking at camera / mirror / ahead / side / back
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

GAZE_KINDS = ("camera", "mirror", "ahead", "side", "back")


def gaze_kind(g: float, rig: dict | None) -> str:
    if rig:
        sg = rig.get("screen_gaze")
        if sg is not None and abs(g - sg) <= 7.0:
            return "camera"
        if any(abs(g - mg) <= 7.0 for mg in (rig.get("mirror_gaze") or {}).values()):
            return "mirror"
    a = abs(g)
    return "ahead" if a <= 35 else "back" if a >= 120 else "side"


def part_of(e: dict) -> str:
    corner = e.get("corner") or e.get("edge") or ""
    if e.get("body") == "vehicle":
        return "own front corner" if "front" in corner else "own rear corner"
    return "trailer front" if "front" in corner else "trailer rear"


def failure_mode(m: dict) -> str | None:
    if m["outcome"] == "FINISHED":
        return None
    r = (m.get("reason") or "").lower()
    if "touch" in r:
        return "rail-hit loop"
    if "clock" in r or m["outcome"] == "TIME_UP":
        return "out of time"
    if "command limit" in r:
        return "command limit"
    if "stopped calling" in r or "turn limit" in r:
        return "gave up"
    if "jackknif" in r:
        return "jackknife"
    return "error"


class _Course:
    """Arc length, curvature and goal of a track dict (from the record)."""

    def __init__(self, td: dict):
        from .track import Track

        self.t = Track.from_dict(td)
        h = np.unwrap(self.t.headings)
        ds = np.maximum(np.gradient(self.t.s), 1e-6)
        self.kappa = np.abs(np.gradient(h) / ds)
        try:
            self.goal = float(self.t.goal_s)
        except Exception:  # noqa: BLE001 - turnaround tracks
            self.goal = float(self.t.s[-1])

    def where(self, x: float, y: float) -> tuple[float, bool]:
        s, _ = self.t.project(np.array([x, y]), None, 1e9)
        i = int(np.searchsorted(self.t.s, s))
        lo, hi = np.searchsorted(self.t.s, [s - 3.0, s + 3.0])
        bend = bool(self.kappa[max(0, lo):max(lo + 1, hi)].max() > 0.03)
        return round(min(1.0, max(0.0, s / max(self.goal, 1e-6))), 3), bend and i >= 0


def _leg_row(rec: dict, L: dict, courses: dict) -> dict:
    m = L["metrics"]
    rig = ((rec.get("vehicles") or {}).get(L["vehicle"]) or {}).get("rig")
    tr = L.get("trace") or {}
    gz = tr.get("gaze") or []
    obs = ((rec.get("config") or {}).get("agent") or {}).get("obs_level")
    driver = obs == "driver" and len(gz) == len(tr.get("t", []))
    course = courses.get(L["track"])
    contacts = []
    for e in L.get("events") or []:
        if e["type"] != "contact":
            continue
        c = {"t": e["t"], "part": part_of(e)}
        if course is not None and "x" in e:
            try:
                c["frac"], c["bend"] = course.where(e["x"], e["y"])
            except Exception:  # noqa: BLE001 - best effort
                pass
        if driver:
            k = min(len(gz) - 1, int(np.searchsorted(tr["t"], e["t"])))
            c["gaze"] = gaze_kind(gz[k], rig)
        contacts.append(c)
    parts: dict = {}
    for c in contacts:
        parts[c["part"]] = parts.get(c["part"], 0) + 1
    row = {"index": L["index"], "vehicle": L["vehicle"], "track": L["track"], "outcome": m["outcome"], "reason": m.get("reason"),
           "time": m["time"], "par": m.get("par_time"), "budget": m.get("budget"), "progress": m["progress"],
           "touches": m["touches"], "shuttles": m["shuttles"], "jackknifes": m["jackknifes"], "commands": m["commands"],
           "max_hitch": m.get("max_abs_hitch_deg"), "mode": failure_mode(m),
           "culprit": max(parts, key=parts.get) if parts else None, "parts": parts, "contacts": contacts,
           "turns": L.get("turns"), "controllers": sum(1 for e in L.get("events") or [] if e["type"] == "controller")}
    if driver and gz:
        n = len(gz)
        share = {k: 0 for k in GAZE_KINDS}
        for g in gz:
            share[gaze_kind(g, rig)] += 1
        row["gaze"] = {k: round(v / n, 3) for k, v in share.items()}
        row["looks"] = sum(1 for a, b in zip(gz, gz[1:]) if abs(a - b) > 15)  # big head movements
    if "note" in L or "notes_in" in L:
        row["note"] = L.get("note")
        row["notes_in"] = len(L.get("notes_in") or [])
    return row


def build(runs_dir: str | Path = "runs") -> dict:
    d = Path(runs_dir)
    runs, legs = [], []
    for p in sorted(d.glob("*.json")):
        if p.name in ("index.json", "results.json"):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cfg = rec.get("config") or {}
        ag = cfg.get("agent") or {}
        if rec.get("partial") or cfg.get("control") != "continuous" or (cfg.get("harness_version") or 0) < 4:
            continue
        obs = ag.get("obs_level")
        level = ((cfg.get("perception") or {}).get("level") or "v1") if obs == "driver" else None
        courses = {}
        for td in rec["relay"].get("tracks") or []:
            try:
                courses[td["name"]] = _Course(td)
            except Exception:  # noqa: BLE001 - old track formats
                pass
        run = {"run_id": rec["run_id"], "file": p.name, "created": rec["created"], "model": rec["driver"]["spec"],
               "label": rec["driver"]["label"], "trial": rec["driver"].get("trial", 0), "relay": rec["relay"]["id"],
               "relay_name": rec["relay"]["name"], "seed": rec["relay"]["seed"], "obs": obs, "level": level,
               "mode": cfg.get("mode"), "control": cfg.get("control"), "clock": cfg.get("clock"), "learning": rec.get("learning"),
               "versions": {k: cfg.get(k) for k in ("sim_version", "harness_version", "prompt_version", "control_prompt_version")},
               "perception_version": (cfg.get("perception") or {}).get("version"),
               "perception_prompt_version": (cfg.get("perception") or {}).get("prompt_version"),
               "scripted": rec["driver"]["spec"].startswith("scripted:"),
               "parked": rec["totals"]["legs_finished"], "legs": rec["totals"]["legs"], "progress": rec["totals"]["relay_progress"],
               "touches": rec["totals"]["touches"], "time": rec["totals"]["relay_time"], "wall_s": rec["totals"].get("wall_s")}
        runs.append(run)
        for L in rec["legs"]:
            legs.append({"run_id": rec["run_id"], **_leg_row(rec, L, courses)})
    studies = []
    sd = d / "studies"
    for p in sorted(sd.glob("*.json")) if sd.is_dir() else []:
        if p.name == "index.json":
            continue
        try:
            studies.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return {"generated": dt.datetime.now().isoformat(timespec="seconds"), "runs": runs, "legs": legs, "studies": studies}


def write(runs_dir: str | Path = "runs") -> Path:
    out = Path(runs_dir) / "results.json"
    data = build(runs_dir)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else o))
    tmp.replace(out)
    return out
