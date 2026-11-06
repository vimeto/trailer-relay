"""Data for the public write-up (story/index.html): one compact file, story/data.js (`window.STORY = {...}`), built from
runs/results.json (python -m relay results) and runs/studies/.

    python -m relay.story [--runs runs] [--out story]

Everything the page shows is computed here, so the page itself only draws:
  matrix    legs parked per model x perception level (Wilson 95% interval), by vehicle too
  failures  how failed legs end and which part hit the rail, per model (easy/medium/hard pooled)
  gaze      where drivers looked, per model x vehicle x parked/failed
  learning  first attempt vs later attempts (with notes), per vehicle; per condition; per study and step; transfer
  notes     every note left in a study, with how its own leg went and how the attempts that read it went
  rigs      the perception geometry of each vehicle at each level (for the driver's-seat diagram)
  cost      tokens and API-list-price cost per leg / per parked leg on the shared core configuration
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict
from dataclasses import asdict
from math import comb
from pathlib import Path

from .evaluation import cohort_reason, study_reason
from .perception import LEVELS, rig_for
from .vehicles import VEHICLES

# key, display name, vendor, model-id prefixes (the part after "route:" in a spec)
MODEL_INFO = [
    ("sol", "GPT-6.1 Sol", "OpenAI", ("gpt-6.1-sol",)),
    ("terra", "GPT-6.1 Terra", "OpenAI", ("gpt-6.1-terra",)),
    ("luna", "GPT-6 Luna", "OpenAI", ("gpt-6-luna",)),
    ("opus", "Claude Opus 5.5", "Anthropic", ("claude-opus-5-5",)),
    ("sonnet", "Claude Sonnet 5.5", "Anthropic", ("claude-sonnet-5-5",)),
    ("haiku", "Claude Haiku 5.5", "Anthropic", ("claude-haiku-5-5",)),
    ("argon", "Gemini 4 Argon", "Google", ("gemini-4-argon",)),
    ("flash", "Gemini 3.8 Flash", "Google", ("gemini-3.8-flash",)),
    ("oss", "GPT-OSS 120B", "Open weights", ("gpt-oss-120b",)),
]
MODELS = [(k, n) for k, n, _, _ in MODEL_INFO]
ROUTE_NAMES = {"codex": "Codex CLI", "claude-code": "Claude Code CLI", "agy": "Antigravity CLI", "openai": "OpenAI API"}
VEH_ORDER = ["car", "tractor", "mower", "truck"]
LEVEL_ORDER = ["easy", "medium", "hard", "v1"]
# API list prices, USD per 1M tokens, looked up 2026-11-06. Runs went through $0 subscription CLIs; these prices say what
# the same tokens would cost on each vendor's API. cache_write None = the vendor bills cache writes at the input price.
PRICE_SHEET = {
    "sol":    {"input": 2.00, "cache_read": 0.10, "cache_write": None, "output": 10.00, "source": "OpenAI standard API pricing", "url": "https://developers.openai.com/api/docs/pricing?tab=suite"},
    "terra":  {"input": 1.00, "cache_read": 0.05, "cache_write": None, "output": 5.00, "source": "OpenAI standard API pricing", "url": "https://developers.openai.com/api/docs/pricing?tab=suite"},
    "luna":   {"input": 0.10, "cache_read": 0.01, "cache_write": None, "output": 0.50, "source": "OpenAI standard API pricing", "url": "https://developers.openai.com/api/docs/pricing?tab=suite"},
    "opus":   {"input": 4.00, "cache_read": 0.20, "cache_write": 5.00, "output": 20.00, "source": "Anthropic standard API pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
    "sonnet": {"input": 2.00, "cache_read": 0.10, "cache_write": 2.50, "output": 10.00, "source": "Anthropic standard API pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
    "haiku":  {"input": 0.10, "cache_read": 0.01, "cache_write": 0.125, "output": 0.50,
               "source": "Anthropic, prompts up to 100K tokens", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
    "flash":  {"input": 0.75, "cache_read": 0.075, "cache_write": None, "output": 3.75,
               "source": "Google Cloud global introductory pricing through 2026-12-31", "url": "https://cloud.google.com/model-garden/pricing"},
    "argon":  {"input": 2.00, "cache_read": 0.10, "cache_write": None, "output": 10.00,
               "source": "Google Cloud global introductory pricing through 2026-11-30", "url": "https://cloud.google.com/model-garden/pricing"},
    "oss":    {"input": 0.15, "cache_read": 0.015, "cache_write": None, "output": 0.60,
               "source": "Fireworks serverless, one hosting provider", "url": "https://fireworks.ai/models/fireworks/gpt-oss-120b"},
}
NORMALIZED_CACHE = 0.5  # the same cache-read share for every model, so routes with poor caching are not penalised
CONDITION_NAMES = {"self": "Retries with own notes", "chain": "Notebook relay", "pool": "Team rounds"}


def model_key(spec: str) -> str | None:
    mid = spec.split(":", 1)[-1]
    for k, _, _, prefixes in MODEL_INFO:
        if mid.startswith(prefixes):
            return k
    return None


def fail_stats(Ls: list) -> dict:
    failed = [L for L in Ls if L["outcome"] != "FINISHED"]
    cons = [c for L in failed for c in L["contacts"]]
    fr = [c["frac"] for c in cons if "frac" in c]
    hist = [0] * 20
    for f in fr:
        hist[min(19, int(f * 20))] += 1
    return {"legs": len(Ls), "failed": len(failed), "modes": _count(L["mode"] for L in failed),
            "culprit": _count(L["culprit"] for L in failed if L["culprit"]), "contacts": len(cons),
            "on_bend": round(sum(1 for c in cons if c.get("bend")) / max(1, len(cons)), 3),
            "first_third": round(sum(1 for f in fr if f < 1 / 3) / max(1, len(fr)), 3), "where": hist,
            "touches_parked": round(sum(L["touches"] for L in Ls if L["outcome"] == "FINISHED") / max(1, len(Ls) - len(failed)), 2)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test: is a/(a+b) larger than c/(c+d)?"""
    n1, k, n = a + b, a + c, a + b + c + d
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(a, min(k, n1) + 1)) / comb(n, n1)


def _kn(legs) -> list[int]:
    legs = list(legs)
    return [sum(L["outcome"] == "FINISHED" for L in legs), len(legs)]


def build(runs_dir: str | Path = "runs") -> dict:
    d = Path(runs_dir)
    R = json.loads((d / "results.json").read_text())
    runs = {r["run_id"]: r for r in R["runs"]}
    legs_by_run = defaultdict(list)
    for L in R["legs"]:
        legs_by_run[L["run_id"]].append(L)

    def api_failed(r):  # a leg the vendor's API refused (quota, outage) says nothing about the model's driving
        return any(L["reason"] == "model API error" for L in legs_by_run[r["run_id"]])

    def is_model_run(r):
        return model_key(r["model"]) and cohort_reason(r, legs_by_run[r["run_id"]], allow_ablations=True) is None

    exclusions = [{"run_id": r["run_id"], "model": r["model"],
                   "reason": cohort_reason(r, legs_by_run[r["run_id"]], allow_ablations=True)}
                  for r in R["runs"] if model_key(r["model"]) and not is_model_run(r)]
    rejected_studies = [{"study": s["study"], "model": s["model"], "reason": study_reason(s, runs, legs_by_run)}
                        for s in R["studies"] if study_reason(s, runs, legs_by_run)]
    rejected_ids = {s["study"] for s in rejected_studies}
    studies = sorted((s for s in R["studies"] if s["study"] not in rejected_ids),
                     key=lambda s: (s["model"], s["condition"], (s.get("policy") or {}).get("numbers", "")))
    valid_study_runs = {r["run_id"] for s in studies for r in s["runs"]}
    base = [r for r in R["runs"] if is_model_run(r) and not r.get("learning")]
    study_runs = [r for r in R["runs"] if is_model_run(r) and r["run_id"] in valid_study_runs]

    # ---- matrix
    matrix = []
    for mk, _ in MODELS:
        for lv in LEVEL_ORDER:
            rs = [r for r in base if model_key(r["model"]) == mk and r["level"] == lv]
            if not rs:
                continue
            Ls = [L for r in rs for L in legs_by_run[r["run_id"]]]
            k, n = _kn(Ls)
            matrix.append({"model": mk, "level": lv, "k": k, "n": n, "ci": wilson(k, n), "runs": len(rs),
                           "progress": round(sum(L["progress"] for L in Ls) / n, 3),
                           "seeds": {str(s): sum(1 for r in rs if r["seed"] == s) for s in sorted({r["seed"] for r in rs})},
                           "vehicles": {v: _kn(L for L in Ls if L["vehicle"] == v) for v in VEH_ORDER}})

    # ---- failures: per level, and easy/medium/hard pooled ("all")
    failures = {}
    for mk, _ in MODELS:
        mine = [r for r in base if model_key(r["model"]) == mk]
        if not mine:
            continue
        failures[mk] = {"all": fail_stats([L for r in mine if r["level"] in ("easy", "medium", "hard") for L in legs_by_run[r["run_id"]]])}
        for lv in ("easy", "medium", "hard"):
            Ls = [L for r in mine if r["level"] == lv for L in legs_by_run[r["run_id"]]]
            if Ls:
                failures[mk][lv] = fail_stats(Ls)

    # ---- gaze: every driver's-seat run (studies included), per model x vehicle x outcome
    gaze = []
    for mk, _ in MODELS:
        for v in VEH_ORDER:
            for parked in (True, False):
                Ls = [L for r in base + study_runs if model_key(r["model"]) == mk and r["level"] in ("easy", "medium", "hard")
                      for L in legs_by_run[r["run_id"]] if L["vehicle"] == v and (L["outcome"] == "FINISHED") == parked and L.get("gaze")]
                if not Ls:
                    continue
                n = len(Ls)
                gaze.append({"model": mk, "vehicle": v, "parked": parked, "n": n,
                             "share": {g: round(sum(L["gaze"][g] for L in Ls) / n, 3) for g in Ls[0]["gaze"]},
                             "looks": round(sum(L.get("looks", 0) for L in Ls) / n, 1),
                             "touches": round(sum(L["touches"] for L in Ls) / n, 2),
                             "shuttles": round(sum(L["shuttles"] for L in Ls) / n, 2)})

    # ---- learning
    learning = {"models": {}, "studies": [], "conditions": [], "excluded": rejected_studies}
    def finished(s):
        return study_reason(s, runs, legs_by_run) is None
    for mk, _ in MODELS:
        S = [s for s in studies if model_key(s["model"]) == mk and finished(s)]
        if not S:
            continue
        groups = defaultdict(list)
        for s in S:
            for r in s["runs"]:
                ph = "transfer" if r["role"] == "transfer" else "first" if r["step"] == 1 else "later"
                groups[ph] += r["legs"]
        if not (groups.get("first") and groups.get("later")):
            continue  # studies still on their first step: nothing to compare yet
        out = {ph: {"all": _kn(Ls), **{v: _kn(L for L in Ls if L["vehicle"] == v) for v in VEH_ORDER}} for ph, Ls in groups.items()}
        b0 = [L for r in base if model_key(r["model"]) == mk and r["level"] == "medium" and r["seed"] == 0 for L in legs_by_run[r["run_id"]]]
        b1 = [L for r in base if model_key(r["model"]) == mk and r["level"] == "medium" and r["seed"] == 1 for L in legs_by_run[r["run_id"]]]
        out["baseline_train"] = _kn(b0)
        out["baseline_transfer"] = _kn(b1)
        (a, n_a), (c, n_c) = out["later"]["all"], out["first"]["all"]
        out["p_later_vs_first"] = round(fisher_greater(a, n_a - a, c, n_c - c), 4)
        out["p_by_vehicle"] = {v: round(fisher_greater(out["later"][v][0], out["later"][v][1] - out["later"][v][0],
                                                       out["first"][v][0], out["first"][v][1] - out["first"][v][0]), 4) for v in VEH_ORDER}
        learning["models"][mk] = out
    for s in studies:
        mk = model_key(s["model"])
        numbers = (s.get("policy") or {}).get("numbers", "allow")
        steps = defaultdict(list)
        seqs = defaultdict(list)
        for r in s["runs"]:
            key = "T" if r["role"] == "transfer" else str(r["step"])
            steps[key].append({"member": r["member"], "run_id": r["run_id"], "parked": sum(L["outcome"] == "FINISHED" for L in r["legs"]),
                               "progress": round(sum(L["progress"] for L in r["legs"]) / len(r["legs"]), 3),
                               "legs": [{"vehicle": L["vehicle"], "outcome": L["outcome"], "progress": L["progress"]} for L in r["legs"]]})
            if r["role"] == "train":
                for L in r["legs"]:
                    seqs[(r["member"] if s["condition"] == "self" else 0, L["vehicle"])].append(
                        (r["step"], 1.0 if L["outcome"] == "FINISHED" else L["progress"]))
        mono = sum(1 for q in seqs.values() if all(y[1] >= x[1] - 1e-9 for x, y in zip(sorted(q), sorted(q)[1:])))
        later = [L for r in s["runs"] if r["role"] == "train" and r["step"] > 1 for L in r["legs"]]
        learning["studies"].append({"id": s["study"], "model": mk, "condition": s["condition"], "numbers": numbers,
                                    "members": s["members"], "steps_n": s["steps"], "steps": dict(steps),
                                    "never_worse": [mono, len(seqs)] if s["condition"] != "pool" else None, "later": _kn(later),
                                    "done": finished(s)})
    for mk, _ in MODELS:
        for cond, numbers in [("self", "allow"), ("self", "none"), ("chain", "allow"), ("pool", "allow")]:
            st = [x for x in learning["studies"] if x["model"] == mk and x["condition"] == cond and x["numbers"] == numbers and x["done"]]
            if st:
                learning["conditions"].append({"model": mk, "condition": cond, "numbers": numbers, "later": st[0]["later"], "study": st[0]["id"]})

    # ---- notes
    notes = []
    for s in studies:
        mk = model_key(s["model"])
        numbers = (s.get("policy") or {}).get("numbers", "allow")
        runs_sorted = sorted(s["runs"], key=lambda r: (r["step"], r["member"]))
        for r in runs_sorted:
            for L in r["legs"]:
                n = L.get("note")
                if not n or not n.get("text"):
                    continue
                read_by = [{"step": r2["step"], "member": r2["member"], "role": r2["role"], "outcome": L2["outcome"],
                            "progress": L2["progress"], "run_id": r2["run_id"]}
                           for r2 in runs_sorted for L2 in r2["legs"]
                           if L2["vehicle"] == L["vehicle"] and n.get("accepted") and any(x.get("text") == n["text"] for x in L2.get("notes_in") or [])]
                notes.append({"study": s["study"], "model": mk, "condition": s["condition"], "numbers": numbers,
                              "step": r["step"], "member": r["member"], "run_id": r["run_id"], "vehicle": L["vehicle"],
                              "outcome": L["outcome"], "progress": L["progress"], "touches": L["touches"], "time": L["time"],
                              "read": len(L.get("notes_in") or []), "text": n["text"], "accepted": n["accepted"],
                              "rejections": n.get("rejections") or [], "read_by": read_by})

    # ---- cost on the shared core configuration (medium perception, code mode, continuous control, paused clock)
    cost = {"prices": PRICE_SHEET, "normalized_cache": NORMALIZED_CACHE, "models": []}
    for mk, _ in MODELS:
        rs = [r for r in base if model_key(r["model"]) == mk and r["level"] == "medium" and r["mode"] == "code" and r["clock"] == "paused"]
        p = PRICE_SHEET.get(mk)
        if not rs or not p:
            continue
        legs, route = [], rs[0]["model"].split(":", 1)[0]
        for r in rs:
            rec = json.loads((d / r["file"]).read_text())
            if (rec["config"].get("agent") or {}).get("control") != "continuous":
                continue
            legs += [L for L in rec["legs"] if (L.get("usage") or {}).get("input_tokens")]
        if not legs:
            continue
        n = len(legs)
        parked = sum(L["metrics"]["outcome"] == "FINISHED" for L in legs)
        tot = lambda f: sum((L["usage"].get(f) or 0) for L in legs)
        inp, cached, out, reas = tot("input_tokens"), tot("cached_tokens"), tot("output_tokens"), tot("reasoning_tokens")
        if route == "agy":  # agy reports input_tokens WITHOUT the cache reads (input + output = total)
            inp += cached

        def usd(cached_share=None, writes=False):
            c = cached if cached_share is None else cached_share * inp
            unc_price = p["cache_write"] if (writes and p["cache_write"]) else p["input"]
            return ((inp - c) * unc_price + c * p["cache_read"] + out * p["output"]) / 1e6 / n

        row = {"model": mk, "route": route, "legs": n, "parked": parked, "ci": wilson(parked, n),
               "turns": round(sum(L.get("turns") or 0 for L in legs) / n, 1),
               "input": round(inp / n), "cached_share": round(cached / inp, 3), "output": round(out / n),
               "reasoning": round(reas / n), "input_per_turn": round(inp / max(1, sum(L.get("turns") or 0 for L in legs))),
               "usd": {"measured": round(usd(), 4), "no_cache": round(usd(0.0), 4),
                       "normalized": round(usd(NORMALIZED_CACHE), 4)}}
        if p["cache_write"]:  # Anthropic: the CLI does not split cache writes from plain input; bill all uncached input as writes
            row["usd"]["measured_writes"] = round(usd(writes=True), 4)
        row["usd_per_parked"] = {k: (round(v * n / parked, 3) if parked else None) for k, v in row["usd"].items()}
        cost["models"].append(row)

    # ---- rigs and vehicles (for the perception diagram)
    vehicles = {}
    for v in VEH_ORDER:
        spec = VEHICLES[v]
        sd = spec.to_dict() if hasattr(spec, "to_dict") else asdict(spec)
        vehicles[v] = {"name": spec.name, "trailer": spec.trailer_name,
                       "geom": {k: sd[k] for k in ("wheelbase", "hitch_offset", "trailer_length", "body_front", "body_rear", "body_width",
                                                    "trailer_front", "trailer_rear", "trailer_width")},
                       "rig": {lv: _rig_dict(rig_for(spec, lv)) for lv in LEVEL_ORDER}}

    present = {model_key(r["model"]) for r in base + study_runs}
    models_out = []
    for k, n, vendor, _ in MODEL_INFO:
        if k not in present:
            continue
        specs = sorted({r["model"] for r in base + study_runs if model_key(r["model"]) == k})
        route = specs[0].split(":", 1)[0] if specs and ":" in specs[0] else "api"
        models_out.append({"key": k, "name": n, "vendor": vendor, "specs": specs, "route": ROUTE_NAMES.get(route, route),
                           "learning": k in learning["models"]})
    counts = {"runs": len(base) + len(study_runs), "matrix_runs": len(base), "study_runs": len(study_runs),
              "legs": sum(len(legs_by_run[r["run_id"]]) for r in base + study_runs), "notes": len(notes), "studies": len(studies),
              "excluded_api_error_runs": sum(1 for r in R["runs"] if model_key(r["model"]) and api_failed(r)),
              "excluded_api_error_studies": sum(s["reason"] == "model API error" for s in rejected_studies),
              "excluded_studies": len(rejected_studies),
              "notes_accepted_first_try": sum(1 for n in notes if n["accepted"] and not n["rejections"])}
    return {"generated": dt.datetime.now().isoformat(timespec="minutes"), "counts": counts,
            "models": models_out, "conditions": CONDITION_NAMES,
            "levels": [{"key": k, "noise": LEVELS[k].noise, "camera_mult": LEVELS[k].camera_range_mult,
                        "periphery": LEVELS[k].periphery_range} for k in LEVEL_ORDER],
            "vehicles": vehicles, "matrix": matrix, "failures": failures, "gaze": gaze, "learning": learning, "notes": notes, "cost": cost,
            "exclusions": exclusions, "core": {"description": "standard, seeds 0/1, driver medium, code, continuous, paused",
            "runs": [r["run_id"] for r in base if r["level"] == "medium"],
            "statistics": "Descriptive leg-level Wilson intervals; repeated legs share runs and courses. Fisher p-values are exploratory and uncorrected."}}


def _count(it) -> dict:
    out: dict = defaultdict(int)
    for x in it:
        out[x] += 1
    return dict(out)


def _rig_dict(r) -> dict:
    d = r.to_dict()
    return {k: d[k] for k in ("eye", "gaze_min", "gaze_max", "fov_deg", "range", "cab_block_deg", "mirrors", "screen",
                              "camera_fov_deg", "camera_range", "mirror_gaze", "screen_gaze", "noise", "periphery_deg", "periphery_range")}


def write(runs_dir: str | Path = "runs", out_dir: str | Path = "story") -> Path:
    out = Path(out_dir) / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build(runs_dir)
    out.write_text("window.STORY = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    wrap_page(out_dir)
    return out


def wrap_page(out_dir: str | Path = "story") -> Path | None:
    """story/page.html is a body fragment (it also publishes as an Artifact preview);
    index.html is the same page as a complete document for static hosting."""
    src = Path(out_dir) / "page.html"
    if not src.exists():
        return None
    body = src.read_text()
    head, sep, rest = body.partition("</style>")
    head += sep
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           "<style>img{max-width:100%}[hidden]{display:none!important}</style>\n"
           + head + "\n</head>\n<body>\n" + rest.lstrip() + "\n</body>\n</html>\n")
    dst = Path(out_dir) / "index.html"
    dst.write_text(doc)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m relay.story")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--runs", default=str(root / "runs"))
    ap.add_argument("--out", default=str(root / "story"))
    a = ap.parse_args()
    p = write(a.runs, a.out)
    print(p, f"{p.stat().st_size / 1024:.0f} KB")
