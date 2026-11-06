"""Static export of the explorer: `python -m relay site [--out site]`.

Layout (every URL relative, so it works from any static host under any sub-path):

    site/index.html        the story page (story/ copied as is), or a redirect to explorer/ while there is none
    site/explorer/         the viewer (viewer/ + vendor/), marked <meta name="relay-static">: no polling, no listing probe
    site/runs/index.json   the exported runs only, no live entries
    site/runs/results.json the Results digest (python -m relay results)
    site/runs/studies/     the learning studies
    site/runs/<id>.json    every run in results.json plus every study run, slimmed (below)

Slimming keeps everything the viewer shows:
  * transcript calls lose `report` (the text observation sent to the model) and `program` (a copy of args.code);
    the viewer reads neither (Inspect rebuilds the view geometry from the recorded pose and gaze).
  * traces (10 Hz) are thinned to 5 Hz. Samples are kept at full rate within 0.5 s of an event (contact, status,
    phase, finish), wherever the gaze changes (head turns), wherever the rig starts, stops or changes direction,
    and at both ends of a stretch where nothing moves; the stretch in between is dropped (the viewer interpolates
    over t, so a still rig needs only its ends). `--full-traces` keeps every sample.
"""
from __future__ import annotations

import bisect
import json
import shutil
from pathlib import Path

VIEWER_SKIP = {"mockup", ".DS_Store"}
STATIC_META = '<meta name="relay-static" content="1">'
CALL_DROP = ("report", "program")
QUIET_EVENTS = {"controller", "command"}  # frequent, and nothing visible happens at them


def _vclass(v: float) -> int:
    return 0 if abs(v) < 0.02 else (1 if v > 0 else -1)


def thin_trace(tr: dict, events: list, hz: float = 5.0, near: float = 0.5) -> dict:
    """Drop trace samples the viewer can interpolate back (see the module docstring)."""
    t = tr.get("t") or []
    n = len(t)
    if n < 3:
        return tr
    keys = [k for k, v in tr.items() if isinstance(v, list) and len(v) == n]
    rows = list(zip(*(tr[k] for k in keys if k != "t")))  # the state at each sample, without its time
    g, v = tr.get("gaze"), tr.get("v")
    ev = sorted(e["t"] for e in events or [] if e.get("t") is not None and e.get("type") not in QUIET_EVENTS)
    forced = [False] * n
    still = [False] * n  # interior of a stretch where no field changes
    forced[0] = forced[-1] = True
    for i in range(1, n - 1):
        same_prev, same_next = rows[i] == rows[i - 1], rows[i] == rows[i + 1]
        if same_prev and same_next:
            still[i] = True
            continue
        if same_prev != same_next:  # an end of a still stretch
            forced[i] = True
        if g is not None and (g[i] != g[i - 1] or g[i] != g[i + 1] or (i > 1 and g[i - 1] != g[i - 2])):  # a head turn, and the sample after it
            forced[i] = True
        if v is not None and (_vclass(v[i]) != _vclass(v[i - 1]) or _vclass(v[i]) != _vclass(v[i + 1])):
            forced[i] = True
        k = bisect.bisect_left(ev, t[i] - near)
        if k < len(ev) and ev[k] <= t[i] + near:
            forced[i] = True
    keep, last = [], -1e9
    step = 1.0 / hz - 1e-6
    for i in range(n):
        if forced[i] or (not still[i] and t[i] - last >= step):
            keep.append(i)
            last = t[i]
    out = dict(tr)
    for k in keys:
        col = tr[k]
        out[k] = [col[i] for i in keep]
    return out


def slim_run(rec: dict, full_traces: bool = False) -> dict:
    for L in rec.get("legs") or []:
        for turn in L.get("transcript") or []:
            for c in turn.get("calls") or []:
                for k in CALL_DROP:
                    c.pop(k, None)
        if not full_traces and isinstance(L.get("trace"), dict):
            L["trace"] = thin_trace(L["trace"], L.get("events") or [])
    return rec


def _ints(x):
    """64.0 -> 64 (the same number in JavaScript, two bytes shorter in JSON)."""
    if isinstance(x, float):
        return int(x) if x.is_integer() and abs(x) < 2**53 else x
    if isinstance(x, list):
        return [_ints(v) for v in x]
    if isinstance(x, dict):
        return {k: _ints(v) for k, v in x.items()}
    return x


def _dump(obj, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_ints(obj), separators=(",", ":"), ensure_ascii=False)
    path.write_text(data, encoding="utf-8")
    return len(data.encode("utf-8"))


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build(out: str | Path, root: str | Path, runs_dir: str | Path | None = None, full_traces: bool = False, log=print,
          valid_only: bool = False, viewer_dir: str | Path | None = None) -> dict:
    root = Path(root)
    runs = Path(runs_dir) if runs_dir else root / "runs"
    out = Path(out)
    if out.exists():
        if any(out.iterdir()) and not (out / "explorer").is_dir():
            raise SystemExit(f"{out} exists and is not a previous export (no explorer/): refusing to overwrite it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # story page at the root, or a stand-in that opens the explorer
    story = root / "story"
    if (story / "index.html").is_file():
        shutil.copytree(story, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*", "__pycache__"))
    else:
        (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Trailer Relay</title>'
                                        '<meta http-equiv="refresh" content="0; url=explorer/">'
                                        '<p><a href="explorer/">Open the explorer</a></p>\n', encoding="utf-8")

    # the viewer, marked static
    ex = out / "explorer"
    shutil.copytree(Path(viewer_dir) if viewer_dir else root / "viewer", ex, ignore=shutil.ignore_patterns(*VIEWER_SKIP, ".*"))
    page = (ex / "index.html").read_text(encoding="utf-8")
    if STATIC_META not in page:
        page = page.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n' + STATIC_META, 1)
    (ex / "index.html").write_text(page, encoding="utf-8")

    # the data: runs in the digest, plus every study run
    R = json.loads((runs / "results.json").read_text())
    excluded_runs, excluded_studies = set(), set()
    if valid_only:
        from collections import defaultdict
        from .evaluation import study_reason

        by_run = defaultdict(list)
        for leg in R["legs"]:
            by_run[leg["run_id"]].append(leg)
        records = {r["run_id"]: r for r in R["runs"]}
        excluded_runs = {rid for rid, legs in by_run.items() if any(l.get("reason") == "model API error" for l in legs)}
        for study in R["studies"]:
            if study_reason(study, records, by_run):
                excluded_studies.add(study["study"])
                excluded_runs.update(r["run_id"] for r in study["runs"])
        R = {**R, "runs": [r for r in R["runs"] if r["run_id"] not in excluded_runs],
             "legs": [l for l in R["legs"] if l["run_id"] not in excluded_runs],
             "studies": [s for s in R["studies"] if s["study"] not in excluded_studies]}
    studies = []
    sdir = runs / "studies"
    names = [p.name for p in sorted(sdir.glob("*.json")) if p.name != "index.json"] if sdir.is_dir() else []
    for nm in names:
        study = json.loads((sdir / nm).read_text())
        if study["study"] not in excluded_studies:
            studies.append((nm, study))
    names = [nm for nm, _ in studies]
    want = {r["run_id"] for r in R.get("runs") or [] if r.get("run_id")}
    for _, s in studies:
        want |= {r["run_id"] for r in s.get("runs") or [] if r.get("run_id")}
    idx = json.loads((runs / "index.json").read_text())
    items, seen = [], set()
    for it in idx.get("runs") or []:
        rid = it.get("run_id") or Path(it["file"]).stem
        if rid not in want or it.get("live") or str(it.get("file", "")).startswith("live/") or rid in seen:
            continue
        if not (runs / it["file"]).is_file():
            continue
        seen.add(rid)
        items.append(it)
    missing = sorted(want - seen)
    raw = slim = 0
    for it in items:
        src = runs / it["file"]
        raw += src.stat().st_size
        slim += _dump(slim_run(json.loads(src.read_text()), full_traces), out / "runs" / it["file"])
    _dump({**idx, "runs": items}, out / "runs" / "index.json")
    _dump(R, out / "runs" / "results.json")
    for nm, s in studies:
        _dump(s, out / "runs" / "studies" / nm)
    _dump(names, out / "runs" / "studies" / "index.json")

    total = _size(out)
    info = {"out": str(out), "runs": len(items), "studies": len(studies), "missing": missing,
            "excluded_runs": len(excluded_runs), "excluded_studies": len(excluded_studies), "runs_raw_mb": raw / 1e6,
            "runs_mb": slim / 1e6, "explorer_mb": _size(ex) / 1e6, "total_mb": total / 1e6, "story": (story / "index.html").is_file()}
    log(f"exported {len(items)} runs ({raw / 1e6:.1f} MB -> {slim / 1e6:.1f} MB{'' if full_traces else ', traces thinned'}), "
        f"{len(studies)} studies, explorer {info['explorer_mb']:.1f} MB, story {'copied' if info['story'] else 'missing (redirect page)'}")
    if missing:
        log(f"warning: {len(missing)} runs named in results.json or the studies have no record in {runs}: {', '.join(missing[:5])}")
    log(f"total {total / 1e6:.1f} MB in {out}  (open {out / 'index.html'} through any static server)")
    return info
