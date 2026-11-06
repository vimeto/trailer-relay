"""python -m relay site: the static export (relay/site.py)."""
import json
from pathlib import Path

from relay.site import build, thin_trace

ROOT = Path(__file__).resolve().parent.parent


def _run(rid, n=200, live=False):
    t = [round(0.1 * i, 2) for i in range(n)]
    # still for the first 5 s, then reversing; one head turn at 12 s
    x = [0.0 if ti < 5 else round(-(ti - 5) * 0.5, 3) for ti in t]
    v = [0.0 if ti < 5 else -0.5 for ti in t]
    gaze = [0.0 if ti < 12 else (20.0 if ti < 12.2 else 40.0) for ti in t]
    leg = {"index": 0, "vehicle": "car", "metrics": {"outcome": "FINISHED", "time": t[-1]},
           "trace": {"t": t, "x": x, "y": [0.0] * n, "th1": [0.0] * n, "th2": [0.0] * n, "steer": [0.0] * n, "v": v, "gaze": gaze},
           "events": [{"t": 15.0, "type": "contact", "part": "trailer rear"}, {"t": 6.0, "type": "controller", "n": 1}],
           "transcript": [{"turn": 1, "t": 0.0, "text": "go", "calls": [{"name": "set_controller", "t0": 0.0, "args": {"code": "def control(o): pass"},
                                                                         "report": "x" * 500, "program": "def control(o): pass"}]}]}
    return {"run_id": rid, "created": "2026-11-05T12:00:00", "driver": {"spec": "codex:gpt-6.1-sol", "label": "GPT-6.1 Sol", "trial": 0},
            "config": {}, "relay": {"id": "standard-s0", "name": "standard", "seed": 0, "tracks": []}, "legs": [leg], "totals": {}, **({"live": True} if live else {})}


def _fixture(tmp_path):
    root = tmp_path / "root"
    runs = root / "runs"
    (runs / "studies").mkdir(parents=True)
    (runs / "live").mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer", target_is_directory=True)
    for rid in ("a1", "b2", "c3"):
        (runs / f"{rid}.json").write_text(json.dumps(_run(rid)))
    (runs / "live" / "d4.json").write_text(json.dumps(_run("d4", live=True)))
    items = [{"file": f"{r}.json", "run_id": r, "relay": {"id": "standard-s0", "name": "standard", "seed": 0}} for r in ("a1", "b2", "c3")]
    items.append({"file": "live/d4.json", "run_id": "d4", "live": True, "relay": {"id": "standard-s0", "name": "standard", "seed": 0}})
    (runs / "index.json").write_text(json.dumps({"runs": items}))
    # a1 is in the digest, b2 only in a study, c3 in neither (not exported)
    (runs / "results.json").write_text(json.dumps({"generated": "x", "runs": [{"run_id": "a1", "file": "a1.json"}], "legs": [], "studies": []}))
    (runs / "studies" / "s1.json").write_text(json.dumps({"study": "s1", "condition": "chain", "runs": [{"run_id": "b2", "role": "train", "step": 1, "member": 0, "legs": []}]}))
    (runs / "studies" / "index.json").write_text(json.dumps(["s1.json"]))
    return root


def test_site_export(tmp_path):
    root = _fixture(tmp_path)
    out = tmp_path / "site"
    info = build(out, root, log=lambda m: None)
    assert (out / "index.html").is_file() and "explorer/" in (out / "index.html").read_text()  # no story/: a redirect page
    page = (out / "explorer" / "index.html").read_text()
    assert '<meta name="relay-static" content="1">' in page
    assert (out / "explorer" / "app.js").is_file() and (out / "explorer" / "vendor" / "three.module.min.js").is_file()
    assert not (out / "explorer" / "mockup").exists()
    idx = json.loads((out / "runs" / "index.json").read_text())
    assert sorted(it["run_id"] for it in idx["runs"]) == ["a1", "b2"] and info["runs"] == 2 and info["missing"] == []
    assert not (out / "runs" / "c3.json").exists() and not (out / "runs" / "live").exists()
    assert json.loads((out / "runs" / "studies" / "index.json").read_text()) == ["s1.json"]
    assert (out / "runs" / "results.json").is_file()
    rec = json.loads((out / "runs" / "a1.json").read_text())
    call = rec["legs"][0]["transcript"][0]["calls"][0]
    assert "report" not in call and "program" not in call and call["args"]["code"]
    tr = rec["legs"][0]["trace"]
    assert len(tr["t"]) < 120 and tr["t"][0] == 0 and tr["t"][-1] == 19.9
    assert all(len(tr[k]) == len(tr["t"]) for k in tr)
    # rebuilding over a previous export is fine; over anything else it is refused
    build(out, root, log=lambda m: None)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    try:
        build(tmp_path / "other", root, log=lambda m: None)
        raise AssertionError("overwrote a foreign directory")
    except SystemExit:
        assert (tmp_path / "other" / "keep.txt").is_file()


def test_thin_trace_keeps_what_the_viewer_needs():
    tr = _run("x")["legs"][0]["trace"]
    th = thin_trace(tr, [{"t": 15.0, "type": "contact"}])
    t = th["t"]
    # the still start is folded to its two ends; motion is kept at about 5 Hz
    assert 0.0 in t and 4.9 in t and not any(0.0 < x < 4.9 for x in t)
    assert 5.0 in t  # the rig starts moving
    # the head turn (12.0 -> 12.2) and the sample after it are kept at full rate
    assert {11.9, 12.0, 12.1, 12.2, 12.3} <= set(t)
    # full rate within 0.5 s of the contact
    assert {round(14.5 + 0.1 * k, 2) for k in range(11)} <= set(t)
    # elsewhere, at most 0.2 s between samples while moving
    moving = [x for x in t if x >= 5.0]
    assert max(b - a for a, b in zip(moving, moving[1:])) <= 0.2 + 1e-9
