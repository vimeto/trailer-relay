"""Browser e2e: the static viewer loads real run records, renders, and responds to seeking.

Needs Playwright with a Chromium/Chrome available; skipped otherwise. Marked `browser`.
"""
from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from relay.agent import LLMDriver
from relay.autopilot import Autopilot
from relay.models.scripted import ScriptedAdapter
from relay.runner import run_relay, save_run
from relay.seeder import load_or_generate_relay

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Serve a copy of the project root whose runs/ holds a fresh autopilot run of the mini relay."""
    root = tmp_path_factory.mktemp("site")
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = load_or_generate_relay("mini", 0, ROOT / "tracks")
    rec = run_relay(relay, Autopilot(), "autopilot", label="Autopilot", log=None)
    save_run(rec, root / "runs")
    # a model-style run whose turns carry reasoning summaries (as reasoning models return them)
    rec2 = run_relay(relay, LLMDriver(ScriptedAdapter("hitch-keeper")), "scripted:hitch-keeper", label="Thinker", log=None)
    for e in rec2["legs"][0]["transcript"]:
        e["text"] = ""
        e["reasoning"] = "**Old plan**\n\nignore me\n\n**Keeping the trailer straight**\n\nHitch is small, so I back up gently."
    save_run(rec2, root / "runs")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    handler.log_message = lambda *a, **k: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rec
    httpd.shutdown()


@pytest.fixture(scope="module")
def page(site, browser):
    url, _ = site
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(url + "/viewer/?paused=1")
    pg.wait_for_function("window.__relayReady === true || !!window.__relayError", timeout=60000)
    pg.errors = errors
    yield pg
    pg.close()


def test_loads_without_errors(page):
    assert page.evaluate("window.__relayError || null") is None
    assert page.errors == []


def test_state_matches_record(page, site):
    _, rec = site
    st = page.evaluate("window.__relay.state()")
    assert st["relay"] == rec["relay"]["id"]
    assert len(st["runs"]) == 2
    auto = next(r for r in st["runs"] if r["label"] == "Autopilot")
    assert abs(auto["total"] - rec["totals"]["relay_time"]) < 0.05


def test_canvas_not_blank(page):
    page.evaluate("window.__relay.seek(5)")
    png = page.locator("#scene").screenshot()
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        assert len(png) > 20000  # a flat image compresses to almost nothing
        return
    im = Image.open(BytesIO(png)).convert("RGB").resize((64, 40))
    colors = {px for px in im.getdata()}
    assert len(colors) > 50


def test_seek_updates_hud(page, site):
    _, rec = site
    T = rec["totals"]["relay_time"]
    page.evaluate(f"window.__relay.seek({T / 2})")
    assert page.inner_text("#clock-time").strip() != "0:00.0"
    page.evaluate(f"window.__relay.seek({T})")
    st = page.evaluate("window.__relay.state()")
    assert next(r for r in st["runs"] if r["label"] == "Autopilot")["status"] == "FINISHED"
    assert "Parked" in page.inner_text("#board-rows")


def test_radio_shows_latest_reasoning_section(page):
    page.evaluate("window.__relay.focus(0)")  # models are listed before the autopilot
    page.evaluate("window.__relay.seek(6)")
    page.wait_for_timeout(100)
    assert page.inner_text("#radio-eyebrow").strip().lower().startswith("thinking")
    txt = page.inner_text("#radio-text")
    assert "Keeping the trailer straight" in txt and "back up gently" in txt and "ignore me" not in txt


def test_camera_modes(page):
    for cam in ("follow", "chase", "overview"):
        page.evaluate(f"window.__relay.cam('{cam}')")
        assert page.evaluate("window.__relay.state().cam") == cam
    assert page.errors == []


# --------------------------------------------------------------------------------------------
# Synthetic multi-variant site: several models/prompts/observation modes, a code-mode run, a
# cone-observation run and a live (unfinished) run. Records are hand-built from the warmup relay
# so these tests only depend on the record format, not on the simulator.

import json
import math
import shutil

WARMUP = ROOT / "tracks" / "warmup-s0.json"
CODE = "import math\n\n# back up while keeping the hitch small\nfor step in range(12):\n    h = hitch()\n    drive(steer=-1.5 * h, distance=-0.5)\nprint('hitch', round(hitch(), 1))\n" + "\n".join(f"# line {k}" for k in range(10))


def _leg(relay, i, *, time, outcome="FINISHED", running=False, start_clock=0.0, cone=False, code=False, progress=None):
    tr = relay["tracks"][i]
    x0, y0, th1, th2 = tr["start_state"]
    n = max(1, int(time / 0.5)) + 1
    ts = [round(min(time, k * 0.5), 3) for k in range(n)]
    ts[-1] = time
    v = -1.0
    trace = {
        "t": ts,
        "x": [x0 + v * t * math.cos(th1) * 0.3 for t in ts],
        "y": [y0 + v * t * math.sin(th1) * 0.3 for t in ts],
        "th1": [th1] * n, "th2": [th2] * n, "steer": [0.0] * n, "v": [v * 0.3] * n,
    }
    events = [{"type": "status", "t": 0.0, "status": "REVERSING"},
              {"type": "command", "t": 0.5, "steer_deg": 0, "distance_m": -3, "speed": 0.3}]
    if cone:
        events += [{"type": "look", "t": 4.0, "yaw_deg": 150.0}, {"type": "look", "t": 8.0, "yaw_deg": -30.0}]
    if not running:
        events.append({"type": "finish", "t": time, "outcome": outcome})
    if code:
        calls1 = [{"name": "run_program", "args": {"code": CODE}, "t0": 1.0, "t1": 6.0, "stdout": "hitch 2.1\nstill 9.8 m to go", "error": None,
                   "commands": [{"steer_deg": 0, "distance_m": -0.5}] * 12, "report": "12 commands", "result": {}}]
        calls2 = [{"name": "run_program", "args": {"code": "drive(steer=foo, distance=-1)"}, "t0": 10.0, "t1": 11.0, "stdout": "",
                   "error": "Traceback (most recent call last):\nNameError: name 'foo' is not defined", "commands": [], "report": "", "result": {}}]
        transcript = [{"turn": 1, "text": "", "reasoning": "**Writing a controller**\n\nA proportional hitch loop.", "calls": calls1},
                      {"turn": 2, "text": "", "reasoning": None, "calls": calls2}]
    else:
        calls = [{"name": "drive", "args": {"steer_deg": 0, "distance_m": -3, "speed_mps": 0.3}, "t0": 0.5, "t1": 10.5, "command": 1,
                  "result": {"traveled": 3.0, "stop_reason": "completed"}}]
        if cone:
            calls = [{"name": "look", "args": {"yaw_deg": 150}, "t0": 4.0, "t1": 4.0, "result": {"yaw_deg": 150}}] + calls
        transcript = [{"turn": 1, "text": "Backing straight in.", "calls": calls}]
    prog = 1.0 if outcome == "FINISHED" and not running else (progress if progress is not None else 0.4)
    leg = {
        "index": i, "vehicle": tr["vehicle"], "start_clock": start_clock, "trace": trace, "events": events, "transcript": transcript,
        "metrics": {"outcome": None if running else outcome, "time": time, "progress": prog, "touches": 1 if outcome != "FINISHED" else 0,
                    "shuttles": 2, "jackknifes": 0, "commands": 3, "short_m": 0 if outcome == "FINISHED" else 4.2},
        "usage": {"cost_usd": 0.01},
    }
    if running:
        leg["running"] = True
    if cone:
        leg["gaze0_deg"] = 180
    return leg


def _record(relay, run_id, label, spec, *, times, outcomes=None, prompt=2, obs="local", mode="tools", route="api", trial=0,
            live=False, running=(), cost=0.05, updated=1000):
    outcomes = outcomes or ["FINISHED"] * len(times)
    legs, clock = [], 0.0
    for i, (t, o) in enumerate(zip(times, outcomes)):
        legs.append(_leg(relay, i, time=t, outcome=o, running=i in running, start_clock=clock, cone=obs.startswith("cone"), code=mode == "code"))
        clock += t
    fin = sum(1 for L in legs if L["metrics"]["outcome"] == "FINISHED")
    rec = {
        "run_id": run_id, "created": f"2026-11-05T12:00:{trial:02d}", "driver": {"spec": spec, "label": label, "trial": trial},
        "partial": live, "relay": relay,
        "config": {"agent": {"obs_level": obs}, "prompt_version": prompt, "sim_version": 2, "harness_version": 2, "mode": mode, "route": route},
        "totals": {"legs": len(relay["tracks"]), "legs_run": len(legs), "legs_finished": fin, "relay_time": clock, "adjusted_time": clock + 5 * (len(legs) - fin),
                   "relay_progress": sum(L["metrics"]["progress"] for L in legs), "touches": len(legs) - fin, "shuttles": 2 * len(legs), "cost_usd": cost},
        "legs": legs,
    }
    if live:
        rec["live"] = True
        rec["updated"] = updated
    return rec


def _summary(rec, file):
    s = {k: rec[k] for k in ("run_id", "created", "driver", "partial", "totals", "config")}
    s["file"] = file
    s["relay"] = {k: rec["relay"][k] for k in ("id", "name", "seed")}
    s["legs"] = [{"vehicle": L["vehicle"], **L["metrics"]} for L in rec["legs"]]
    if rec.get("live"):
        s["live"] = True
        s["updated"] = rec["updated"]
    return s


def _write_site(root: Path, recs: dict[str, dict]):
    runs = root / "runs"
    if runs.exists():
        shutil.rmtree(runs)
    (runs / "live").mkdir(parents=True)
    for file, rec in recs.items():
        (runs / file).write_text(json.dumps(rec))
    (runs / "index.json").write_text(json.dumps({"runs": [_summary(r, f) for f, r in recs.items()]}))


def _variants(relay):
    return {
        "a1.json": _record(relay, "a1", "Alpha", "openai:alpha", times=[20, 30, 25, 40], cost=0.04),
        "a2.json": _record(relay, "a2", "Alpha", "openai:alpha", times=[22, 31, 60, 38], outcomes=["FINISHED", "FINISHED", "TIME_UP", "FINISHED"],
                           prompt=3, obs="cone", cost=0.06),
        "b1.json": _record(relay, "b1", "Beta", "openai:beta", times=[18, 28, 26, 35], obs="sketch", mode="code", route="codex", cost=0.0),
        "b2.json": _record(relay, "b2", "Beta", "openai:beta", times=[19, 27, 30, 36], obs="sketch", mode="code", route="codex", trial=1, cost=0.0),
        "live/g1.json": _record(relay, "g1", "Gamma", "openai:gamma", times=[20, 8, 5, 3], live=True, running=(1, 2, 3), cost=0.02),
    }


def _serve(root: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture(scope="module")
def vsite(tmp_path_factory):
    root = tmp_path_factory.mktemp("vsite")
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(WARMUP.read_text())
    recs = _variants(relay)
    _write_site(root, recs)
    httpd = _serve(root)
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root, relay, recs
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except Exception:  # noqa: BLE001
            try:
                b = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no browser available: {e}")
        yield b
        b.close()


def _open(browser, url):
    pg = browser.new_page(viewport={"width": 1440, "height": 900})
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.set_default_timeout(120000)  # swiftshader under CPU load is slow; a timeout alone is not a regression
    pg.goto(url, timeout=180000)
    pg.wait_for_function("window.__relayReady === true || !!window.__relayError", timeout=180000)
    assert pg.evaluate("window.__relayError || null") is None
    return pg


def _focus(pg, run_id):
    k = [r["runId"] for r in pg.evaluate("window.__relay.state()")["runs"]].index(run_id)
    pg.evaluate(f"window.__relay.focus({k})")


def test_heats_vs_relay_clock(browser, vsite):
    url, *_ = vsite
    pg = _open(browser, url + "/viewer/?paused=1&poll=60000")
    st = pg.evaluate("window.__relay.state()")
    assert st["anyLive"] and st["clockMode"] == "heats"  # a live run is selected -> heats by default
    assert all(l["start"] == 0 for r in st["runs"] for l in r["legs"])
    assert abs(st["Tmax"] - 60) < 1e-6  # longest single leg (a2 leg 3)
    assert not pg.locator("#heatlegs").is_hidden()
    pg.click("#clockmode [data-clock=relay]")
    st = pg.evaluate("window.__relay.state()")
    assert st["clockMode"] == "relay"
    a1 = next(r for r in st["runs"] if r["runId"] == "a1")
    assert [l["start"] for l in a1["legs"]] == [0, 20, 50, 75]
    assert abs(st["Tmax"] - (22 + 31 + 60 + 38)) < 1e-6
    assert pg.locator("#heatlegs").is_hidden()
    assert "clock=relay" in pg.url
    # pinned by the URL
    pg2 = _open(browser, url + "/viewer/?paused=1&poll=60000&clock=relay")
    assert pg2.evaluate("window.__relay.state().clockMode") == "relay"
    # heats + leg 3: the focused rig is on course 3
    pg2.evaluate("window.__relay.setClock('heats'); window.__relay.heatLeg(2); window.__relay.seek(10)")
    st = pg2.evaluate("window.__relay.state()")
    assert st["heatLeg"] == 2 and all(r["active"] == 2 for r in st["runs"])
    assert "Leg 3/4" in pg2.inner_text("#radio-leg")
    assert pg.errors == [] and pg2.errors == []
    pg.close(); pg2.close()


def test_run_picker_filtering(browser, vsite):
    url, *_ = vsite
    pg = _open(browser, url + "/viewer/?paused=1&poll=60000")
    assert len(pg.evaluate("window.__relay.state().runs")) == 5
    pg.click("#runs-btn")
    assert pg.locator("#runs-panel").is_visible()
    assert pg.locator("#run-list .run-item").count() == 5
    rl = pg.inner_text("#run-list")
    assert "Vision cone" in rl and "Writes its own controller" in rl and "driving now" in rl
    # facet chip: only Beta
    pg.click("#facets [data-facet=model] .fchip[data-value=Beta]")
    pg.wait_for_function("window.__relay.state().runs.length === 2")
    st = pg.evaluate("window.__relay.state()")
    assert sorted(r["runId"] for r in st["runs"]) == ["b1", "b2"]
    assert "f_model=Beta" in pg.url
    assert pg.locator("#board-rows .row:visible").count() == 2
    pg.click("#sel-clear")
    pg.wait_for_function("window.__relay.state().runs.length === 5")
    # uncheck one run -> persisted in ?runs=
    pg.click("#run-list .run-item[data-run=a2]")
    pg.wait_for_function("window.__relay.state().runs.length === 4")
    assert "runs=" in pg.url and "a2" not in pg.url.split("runs=")[1].split("&")[0]
    # a fresh page with that URL shows the same selection
    pg2 = _open(browser, pg.url)
    assert sorted(r["runId"] for r in pg2.evaluate("window.__relay.state().runs")) == ["a1", "b1", "b2", "g1"]
    # select none / all
    pg.click("#sel-none")
    pg.wait_for_function("window.__relay.state().runs.length === 0")
    pg.click("#sel-all")
    pg.wait_for_function("window.__relay.state().runs.length === 5")
    assert pg.errors == [] and pg2.errors == []
    pg.close(); pg2.close()


def test_compare_table(browser, vsite):
    url, *_ = vsite
    pg = _open(browser, url + "/viewer/?paused=1&poll=60000")
    pg.click("#compare-btn")
    assert pg.locator("#compare").is_visible()
    rows = pg.locator("#compare-table tbody tr")
    assert rows.count() == 4  # Alpha p2 local, Alpha p3 cone, Beta code (x2), Gamma live
    beta = pg.locator("#compare-table tbody tr", has_text="Beta")
    assert beta.locator("td[data-col=runs]").inner_text().strip() == "2"
    assert "4/4" in beta.locator("td[data-col=parked]").inner_text()
    assert "codex" in beta.inner_text()
    cone = pg.locator("#compare-table tbody tr", has_text="cone")
    assert "3/4" in cone.locator("td[data-col=parked]").inner_text()
    assert cone.locator("td[data-col=cost]").inner_text().startswith("$0.06")
    # first row is the best: everything parked, fastest adjusted time among fully parked variants
    first = rows.nth(0)
    assert "Beta" in first.inner_text()
    assert first.locator("td.best").count() >= 1
    # per-leg mini cells: 4 legs, one cell per run
    assert beta.locator(".legcell").count() == 4 and beta.locator(".legcell b.ok").count() == 8
    gamma = pg.locator("#compare-table tbody tr", has_text="Gamma")
    assert gamma.locator(".legcell b.run").count() == 3
    pg.keyboard.press("Escape")
    assert pg.locator("#compare").is_hidden()
    assert pg.errors == []
    pg.close()


def test_code_mode_radio(browser, vsite):
    url, *_ = vsite
    pg = _open(browser, url + "/viewer/?paused=1&poll=60000&clock=heats&leg=1")
    _focus(pg, "b1")
    pg.evaluate("window.__relay.seek(3)")  # program still running
    assert pg.inner_text("#radio-eyebrow").lower().startswith("program")
    assert pg.locator("#radio-text pre.prog").count() == 1
    assert "back up while keeping the hitch small" in pg.inner_text("#radio-text pre.prog")
    assert "more lines" in pg.inner_text("#radio-text pre.prog")
    assert "running" in pg.inner_text("#radio-text .prog-out")
    pg.evaluate("window.__relay.seek(7)")  # finished: stdout excerpt
    assert "still 9.8 m to go" in pg.inner_text("#radio-text .prog-out")
    assert "run_program" in pg.inner_text("#radio-cmd") and "12 commands" in pg.inner_text("#radio-cmd")
    pg.evaluate("window.__relay.seek(12)")  # second program raised
    assert pg.locator("#radio-text .prog-out.err").count() == 1
    assert "NameError" in pg.inner_text("#radio-text .prog-out.err")
    assert "error" in pg.inner_text("#radio-cmd")
    # trace.v -> speed readout
    assert pg.locator("#speed-read").is_visible() and "rev" in pg.inner_text("#speed-dir").lower() and "0.3 m/s" in pg.inner_text("#speed-val")
    assert pg.errors == []
    pg.close()


def test_gaze_cone(browser, vsite):
    url, *_ = vsite
    pg = _open(browser, url + "/viewer/?paused=1&poll=60000&clock=heats&leg=1")
    _focus(pg, "a2")
    pg.evaluate("window.__relay.seek(2)")
    g = pg.evaluate("window.__relay.gaze()")
    assert g["cone"] and g["visible"] and abs(g["yaw"] - 180) < 1e-6
    assert g["cones"] == 4  # heats: the focused run is driving all four courses, one cone each
    assert abs(g["R"] - 1.6 * 14) < 1e-6  # car: 1.6 x view_radius fallback
    pg.evaluate("window.__relay.seek(6)")
    assert abs(pg.evaluate("window.__relay.gaze().yaw") - 150) < 1e-6
    assert pg.locator("#gaze-read").is_visible() and "behind-left" in pg.inner_text("#gaze-val")
    assert "look(150°)" in pg.inner_text("#radio-cmd")
    pg.evaluate("window.__relay.seek(9)")
    assert abs(pg.evaluate("window.__relay.gaze().yaw") + 30) < 1e-6
    # leg over -> cone gone; non-cone runs never draw one
    pg.evaluate("window.__relay.seek(23)")
    assert not pg.evaluate("window.__relay.gaze().visible")
    _focus(pg, "a1")
    pg.evaluate("window.__relay.seek(6)")
    g = pg.evaluate("window.__relay.gaze()")
    assert not g["cone"] and g["cones"] == 0
    assert pg.errors == []
    pg.close()


def test_live_polling(browser, tmp_path):
    root = tmp_path / "live-site"
    root.mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(WARMUP.read_text())
    recs = {"a1.json": _record(relay, "a1", "Alpha", "openai:alpha", times=[20, 30, 25, 40]),
            "live/g1.json": _record(relay, "g1", "Gamma", "openai:gamma", times=[20, 8, 5, 3], live=True, running=(1, 2, 3), updated=1000)}
    _write_site(root, recs)
    httpd = _serve(root)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        pg = _open(browser, url + "/viewer/?poll=60000&focus=Gamma")  # playing: starts at the live edge
        st = pg.evaluate("window.__relay.state()")
        assert st["anyLive"] and st["clockMode"] == "heats" and st["followLive"]
        assert pg.locator("#live-pill").is_visible()
        g = next(r for r in st["runs"] if r["runId"] == "g1")
        assert g["live"] and g["legs"][1]["running"] and g["legs"][1]["time"] == 8
        pg.evaluate("window.__relay.cam('overview')")
        # nothing changed -> no rebuild
        assert pg.evaluate("window.__relay.refresh()") is False
        # the backend checkpoints: legs advanced, 'updated' bumped
        recs["live/g1.json"] = _record(relay, "g1", "Gamma", "openai:gamma", times=[20, 30, 12, 9], live=True, running=(2, 3), updated=1003)
        _write_site(root, recs)
        assert pg.evaluate("window.__relay.refresh()") is True
        st = pg.evaluate("window.__relay.state()")
        g = next(r for r in st["runs"] if r["runId"] == "g1")
        assert g["legs"][1]["time"] == 30 and not g["legs"][1]["running"] and g["legs"][2]["time"] == 12
        assert st["Tmax"] == 40 and st["T"] == pytest.approx(40, abs=1.0)  # kept following the live edge
        assert st["cam"] == "overview" and st["runs"][st["focus"]]["runId"] == "g1"  # view preserved
        # finished: the live file disappears and runs/g1.json appears; same run id, no longer live
        del recs["live/g1.json"]
        recs["g1.json"] = _record(relay, "g1", "Gamma", "openai:gamma", times=[20, 30, 26, 41])
        _write_site(root, recs)
        assert pg.evaluate("window.__relay.refresh()") is True
        st = pg.evaluate("window.__relay.state()")
        assert not st["anyLive"] and len(st["runs"]) == 2
        g = next(r for r in st["runs"] if r["runId"] == "g1")
        assert not g["live"] and g["file"] == "g1.json" and st["runs"][st["focus"]]["runId"] == "g1"
        assert pg.locator("#live-pill").is_hidden()
        # timed polling also works (no manual refresh)
        recs["a3.json"] = _record(relay, "a3", "Alpha", "openai:alpha", times=[21, 30, 25, 40], trial=1)
        _write_site(root, recs)
        pg2 = _open(browser, url + "/viewer/?paused=1&poll=300")
        recs["a4.json"] = _record(relay, "a4", "Alpha", "openai:alpha", times=[21, 30, 25, 41], trial=2)
        _write_site(root, recs)
        pg2.wait_for_function("window.__relay.state().runs.length === 4", timeout=10000)
        assert pg.errors == [] and pg2.errors == []
        pg.close(); pg2.close()
    finally:
        httpd.shutdown()


def test_record_with_missing_legs_and_many_runs(browser, tmp_path):
    """A record may skip courses (a leg crashed before its first snapshot); with many runs the standings fold into "+N more"."""
    root = tmp_path / "gap-site"
    root.mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(WARMUP.read_text())
    gap = _record(relay, "gap", "Gappy", "codex:gappy", times=[20, 30, 25, 40], live=True, running=(3,), route="codex", obs="cone")
    gap["legs"] = gap["legs"][2:]  # only courses 3 and 4
    recs = {"live/gap.json": gap}
    for k in range(9):
        recs[f"r{k}.json"] = _record(relay, f"r{k}", f"Model {k}", f"openai:m{k}", times=[20 + k, 30, 25, 40])
    _write_site(root, recs)
    httpd = _serve(root)
    try:
        pg = _open(browser, f"http://127.0.0.1:{httpd.server_address[1]}/viewer/?paused=1&poll=60000&focus=Gappy")
        for clock in ("heats", "relay"):
            pg.evaluate(f"window.__relay.setClock('{clock}')")
            for t in (0, 10, 35, 80):
                pg.evaluate(f"window.__relay.seek({t})")
                pg.wait_for_timeout(100)
        st = pg.evaluate("window.__relay.state()")
        assert len(st["runs"]) == 10
        # Broadcast standings show the top five (focus included) and fold the rest; the Grid has every run
        assert pg.locator("#board-rows .row:visible").count() == 5
        assert pg.locator("#board-rows .row.focus:visible").count() == 1
        assert "+5 more" in pg.inner_text("#board-more")
        pg.evaluate("window.__relay.view('grid')")
        assert pg.locator("#grid-tiles .gtile").count() == 10
        assert pg.errors == [], pg.errors
        pg.close()
    finally:
        httpd.shutdown()


# --------------------------------------------------------------------------------------------
# Views (Broadcast / Grid / Inspect) and in-scene chips


def _chip_site(relay):
    """Records exercising every chip: pinned on a rail, committed hitch, realtime thinking, phases."""
    recs = _variants(relay)
    # pinned: a contact at t=4, status PINNED from then on, a woke turn
    pin = _record(relay, "pin", "Pinner", "openai:pinner", times=[20, 30, 25, 40], outcomes=["TIME_UP", "FINISHED", "FINISHED", "FINISHED"])
    L = pin["legs"][0]
    tr = L["trace"]
    k = tr["t"].index(4.0)
    L["events"] += [{"type": "contact", "t": 4.0, "x": tr["x"][k], "y": tr["y"][k], "body": "trailer", "corner": "rear-left", "rail": "left"},
                    {"type": "status", "t": 4.0, "status": "PINNED"},
                    {"type": "phase", "t": 2.0, "phase": 1, "of": 3, "label": "back into the alley"}]
    L["events"].sort(key=lambda e: e["t"])
    # realtime code run: turn 2 asked at t=5 and answered at t=12; the controller keeps driving meanwhile
    rt = _record(relay, "rt", "Realtimer", "codex:rt", times=[20, 30, 25, 40], mode="code", route="codex")
    rt["config"].update(control="continuous", clock="realtime")
    R = rt["legs"][0]
    R["transcript"] = [
        {"turn": 1, "t": 0.0, "t_reply": 0.0, "think_sim_s": 0.0, "latency_s": 3.0, "text": "Installing a hitch controller.", "reasoning": "",
         "calls": [{"name": "set_controller", "args": {"code": "def control(obs):\n    return (0.0, -0.3, 30)"}, "t0": 0.0, "t1": 0.0, "result": {"status": "driving"}}]},
        {"turn": 2, "t": 5.0, "t_reply": 12.0, "think_sim_s": 7.0, "latency_s": 7.0, "text": "Keep going.", "reasoning": "",
         "calls": [{"name": "keep_driving", "args": {"check_in_s": 5}, "t0": 12.0, "t1": 12.0, "woke": ["controller check-in"], "result": {"status": "driving"}}]},
    ]
    R["events"] += [{"type": "controller", "t": 0.0, "n": 1}, {"type": "status", "t": 6.0, "status": "COMMITTED"}]
    R["events"].sort(key=lambda e: e["t"])
    recs["pin.json"] = pin
    recs["rt.json"] = rt
    return recs


@pytest.fixture(scope="module")
def csite(tmp_path_factory):
    root = tmp_path_factory.mktemp("csite")
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(WARMUP.read_text())
    _write_site(root, _chip_site(relay))
    httpd = _serve(root)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_view_switcher(browser, csite):
    pg = _open(browser, csite + "/viewer/?paused=1&poll=60000&clock=relay")
    assert pg.evaluate("window.__relay.state().view") == "broadcast"
    assert pg.locator("#board").is_visible() and pg.locator("#grid-view").is_hidden()
    pg.keyboard.press("g")
    assert pg.evaluate("window.__relay.view()") == "grid"
    assert pg.locator("#grid-view").is_visible() and pg.locator("#board").is_hidden() and pg.locator("#radio").is_hidden()
    assert "view=grid" in pg.url
    pg.keyboard.press("i")
    assert pg.evaluate("window.__relay.view()") == "inspect" and pg.locator("#inspect-view").is_visible()
    pg.keyboard.press("Escape")
    assert pg.evaluate("window.__relay.view()") == "broadcast" and "view=" not in pg.url
    pg.click("#views [data-view=grid]")
    assert pg.evaluate("window.__relay.view()") == "grid"
    pg.keyboard.press("b")
    assert pg.evaluate("window.__relay.view()") == "broadcast"
    # the view survives a reload through the URL
    pg2 = _open(browser, csite + "/viewer/?paused=1&poll=60000&view=inspect")
    assert pg2.evaluate("window.__relay.view()") == "inspect" and pg2.locator("#inspect-view").is_visible()
    assert pg.errors == [] and pg2.errors == []
    pg.close(); pg2.close()


def test_scene_chips(browser, csite):
    pg = _open(browser, csite + "/viewer/?paused=1&poll=60000&clock=heats&leg=1")
    pg.evaluate("window.__relay.cam('overview')")
    pg.evaluate("window.__relay.seek(8)")
    pg.wait_for_timeout(200)
    shown = pg.evaluate("window.__relay.chips()")
    labels = {c["run"]: c for c in shown if not c.get("ring")}
    rings = [c for c in shown if c.get("ring")]
    assert "pin" in labels and labels["pin"]["cls"] == "bad" and "Pinned" in labels["pin"]["text"]
    assert "touch #1" in labels["pin"]["text"] and "on the rail 4 s" in labels["pin"]["text"]
    assert any(r["key"].startswith("pin:") and "bad" in r["cls"] for r in rings)
    assert pg.locator("#chips .callout.bad:visible").count() >= 1
    # realtime run: committed hitch ring (warn) and the thinking bubble, but a single label per rig beyond the top one is allowed
    rt = [c for c in shown if c.get("run") == "rt" or c["key"].startswith("rt:")]
    assert any("warn" in c["cls"] for c in rt if c.get("ring"))
    assert any(c["cls"] == "think" and "Thinking 3 s" in c["text"] and "meanwhile" in c["text"] for c in shown if not c.get("ring"))
    # phase in the lower-third
    _focus(pg, "pin")
    assert "Phase 1/3" in pg.inner_text("#radio-leg") and "back into the alley" in pg.inner_text("#radio-leg")
    # normal driving shows no state chip; outside Broadcast nothing is drawn
    pg.evaluate("window.__relay.seek(1)")
    assert not [c for c in pg.evaluate("window.__relay.chips()") if c.get("cls") in ("bad", "warn", "think")]
    pg.evaluate("window.__relay.seek(8); window.__relay.view('grid')")
    assert pg.evaluate("window.__relay.chips()") == []
    assert pg.errors == []
    pg.close()


def test_grid_opens_inspect(browser, csite):
    pg = _open(browser, csite + "/viewer/?paused=1&poll=60000&clock=heats&leg=1&view=grid")
    pg.evaluate("window.__relay.seek(14)")
    tiles = pg.locator("#grid-tiles .gtile")
    assert tiles.count() == 7
    assert pg.locator("#grid-tiles .gtile[data-run=pin] .gt-map").count() == 1
    pg.click("#grid-tiles .gtile[data-run=rt]")
    st = pg.evaluate("window.__relay.state()")
    assert st["view"] == "inspect" and st["runs"][st["focus"]]["runId"] == "rt"
    pg.wait_for_timeout(300)
    assert pg.locator("#insp-svg .path").count() == 1 and pg.locator("#insp-svg .vehicle").count() == 1
    assert pg.locator("#insp-svg .thinkpath").count() >= 1  # realtime: path driven while thinking
    assert pg.locator("#insp-svg .ray").count() == 10  # 5 trailer + 5 vehicle rays (local observation)
    assert pg.locator("#insp-log .turn").count() == 2
    log = pg.inner_text("#insp-log")
    assert "Installing a hitch controller." in log and "thought 7.0 s" in log and "woke: controller check-in" in log
    assert pg.locator("#insp-charts polyline").count() >= 2  # hitch and speed
    assert pg.errors == []
    pg.close()


TURN = ROOT / "tracks" / "turnaround-s0.json"


@pytest.mark.skipif(not TURN.exists(), reason="no turnaround relay cached in tracks/")
def test_turnaround_course(browser, tmp_path):
    """Turnaround (intiaanikäännös) courses: one closed kerb polygon, two phase zones and routes, phase events."""
    root = tmp_path / "turn-site"
    root.mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(TURN.read_text())
    rec = _record(relay, "t1", "Turner", "openai:turner", times=[12, 12, 12, 12])
    L = rec["legs"][0]
    L["events"].append({"type": "phase", "t": 5.0, "phase": 2, "of": 2, "label": "drive out forward"})
    L["events"].sort(key=lambda e: e["t"])
    _write_site(root, {"t1.json": rec})
    httpd = _serve(root)
    try:
        pg = _open(browser, f"http://127.0.0.1:{httpd.server_address[1]}/viewer/?paused=1&poll=60000&clock=relay")
        pg.evaluate("window.__relay.seek(2)")
        assert "Phase 1/2 · reverse" in pg.inner_text("#radio-leg")
        pg.evaluate("window.__relay.seek(8)")
        assert "Phase 2/2 · forward" in pg.inner_text("#radio-leg") and "drive out forward" in pg.inner_text("#radio-leg")
        pg.evaluate("window.__relay.view('inspect')")
        pg.wait_for_timeout(300)
        assert pg.locator("#insp-svg .course .rail").count() == 1  # the kerb polygon, not left/right halves + end walls
        assert pg.locator("#insp-svg .course .guide").count() == 2  # phase routes
        assert pg.locator("#insp-svg .course .zone").count() == 1 and pg.locator("#insp-svg .course .bay").count() == 1  # side road + exit zone (= bay)
        assert pg.locator("#insp-svg .route").count() == 1  # phase-aware route arrow
        assert "Phase 2/2" in pg.inner_text("#insp-var")
        pg.evaluate("window.__relay.view('grid')")
        assert "phase 2/2" in pg.inner_text("#grid-tiles .gt-foot")
        assert pg.errors == [], pg.errors
        pg.close()
    finally:
        httpd.shutdown()


# --------------------------------------------------------------------------------------------
# Driver's-seat observation ("driver"): head, mirrors, reversing camera; asymmetric courses


def _driver_record(relay):
    """A driver-mode record: the car leg looks ahead, at the left mirror, turns, watches the reversing-camera screen,
    then looks over the left shoulder. The record carries no rig, so the viewer derives it (rig_for, ported)."""
    from relay.perception import rig_for
    from relay.vehicles import VEHICLES
    rec = _record(relay, "drv", "Driver", "openai:driver", times=[20, 30, 25, 40], obs="driver", mode="code")
    car = rig_for(VEHICLES["car"], "v1")  # records without a rig predate the difficulty levels
    lm, scr = car.mirror_gaze["left_mirror"], car.screen_gaze

    def gaze(t):
        if t < 2.0:
            return 0.0
        if t < 2.5:
            return 20.0  # turning towards the mirror
        if t <= 6.0:
            return lm
        if t < 7.0:
            return 10.0  # turning towards the screen
        if t < 12.0:
            return scr
        return 152.0
    for L in rec["legs"]:
        L["trace"]["gaze"] = [gaze(t) if L["index"] == 0 else 0.0 for t in L["trace"]["t"]]
    return rec, car


def test_driver_view(browser, tmp_path):
    root = tmp_path / "drv-site"
    root.mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(WARMUP.read_text())
    rec, car = _driver_record(relay)
    other = _record(relay, "loc", "Local", "openai:local", times=[20, 30, 25, 40])
    _write_site(root, {"drv.json": rec, "loc.json": other})
    httpd = _serve(root)
    try:
        pg = _open(browser, f"http://127.0.0.1:{httpd.server_address[1]}/viewer/?paused=1&poll=60000&clock=heats&leg=1&focus=Driver")
        # every run's vehicles carry the rig's mirrors (none on the mower); the truck's are clearly tall
        models = pg.evaluate("window.__relay.models()")
        assert len(models) == 2
        for m in models:
            by = {L["vehicle"]: L for L in m["legs"]}
            assert {k: len(v["mirrors"]) for k, v in by.items()} == {"car": 2, "tractor": 2, "mower": 0, "truck": 2}
            assert by["car"]["screen"] and by["truck"]["screen"] and not by["tractor"]["screen"]
            pos = {x["name"]: x["pos"] for x in by["car"]["mirrors"]}
            assert pos["left_mirror"] == pytest.approx(list(car.mirrors[0].pos)) and pos["right_mirror"] == pytest.approx(list(car.mirrors[1].pos))
            assert min(x["height"] for x in by["truck"]["mirrors"]) > 1.8 * max(x["height"] for x in by["car"]["mirrors"])
        # looking at the left mirror: the mirror glows and its view cone (axis +-fov/2, out to range) is drawn from it
        pg.evaluate("window.__relay.seek(4)")
        g = pg.evaluate("window.__relay.gaze()")
        assert g["driver"] and g["visible"] and g["at"] == "left_mirror" and not g["turning"]
        assert g["aid"] == "left_mirror" and g["glow"] == ["left_mirror"] and not g["screenGlow"]
        assert g["aidFov"] == pytest.approx(car.mirrors[0].fov_deg) and g["aidRange"] == pytest.approx(car.range)
        assert g["direct"] and g["directFov"] == pytest.approx(car.fov_deg)
        assert g["views"] == 4  # heats: the focused run only, one view per course it is driving
        assert pg.locator("#gaze-read").is_visible() and pg.inner_text("#gaze-val").strip() == "left mirror"
        # turning: nothing seen -> no mirror / camera view, nothing glows
        pg.evaluate("window.__relay.seek(6.6)")
        g = pg.evaluate("window.__relay.gaze()")
        assert g["turning"] and g["aid"] is None and g["glow"] == [] and g["words"] == "turning → camera"
        # watching the reversing-camera screen: the camera wedge from the trailer's rear, the screen glows
        pg.evaluate("window.__relay.seek(9)")
        g = pg.evaluate("window.__relay.gaze()")
        assert g["at"] == "camera" and g["aid"] == "camera" and g["screenGlow"] and g["glow"] == []
        assert g["aidRange"] == pytest.approx(car.camera_range) and g["aidFov"] == pytest.approx(car.camera_fov_deg)
        assert pg.inner_text("#gaze-val").strip() == "camera"
        pg.evaluate("window.__relay.seek(14)")
        assert pg.inner_text("#gaze-val").strip() == "over left shoulder 152°"
        assert pg.evaluate("window.__relay.gaze().aid") is None
        # the canvas shows the views (rendered, not blank)
        png = pg.locator("#scene").screenshot()
        assert len(png) > 20000
        # Inspect: view geometry instead of the local-map reconstruction
        pg.evaluate("window.__relay.seek(4); window.__relay.view('inspect')")
        pg.wait_for_timeout(300)
        assert pg.locator("#insp-svg .vcone.direct").count() == 1 and pg.locator("#insp-svg .vcone.aid.left_mirror").count() == 1
        assert pg.locator("#insp-svg .mirror").count() == 2 and pg.locator("#insp-svg .mirror.on").count() == 1 and pg.locator("#insp-svg .eye").count() == 1
        assert pg.locator("#insp-svg .ray").count() == 0 and pg.locator("#insp-svg .mapdot").count() == 0
        assert "looking: left mirror" in pg.inner_text("#insp-chips")
        assert "not recorded" in pg.inner_text("#insp-legend")
        pg.evaluate("window.__relay.seek(9)")
        pg.wait_for_timeout(300)
        assert pg.locator("#insp-svg .vcone.aid.camera").count() == 1 and "looking: camera" in pg.inner_text("#insp-chips")
        # other observation levels keep the reconstruction and never draw driver views
        pg.evaluate("window.__relay.view('broadcast')")
        _focus(pg, "loc")
        pg.evaluate("window.__relay.seek(4)")
        g = pg.evaluate("window.__relay.gaze()")
        assert not g.get("driver") and pg.locator("#gaze-read").is_hidden()
        pg.evaluate("window.__relay.view('inspect')")
        pg.wait_for_timeout(300)
        assert pg.locator("#insp-svg .ray").count() == 10 and pg.locator("#insp-svg .vcone").count() == 0
        assert pg.errors == [], pg.errors
        pg.close()
    finally:
        httpd.shutdown()


ASYM = ROOT / "tracks" / "asym-s0.json"


@pytest.mark.skipif(not ASYM.exists(), reason="no asym relay cached in tracks/")
def test_asymmetric_course(browser, tmp_path):
    """Asymmetric corridors: the bay is its own rectangle against one rail of a wider yard, the surface follows the
    rails, and the themed scenery keeps out of the real (wider) corridor."""
    root = tmp_path / "asym-site"
    root.mkdir()
    (root / "viewer").symlink_to(ROOT / "viewer")
    relay = json.loads(ASYM.read_text())
    _write_site(root, {"a.json": _record(relay, "a", "Asym", "openai:asym", times=[12, 12, 12, 12])})
    httpd = _serve(root)
    try:
        pg = _open(browser, f"http://127.0.0.1:{httpd.server_address[1]}/viewer/?paused=1&poll=60000&cam=overview")
        for i, tr in enumerate(relay["tracks"]):
            g = tr["geometry"]
            c = pg.evaluate(f"window.__relay.course({i})")
            assert c["asym"] is not None
            assert sum(c["bay"], []) == pytest.approx(sum(g["dock_corners"], []), abs=1e-3)  # painted exactly where the dock is
            L, R = g["left_rail"][-1], g["right_rail"][-1]
            corridor = math.dist(L, R)
            bay_w = math.dist(g["dock_corners"][0], g["dock_corners"][1])
            assert c["endWidth"] == pytest.approx(corridor, abs=1e-3)  # surface out to the real rails
            assert bay_w < corridor - 0.5, (tr["vehicle"], bay_w, corridor)  # the bay does not span the corridor
            assert c["propsChecked"] > 0 and c["propsInside"] == 0, (tr["vehicle"], c["propsInside"])
        # Grid and Inspect draw the same quad as the bay, inside the road polygon
        pg.evaluate("window.__relay.view('grid')")
        pg.wait_for_timeout(300)
        assert pg.locator("#grid-tiles .gt-map .bay").count() == 1
        assert len(pg.get_attribute("#grid-tiles .gt-map .bay", "points").split()) == 4
        pg.evaluate("window.__relay.view('inspect')")
        pg.wait_for_timeout(300)
        bb = pg.evaluate("[document.querySelector('#insp-svg .course .bay').getBBox(), document.querySelector('#insp-svg .course .road').getBBox()].map(b => b.width * b.height)")
        assert 0 < bb[0] < 0.25 * bb[1]
        assert pg.errors == [], pg.errors
        pg.close()
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------- Results view (runs/results.json + studies)
STUDY = "20261105-1300-self-gpt-6.1-sol-allow-ab12"


def _drv(rec, level="medium", created="2026-11-05T12:00:00", learning=None):
    rec["config"].update({"agent": {"obs_level": "driver"}, "perception": {"level": level}, "control": "continuous", "clock": "paused",
                          "harness_version": 4, "mode": "code", "route": "codex"})
    rec["created"] = created
    if learning:
        rec["learning"] = learning
    return rec


def _results_site(root: Path):
    """Calibration runs (Sol x2, Terra, a scripted baseline), a two-attempt 'self' learning study with an accepted note,
    a note accepted after a rejected draft and a fully rejected note, plus older runs so the course set has > 10 runs."""
    relay = json.loads(WARMUP.read_text())
    relay = {**relay, "id": "standard-s0", "name": "standard", "seed": 0}
    R = lambda *a, **k: _record(relay, *a, **k)  # noqa: E731
    fail = ["FINISHED", "FINISHED", "DNF", "TIME_UP"]
    recs = {
        "m1.json": _drv(R("m1", "GPT-6.1 Sol", "codex:gpt-6.1-sol", times=[20, 30, 25, 40], outcomes=fail), created="2026-11-05T12:00:01"),
        "m2.json": _drv(R("m2", "GPT-6.1 Sol", "codex:gpt-6.1-sol", times=[22, 31, 26, 41], trial=1), created="2026-11-05T12:00:02"),
        "m3.json": _drv(R("m3", "GPT-6.1 Terra", "codex:gpt-6.1-terra", times=[20, 30, 25, 40], outcomes=["DNF"] * 4), created="2026-11-05T12:00:03"),
        "b1.json": _drv(R("b1", "ctrl-camera", "scripted:ctrl-camera", times=[20, 30, 25, 40], outcomes=["TIME_UP"] * 4), level="hard", created="2026-11-05T12:00:04"),
    }
    L = lambda step: {"study": STUDY, "condition": "self", "role": "train", "step": step, "member": 0, "numbers": "allow", "level": "medium"}  # noqa: E731
    recs["l1.json"] = _drv(R("l1", "GPT-6.1 Sol", "codex:gpt-6.1-sol", times=[20, 30, 25, 40], outcomes=fail), created="2026-11-05T13:00:01", learning=L(1))
    recs["l2.json"] = _drv(R("l2", "GPT-6.1 Sol", "codex:gpt-6.1-sol", times=[20, 30, 25, 40]), created="2026-11-05T13:10:01", learning=L(2))
    for k in range(8):  # older runs on the same courses: Broadcast then opens on the newest experiment only
        recs[f"o{k}.json"] = R(f"o{k}", "GPT-6 Luna", "openai:gpt-6-luna", times=[20, 30, 25, 40], trial=k)
        recs[f"o{k}.json"]["created"] = f"2026-11-04T09:00:{k:02d}"
    notes = {
        "l1": [{"text": "Swing wide early, then hold the angle.", "accepted": True, "rejections": [["it contains 9 numbers; at most 6 are allowed"]],
                "drafts": ["Steer 0.3 for 2 s, then 0.1 for 4 s, 1 2 3 4 5 6 7.", "Swing wide early, then hold the angle."]},
               {"text": "steer(0.3)", "accepted": False, "rejections": [["too long (700 characters; the limit is 600)"], ["it contains code ('steer(0.3)')"]],
                "drafts": ["A very long note.", "steer(0.3)"]}, None, None],
        "l2": [None] * 4,
    }
    for rid in ("l1", "l2"):
        for i, leg in enumerate(recs[f"{rid}.json"]["legs"]):
            leg["note"] = notes[rid][i]
            leg["notes_in"] = [{"from": "you, attempt 1", "text": notes["l1"][i]["text"]}] if rid == "l2" and notes["l1"][i] and notes["l1"][i]["accepted"] else []
    _write_site(root, recs)
    idx = json.loads((root / "runs" / "index.json").read_text())
    for s in idx["runs"]:
        s["created"] = recs[s["file"]]["created"]
        if recs[s["file"]].get("learning"):
            s["learning"] = recs[s["file"]]["learning"]
    (root / "runs" / "index.json").write_text(json.dumps(idx))
    mode = {"FINISHED": None, "DNF": "rail-hit loop", "TIME_UP": "out of time"}
    runs, legs = [], []
    for file, rec in recs.items():
        if rec["config"].get("harness_version") != 4:
            continue
        runs.append({"run_id": rec["run_id"], "file": file, "created": rec["created"], "model": rec["driver"]["spec"], "label": rec["driver"]["label"],
                     "trial": rec["driver"]["trial"], "relay": relay["id"], "relay_name": "standard", "seed": 0, "obs": "driver",
                     "level": rec["config"]["perception"]["level"], "mode": "code", "clock": "paused", "learning": rec.get("learning"),
                     "scripted": rec["driver"]["spec"].startswith("scripted:"), "parked": rec["totals"]["legs_finished"], "legs": 4,
                     "progress": rec["totals"]["relay_progress"], "touches": rec["totals"]["touches"], "time": rec["totals"]["relay_time"]})
        for lg in rec["legs"]:
            m = lg["metrics"]
            contacts = [] if m["outcome"] == "FINISHED" else [{"t": 3.0, "part": "trailer rear", "frac": 0.62, "bend": True, "gaze": "camera"}]
            legs.append({"run_id": rec["run_id"], "index": lg["index"], "vehicle": lg["vehicle"], "track": f"t{lg['index']}", "outcome": m["outcome"],
                         "reason": None, "time": m["time"], "progress": m["progress"], "touches": m["touches"], "mode": mode[m["outcome"]],
                         "culprit": contacts[0]["part"] if contacts else None, "contacts": contacts,
                         "gaze": {"camera": 0.5, "mirror": 0.2, "ahead": 0.2, "side": 0.05, "back": 0.05}})
    study = {"study": STUDY, "model": "codex:gpt-6.1-sol", "condition": "self", "level": "medium", "seed": 0, "transfer_seed": None, "members": 1,
             "steps": 2, "policy": {"numbers": "allow", "max_chars": 600, "max_numbers": 6, "retries": 1}, "prompt_version": 1,
             "runs": [{"run_id": rid, "role": "train", "step": k + 1, "member": 0, "relay": relay["id"],
                       "legs": [{"index": lg["index"], "vehicle": lg["vehicle"], "outcome": lg["metrics"]["outcome"], "time": lg["metrics"]["time"],
                                 "progress": lg["metrics"]["progress"], "touches": lg["metrics"]["touches"], "notes_in": lg["notes_in"], "note": lg["note"]}
                                for lg in recs[f"{rid}.json"]["legs"]]} for k, rid in enumerate(("l1", "l2"))]}
    (root / "runs" / "studies").mkdir()
    (root / "runs" / "studies" / f"{STUDY}.json").write_text(json.dumps(study))
    (root / "runs" / "studies" / "index.json").write_text(json.dumps([f"{STUDY}.json"]))
    (root / "runs" / "results.json").write_text(json.dumps({"generated": "2026-11-05T14:00:00", "runs": runs, "legs": legs, "studies": [study]}))


@pytest.fixture(scope="module")
def rsite(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsite")
    (root / "viewer").symlink_to(ROOT / "viewer")
    _results_site(root)
    httpd = _serve(root)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_results_landing_matrix_and_runs(browser, rsite):
    pg = _open(browser, rsite + "/viewer/?paused=1&poll=60000")
    # no runs named in the URL: Results is the landing view, and the 3D scene has not been loaded
    assert pg.evaluate("window.__relay.view()") == "results"
    st = pg.evaluate("window.__relay.state()")
    assert st["lazy"] is True and st["runs"] == []
    res = pg.evaluate("window.__relay.results()")
    assert res["section"] == "overview" and res["digest"] == 6 and res["studies"] == 1 and res["runs"] == 14
    m = pg.locator(".matrix")
    assert "Sol" in m.inner_text() and "Terra" in m.inner_text() and "Naive controller (camera)" in m.inner_text()
    sol = pg.locator(".matrix tbody tr", has_text="Sol").first
    # Sol medium: 6 of 8 legs parked over two runs (learning runs are not in the matrix)
    assert "6/8 legs" in sol.inner_text() and "75%" in sol.inner_text()
    sol.locator(".mc").first.click()
    res = pg.evaluate("window.__relay.results()")
    assert res["section"] == "runs" and sorted(res["ids"]) == ["m1", "m2"] and sorted(res["shown"]) == ["m1", "m2"]
    assert pg.locator("#rt-body tr").count() == 2
    # split by vehicle shows a row per vehicle in each cell
    pg.click(".res-nav [data-sec=overview]")
    pg.click(".rseg[data-ctl=split] button[data-v=vehicle]")
    assert pg.locator(".matrix .msrow").count() >= 4 * 3
    # runs table: select two runs and open exactly those in Broadcast
    pg.click(".res-nav [data-sec=runs]")
    assert pg.locator("#rt-body tr").count() == 14
    pg.fill("input[data-ctl=rt-q]", "terra")
    assert pg.locator("#rt-body tr").count() == 1
    pg.fill("input[data-ctl=rt-q]", "")
    pg.click("#rt-body tr[data-id=m1]")
    pg.click("#rt-body tr[data-id=m3]")
    assert sorted(pg.evaluate("window.__relay.results().selected")) == ["m1", "m3"]
    pg.click(".selbar [data-act=broadcast]")
    pg.wait_for_function("window.__relay.state().runs.length === 2 && window.__relay.view() === 'broadcast'")
    st = pg.evaluate("window.__relay.state()")
    assert st["view"] == "broadcast" and sorted(st["shown"]) == ["m1", "m3"] and st["lazy"] is False
    assert sorted(r["name"] for r in st["runs"]) == ["GPT-6.1 Sol · Driver's seat, Medium · Course set A · run 1", "GPT-6.1 Terra · Driver's seat, Medium · Course set A · run 1"]
    assert "GPT-6.1 Sol · Driver's seat, Medium" in pg.inner_text("#radio-name") or "Terra" in pg.inner_text("#radio-name")
    # R goes back to Results; B / G / I still work
    pg.keyboard.press("r")
    assert pg.evaluate("window.__relay.view()") == "results" and pg.locator("#results-view").is_visible() and pg.locator("#board").is_hidden()
    pg.keyboard.press("g")
    assert pg.evaluate("window.__relay.view()") == "grid"
    pg.wait_for_function("document.querySelectorAll('#grid-tiles .gtile').length === 2")  # tiles draw on the next frame
    assert pg.errors == [], pg.errors
    pg.close()


def test_results_failures_learning_and_notes(browser, rsite):
    pg = _open(browser, rsite + "/viewer/?view=results&sec=failures&paused=1&poll=60000")
    assert pg.evaluate("window.__relay.results().section") == "failures"
    assert pg.locator(".sbar b").count() >= 3 and pg.locator(".hist svg").count() >= 1 and pg.locator(".gpair").count() >= 1
    pg.evaluate("window.__relay.section('learning')")
    assert pg.locator("svg.curve").count() == 2  # parked share and progress for the one 'self' study
    assert pg.locator("svg.curve line.ref").count() == 2  # the no-notes baseline (Sol, medium, seed 0) in each
    assert "1/1" in pg.inner_text(".mono")
    # notes browser: accepted note with its rejected draft, a fully rejected note, and who read the accepted one
    pg.evaluate("window.__relay.section('notes')")
    txt = pg.inner_text(".ncols")
    assert "Swing wide early, then hold the angle." in txt and "ACCEPTED" in txt.upper()
    assert "Attempt 1" in txt and "Attempt 2" in txt and "READ BY" in txt.upper()
    pg.click(".drafts summary")
    assert "at most 6 are allowed" in pg.inner_text(".ncols")
    pg.click(".rseg[data-ctl=nt-leg] button[data-v='1']")
    txt = pg.inner_text(".ncols").lower()
    assert "nothing passed on" in txt and "it contains code" in txt and "too long" in txt
    # a note's attempt opens in Inspect at that leg, where the notes card shows the same note
    pg.evaluate("window.__relay.section('notes')")
    pg.click(".rseg[data-ctl=nt-leg] button[data-v='0']")
    pg.locator(".nentry").first.locator("[data-view=inspect]").click()
    pg.wait_for_function("window.__relay.state().view === 'inspect' && window.__relay.state().runs.length > 0")
    pg.wait_for_function("document.querySelector('#insp-notes') && !document.querySelector('#insp-notes').hidden")
    notes = pg.inner_text("#insp-notes")
    assert "note accepted" in notes.lower() and "Swing wide early" in notes
    assert pg.evaluate("window.__relay.state().heatLeg") == 0
    assert pg.errors == [], pg.errors
    pg.close()


def test_broadcast_default_selection(browser, rsite):
    # Broadcast named in the URL, no runs: 14 runs on these courses -> only the newest experiment (the study), with a hint
    pg = _open(browser, rsite + "/viewer/?view=broadcast&paused=1&poll=60000")
    st = pg.evaluate("window.__relay.state()")
    assert st["view"] == "broadcast" and sorted(st["shown"]) == ["l1", "l2"] and len(st["groupRuns"]) == 14
    assert st["hint"]["n"] == 2 and pg.locator("#bc-hint").is_visible() and "2 of 14" in pg.inner_text("#bc-hint")
    names = [r["name"] for r in st["runs"]]
    assert sorted(names) == ["GPT-6.1 Sol · Attempt 1 · driver 1", "GPT-6.1 Sol · Attempt 2 · driver 1"]
    # P opens the picker (search narrows the list)
    pg.keyboard.press("p")
    assert pg.locator("#runs-panel").is_visible()
    pg.fill("#rp-search", "luna")
    assert pg.locator("#run-list .run-item").count() == 8
    assert pg.errors == [], pg.errors
    pg.close()


def test_run_labels(browser, rsite):
    pg = _open(browser, rsite + "/viewer/?paused=1&poll=60000")
    out = pg.evaluate("""async () => {
      const L = await import('./labels.js');
      const rec = (id, spec, label, trial, extra = {}) => ({ run_id: id, created: '2026-11-05T12:00:00', driver: { spec, label, trial },
        relay: { id: 'standard-s1', name: 'standard', seed: 1 },
        config: { control: 'continuous', clock: 'paused', mode: 'code', route: 'codex', agent: { obs_level: 'driver' }, perception: { level: 'medium' } }, ...extra });
      const learn = (cond, step, member, role = 'train') => ({ learning: { study: 's', condition: cond, role, step, member, numbers: 'allow' } });
      const row = { run_id: 'r1', model: 'scripted:ctrl-camera', label: 'ctrl-camera', trial: 0, relay: 'standard-s0', relay_name: 'standard', seed: 0, obs: 'driver', level: 'hard', clock: 'paused' };
      const twins = [rec('a', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 0), rec('b', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 0)];
      twins[1].created = '2026-11-05T12:07:00';
      return {
        driver: L.runLabel(rec('x', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 1)).text,
        mini: L.runLabel(rec('x', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 1)).mini,
        self: L.runLabel(rec('x', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 1, learn('self', 2, 0))).text,
        pool: L.runLabel(rec('x', 'codex:gpt-6.1-terra', 'GPT-6.1 Terra', 0, learn('pool', 1, 1))).text,
        chain: L.runLabel(rec('x', 'codex:gpt-6-luna', 'GPT-6 Luna', 0, learn('chain', 3, 0))).text,
        transfer: L.runLabel(rec('x', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 0, learn('self', 4, 1, 'transfer'))).text,
        row: L.runLabel(row).text,
        twins: L.uniqueLabels(twins).map((l) => l.text),
        colors: [L.modelColor({ spec: 'codex:gpt-6.1-sol', model: 'GPT-6.1 Sol' }), L.modelColor({ spec: 'scripted:ctrl-camera', model: 'x' })],
        hook: window.__relay.label(rec('x', 'codex:gpt-6.1-sol', 'GPT-6.1 Sol', 1)).text,
      };
    }""")
    assert out["driver"] == "GPT-6.1 Sol · Driver's seat, Medium · Course set B · run 2" and out["mini"] == "Sol · run 2" and out["hook"] == out["driver"]
    assert out["self"] == "GPT-6.1 Sol · Attempt 2 · driver 1"
    assert out["pool"] == "GPT-6.1 Terra · Team round 1 · driver 2"
    assert out["chain"] == "GPT-6 Luna · Notebook relay · generation 3"
    assert out["transfer"] == "GPT-6.1 Sol · New courses (transfer) · driver 2"
    assert out["row"] == "Naive controller (camera) · Driver's seat, Hard · Course set A · run 1"
    assert out["twins"] == ["GPT-6.1 Sol · Driver's seat, Medium · Course set B · run 1 · 12:00", "GPT-6.1 Sol · Driver's seat, Medium · Course set B · run 1 · 12:07"]
    assert out["colors"][0] == "#3987e5" and out["colors"][1] != out["colors"][0]
    assert pg.errors == [], pg.errors
    pg.close()
