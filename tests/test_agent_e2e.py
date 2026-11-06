"""End-to-end: model adapter -> agent loop -> simulator -> run record -> CLI."""
import json
import os
import subprocess
import sys

import pytest

from relay.agent import AgentConfig, LLMDriver, compact_history
from relay.autopilot import Autopilot
from relay.models import Message, ModelError, ScriptedAdapter
from relay.runner import run_relay, save_run
from relay.sim import Leg, LegConfig

from .conftest import ROOT


def oracle_policy(leg_ref):
    """A scripted 'model' that secretly uses the autopilot: proves the loop can finish a leg."""
    ap = Autopilot()
    state = {"setup": False}

    def policy(system, messages):
        leg = leg_ref["leg"]
        if not state["setup"]:
            ap._setup(leg)
            state["setup"] = True
        s, d = ap.plan(leg)
        return [(round(s, 2), round(d, 3))]

    return policy


def test_llm_driver_finishes_leg_with_oracle_policy(car_lturn):
    leg = Leg(car_lturn)
    ref = {"leg": leg}
    drv = LLMDriver(ScriptedAdapter("oracle", policy=oracle_policy(ref)))
    out = drv.run_leg(leg)
    assert leg.outcome == "FINISHED", leg.metrics()
    assert out["transcript"] and all(c["name"] == "drive" for e in out["transcript"] for c in e["calls"])
    first = out["transcript"][0]["calls"][0]
    assert "STATUS:" in first["report"] and first["result"]["traveled"] > 0
    assert out["messages"][0]["text"].startswith("LEG 1 of 1")


def test_idle_model_is_nudged_then_dnf(straight_car):
    leg = Leg(straight_car)
    ad = ScriptedAdapter("idle")
    out = LLMDriver(ad, AgentConfig(max_no_tool_turns=3)).run_leg(leg)
    assert leg.outcome == "DNF" and "stopped calling the tool" in leg.outcome_reason
    assert ad.calls == 3 and sum(1 for e in out["transcript"] if e.get("nudged")) == 2


def test_multiple_calls_and_leg_over_mid_turn(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, max_commands=2))
    pol = lambda s, m: [(0, -1), (0, -1), (0, -1)]  # noqa: E731
    out = LLMDriver(ScriptedAdapter("x", policy=pol)).run_leg(leg)
    calls = out["transcript"][0]["calls"]
    assert [c.get("command") for c in calls] == [1, 2, None]
    assert calls[2]["report"].startswith("LEG OVER")
    assert "LEG OVER: DNF" in calls[1]["report"]


def test_bad_tool_calls_are_reported_not_executed(straight_car):
    from relay.models.base import ToolCall
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, max_commands=1))
    seq = iter([Message("assistant", tool_calls=[ToolCall("a", "teleport", {}), ToolCall("b", "drive", {"steer_deg": "left"})]),
                [(0, -1)]])
    out = LLMDriver(ScriptedAdapter("x", policy=lambda s, m: next(seq))).run_leg(leg)
    reps = [c["report"] for c in out["transcript"][0]["calls"]]
    assert "Unknown tool" in reps[0] and "Invalid arguments" in reps[1]
    assert leg.commands == 1


def test_api_error_aborts_leg(straight_car):
    class Boom(ScriptedAdapter):
        def complete(self, *a, **k):
            raise ModelError("HTTP 500 forever")

    leg = Leg(straight_car)
    out = LLMDriver(Boom("x", policy=lambda s, m: [])).run_leg(leg)
    assert leg.outcome == "DNF" and out["api_error"] and "model API error" in leg.outcome_reason


def test_history_compaction_keeps_recent_reports():
    msgs = [Message("user", "brief")]
    for i in range(10):
        msgs.append(Message("assistant", tool_calls=[]))
        msgs.append(Message("tool", f"RESULT: r{i}\nSTATUS: REVERSING\nPROXIMITY lots of numbers", tool_call_id=str(i)))
    out = compact_history(msgs, keep_full=3)
    tools = [m for m in out if m.role == "tool"]
    assert sum(t.text.startswith("[older report") for t in tools) == 7
    assert "PROXIMITY" in tools[-1].text and "PROXIMITY" not in tools[0].text
    assert msgs[2].text.startswith("RESULT")  # original history untouched


def test_think_cost_charges_clock(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, think_s_per_1k_tokens=50.0, max_commands=1))
    LLMDriver(ScriptedAdapter("straight-back")).run_leg(leg)
    assert any(e["type"] == "think" for e in leg.events)


def test_run_relay_record_and_index(mini_relay, tmp_path):
    rec = run_relay(mini_relay, Autopilot(), "autopilot", log=None)
    assert rec["totals"]["legs_finished"] == 1 and rec["totals"]["relay_progress"] == pytest.approx(1.0, abs=0.01)
    L = rec["legs"][0]
    assert L["metrics"]["outcome"] == "FINISHED" and len(L["trace"]["t"]) > 10
    rec2 = run_relay(mini_relay, LLMDriver(ScriptedAdapter("straight-back")), "scripted:straight-back", log=None)
    assert rec2["legs"][0]["transcript"]
    p = save_run(rec, tmp_path)
    save_run(rec2, tmp_path)
    idx = json.loads((tmp_path / "index.json").read_text())
    assert len(idx["runs"]) == 2 and {r["file"] for r in idx["runs"]} >= {p.name}
    json.dumps(rec2)  # fully serialisable


def test_cli_end_to_end(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    runs, tracks = tmp_path / "runs", tmp_path / "tracks"
    base = [sys.executable, "-m", "relay", "--runs", str(runs), "--tracks", str(tracks)]
    r = subprocess.run(base + ["run", "--models", "autopilot", "scripted:hitch-keeper", "--relay", "mini", "--parallel", "2"],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "Autopilot" in r.stdout and "hitch-keeper" in r.stdout
    files = [p for p in runs.glob("*.json") if p.name not in ("index.json", "results.json")]
    assert len(files) == 2 and (tracks / "mini-s0.json").exists()
    r = subprocess.run(base + ["report"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 0 and "legs✓" in r.stdout


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RELAY_LIVE") != "1", reason="set RELAY_LIVE=1 to call the real API")
def test_live_gpt6_luna_drives(straight_car):
    from relay.models import make_adapter
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, max_commands=3))
    out = LLMDriver(make_adapter("openai:gpt-6-luna")).run_leg(leg)
    assert leg.commands >= 1 and out["usage"]["output_tokens"] > 0


def test_parallel_legs_match_sequential(mini_relay):
    """Legs are independent episodes: driving them concurrently must give the same record as in order."""
    relay = dict(mini_relay, tracks=list(mini_relay["tracks"]) * 3)

    def go(par):
        drv = LLMDriver(ScriptedAdapter("hitch-keeper"), AgentConfig(max_turns=30))
        return run_relay(relay, drv, "scripted:hk", log=None, parallel_legs=par)

    a, b = go(False), go(True)
    assert [L["metrics"] for L in a["legs"]] == [L["metrics"] for L in b["legs"]]
    assert [L["start_clock"] for L in a["legs"]] == [L["start_clock"] for L in b["legs"]]
    clocks = [L["start_clock"] for L in b["legs"]]
    assert clocks[0] == 0 and clocks[1] == round(b["legs"][0]["metrics"]["time"], 2)
    assert a["totals"]["relay_time"] == b["totals"]["relay_time"]


def test_checkpoints_are_partial_prefixes(mini_relay):
    relay = dict(mini_relay, tracks=list(mini_relay["tracks"]) * 2)
    seen = []
    run_relay(relay, Autopilot(), "autopilot", log=None, checkpoint=seen.append)
    assert len(seen) == 1 and seen[0]["partial"] is True and len(seen[0]["legs"]) == 1
    assert seen[0]["totals"]["legs"] == 2 and seen[0]["totals"]["legs_run"] == 1


def test_parallel_checkpoint_when_prefix_grows():
    """Drive completion order explicitly; simulator speed and scheduling must not decide the assertion."""
    from threading import Event
    from relay.runner import _drive_all

    gates = [Event() for _ in range(3)]
    gates[0].set()
    done, seen, completion = {}, [], []

    def run_one(i, track):
        assert gates[i].wait(10), f"leg {i} was never released"
        return i

    def finish(i, out):
        done[i] = out
        completion.append(i)
        if i == 0:
            gates[2].set()
        elif i == 2:
            gates[1].set()

    def prefix(records):
        out = []
        for i in range(3):
            if i not in records:
                break
            out.append(records[i])
        return out

    _drive_all([0, 1, 2], run_one, finish, done, prefix, True, seen.append, lambda: prefix(done))
    assert completion == [0, 2, 1]
    assert seen == [[0]]  # leg 2 adds no prefix; the complete relay is returned without a partial checkpoint
    assert prefix(done) == [0, 1, 2]
