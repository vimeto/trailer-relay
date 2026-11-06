"""Code mode (programs), look/cone tools, the codex CLI adapter, the cost ledger and live run files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import threading
import time

import pytest

from relay.agent import AgentConfig, LLMDriver, compact_history
from relay.costs import Budget, BudgetExceeded, usd
from relay.models.base import Message, ToolCall
from relay.models.codex import CodexAdapter, render_conversation
from relay.models.scripted import ScriptedAdapter
from relay.program import ProgramRunner
from relay.prompts import system_prompt, tools_for
from relay.runner import leg_config_for, route_of, run_relay, run_trials, save_run, update_index
from relay.sim import Leg, LegConfig

P_CONTROLLER = """
while True:
    o = observe()
    if o["progress"]["remaining"] <= 0.2:
        break
    h = o["hitch_deg"]
    try:
        drive(max(-20, min(20, 1.5 * h)), -1.0)
    except LegOver as e:
        print("over:", e)
        break
print("done", observe()["status"])
"""


def program_policy(code):
    def policy(system, messages):
        if any(m.role == "tool" for m in messages):
            return Message(role="assistant", text="", tool_calls=[ToolCall(id="p2", name="run_program", arguments={"code": "print(1/0)"})])
        return Message(role="assistant", text="writing a controller",
                       tool_calls=[ToolCall(id="p1", name="run_program", arguments={"code": code})])
    return policy


# ----------------------------------------------------------------------------- program runner
def test_program_drives_and_reports(mini_relay):
    leg = Leg(mini_relay["tracks"][0], config=leg_config_for("code"))
    res = ProgramRunner(leg, "local").run(P_CONTROLLER)
    assert res.error is None and not res.timed_out
    assert len(res.commands) > 5 and res.calls > len(res.commands)
    assert leg.outcome == "FINISHED" and "done FINISHED" in res.stdout or "over:" in res.stdout


def test_program_errors_and_limits(mini_relay):
    leg = Leg(mini_relay["tracks"][0], config=leg_config_for("code"))
    res = ProgramRunner(leg, "local").run("drive('x', 1)")
    assert res.error and "ValueError" in res.error
    res = ProgramRunner(leg, "local", wall_timeout_s=2).run("while True:\n    pass")
    assert res.timed_out and "stopped" in res.error
    res = ProgramRunner(leg, "local", max_calls=3).run("for i in range(10):\n    observe()\nprint('never')")
    assert "program stopped" in res.stdout and "never" not in res.stdout
    res = ProgramRunner(leg, "local", stdout_limit=200).run("print('x' * 5000)")
    assert len(res.stdout) < 300 and "truncated" in res.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_program_sandbox_blocks_secrets_network_and_writes(mini_relay, tmp_path):
    leg = Leg(mini_relay["tracks"][0])
    secret = Path.home() / ".secrets"
    code = f"""
import socket
for what, fn in [
    ("read-home", lambda: open({str(secret)!r}).read()),
    ("listdir-home", lambda: __import__('os').listdir({str(Path.home())!r})),
    ("write-outside", lambda: open({str(tmp_path / 'pwned.txt')!r}, 'w').write('x')),
    ("network", lambda: socket.create_connection(("1.1.1.1", 80), timeout=3)),
]:
    try:
        fn()
        print(what, "ALLOWED")
    except Exception as e:
        print(what, "blocked", type(e).__name__)
"""
    res = ProgramRunner(leg, "local").run(code)
    assert "ALLOWED" not in res.stdout, res.stdout
    assert res.stdout.count("blocked") == 4
    assert not (tmp_path / "pwned.txt").exists()


def test_program_env_has_no_secrets(mini_relay, monkeypatch):
    monkeypatch.setenv("OAI_API_KEY", "sk-should-not-leak")
    leg = Leg(mini_relay["tracks"][0])
    res = ProgramRunner(leg, "local").run("import os; print(sorted(os.environ))")
    assert "OAI_API_KEY" not in res.stdout and "sk-should" not in res.stdout


# ----------------------------------------------------------------------------- agent: code mode + look
def test_agent_code_mode(mini_relay):
    cfg = AgentConfig(mode="code")
    leg = Leg(mini_relay["tracks"][0], config=leg_config_for("code"))
    out = LLMDriver(ScriptedAdapter(policy=program_policy(P_CONTROLLER)), cfg).run_leg(leg)
    assert leg.outcome == "FINISHED"
    call = out["transcript"][0]["calls"][0]
    assert call["name"] == "run_program" and call["commands"] and call["t1"] > call["t0"]
    assert call["report"].startswith("PROGRAM FINISHED") and "OUTPUT:" in call["report"]
    assert any(e["type"] == "program" for e in leg.events)  # the 2 s start charge is on the clock


def test_agent_code_mode_reports_errors(mini_relay):
    def policy(system, messages):
        return Message(role="assistant", text="", tool_calls=[ToolCall(id="x", name="run_program", arguments={"code": "drive(0, -1)\nraise RuntimeError('boom')"})])
    leg = Leg(mini_relay["tracks"][0], config=leg_config_for("code", LegConfig(time_budget=12)))
    out = LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(mode="code")).run_leg(leg)
    rep = out["transcript"][0]["calls"][0]["report"]
    assert rep.startswith("PROGRAM FAILED") and "RuntimeError: boom" in rep


def test_agent_rejects_tools_of_other_mode(mini_relay):
    def policy(system, messages):
        return Message(role="assistant", text="", tool_calls=[ToolCall(id="x", name="drive", arguments={"steer_deg": 0, "distance_m": -1})])
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(time_budget=20))
    out = LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(mode="code", max_turns=2)).run_leg(leg)
    assert "Unknown tool 'drive'" in out["transcript"][0]["calls"][0]["report"] and leg.commands == 0


def test_agent_look_tool_in_cone_mode(mini_relay):
    def policy(system, messages):
        n = sum(1 for m in messages if m.role == "assistant")
        if n == 0:
            return Message(role="assistant", text="", tool_calls=[ToolCall(id="l", name="look", arguments={"yaw_deg": 90})])
        return [(0.0, -2.0)]
    leg = Leg(mini_relay["tracks"][0])
    out = LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(obs_level="cone", max_turns=3)).run_leg(leg)
    c = out["transcript"][0]["calls"][0]
    assert c["name"] == "look" and c["report"].startswith("RESULT: you now look +90")
    assert c["t1"] > c["t0"] and any(e["type"] == "look" for e in leg.events)


def test_prompts_and_tools_per_mode():
    assert [t.name for t in tools_for("local", "tools")] == ["drive"]
    assert [t.name for t in tools_for("cone", "tools")] == ["drive", "look"]
    assert [t.name for t in tools_for("cone", "code")] == ["run_program"]
    code = system_prompt("local", "code")
    assert "CODE MODE" in code and "LegOver" in code and "every drive() costs 0.25 s" in code
    assert "what you SEE" in system_prompt("cone", "tools") and "LOCAL MAP of nearby rail points" not in system_prompt("cone")


def test_history_is_bounded():
    msgs = [Message(role="user", text="brief")]
    for i in range(60):
        msgs += [Message(role="assistant", text="", tool_calls=[ToolCall(id=str(i), name="drive", arguments={})]),
                 Message(role="tool", text=f"STATUS: REVERSING\nPROGRESS: {i}\nnoise " * 20, tool_call_id=str(i))]
    out = compact_history(msgs, keep_full=4, max_abridged=10)
    tools = [m for m in out if m.role == "tool"]
    assert sum(1 for m in tools if m.text == "[old report omitted]") == 46
    assert sum(1 for m in tools if m.text.startswith("[older report, abridged]")) == 10
    assert tools[-1].text == msgs[-1].text


# ----------------------------------------------------------------------------- codex adapter
FAKE_CODEX = r'''#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
out = args[args.index("-o") + 1]
schema = json.load(open(args[args.index("--output-schema") + 1]))
assert "-s" in args and args[args.index("-s") + 1] == "read-only"
assert "--ephemeral" in args and "--ignore-user-config" in args
tools = schema["properties"]["calls"]["items"]["properties"]["tool"]["enum"]
open(out, "w").write(json.dumps({"note": "fake says hi", "calls": [{"tool": tools[0], "args_json": json.dumps({"steer_deg": 0, "distance_m": -2, "speed_mps": None})}]}))
print(json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "thinking about " + str(len(prompt))}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 50, "reasoning_output_tokens": 30}}))
'''


@pytest.fixture
def fake_codex(tmp_path):
    p = tmp_path / "codex"
    p.write_text(FAKE_CODEX.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_codex_adapter_round_trip(fake_codex):
    ad = CodexAdapter("gpt-6-luna", codex_bin=fake_codex)
    msgs = [Message(role="user", text="LEG 1 briefing")]
    r = ad.complete("SYS", msgs, tools_for("local"))
    tc = r.message.tool_calls[0]
    assert tc.name == "drive" and tc.arguments == {"steer_deg": 0, "distance_m": -2, "speed_mps": None}
    assert r.message.text == "fake says hi" and r.message.reasoning.startswith("thinking about")
    assert (r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens) == (1000, 400, 50)


def test_codex_drives_a_leg(fake_codex, mini_relay):
    from relay.models.registry import make_adapter
    os.environ["RELAY_CODEX_BIN"] = fake_codex
    try:
        ad = make_adapter("codex:gpt-6-luna")
    finally:
        del os.environ["RELAY_CODEX_BIN"]
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(time_budget=30))
    out = LLMDriver(ad, AgentConfig(max_turns=4)).run_leg(leg)
    assert leg.commands >= 1 and out["usage"]["input_tokens"] >= 1000


def test_codex_prompt_rendering():
    msgs = [Message(role="user", text="BRIEF"), Message(role="assistant", text="go", tool_calls=[ToolCall(id="a", name="drive", arguments={"steer_deg": 1})]),
            Message(role="tool", text="RESULT ok", tool_call_id="a", tool_name="drive")]
    s = render_conversation("SYS", msgs, tools_for("cone"))
    assert s.index("# SYSTEM INSTRUCTIONS") < s.index("- drive:") < s.index("- look:") < s.index("## USER") < s.index("## YOU (turn 1)")
    assert 'drive({"steer_deg":1})' in s and "RESULT ok" in s and "Do NOT run shell commands" in s


def test_codex_missing_binary_is_model_error():
    from relay.models.base import ModelError
    with pytest.raises(ModelError):
        CodexAdapter("m", codex_bin="/nonexistent/codex", max_retries=0).complete("S", [Message(role="user", text="x")], tools_for("local"))


# ----------------------------------------------------------------------------- costs
def test_costs_and_budget(tmp_path):
    u = {"input_tokens": 1_000_000, "cached_tokens": 500_000, "output_tokens": 100_000}
    billed, eq = usd("openai:gpt-6.1-sol", u)
    assert billed == eq == pytest.approx(0.5 * 2 + 0.5 * 0.1 + 0.1 * 10)
    assert usd("codex:gpt-6.1-sol", u) == (0.0, pytest.approx(eq))
    assert usd("openai:unknown-model", u) == (0.0, 0.0)
    b = Budget(tmp_path / "ledger" / "spend.json", 2.0, "t")
    b.check()
    assert b.charge("openai:gpt-6.1-sol", u) == pytest.approx(2.05)
    with pytest.raises(BudgetExceeded):
        b.check()
    b.charge("codex:gpt-6.1-sol", u)
    d = json.loads((tmp_path / "ledger" / "spend.json").read_text())
    assert d["by_model"]["codex:gpt-6.1-sol"]["api_equiv_usd"] == pytest.approx(2.05) and d["by_model"]["codex:gpt-6.1-sol"]["usd"] == 0
    assert Budget(tmp_path / "ledger" / "spend.json", None, "other").spent() == 0


def test_budget_ledger_is_safe_under_threads(tmp_path):
    b = Budget(tmp_path / "s.json", None, "t")
    u = {"input_tokens": 1000, "cached_tokens": 0, "output_tokens": 0}
    ths = [threading.Thread(target=lambda: [b.charge("openai:gpt-6.1-terra", u) for _ in range(20)]) for _ in range(5)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert b.spent() == pytest.approx(100 * 1000 * 1.0 / 1e6)


def test_exhausted_budget_aborts_leg(mini_relay, tmp_path):
    b = Budget(tmp_path / "s.json", 0.0, "t")
    leg = Leg(mini_relay["tracks"][0])
    out = LLMDriver(ScriptedAdapter("straight-back"), AgentConfig(), budget=b).run_leg(leg)
    assert leg.outcome == "DNF" and "budget" in out["api_error"] and leg.commands == 0


# ----------------------------------------------------------------------------- runner: mode/route, live files
def test_routes_and_code_leg_config():
    assert route_of("codex:gpt-6.1-sol") == "codex" and route_of("openai:gpt-6-luna") == "api" and route_of("autopilot") == "local"
    c = leg_config_for("code")
    assert c.command_overhead_s == 0.25 and c.max_commands >= 400
    assert leg_config_for("tools").command_overhead_s == 1.0


def test_live_file_during_run(mini_relay, tmp_path):
    seen = []

    def slow_policy(system, messages):
        time.sleep(0.25)
        f = tmp_path / "live"
        for p in f.glob("*.json") if f.is_dir() else []:
            seen.append(json.loads(p.read_text()))
        return [(0.0, -2.0)]

    import relay.runner as R
    old = R.LIVE_INTERVAL_S
    R.LIVE_INTERVAL_S = 0.1
    try:
        rec = run_relay(mini_relay, LLMDriver(ScriptedAdapter(policy=slow_policy), AgentConfig()), "scripted:slow", log=None,
                        agent_cfg=AgentConfig(), live_dir=tmp_path)
    finally:
        R.LIVE_INTERVAL_S = old
    assert seen, "no live snapshot was written"
    snap = seen[-1]
    assert snap["live"] and snap["updated"] and snap["run_id"] == rec["run_id"]
    L = snap["legs"][0]
    assert L["running"] and L["metrics"]["outcome"] is None and len(L["trace"]["t"]) == len(L["trace"]["x"])
    assert not (tmp_path / "live" / f"{rec['run_id']}.json").exists()  # removed when the run ends
    assert rec["config"]["mode"] == "tools" and rec["config"]["route"] == "local"
    save_run(rec, tmp_path)
    idx = json.loads(update_index(tmp_path).read_text())
    assert [r.get("live", False) for r in idx["runs"]] == [False]


def test_index_lists_live_and_marks_stale(tmp_path, mini_relay):
    rec = run_relay(mini_relay, LLMDriver(ScriptedAdapter("straight-back")), "scripted:straight-back", log=None)
    (tmp_path / "live").mkdir()
    fresh = dict(rec, run_id="fresh", live=True, updated=time.time())
    stale = dict(rec, run_id="stale", live=True, updated=time.time() - 3600)
    (tmp_path / "live" / "fresh.json").write_text(json.dumps(fresh))
    (tmp_path / "live" / "stale.json").write_text(json.dumps(stale))
    idx = {r["run_id"]: r for r in json.loads(update_index(tmp_path).read_text())["runs"]}
    assert idx["fresh"]["live"] and idx["fresh"]["file"] == "live/fresh.json"
    assert not idx["stale"]["live"] and idx["stale"]["partial"]


def test_run_trials_code_mode_records_cost(tmp_path, mini_relay):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "mini-s0.json").write_text(json.dumps({**__import__("relay.seeder", fromlist=["x"]).relay_to_dict(mini_relay), "sim_version": __import__("relay.sim", fromlist=["x"]).SIM_VERSION}))
    from relay.models import registry
    paths = run_trials(["scripted:straight-back"], "mini", runs_dir=tmp_path / "runs", tracks_dir=tracks,
                       agent_cfg=AgentConfig(mode="tools"), log=lambda m: None)
    rec = json.loads(paths[0].read_text())
    assert "cost_usd" in rec["totals"] and rec["config"]["leg"]["command_overhead_s"] == 1.0
    assert (tmp_path / "runs" / "ledger" / "spend.json").exists()


def test_cone_report_with_partial_bay_view(mini_relay):
    from relay.observe import observe, render_report
    obs = observe(Leg(mini_relay["tracks"][0]), "cone")
    obs["local_map"].update(bay_corners=[[3.0, 1.0]])
    obs["local_map"].pop("dock_wall", None)
    assert "only part of the bay" in render_report(obs)


def test_harness_error_ends_only_that_leg(mini_relay):
    def bad_policy(system, messages):
        raise RuntimeError("adapter exploded")
    rec = run_relay(mini_relay, LLMDriver(ScriptedAdapter(policy=bad_policy)), "scripted:bad", log=None)
    L = rec["legs"][0]
    assert L["metrics"]["outcome"] == "DNF" and "adapter exploded" in L["api_error"]


def test_codex_agent_tools_are_disabled(tmp_path):
    """codex's read-only sandbox still lets a shell READ the disk, so the shell (and every other agent tool) is turned off."""
    from relay.models.codex import NO_TOOLS
    cmd = CodexAdapter("gpt-6-luna", codex_bin="codex").command(tmp_path, tmp_path / "s.json", tmp_path / "o.json")
    disabled = {cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"}
    assert {"shell_tool", "unified_exec", "code_mode_host"} <= disabled and set(NO_TOOLS) == disabled
    assert 'web_search="disabled"' in cmd
