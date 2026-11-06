"""Continuous control: Leg.control_step physics, the sandboxed controller process, the Engine and the agent loop
(paused and realtime clocks)."""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
import time

import pytest

from relay.agent import AgentConfig, LLMDriver
from relay.continuous import Engine
from relay.controller import ControllerError, ControllerProcess, parse_command
from relay.models.base import Message, ToolCall
from relay.models.scripted import ScriptedAdapter
from relay.observe import observe
from relay.prompts import system_prompt, tools_for
from relay.runner import leg_config_for, run_relay
from relay.sim import Leg, LegConfig

GOOD = """
n = 0
def control(obs):
    global n
    n += 1
    h, b = obs["hitch_deg"], obs["route"]["bearing_deg"]
    want = max(-25.0, min(25.0, 1.2 * b))
    if n == 1:
        print("controller up, v =", obs["motion"]["v_mps"])
    return (2.0 * (h - want), -obs["speed"]["default_rev"])
"""


def ideal(track, **kw):
    return Leg(track, config=LegConfig(ground_rms_deg_per_m=0.0, **kw))


def msg(*calls):
    return Message(role="assistant", text="", tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)])


# ----------------------------------------------------------------------------- physics
def test_control_step_keeps_rolling_and_accelerates(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    sp = leg.spec
    assert leg.set_controls(0, -sp.speed_rev) == []
    leg.control_step(0.5)
    assert -sp.speed_rev <= leg.v < 0 and abs(leg.v) == pytest.approx(min(sp.speed_rev, sp.accel * 0.5), abs=1e-6)
    leg.control_step(3.0)
    assert leg.v == pytest.approx(-sp.speed_rev)
    assert leg.stops == 0 and leg.reverse_m > 0 and leg.status == "REVERSING"
    assert observe(leg)["motion"]["v_mps"] == pytest.approx(-sp.speed_rev, abs=1e-3)
    m = leg.metrics()
    assert m["stops"] == 0 and m["moving_time"] == pytest.approx(leg.t, abs=0.06) and m["mean_speed"] > 0


def test_control_step_gear_change_brakes_and_waits(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    sp = leg.spec
    leg.set_controls(0, -sp.speed_rev)
    leg.control_step(2.0)
    leg.set_controls(0, sp.speed_fwd)
    leg.control_step(sp.speed_rev / sp.accel + 0.1)  # braking
    assert leg.v == 0.0 and leg.shuttles == 1 and leg.stops == 1 and leg.hold_s > 0
    assert any(e["type"] == "shuttle" and e["to"] == "forward" for e in leg.events)
    leg.control_step(sp.shuttle_s + 1.0)
    assert leg.v > 0 and leg.forward_m > 0


def test_control_step_clamps_and_rejects_garbage(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    sp = leg.spec
    notes = leg.set_controls(90, -99)
    assert any("steering clamped" in n for n in notes) and any("speed capped" in n for n in notes)
    assert math.degrees(leg.target_steer) == pytest.approx(sp.max_steer_deg) and leg.target_speed == -sp.speed_rev_max
    notes = leg.set_controls("left", float("nan"))
    assert leg.target_speed == 0.0 and len(notes) == 2


def test_continuous_contact_pins_then_blocks_once(mini_relay):
    leg = ideal(mini_relay["tracks"][0], touch_penalty_s=1.0)
    sp = leg.spec
    leg.set_controls(sp.max_steer_deg, sp.speed_fwd)  # drive forward on full lock into the start wall / rails
    leg.control_step(60)
    assert leg.touches == 1 and any(e["type"] == "contact" for e in leg.events)
    assert leg.pinned and leg.v == 0.0
    pinned0 = leg.stall_time["PINNED"]
    leg.control_step(10)  # still pushing the same way: the same episode, no second touch, time lost while PINNED
    assert leg.touches == 1 and leg.pinned and leg.v == 0.0 and leg.stall_time["PINNED"] > pinned0 + 9
    leg.set_controls(0, -sp.speed_rev)  # back away
    leg.control_step(sp.shuttle_s + 3)
    assert not leg.pinned and leg.reverse_m > 0.05


def test_wheels_turn_while_blocked_so_a_pinned_rig_can_steer_free(mini_relay):
    leg = ideal(mini_relay["tracks"][0], touch_penalty_s=0.5)
    sp = leg.spec
    leg.set_controls(sp.max_steer_deg, sp.speed_fwd)
    leg.control_step(60)
    assert leg.pinned
    leg.set_controls(-sp.max_steer_deg, sp.speed_fwd)  # keep pushing forward but turn the wheels the other way
    leg.control_step(1.0)
    assert math.degrees(leg.steer) < 0  # the wheels turned although the rig is up against the rail
    leg.control_step(2 * sp.max_steer_deg / sp.steer_rate_deg_per_s + 5)
    assert math.degrees(leg.steer) == pytest.approx(-sp.max_steer_deg)


def test_continuous_jackknife(mini_relay):
    leg = ideal(mini_relay["tracks"][0], jackknife_penalty_s=1.0)
    leg.set_controls(-leg.spec.max_steer_deg, -leg.spec.speed_rev)
    for _ in range(600):
        leg.control_step(0.1)
        if leg.jackknifes or leg.touches or leg.done:
            break
    assert leg.jackknifes == 1 or leg.touches >= 1  # a tight course may stop it at a rail first
    if leg.jackknifes:
        assert leg.jackknifed and leg.v == 0.0 and abs(math.degrees(leg.hitch)) >= leg.spec.max_hitch_deg - 1


def test_discrete_drive_counts_stops(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    leg.drive(0, -1)
    leg.drive(0, -1)
    leg.drive(5, 0)  # dry steering: no stop added
    m = leg.metrics()
    assert m["stops"] == 2 and m["stopped_time"] > 0 and m["moving_time"] > 0


# ----------------------------------------------------------------------------- controller process
def test_controller_ticks_and_keeps_state(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    leg.continuous = True
    c = ControllerProcess("k = 0\ndef control(obs):\n    global k\n    k += 1\n    print('tick', k)\n    if k == 2: wake('second')\n    return {'steer_deg': k, 'speed_mps': -1, 'look_deg': 90}\n")
    try:
        (s, v, look), out, wake = c.tick(observe(leg))
        assert (s, v, look) == (1.0, -1.0, 90.0) and out.strip() == "tick 1" and wake == []
        (s, *_), out, wake = c.tick(observe(leg))
        assert s == 2.0 and wake == ["second"]
    finally:
        c.close()


@pytest.mark.parametrize("code, needle", [
    ("x = 1", "must define a function control"),
    ("def control(obs:\n  pass", "SyntaxError"),
    ("raise ValueError('at import')\ndef control(o): return (0, 0)", "at import"),
])
def test_controller_start_errors(code, needle):
    with pytest.raises(ControllerError) as ei:
        ControllerProcess(code)
    assert needle in str(ei.value)


@pytest.mark.parametrize("code, needle", [
    ("def control(obs):\n    return obs['nope']", "KeyError"),
    ("def control(obs):\n    return 'fast'", "must return"),
    ("def control(obs):\n    return (float('nan'), 0)", "non-finite"),
    ("def control(obs):\n    while True: pass", "longer than"),
])
def test_controller_tick_errors(mini_relay, code, needle):
    leg = ideal(mini_relay["tracks"][0])
    c = ControllerProcess(code, tick_timeout_s=0.3)
    with pytest.raises(ControllerError) as ei:
        c.tick(observe(leg))
    assert needle in str(ei.value) and not c.alive


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec isolation only")
def test_controller_sandbox_blocks_home_and_network():
    import socket
    import tempfile

    # A real readable canary and listening local socket make denial observable without personal files or internet.
    with tempfile.TemporaryDirectory(prefix=".relay-canary-", dir=Path.home()) as home_dir, socket.socket() as listener:
        canary = Path(home_dir) / "canary.txt"
        canary.write_text("isolated-test-canary")
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = f"""
import socket
r = []
try:
    open({str(canary)!r}).read(); r.append("READ")
except Exception as e:
    r.append(type(e).__name__)
try:
    socket.create_connection(("127.0.0.1", {port}), timeout=2); r.append("NET")
except Exception as e:
    r.append(type(e).__name__)
def control(obs):
    return (0, 0)
print(r)
"""
        c = ControllerProcess(code)
        try:
            assert c.sandbox
            assert "READ" not in c.startup_out and "NET" not in c.startup_out
        finally:
            c.close()


def test_parse_command_forms():
    assert parse_command([1, -2]) == (1.0, -2.0, None)
    assert parse_command({"steer_deg": 1, "speed_mps": 0, "look_deg": 180}) == (1.0, 0.0, 180.0)
    with pytest.raises(ControllerError):
        parse_command({"steer": 1})


# ----------------------------------------------------------------------------- engine
def test_engine_controller_parks_without_stopping(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    eng = Engine(leg, "local")
    try:
        assert eng.install(GOOD) is None
        eng.run(leg.budget)
    finally:
        eng.close()
    assert leg.outcome == "FINISHED" and leg.stops == 0 and leg.touches == 0
    assert any(k == "leg over" for k, _ in eng.wakes)
    assert "controller up" in "".join(o for _, o in eng.output)


def test_engine_controller_error_brakes_and_wakes(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    eng = Engine(leg, "local")
    code = "def control(obs):\n    if obs['t'] > 2: raise RuntimeError('lost it')\n    return (0, -1)\n"
    try:
        assert eng.install(code) is None
        eng.run(10)
        wakes = eng.take_wakes()
        assert wakes[0][0] == "controller error" and "lost it" in wakes[0][1]
        assert eng.controller is None and leg.target_speed == 0.0
        eng.run(3)
        assert leg.v == 0.0 and leg.stops == 1
    finally:
        eng.close()


def test_engine_failed_install_keeps_old_controller(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    eng = Engine(leg, "local")
    try:
        assert eng.install(GOOD) is None
        err = eng.install("def control(obs) return 1")
        assert "SyntaxError" in err and eng.controller is not None and eng.controller_n == 1
    finally:
        eng.close()


def test_engine_realtime_drives_while_thinking(mini_relay):
    leg = ideal(mini_relay["tracks"][0])
    eng = Engine(leg, "local")
    leg.set_controls(0, -1.0)
    res, wall, sim = eng.run_while(lambda: time.sleep(0.4) or "answer", scale=5.0)
    assert res == "answer" and wall >= 0.4 and sim == pytest.approx(wall * 5.0, abs=0.15)
    assert leg.t == pytest.approx(sim, abs=0.11) and leg.reverse_m > 0.5
    # offset: the first wall seconds are free
    t0 = leg.t
    eng.run_while(lambda: time.sleep(0.2), scale=5.0, offset_s=10)
    assert leg.t == t0


# ----------------------------------------------------------------------------- agent loop
def test_agent_continuous_code_parks_in_one_decision(mini_relay):
    def policy(system, messages):
        if not any(m.role == "tool" for m in messages):
            return msg(("set_controller", {"code": GOOD, "check_in_s": 120}))
        return msg(("keep_driving", {"check_in_s": 120}))
    cfg = AgentConfig(mode="code", control="continuous")
    leg = Leg(mini_relay["tracks"][0], config=leg_config_for("code", None, "continuous"))
    out = LLMDriver(ScriptedAdapter(policy=policy), cfg).run_leg(leg)
    assert leg.outcome == "FINISHED" and leg.stops == 0
    c = out["transcript"][0]["calls"][0]
    assert c["name"] == "set_controller" and c["report"].startswith("WOKE") and "LEG OVER" in c["report"]
    assert "controller up" in c["report"] and out["control"]["controllers"] == 1


def test_agent_continuous_code_reports_bad_controller(mini_relay):
    def policy(system, messages):
        return msg(("set_controller", {"code": "def control(obs):\n    return 1/0", "check_in_s": 5}))
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(time_budget=30))
    out = LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(mode="code", control="continuous", max_turns=2)).run_leg(leg)
    rep = out["transcript"][0]["calls"][0]["report"]
    assert "CONTROLLER ERROR" in rep and "ZeroDivisionError" in rep


def test_agent_continuous_tools_keep_moving_between_turns(mini_relay):
    seen = []

    def policy(system, messages):
        seen.append(messages[-1].text if messages[-1].role == "tool" else "")
        return msg(("drive", {"steer_deg": 0, "speed_mps": -1.0, "duration_s": 1.0}))
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(ground_rms_deg_per_m=0.0))
    LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(control="continuous", max_turns=6)).run_leg(leg)
    assert leg.commands == 6 and leg.stops == 0 or leg.touches  # straight back may reach a rail on a curved course
    assert "MOTION: reversing at" in seen[2] and "DURING THIS COMMAND" in seen[2]


def test_agent_continuous_wake_skips_rest_of_plan(mini_relay):
    def policy(system, messages):
        return msg(("drive", {"steer_deg": 34, "speed_mps": 2.0, "duration_s": 30}),
                   ("drive", {"steer_deg": 0, "speed_mps": -1, "duration_s": 5}))
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(time_budget=80))
    out = LLMDriver(ScriptedAdapter(policy=policy), AgentConfig(control="continuous", max_turns=1)).run_leg(leg)
    c1, c2 = out["transcript"][0]["calls"]
    assert c1["report"].startswith("WOKE") and "CONTACT" in c1["report"] and c1["woke"] == ["contact"]
    assert c2["report"].startswith("Not executed")


def test_agent_realtime_reports_what_happened_while_thinking(mini_relay):
    def policy(system, messages):
        time.sleep(0.25)
        return msg(("drive", {"steer_deg": 0, "speed_mps": -1.0, "duration_s": 0.5}))
    leg = Leg(mini_relay["tracks"][0], config=LegConfig(ground_rms_deg_per_m=0.0))
    cfg = AgentConfig(control="continuous", clock="realtime", latency_scale=8.0, max_turns=3)
    out = LLMDriver(ScriptedAdapter(policy=policy), cfg).run_leg(leg)
    t = out["transcript"]
    assert t[1]["think_sim_s"] >= 1.5 and t[1]["t_reply"] > t[1]["t"]
    assert "WHILE YOU WERE THINKING" in t[1]["calls"][0]["report"]


def test_continuous_prompts_and_tools():
    s = system_prompt("local", "tools", "continuous", "realtime")
    assert "duration_s" in s and "THE CLOCK RUNS WHILE YOU THINK" in s and "distance_m" not in s
    s2 = system_prompt("cone", "code", "continuous", "paused", 0.5)
    assert "control(obs)" in s2 and "look_deg" in s2 and "0.5 s reaction" in s2 and "look(yaw_deg)" not in s2
    assert "duration_s" not in system_prompt("local", "tools")  # discrete text unchanged
    assert [t.name for t in tools_for("local", "tools", "continuous")] == ["drive"]
    assert [t.name for t in tools_for("cone", "tools", "continuous")] == ["drive", "look"]
    assert [t.name for t in tools_for("local", "code", "continuous")] == ["set_controller", "keep_driving"]


def test_run_relay_records_control_and_stops(mini_relay):
    def policy(system, messages):
        return msg(("set_controller", {"code": GOOD, "check_in_s": 120}))
    cfg = AgentConfig(mode="code", control="continuous")
    rec = run_relay(mini_relay, LLMDriver(ScriptedAdapter(policy=policy), cfg), "scripted:x", leg_config_for("code", None, "continuous"), cfg)
    assert rec["config"]["control"] == "continuous" and rec["config"]["clock"] == "paused"
    assert rec["config"]["control_prompt_version"] >= 1
    assert "stops" in rec["totals"] and all("stops" in L["metrics"] for L in rec["legs"])
    assert any(e["type"] == "controller" for L in rec["legs"] for e in L["events"])
