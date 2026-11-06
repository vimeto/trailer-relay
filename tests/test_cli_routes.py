"""The claude-code: and agy: subscription routes (fake CLIs; no network)."""
from __future__ import annotations

import json
import stat
import sys

import pytest

from relay.costs import usd
from relay.models.base import Message, ModelError
from relay.models.cli import AgyAdapter, ClaudeCodeAdapter
from relay.models.registry import make_adapter
from relay.prompts import tools_for
from relay.runner import route_of

FAKE_CLAUDE = r'''#!PY
import json, os, sys
a = sys.argv[1:]
prompt = sys.stdin.read()
assert a[0] == "-p" and a[a.index("--tools") + 1] == "" and "--strict-mcp-config" in a and "--no-session-persistence" in a
assert a[a.index("--setting-sources") + 1] == "" and "decision function" in a[a.index("--system-prompt") + 1]
assert "CLAUDECODE" not in os.environ and os.listdir(".") == []
assert "# SYSTEM INSTRUCTIONS" in prompt
schema = json.loads(a[a.index("--json-schema") + 1])
tool = schema["properties"]["calls"]["items"]["properties"]["tool"]["enum"][0]
eff = a[a.index("--effort") + 1] if "--effort" in a else "none"
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "permission_denials": [],
  "usage": {"input_tokens": 3, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 300, "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 25}},
  "structured_output": {"note": "effort " + eff, "calls": [{"tool": tool, "args_json": "{\"steer_deg\": 5, \"distance_m\": -1, \"speed_mps\": null}"}]}}))
'''

FAKE_AGY = r'''#!PY
import json, os, sys
a = sys.argv[1:]
prompt = sys.stdin.read()
assert "-p" not in a and a[a.index("--output-format") + 1] == "json" and "--disable-slash-commands" in a
assert os.listdir(".") == []
mode = os.environ.get("FAKE_AGY_MODE", "ok")
if mode == "tool":
    print("jetski: no output produced — a tool required the \"command\" permission that headless mode cannot prompt for, so it was auto-denied.", file=sys.stderr)
    sys.exit(1)
if mode == "denied":
    print(json.dumps({"status": "SUCCESS", "response": "", "usage": {}, "denied_actions": [{"action": "command", "display_name": "RunCommand"}]}))
    sys.exit(0)
schema = json.loads(a[a.index("--json-schema") + 1])
tool = schema["properties"]["calls"]["items"]["properties"]["tool"]["enum"][0]
print(json.dumps({"status": "SUCCESS", "response": "```json ...```",
  "usage": {"input_tokens": 9000, "output_tokens": 60, "thinking_tokens": 20, "cache_read_tokens": 4000},
  "structured_output": {"note": "agy " + a[a.index("--model") + 1], "calls": [{"tool": tool, "args_json": "{}"}]}}))
'''


def _bin(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src.replace("#!PY", f"#!{sys.executable}"))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_claude_code_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")  # a parent Claude Code session must not leak into the child
    ad = ClaudeCodeAdapter("claude-opus-5-5", cli_bin=_bin(tmp_path, "claude", FAKE_CLAUDE), reasoning_effort="high")
    r = ad.complete("SYS", [Message(role="user", text="LEG 1")], tools_for("local"))
    tc = r.message.tool_calls[0]
    assert tc.name == "drive" and tc.arguments == {"steer_deg": 5, "distance_m": -1, "speed_mps": None}
    assert r.message.text == "effort high" and r.message.provider == "claude-code"
    assert (r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens, r.usage.reasoning_tokens) == (1003, 300, 40, 25)
    assert ad.tool_items == 0


def test_agy_round_trip(tmp_path):
    ad = AgyAdapter("gemini-4-argon-high", cli_bin=_bin(tmp_path, "agy", FAKE_AGY))
    r = ad.complete("SYS", [Message(role="user", text="LEG 1")], tools_for("local"))
    assert r.message.text == "agy gemini-4-argon-high" and r.message.tool_calls[0].name == "drive"
    assert (r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens) == (9000, 4000, 80)


def test_agy_tool_attempt_is_counted_and_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AGY_MODE", "tool")
    ad = AgyAdapter("gemini-3.8-flash-low", cli_bin=_bin(tmp_path, "agy", FAKE_AGY), max_retries=1)
    ad._sleep = lambda *a, **k: None
    with pytest.raises(ModelError):
        ad.complete("SYS", [Message(role="user", text="x")], tools_for("local"))
    assert ad.tool_items == 2 and ad.retries == 2


def test_specs_routes_and_billing(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_AGY_BIN", _bin(tmp_path, "agy", FAKE_AGY))
    assert isinstance(make_adapter("agy:gemini-4-argon-high"), AgyAdapter)
    assert isinstance(make_adapter("claude-code:claude-sonnet-5-5"), ClaudeCodeAdapter)
    assert route_of("agy:gemini-4-argon-high") == "agy" and route_of("claude-code:claude-opus-5-5") == "claude-code"
    assert usd("claude-code:claude-opus-5-5", {"input_tokens": 10**6, "output_tokens": 10**6})[0] == 0.0
    assert usd("agy:gemini-4-argon-high", {"input_tokens": 10**6, "output_tokens": 10**6})[0] == 0.0


def test_agy_denied_action_without_answer_is_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AGY_MODE", "denied")
    ad = AgyAdapter("gemini-3.8-flash-low", cli_bin=_bin(tmp_path, "agy", FAKE_AGY), max_retries=0)
    with pytest.raises(ModelError, match="RunCommand"):
        ad.complete("SYS", [Message(role="user", text="x")], tools_for("local"))
    assert ad.tool_items == 1


def test_agy_python_repr_and_prose_fallback(tmp_path):
    """GPT-OSS on agy: fields come back as Python-repr strings, and sometimes the answer is only in `response`."""
    ad = AgyAdapter("gpt-oss-120b-medium", cli_bin="unused")
    out = json.dumps({"status": "SUCCESS", "response": "x", "structured_output": "{'calls': [], 'note': 'hi'}",
                      "usage": "{'input_tokens': 5385, 'output_tokens': 406, 'thinking_tokens': 0, 'cache_read_tokens': 6528}"})
    data, u = ad.parse_result(out)
    assert data == {"calls": [], "note": "hi"} and (u.input_tokens, u.cached_tokens) == (5385, 6528)
    out = json.dumps({"status": "SUCCESS", "usage": {"input_tokens": 1, "output_tokens": 1},
                      "response": 'Sure. {"a": 1} then {"note": "n", "calls": [{"tool": "keep_driving", "args_json": "{}"}]}'})
    data, _ = ad.parse_result(out)
    assert data["calls"][0]["tool"] == "keep_driving"


def test_index_initializes_an_empty_workspace(tmp_path):
    import json
    from relay.runner import update_index

    out = update_index(tmp_path / 'new-workspace' / 'runs')
    assert json.loads(out.read_text()) == {'runs': []}
