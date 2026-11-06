"""Drive through the `codex` CLI (`codex exec`), e.g. on a ChatGPT subscription instead of API billing.

Every model turn is one stateless `codex exec` call:
  * the system prompt, the tool list and the (already compacted) conversation are rendered as text;
  * `--output-schema` forces a JSON answer {note, calls:[{tool, args_json}]}, which we turn into tool calls;
  * the agent runs in an empty temporary directory, with no user config or rules, and is told it is a pure
    decision function. Its shell / code / browser / web tools are DISABLED (NO_TOOLS): codex's "read-only"
    sandbox only blocks writes, so with a shell the model could read the whole disk (this repo, other runs);
  * any tool item that still shows up in the event stream is counted and warned about (tool_items);
  * token usage is read from the `--json` event stream.

Spec: `codex:<model>` (e.g. codex:gpt-6.1-sol). Reasoning effort, if set, is passed as model_reasoning_effort.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .base import Message, ModelAdapter, ModelError, ModelResponse, ToolCall, ToolSpec, Usage, _warn, loads_args

ANSWER_RULES = """\
# HOW TO ANSWER
You are being used as a pure decision function inside a driving simulator. Do NOT run shell commands, \
read or write files, or browse; everything you need is below. Answer ONLY with JSON matching the schema:
  note  = one short sentence for the team radio (may be empty);
  calls = the tool calls to make now, in order. Each has `tool` (a tool name from the list) and `args_json`
          (the JSON object of that tool's arguments, as a string, e.g. "{\\"steer_deg\\": 10, \\"distance_m\\": -2, \\"speed_mps\\": null}").
Give at least one call unless you truly have nothing to do."""


def _schema(tools: list[ToolSpec]) -> dict:
    return {
        "type": "object",
        "properties": {
            "note": {"type": "string"},
            "calls": {"type": "array", "items": {
                "type": "object",
                "properties": {"tool": {"type": "string", "enum": [t.name for t in tools]}, "args_json": {"type": "string"}},
                "required": ["tool", "args_json"], "additionalProperties": False}},
        },
        "required": ["note", "calls"], "additionalProperties": False,
    }


def render_conversation(system: str, messages: list[Message], tools: list[ToolSpec]) -> str:
    out = ["# SYSTEM INSTRUCTIONS", system, "", "# TOOLS"]
    for t in tools:
        out.append(f"- {t.name}: {t.description}\n  arguments (JSON schema): {json.dumps(t.parameters, separators=(',', ':'))}")
    out += ["", ANSWER_RULES, "", "# CONVERSATION SO FAR"]
    turn = 0
    for m in messages:
        if m.role == "user":
            out += ["## USER", m.text]
        elif m.role == "assistant":
            turn += 1
            calls = "; ".join(f"{tc.name}({json.dumps(tc.arguments, separators=(',', ':'))})" for tc in m.tool_calls) or "(no calls)"
            out += [f"## YOU (turn {turn})", (f"note: {m.text}\n" if m.text else "") + f"calls: {calls}"]
        elif m.role == "tool":
            out += [f"## TOOL RESULT ({m.tool_name or 'tool'})", m.text]
    out += ["", "# YOUR TURN", "Decide your next tool calls now and answer with the JSON."]
    return "\n".join(out)


# every codex feature that lets the agent act or look outside the prompt (verified: `ls` -> "code-mode host is disabled")
NO_TOOLS = ["shell_tool", "unified_exec", "code_mode_host", "browser_use", "browser_use_external", "in_app_browser",
            "computer_use", "image_generation", "view_image", "apps", "plugins", "tool_suggest", "skill_search"]
TOOL_ITEMS = ("command_execution", "file_change", "mcp_tool_call", "web_search", "collab_tool_call")


class CodexAdapter(ModelAdapter):
    provider = "codex"

    def __init__(self, model: str, codex_bin: str | None = None, **kw):
        kw.pop("api_key", None)
        super().__init__(model, **kw)
        self.codex_bin = codex_bin or os.environ.get("RELAY_CODEX_BIN") or shutil.which("codex") or "codex"
        self.timeout_s = float(kw.get("timeout", 120.0)) * 2.5  # an exec call includes agent start-up
        self.tool_items = 0  # tool items seen in the event stream (should stay 0)

    def build_request(self, system, messages, tools):  # pragma: no cover - not HTTP based
        raise NotImplementedError

    def parse_response(self, data):  # pragma: no cover - not HTTP based
        raise NotImplementedError

    def command(self, workdir: Path, schema_path: Path, out_path: Path) -> list[str]:
        cmd = [self.codex_bin, "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
               "-s", "read-only", "-C", str(workdir), "-m", self.model, "--output-schema", str(schema_path),
               "-o", str(out_path), "--json", "--color", "never", "-c", 'web_search="disabled"']
        for f in NO_TOOLS:
            cmd += ["--disable", f]
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return cmd + ["-"]

    def parse_output(self, answer: str, events: str) -> tuple[Message, Usage, str]:
        usage = Usage()
        thoughts = []
        for line in events.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                u = ev.get("usage") or {}
                usage.add(Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                                reasoning_tokens=u.get("reasoning_output_tokens", 0), cached_tokens=u.get("cached_input_tokens", 0)))
            elif ev.get("type") == "item.completed" and (ev.get("item") or {}).get("type") in TOOL_ITEMS:
                self.tool_items += 1
                _warn(f"codex agent used a tool despite NO_TOOLS: {str(ev['item'])[:200]}")
            elif ev.get("type") == "item.completed" and (ev.get("item") or {}).get("type") == "reasoning":
                if ev["item"].get("text"):
                    thoughts.append(ev["item"]["text"])
        data = json.loads(answer)
        calls = [ToolCall(id=f"cx_{uuid.uuid4().hex[:10]}", name=c.get("tool", ""), arguments=loads_args(c.get("args_json")))
                 for c in data.get("calls") or []]
        msg = Message(role="assistant", text=(data.get("note") or "").strip(), tool_calls=calls, reasoning="\n\n".join(thoughts))
        return msg, usage, "tool_use" if calls else "end_turn"

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        prompt = render_conversation(system, messages, tools)
        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix="relay-codex-") as td:
                td = Path(td)
                work = td / "empty"
                work.mkdir()
                schema_path, out_path = td / "schema.json", td / "answer.json"
                schema_path.write_text(json.dumps(_schema(tools)))
                try:
                    p = subprocess.run(self.command(work, schema_path, out_path), input=prompt, capture_output=True, text=True,
                                       timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    last_err = f"codex exec timed out after {self.timeout_s:.0f}s"
                except FileNotFoundError as e:
                    raise ModelError(f"codex CLI not found ({e}); install it or set RELAY_CODEX_BIN") from e
                else:
                    answer = out_path.read_text() if out_path.exists() else ""
                    if p.returncode == 0 and answer.strip():
                        try:
                            msg, usage, stop = self.parse_output(answer, p.stdout)
                        except (json.JSONDecodeError, AttributeError, TypeError) as e:
                            last_err = f"unparseable answer ({e}): {answer[:200]!r}"
                        else:
                            msg.provider = self.provider
                            return ModelResponse(message=msg, usage=usage, stop_reason=stop, latency_s=time.time() - t0, raw=None)
                    else:
                        last_err = f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}"
                        o = p.stdout + p.stderr  # wrong model name / plan, or no credits left: retrying will not help
                        if "is not supported" in o or "out of credits" in o:
                            raise ModelError(f"{self.spec}: {last_err}")
            self.retries += 1
            _warn(f"{self.spec}: {last_err}; retry {attempt + 1}/{self.max_retries}")
            self._sleep(attempt)
        raise ModelError(f"{self.spec}: giving up after {self.max_retries + 1} attempts ({last_err})")
