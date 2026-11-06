"""Drive through other vendors' agent CLIs on the user's subscriptions, the same way the codex route does.

Every model turn is one stateless CLI call: the system prompt, tools and (compacted) conversation are rendered as
text by codex.render_conversation (identical across routes), a JSON schema forces the answer
{note, calls:[{tool, args_json}]}, and the call runs in an empty temporary directory.

* `claude-code:<model>` — the Claude Code CLI (`claude -p`).
    `--tools ""` removes every built-in tool (verified: the model reports "NONE" and cannot read a canary file);
    `--strict-mcp-config` with no config loads no MCP servers; `--setting-sources ""` ignores user/project settings;
    `--system-prompt` replaces Claude Code's coding-agent prompt with a one-line decision-function prompt.
    Reasoning effort maps to `--effort`.
* `agy:<model>` — the Antigravity CLI (Gemini, and the Claude/GPT-OSS models it hosts). Effort is part of the model
    name (e.g. agy:gemini-4-argon-high). agy has no switch to remove tools, but in headless mode every tool that needs a
    permission is auto-denied and the turn ends with no output (verified for shell commands, web search and reading a
    file outside the workspace). Such a turn is counted in `tool_items` and retried; nothing runs. Reading inside the
    workspace is allowed, and the workspace is an empty temporary directory.

Both CLIs return `structured_output` and token usage in their JSON result. Billed cost is $0 (subscription).
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .base import Message, ModelAdapter, ModelError, ModelResponse, ToolCall, ToolSpec, Usage, _warn, loads_args
from .codex import _schema, render_conversation

DECISION_SYSTEM = ("You are a pure decision function inside a driving simulator. You have no tools. "
                   "Everything you need is in the user message. Answer only with JSON matching the schema.")

# environment variables of a parent Claude Code session that would change how a nested `claude -p` behaves
_PARENT_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT", "CLAUDE_AGENT_SDK_VERSION")


class _CLIAdapter(ModelAdapter):
    provider = "cli"
    bin_env = ""
    bin_name = ""

    def __init__(self, model: str, cli_bin: str | None = None, **kw):
        kw.pop("api_key", None)
        super().__init__(model, **kw)
        self.cli_bin = cli_bin or os.environ.get(self.bin_env) or shutil.which(self.bin_name) or self.bin_name
        self.timeout_s = float(kw.get("timeout", 120.0)) * 2.5  # a CLI call includes agent start-up
        self.tool_items = 0  # turns where the agent tried a tool (should stay 0)

    def build_request(self, system, messages, tools):  # pragma: no cover - not HTTP based
        raise NotImplementedError

    def parse_response(self, data):  # pragma: no cover - not HTTP based
        raise NotImplementedError

    def command(self, schema: str) -> list[str]:
        raise NotImplementedError

    def env(self) -> dict:
        return dict(os.environ)

    def parse_result(self, out: str) -> tuple[dict, Usage]:
        raise NotImplementedError

    def tool_blocked(self, out: str) -> bool:
        return False

    def prepare_workspace(self, td: Path) -> None:
        """Files the CLI should find in its (otherwise empty) working directory."""

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        prompt = render_conversation(system, messages, tools)
        schema = json.dumps(_schema(tools), separators=(",", ":"))
        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix=f"relay-{self.provider}-") as td:
                self.prepare_workspace(Path(td))
                try:
                    p = subprocess.run(self.command(schema), input=prompt, capture_output=True, text=True, cwd=td,
                                       env=self.env(), timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    last_err = f"{self.provider} timed out after {self.timeout_s:.0f}s"
                except FileNotFoundError as e:
                    raise ModelError(f"{self.provider} CLI not found ({e}); install it or set {self.bin_env}") from e
                else:
                    out = (p.stdout or "") + (p.stderr or "")
                    if self.tool_blocked(out):
                        self.tool_items += 1
                        last_err = "the agent tried a tool (auto-denied, no answer)"
                        _warn(f"{self.spec}: {last_err}")
                    else:
                        try:
                            data, usage = self.parse_result(p.stdout)
                            msg = self.to_message(data)
                        except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError) as e:
                            last_err = f"exit {p.returncode}, unparseable result ({e}): {out[-300:]!r}"
                            if ("not supported" in out or "unknown model" in out.lower() or "invalid model" in out.lower()
                                    or "quota reached" in out.lower()):  # a subscription quota does not come back within a run
                                raise ModelError(f"{self.spec}: {last_err}")
                        else:
                            msg.provider = self.provider
                            stop = "tool_use" if msg.tool_calls else "end_turn"
                            return ModelResponse(message=msg, usage=usage, stop_reason=stop, latency_s=time.time() - t0, raw=None)
            self.retries += 1
            _warn(f"{self.spec}: {last_err}; retry {attempt + 1}/{self.max_retries}")
            self._sleep(attempt)
        raise ModelError(f"{self.spec}: giving up after {self.max_retries + 1} attempts ({last_err})")

    @staticmethod
    def to_message(data: dict) -> Message:
        if not isinstance(data, dict):
            raise TypeError(f"structured output is {type(data).__name__}, not an object")
        calls = [ToolCall(id=f"cl_{uuid.uuid4().hex[:10]}", name=c.get("tool", ""), arguments=loads_args(c.get("args_json")))
                 for c in data.get("calls") or []]
        return Message(role="assistant", text=(data.get("note") or "").strip(), tool_calls=calls)


class ClaudeCodeAdapter(_CLIAdapter):
    provider = "claude-code"
    bin_env = "RELAY_CLAUDE_BIN"
    bin_name = "claude"

    def command(self, schema: str) -> list[str]:
        cmd = [self.cli_bin, "-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
               "--output-format", "json", "--json-schema", schema, "--model", self.model, "--system-prompt", DECISION_SYSTEM]
        if self.reasoning_effort:
            cmd += ["--effort", self.reasoning_effort]
        return cmd

    def env(self) -> dict:
        e = dict(os.environ)
        for k in _PARENT_ENV:
            e.pop(k, None)
        return e

    def parse_result(self, out: str) -> tuple[dict, Usage]:
        r = json.loads(out)
        if r.get("is_error") or r.get("subtype") not in (None, "success"):
            raise ValueError(f"claude error: {r.get('subtype')} {str(r.get('result'))[:200]}")
        if r.get("permission_denials"):
            self.tool_items += len(r["permission_denials"])
        u = r.get("usage") or {}
        cached = u.get("cache_read_input_tokens", 0)
        usage = Usage(input_tokens=u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + cached,
                      output_tokens=u.get("output_tokens", 0), cached_tokens=cached,
                      reasoning_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens", 0))
        data = r.get("structured_output")
        if data is None:
            data = json.loads(r.get("result") or "")
        return data, usage


class AgyAdapter(_CLIAdapter):
    provider = "agy"
    bin_env = "RELAY_AGY_BIN"
    bin_name = "agy"

    def __init__(self, model: str, cli_bin: str | None = None, **kw):
        super().__init__(model, cli_bin=cli_bin, **kw)
        # high-effort Gemini turns think for 40k+ tokens (often 250-300 s); a timeout throws that work away
        self.timeout_s = max(self.timeout_s, 720.0)

    def command(self, schema: str) -> list[str]:
        # the prompt goes on stdin (print mode is implied by --output-format)
        return [self.cli_bin, "--model", self.model, "--output-format", "json", "--json-schema", schema,
                "--disable-slash-commands", "--print-timeout", f"{int(self.timeout_s)}s"]

    def tool_blocked(self, out: str) -> bool:
        return "no output produced" in out and "auto-denied" in out

    @staticmethod
    def _obj(x):
        """agy returns some fields as JSON and, for some hosted models (GPT-OSS), as Python-repr strings."""
        if isinstance(x, str):
            try:
                return json.loads(x)
            except ValueError:
                return ast.literal_eval(x)
        return x

    @staticmethod
    def _json_in_text(text: str):
        """The last JSON object with a "calls" key in free text (a model that answered in prose instead of the schema)."""
        dec = json.JSONDecoder()
        found = None
        for i, ch in enumerate(text or ""):
            if ch == "{":
                try:
                    obj, _ = dec.raw_decode(text, i)
                except ValueError:
                    continue
                if isinstance(obj, dict) and "calls" in obj:
                    found = obj
        return found

    def parse_result(self, out: str) -> tuple[dict, Usage]:
        r = json.loads(out)
        if r.get("status") != "SUCCESS":
            raise ValueError(f"agy status {r.get('status')}: {str(r.get('response'))[:200]}")
        # NOTE: agy's input_tokens EXCLUDES cache reads (input + output = total); usage is stored as reported,
        # so total prompt tokens = input_tokens + cached_tokens for this route (relay.story accounts for it).
        u = self._obj(r.get("usage")) or {}
        cached = u.get("cache_read_tokens", 0)
        usage = Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0) + u.get("thinking_tokens", 0),
                      cached_tokens=cached, reasoning_tokens=u.get("thinking_tokens", 0))
        denied = r.get("denied_actions") or []
        self.tool_items += len(denied)  # tools the agent tried; headless agy denied them, nothing ran
        data = self._obj(r.get("structured_output"))
        if data is None:
            data = self._json_in_text(r.get("response") or "")
        if data is None:
            what = ", ".join(d.get("display_name") or d.get("action", "?") for d in denied)
            raise ValueError(f"no structured_output in agy result" + (f" (the agent tried a denied tool: {what})" if what else ""))
        return data, usage
