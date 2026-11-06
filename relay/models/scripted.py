"""Offline adapters for tests and baselines (no network)."""
from __future__ import annotations

import itertools
import re
from typing import Callable

from .base import Message, ModelAdapter, ModelResponse, ToolCall, Usage


class ScriptedAdapter(ModelAdapter):
    """Plays back a policy: `policy(system, messages) -> Message | list[(steer, dist)] | str`.

    Named policies (model name in 'scripted:<name>'):
      straight-back   reverse straight 2 m at a time
      idle            never calls a tool (exercises the nudge / DNF path)
      wiggle          alternates steering (exercises shuttles/contacts)
    """

    provider = "scripted"

    def __init__(self, model: str = "straight-back", policy: Callable | None = None, **kw):
        kw.pop("api_key", None)
        kw.pop("base_url", None)
        super().__init__(model, api_key=None, client=kw.pop("client", None), **{k: v for k, v in kw.items() if k in ("label",)})
        if policy is None and model.startswith("replay="):  # scripted:replay=<run file>: re-install that run's controllers
            from ..baselines import replay_policy

            policy = replay_policy(model.split("=", 1)[1])
        self.policy = policy or NAMED[model]
        self._ids = itertools.count()
        self.calls = 0

    def build_request(self, system, messages, tools):  # pragma: no cover - not used
        return "", {}, {}

    def parse_response(self, data):  # pragma: no cover - not used
        raise NotImplementedError

    def complete(self, system, messages, tools) -> ModelResponse:
        self.calls += 1
        if [t.name for t in tools] == ["leave_note"] and not getattr(self.policy, "writes_notes", False):
            # learning studies: offline drivers leave a fixed, rule-abiding note
            n = sum(1 for m in messages if m.role == "assistant" and any(c.name == "leave_note" for c in m.tool_calls))
            out = Message(role="assistant", text="", tool_calls=[ToolCall(id=f"sc_{next(self._ids)}", name="leave_note", arguments={
                "note": f"Scripted note {n + 1}: start the bend early, keep the speed low, and watch the front corner."})])
        else:
            out = self.policy(system, messages)
        if isinstance(out, Message):
            msg = out
        elif isinstance(out, str):
            msg = Message(role="assistant", text=out)
        else:
            msg = Message(role="assistant", text="", tool_calls=[
                ToolCall(id=f"sc_{next(self._ids)}", name="drive", arguments={"steer_deg": s, "distance_m": d}) for s, d in out])
        msg.provider = self.provider
        return ModelResponse(message=msg, usage=Usage(input_tokens=sum(len(m.text) for m in messages) // 4, output_tokens=20),
                             stop_reason="tool_use" if msg.tool_calls else "end_turn", latency_s=0.0)


def _straight_back(system, messages):
    return [(0.0, -2.0)]


def _idle(system, messages):
    return "I am thinking about it."


def _wiggle(system, messages):
    n = sum(1 for m in messages if m.role == "assistant")
    return [((-20.0 if n % 2 else 20.0), -1.5 if n % 3 else 1.0)]


def _hitch_keeper(system, messages):
    """A tiny reactive policy that reads the report text: keep the hitch near zero."""
    last = next((m.text for m in reversed(messages) if m.role in ("tool", "user")), "")
    mh = re.search(r"HITCH ANGLE: ([+-]?\d+(?:\.\d+)?)", last)
    h = float(mh.group(1)) if mh else 0.0
    # reversing: steer in the same direction as the hitch error to bring the trailer back
    return [(max(-30.0, min(30.0, 2.0 * h)), -1.0)]


NAMED = {"straight-back": _straight_back, "idle": _idle, "wiggle": _wiggle, "hitch-keeper": _hitch_keeper}


def _baseline(name):
    def policy(system, messages):
        from ..baselines import POLICIES

        return POLICIES[name](system, messages)
    return policy


NAMED.update({k: _baseline(k) for k in ("ctrl-local", "ctrl-shoulder", "ctrl-camera")})
