"""Provider-neutral chat + tool-calling interface.

Every adapter turns the same neutral conversation (system prompt, list of
`Message`s, list of `ToolSpec`s) into one provider request and parses the reply
back into a neutral `Message`. Adapters are split into two pure functions,
`build_request` and `parse_response`, so they can be unit-tested without network.

Assistant messages keep the provider's raw output (`raw`, `provider`). When the
history is replayed to the *same* provider, the raw blocks are sent back verbatim
so that reasoning state (OpenAI encrypted reasoning items, Anthropic thinking
signatures, Gemini thought signatures) survives across turns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
import random
import sys
import time
from typing import Any

import httpx


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema (object)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # role == "tool"
    tool_name: str | None = None  # role == "tool" (Gemini needs it)
    raw: Any = None  # provider-native assistant output, for faithful replay
    provider: str | None = None
    reasoning: str = ""  # human-readable reasoning summary/thinking, when the provider returns one (display only)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: "Usage"):
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class ModelResponse:
    message: Message
    usage: Usage
    stop_reason: str
    latency_s: float
    raw: Any = None


def _warn(msg: str):
    print(f"  ! {msg}", file=sys.stderr, flush=True)


class ModelError(RuntimeError):
    pass


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ModelAdapter(ABC):
    provider: str = "base"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 temperature: float | None = None, max_output_tokens: int = 4096,
                 reasoning_effort: str | None = None, extra_body: dict | None = None,
                 client: httpx.Client | None = None, max_retries: int = 5, timeout: float = 120.0,
                 label: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}
        # a short connect timeout; the read timeout bounds a hung request (reasoning models may need it larger)
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0))
        self.max_retries = max_retries
        self.retries = 0
        self.label = label or model

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    # ---- to implement -------------------------------------------------------
    @abstractmethod
    def build_request(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> tuple[str, dict, dict]:
        """Return (url, headers, json_body)."""

    @abstractmethod
    def parse_response(self, data: dict) -> tuple[Message, Usage, str]:
        """Return (assistant message, usage, stop_reason)."""

    # ---- shared -------------------------------------------------------------
    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        url, headers, body = self.build_request(system, messages, tools)
        if self.extra_body:
            body = {**body, **self.extra_body}
        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self.client.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"{type(e).__name__}: {e}"
                self.retries += 1
                _warn(f"{self.spec}: {last_err} after {time.time() - t0:.0f}s; retry {attempt + 1}/{self.max_retries}")
                self._sleep(attempt)
                continue
            if r.status_code in RETRY_STATUS:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                self.retries += 1
                _warn(f"{self.spec}: HTTP {r.status_code}; retry {attempt + 1}/{self.max_retries}")
                ra = r.headers.get("retry-after")
                self._sleep(attempt, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
                continue
            if r.status_code >= 400:
                raise ModelError(f"{self.spec} HTTP {r.status_code}: {r.text[:1000]}")
            data = r.json()
            msg, usage, stop = self.parse_response(data)
            msg.provider = self.provider
            return ModelResponse(message=msg, usage=usage, stop_reason=stop, latency_s=time.time() - t0, raw=data)
        raise ModelError(f"{self.spec}: giving up after {self.max_retries + 1} attempts ({last_err})")

    def _sleep(self, attempt: int, retry_after: float | None = None):
        if self.max_retries == 0:
            return
        delay = retry_after if retry_after is not None else min(30.0, 1.5 * 2 ** attempt) * (0.5 + random.random())
        time.sleep(delay)

    def same_provider(self, m: Message) -> bool:
        return m.provider == self.provider and m.raw is not None


def dumps_args(args: dict) -> str:
    return json.dumps(args, separators=(",", ":"))


def loads_args(s: str | dict | None) -> dict:
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_value": v}
    except json.JSONDecodeError:
        return {"_unparsed": s}
