"""Model specs -> adapters, and API-key discovery (env vars first, then ~/.secrets)."""
from __future__ import annotations

import os
from pathlib import Path

from .base import ModelAdapter
from .providers import AnthropicAdapter, GeminiAdapter, OpenAIChatAdapter, OpenAIResponsesAdapter, OpenRouterAdapter
from .codex import CodexAdapter
from .cli import AgyAdapter, ClaudeCodeAdapter
from .scripted import ScriptedAdapter

PROVIDERS = {
    "openai": (OpenAIResponsesAdapter, ["OPENAI_API_KEY", "OAI_API_KEY"]),
    "openai-chat": (OpenAIChatAdapter, ["OPENAI_API_KEY", "OAI_API_KEY"]),
    "openrouter": (OpenRouterAdapter, ["OPENROUTER_API_KEY"]),
    "anthropic": (AnthropicAdapter, ["ANTHROPIC_API_KEY"]),
    "gemini": (GeminiAdapter, ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "scripted": (ScriptedAdapter, []),
    "codex": (CodexAdapter, []),  # the `codex` CLI (its own login), no API key needed
    "claude-code": (ClaudeCodeAdapter, []),  # the `claude` CLI (Claude Code login)
    "agy": (AgyAdapter, []),  # the Antigravity CLI (Google login): Gemini and the models it hosts
}

SECRETS_PATH = Path(os.environ.get("RELAY_SECRETS", str(Path.home() / ".secrets")))


def read_secrets(path: Path | str | None = None) -> dict[str, str]:
    p = Path(path) if path else SECRETS_PATH
    out: dict[str, str] = {}
    try:
        text = p.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def find_key(names: list[str], secrets_path: Path | str | None = None) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    sec = read_secrets(secrets_path)
    for n in names:
        if sec.get(n):
            return sec[n]
    return None


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """'provider:model' or 'openai-chat:model@https://base.url/v1' -> (provider, model, base_url)."""
    if ":" not in spec:
        raise ValueError(f"model spec must look like 'provider:model', got {spec!r}")
    provider, rest = spec.split(":", 1)
    base_url = None
    if "@" in rest and provider in ("openai-chat", "openai", "anthropic", "gemini", "openrouter"):
        rest, base_url = rest.split("@", 1)
    return provider, rest, base_url


def make_adapter(spec: str, api_key: str | None = None, secrets_path=None, **kw) -> ModelAdapter:
    provider, model, base_url = parse_spec(spec)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    cls, key_names = PROVIDERS[provider]
    if provider in ("scripted", "codex", "claude-code", "agy"):
        return cls(model, **kw)
    key = api_key or find_key(key_names, secrets_path)
    if not key:
        raise RuntimeError(f"no API key for {provider}: set one of {key_names} in the environment or {SECRETS_PATH}")
    return cls(model, api_key=key, base_url=base_url or kw.pop("base_url", None), **kw)
