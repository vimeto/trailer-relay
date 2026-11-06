"""Concrete adapters: OpenAI Responses, OpenAI-compatible Chat Completions
(OpenAI, OpenRouter, vLLM, DeepSeek, Moonshot, Together, ...), Anthropic Messages
and Google Gemini generateContent."""
from __future__ import annotations

import hashlib
import re
import uuid

from .base import Message, ModelAdapter, ToolCall, ToolSpec, Usage, dumps_args, loads_args


# ============================================================================ OpenAI Responses
class OpenAIResponsesAdapter(ModelAdapter):
    provider = "openai"

    def build_request(self, system, messages, tools):
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/responses"
        items: list[dict] = []
        for m in messages:
            if m.role == "user":
                items.append({"role": "user", "content": [{"type": "input_text", "text": m.text}]})
            elif m.role == "assistant":
                if self.same_provider(m):
                    for it in m.raw:
                        it = dict(it)
                        if it.get("type") in ("function_call", "message"):
                            it.pop("id", None)  # stateless replay (store=false)
                            it.pop("status", None)
                        items.append(it)
                else:
                    if m.text:
                        items.append({"role": "assistant", "content": [{"type": "output_text", "text": m.text}]})
                    for tc in m.tool_calls:
                        items.append({"type": "function_call", "call_id": tc.id, "name": tc.name, "arguments": dumps_args(tc.arguments)})
            elif m.role == "tool":
                items.append({"type": "function_call_output", "call_id": m.tool_call_id, "output": m.text})
        body = {
            "model": self.model,
            "instructions": system,
            "input": items,
            "tools": [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": True} for t in tools],
            "store": False,
            "max_output_tokens": self.max_output_tokens,
            # one cache key per leg (same instructions + briefing) so the growing conversation prefix is reused
            "prompt_cache_key": "relay-" + hashlib.sha1((system + (messages[0].text if messages else "")).encode()).hexdigest()[:16],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort or self.is_reasoning_model:
            # carry the model's reasoning across turns (store=false needs the encrypted content)
            # and ask for a readable summary for the transcript / viewer radio
            body["reasoning"] = {"summary": "auto", **({"effort": self.reasoning_effort} if self.reasoning_effort else {})}
            body["include"] = ["reasoning.encrypted_content"]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return url, headers, body

    @property
    def is_reasoning_model(self) -> bool:
        return bool(re.match(r"^(o\d|gpt-5|gpt-6)", self.model))

    def parse_response(self, data):
        out = data.get("output") or []
        texts, calls, thoughts = [], [], []
        for it in out:
            if it.get("type") == "reasoning":
                thoughts += [s.get("text", "") for s in it.get("summary") or [] if s.get("text")]
            elif it.get("type") == "message":
                for c in it.get("content") or []:
                    if c.get("type") in ("output_text", "text"):
                        texts.append(c.get("text", ""))
            elif it.get("type") == "function_call":
                calls.append(ToolCall(id=it.get("call_id") or it.get("id"), name=it.get("name"), arguments=loads_args(it.get("arguments"))))
        u = data.get("usage") or {}
        usage = Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                      reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
                      cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens", 0))
        stop = "tool_use" if calls else (data.get("status") or "completed")
        if data.get("incomplete_details"):
            stop = "incomplete:" + str((data["incomplete_details"] or {}).get("reason"))
        return Message(role="assistant", text="\n".join(t for t in texts if t), tool_calls=calls, raw=out,
                       reasoning="\n\n".join(thoughts)), usage, stop


# ============================================================================ Chat Completions
class OpenAIChatAdapter(ModelAdapter):
    """Any OpenAI-compatible /chat/completions endpoint."""

    provider = "openai-chat"

    def build_request(self, system, messages, tools):
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                msgs.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                if self.same_provider(m):
                    msgs.append(m.raw)
                else:
                    a: dict = {"role": "assistant", "content": m.text or None}
                    if m.tool_calls:
                        a["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": dumps_args(tc.arguments)}} for tc in m.tool_calls]
                    msgs.append(a)
            elif m.role == "tool":
                msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
        body = {
            "model": self.model,
            "messages": msgs,
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools],
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return url, headers, body

    def parse_response(self, data):
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        calls = [ToolCall(id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", name=tc["function"]["name"],
                          arguments=loads_args(tc["function"].get("arguments")))
                 for tc in (msg.get("tool_calls") or [])]
        raw = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_content")}
        raw.setdefault("role", "assistant")
        u = data.get("usage") or {}
        usage = Usage(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                      reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                      cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        return Message(role="assistant", text=msg.get("content") or "", tool_calls=calls, raw=raw), usage, ch.get("finish_reason") or "stop"


class OpenRouterAdapter(OpenAIChatAdapter):
    provider = "openrouter"

    def build_request(self, system, messages, tools):
        if not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"
        url, headers, body = super().build_request(system, messages, tools)
        headers["X-Title"] = "trailer-relay"
        return url, headers, body


# ============================================================================ Anthropic
class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def build_request(self, system, messages, tools):
        url = (self.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/messages"
        msgs: list[dict] = []

        def push(role, blocks):
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"].extend(blocks)
            else:
                msgs.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m.role == "user":
                push("user", [{"type": "text", "text": m.text}])
            elif m.role == "assistant":
                if self.same_provider(m):
                    push("assistant", m.raw)
                else:
                    blocks = ([{"type": "text", "text": m.text}] if m.text else [])
                    blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls]
                    push("assistant", blocks or [{"type": "text", "text": "(no output)"}])
            elif m.role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text}])
        # cache the stable prefix (system + tools) and the conversation so far
        body = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": msgs,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
        }
        if msgs and msgs[-1]["content"]:
            last = dict(msgs[-1]["content"][-1])
            last["cache_control"] = {"type": "ephemeral"}
            msgs[-1]["content"][-1] = last
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            budget = {"low": 2048, "medium": 8192, "high": 24000}.get(self.reasoning_effort, 8192)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(self.max_output_tokens, budget + 2048)
        headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return url, headers, body

    def parse_response(self, data):
        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {}) for b in blocks if b.get("type") == "tool_use"]
        u = data.get("usage") or {}
        usage = Usage(input_tokens=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                      output_tokens=u.get("output_tokens", 0), cached_tokens=u.get("cache_read_input_tokens", 0))
        thinking = "\n\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking" and b.get("thinking"))
        return Message(role="assistant", text="\n".join(texts), tool_calls=calls, raw=blocks, reasoning=thinking), usage, data.get("stop_reason") or "end_turn"


# ============================================================================ Gemini
class GeminiAdapter(ModelAdapter):
    provider = "gemini"

    def build_request(self, system, messages, tools):
        base = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"
        contents: list[dict] = []

        def push(role, parts):
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        for m in messages:
            if m.role == "user":
                push("user", [{"text": m.text}])
            elif m.role == "assistant":
                if self.same_provider(m):
                    push("model", m.raw)
                else:
                    parts = ([{"text": m.text}] if m.text else [])
                    parts += [{"functionCall": {"name": tc.name, "args": tc.arguments}} for tc in m.tool_calls]
                    push("model", parts or [{"text": "(no output)"}])
            elif m.role == "tool":
                fr = {"name": m.tool_name or "tool", "response": {"report": m.text}}
                if m.tool_call_id and not m.tool_call_id.startswith("gem_"):
                    fr["id"] = m.tool_call_id
                push("user", [{"functionResponse": fr}])
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": [{"name": t.name, "description": t.description, "parameters": _gemini_schema(t.parameters)} for t in tools]}],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }
        if self.temperature is not None:
            body["generationConfig"]["temperature"] = self.temperature
        if self.reasoning_effort:
            budget = {"low": 1024, "medium": 8192, "high": 24576}.get(self.reasoning_effort, 8192)
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget, "includeThoughts": True}
        headers = {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}
        return url, headers, body

    def parse_response(self, data):
        cands = data.get("candidates") or [{}]
        content = cands[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        calls = []
        for p in parts:
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(ToolCall(id=fc.get("id") or f"gem_{uuid.uuid4().hex[:10]}", name=fc["name"], arguments=fc.get("args") or {}))
        u = data.get("usageMetadata") or {}
        usage = Usage(input_tokens=u.get("promptTokenCount", 0), output_tokens=u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                      reasoning_tokens=u.get("thoughtsTokenCount", 0), cached_tokens=u.get("cachedContentTokenCount", 0))
        thoughts = "\n\n".join(p["text"] for p in parts if p.get("thought") and p.get("text"))
        return Message(role="assistant", text="\n".join(texts), tool_calls=calls, raw=parts, reasoning=thoughts), usage, cands[0].get("finishReason") or "STOP"


def _gemini_schema(schema: dict) -> dict:
    """Gemini's function schema is an OpenAPI subset: drop unsupported keys."""
    drop = {"additionalProperties", "$schema", "strict"}
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k not in drop}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema
