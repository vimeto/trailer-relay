import json

import httpx
import pytest

from relay.models import (AnthropicAdapter, GeminiAdapter, Message, ModelError, OpenAIChatAdapter, OpenAIResponsesAdapter,
                          OpenRouterAdapter, ToolCall, find_key, make_adapter, parse_spec, read_secrets)
from relay.prompts import DRIVE_TOOL


def history(provider_raw=None, provider=None):
    """user -> assistant(text + tool call) -> tool -> assistant(2 calls) -> tool -> tool"""
    a1 = Message(role="assistant", text="Backing up a bit.", tool_calls=[ToolCall("c1", "drive", {"steer_deg": 0, "distance_m": -2})],
                 raw=provider_raw, provider=provider)
    a2 = Message(role="assistant", text="", tool_calls=[ToolCall("c2", "drive", {"steer_deg": 5, "distance_m": -1}),
                                                       ToolCall("c3", "drive", {"steer_deg": -5, "distance_m": -1})])
    return [Message(role="user", text="LEG 1 briefing"), a1, Message(role="tool", text="REPORT 1", tool_call_id="c1", tool_name="drive"),
            a2, Message(role="tool", text="REPORT 2", tool_call_id="c2", tool_name="drive"),
            Message(role="tool", text="REPORT 3", tool_call_id="c3", tool_name="drive")]


# ----------------------------------------------------------------------------- OpenAI Responses
def test_openai_responses_request_shape():
    ad = OpenAIResponsesAdapter("gpt-6-luna", api_key="k", reasoning_effort="low")
    url, h, body = ad.build_request("SYS", history(), [DRIVE_TOOL])
    assert url.endswith("/v1/responses") and h["Authorization"] == "Bearer k"
    assert body["instructions"] == "SYS" and body["store"] is False
    types = [it.get("type", it.get("role")) for it in body["input"]]
    assert types == ["user", "assistant", "function_call", "function_call_output", "function_call", "function_call", "function_call_output", "function_call_output"]
    assert json.loads(body["input"][2]["arguments"]) == {"steer_deg": 0, "distance_m": -2}
    assert body["tools"][0]["name"] == "drive" and body["tools"][0]["strict"] is True
    assert body["reasoning"] == {"summary": "auto", "effort": "low"} and "reasoning.encrypted_content" in body["include"]
    # the cache key depends only on the instructions + briefing, so it is stable across a leg's turns
    k = body["prompt_cache_key"]
    assert k.startswith("relay-") and ad.build_request("SYS", history()[:1], [DRIVE_TOOL])[2]["prompt_cache_key"] == k
    assert ad.build_request("SYS2", history(), [DRIVE_TOOL])[2]["prompt_cache_key"] != k


def test_openai_responses_reasoning_defaults_by_model():
    # reasoning families always replay encrypted reasoning and ask for a summary; others send no reasoning block
    _, _, body = OpenAIResponsesAdapter("gpt-6-luna", api_key="k").build_request("S", history(), [DRIVE_TOOL])
    assert body["reasoning"] == {"summary": "auto"} and body["include"] == ["reasoning.encrypted_content"]
    _, _, body = OpenAIResponsesAdapter("gpt-4.1", api_key="k").build_request("S", history(), [DRIVE_TOOL])
    assert "reasoning" not in body and "include" not in body


def test_openai_responses_native_replay_keeps_reasoning_strips_ids():
    raw = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "XYZ", "summary": []},
           {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "drive", "arguments": "{}", "status": "completed"}]
    ad = OpenAIResponsesAdapter("m", api_key="k")
    _, _, body = ad.build_request("S", history(raw, "openai"), [DRIVE_TOOL])
    assert body["input"][1] == raw[0]  # reasoning item verbatim
    assert body["input"][2] == {"type": "function_call", "call_id": "c1", "name": "drive", "arguments": "{}"}


def test_openai_responses_parse():
    data = {"status": "completed", "output": [
        {"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": "Hitch is +20, steer left."}]},
        {"type": "message", "content": [{"type": "output_text", "text": "Reversing."}]},
        {"type": "function_call", "call_id": "call_9", "name": "drive", "arguments": "{\"steer_deg\": 3, \"distance_m\": -1.5}"}],
        "usage": {"input_tokens": 100, "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 12}, "input_tokens_details": {"cached_tokens": 64}}}
    msg, u, stop = OpenAIResponsesAdapter("m", api_key="k").parse_response(data)
    assert msg.text == "Reversing." and msg.tool_calls[0].arguments == {"steer_deg": 3, "distance_m": -1.5}
    assert msg.tool_calls[0].id == "call_9" and msg.raw == data["output"]
    assert msg.reasoning == "Hitch is +20, steer left."
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens, u.cached_tokens) == (100, 30, 12, 64)
    assert stop == "tool_use"


# ----------------------------------------------------------------------------- Chat Completions
def test_chat_request_and_parse():
    ad = OpenAIChatAdapter("deepseek-chat", api_key="k", base_url="https://api.deepseek.com/v1")
    url, _, body = ad.build_request("SYS", history(), [DRIVE_TOOL])
    assert url == "https://api.deepseek.com/v1/chat/completions"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "tool"]
    assert body["messages"][4]["tool_calls"][1]["function"]["name"] == "drive"
    assert body["tools"][0]["function"]["parameters"]["required"] == ["steer_deg", "distance_m", "speed_mps"]
    data = {"choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "hmm",
                                     "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "drive", "arguments": "{\"steer_deg\":1,\"distance_m\":-1}"}}]},
                         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
    msg, u, stop = ad.parse_response(data)
    assert msg.tool_calls[0].arguments == {"steer_deg": 1, "distance_m": -1} and stop == "tool_calls"
    assert msg.raw["reasoning_content"] == "hmm" and u.output_tokens == 4


def test_openrouter_defaults():
    url, h, _ = OpenRouterAdapter("moonshotai/kimi-k3", api_key="k").build_request("S", [Message("user", "hi")], [DRIVE_TOOL])
    assert url == "https://openrouter.ai/api/v1/chat/completions" and "X-Title" in h


# ----------------------------------------------------------------------------- Anthropic
def test_anthropic_request_merges_tool_results_and_caches():
    ad = AnthropicAdapter("claude-opus-5-5", api_key="k")
    url, h, body = ad.build_request("SYS", history(), [DRIVE_TOOL])
    assert url.endswith("/v1/messages") and h["anthropic-version"] == "2023-06-01"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]  # alternating; tool results merged
    last = body["messages"][-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result"]
    assert last[-1]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert body["messages"][1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "drive", "input": {"steer_deg": 0, "distance_m": -2}}


def test_anthropic_parse_and_native_replay():
    blocks = [{"type": "thinking", "thinking": "...", "signature": "sig"}, {"type": "text", "text": "ok"},
              {"type": "tool_use", "id": "tu1", "name": "drive", "input": {"steer_deg": 2, "distance_m": -3}}]
    ad = AnthropicAdapter("m", api_key="k", reasoning_effort="low")
    msg, u, stop = ad.parse_response({"content": blocks, "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 100}})
    assert msg.text == "ok" and msg.tool_calls[0].id == "tu1" and stop == "tool_use"
    assert u.input_tokens == 105 and u.cached_tokens == 100
    msg.provider = "anthropic"
    _, _, body = ad.build_request("S", [Message("user", "go"), msg, Message("tool", "R", tool_call_id="tu1")], [DRIVE_TOOL])
    assert body["messages"][1]["content"][0]["signature"] == "sig"
    assert body["thinking"]["type"] == "enabled" and body["max_tokens"] > body["thinking"]["budget_tokens"]


# ----------------------------------------------------------------------------- Gemini
def test_gemini_request_and_parse():
    ad = GeminiAdapter("gemini-3-pro", api_key="k")
    url, h, body = ad.build_request("SYS", history(), [DRIVE_TOOL])
    assert url.endswith("/models/gemini-3-pro:generateContent") and h["x-goog-api-key"] == "k"
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user", "model", "user"]
    fr = body["contents"][-1]["parts"]
    assert fr[0]["functionResponse"]["name"] == "drive" and fr[0]["functionResponse"]["response"]["report"] == "REPORT 2"
    params = body["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "additionalProperties" not in params
    data = {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking...", "thought": True}, {"text": "Go."},
            {"functionCall": {"name": "drive", "args": {"steer_deg": -4, "distance_m": -2}}, "thoughtSignature": "ts"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 5}}
    msg, u, stop = ad.parse_response(data)
    assert msg.text == "Go." and msg.tool_calls[0].arguments["steer_deg"] == -4 and msg.reasoning == "thinking..."
    assert u.output_tokens == 15 and u.reasoning_tokens == 5
    msg.provider = "gemini"
    _, _, b2 = ad.build_request("S", [Message("user", "x"), msg], [DRIVE_TOOL])
    assert b2["contents"][1]["parts"][2]["thoughtSignature"] == "ts"


# ----------------------------------------------------------------------------- transport
def _client(responses):
    it = iter(responses)

    def handler(request):
        code, payload = next(it)
        return httpx.Response(code, json=payload, headers={"retry-after": "0"} if code == 429 else {})

    return httpx.Client(transport=httpx.MockTransport(handler))


OK = {"status": "completed", "output": [{"type": "function_call", "call_id": "c", "name": "drive", "arguments": "{\"steer_deg\":0,\"distance_m\":-1}"}],
      "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_retries_then_succeeds():
    ad = OpenAIResponsesAdapter("m", api_key="k", client=_client([(429, {}), (503, {}), (200, OK)]))
    r = ad.complete("S", [Message("user", "go")], [DRIVE_TOOL])
    assert r.message.tool_calls[0].name == "drive" and r.message.provider == "openai"


def test_client_error_is_not_retried():
    ad = OpenAIResponsesAdapter("m", api_key="k", client=_client([(400, {"error": "bad"})]))
    with pytest.raises(ModelError, match="HTTP 400"):
        ad.complete("S", [Message("user", "go")], [DRIVE_TOOL])


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    ad = OpenAIResponsesAdapter("m", api_key="k", max_retries=2, client=_client([(500, {})] * 3))
    with pytest.raises(ModelError, match="giving up"):
        ad.complete("S", [Message("user", "go")], [DRIVE_TOOL])


# ----------------------------------------------------------------------------- registry / keys
def test_parse_spec():
    assert parse_spec("openai:gpt-6-luna") == ("openai", "gpt-6-luna", None)
    assert parse_spec("openai-chat:qwen@http://localhost:8000/v1") == ("openai-chat", "qwen", "http://localhost:8000/v1")
    with pytest.raises(ValueError):
        parse_spec("gpt-6-luna")


def test_keys_from_secrets_file(tmp_path, monkeypatch):
    for k in ("OPENAI_API_KEY", "OAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    sec = tmp_path / "secrets"
    sec.write_text("# comment\nexport OAI_API_KEY='sk-test'\nHF_TOKEN=x\n")
    assert read_secrets(sec)["OAI_API_KEY"] == "sk-test"
    assert find_key(["OPENAI_API_KEY", "OAI_API_KEY"], sec) == "sk-test"
    ad = make_adapter("openai:gpt-6-luna", secrets_path=sec)
    assert isinstance(ad, OpenAIResponsesAdapter) and ad.api_key == "sk-test"
    monkeypatch.setenv("OPENAI_API_KEY", "env-wins")
    assert make_adapter("openai-chat:x", secrets_path=sec).api_key == "env-wins"
    with pytest.raises(RuntimeError, match="no API key"):
        make_adapter("anthropic:claude-opus-5-5", secrets_path=sec)
    with pytest.raises(ValueError):
        make_adapter("nope:x", secrets_path=sec)
