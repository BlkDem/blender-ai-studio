"""HTTP-level tests for the providers.

Real request shapes and real response parsing, driven through a mock transport.
A provider adapter is a translation layer, and the only way to test a translation
is to check both directions of it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.errors import AuthenticationError, ProviderError, RateLimitError
from app.llm.base import ChatRequest, Message, ToolCall, ToolSpec
from app.llm.models import ModelInfo, cost_of
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.anthropic import _to_wire as anthropic_wire
from app.llm.providers.gemini import GeminiProvider, _parse_candidates
from app.llm.providers.gemini import _to_wire as gemini_wire
from app.llm.providers.openai_compatible import OpenAICompatibleProvider, _to_wire


def provider_with(handler: Any, **kwargs: Any) -> OpenAICompatibleProvider:
    """A provider whose HTTP goes to ``handler`` instead of a network."""
    provider = OpenAICompatibleProvider(api_key="k", base_url="https://api.test/v1", **kwargs)
    provider._client = httpx.AsyncClient(
        base_url="https://api.test/v1", transport=httpx.MockTransport(handler)
    )
    return provider


# --- OpenAI-compatible ------------------------------------------------------


async def test_a_chat_request_is_built_the_way_openai_expects() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    provider = provider_with(handler)
    response = await provider.chat(
        ChatRequest(
            messages=[Message.system("be brief"), Message.user("hello")],
            model="m",
            tools=[
                ToolSpec("t", "does a thing", {"type": "object", "properties": {"a": {"type": "string"}}})
            ],
        )
    )

    assert response.text == "hi"
    assert response.usage.input_tokens == 12
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert seen["tools"][0]["function"]["name"] == "t"
    assert seen["stream"] is False


async def test_a_tool_call_is_parsed_with_its_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "create_object", "arguments": '{"name": "Cube"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9},
            },
        )

    response = await provider_with(handler).chat(ChatRequest(messages=[Message.user("go")], model="m"))
    assert response.wants_tools()
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].arguments == {"name": "Cube"}


async def test_truncated_tool_arguments_do_not_kill_the_turn() -> None:
    """A model that streams half a JSON object is common; losing the run is not
    an acceptable way to deal with it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "c", "function": {"name": "t", "arguments": '{"name": "Cu'}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    response = await provider_with(handler).chat(ChatRequest(messages=[], model="m"))
    assert response.tool_calls[0].arguments == {"__raw__": '{"name": "Cu'}


async def test_streaming_yields_text_then_the_finished_tool_call() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Look"}}]},
        {"choices": [{"delta": {"content": "ing"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_x", "function": {"name": "get_scene", "arguments": ""}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"object'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '_limit": 5}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"usage": {"prompt_tokens": 7, "completion_tokens": 11}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = provider_with(handler)
    received = [
        chunk
        async for chunk in provider.stream(ChatRequest(messages=[], model="m", tools=[ToolSpec("get_scene")]))
    ]

    text = "".join(c.text for c in received if c.type == "text")
    assert text == "Looking"
    ends = [c for c in received if c.type == "tool_end"]
    assert len(ends) == 1, "the tool call is reported once, however many chunks carried it"
    end = ends[0]
    assert end.name == "get_scene"
    assert end.arguments == {"object_limit": 5}
    usage = next(c for c in received if c.type == "usage")
    assert (usage.usage.input_tokens, usage.usage.output_tokens) == (7, 11)
    assert next(c for c in received if c.type == "end").finish_reason == "tool_calls"


async def test_a_model_that_refuses_max_tokens_is_answered_in_its_own_words() -> None:
    """OpenAI's newer models name the parameter they want; ask rather than guess.

    Every OpenAI-compatible server in common use knows only ``max_tokens``, and
    the model -- not the config -- decides which spelling it is. So the refusal
    is read and answered once.
    """
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
                    }
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    response = await provider_with(handler).chat(ChatRequest(messages=[], model="gpt-5-mini", max_tokens=400))
    assert response.text == "ready"
    assert sent[0]["max_tokens"] == 400
    assert "max_completion_tokens" not in sent[0]
    assert sent[1]["max_completion_tokens"] == 400
    assert "max_tokens" not in sent[1], "sending both is what the model refused in the first place"


async def test_a_refusal_that_is_not_about_the_parameter_is_not_retried() -> None:
    """A retry here would spend the same money twice and hide the real error."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    with pytest.raises(ProviderError, match="model not found"):
        await provider_with(handler).chat(ChatRequest(messages=[], model="m", max_tokens=8))
    assert calls == 1


async def test_streaming_retries_once_when_the_model_rejects_the_parameter() -> None:
    """Streaming has to survive the same refusal, and report the tool once."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(
                400,
                json={"error": {"message": "Use 'max_completion_tokens' instead of 'max_tokens'"}},
            )
        chunks = [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    received = [
        chunk
        async for chunk in provider_with(handler).stream(
            ChatRequest(messages=[], model="gpt-5-mini", max_tokens=32)
        )
    ]
    assert "".join(c.text for c in received if c.type == "text") == "ok"
    assert next(c for c in received if c.type == "end").finish_reason == "stop"
    assert sent[1]["max_completion_tokens"] == 32
    assert "max_tokens" not in sent[1]


async def test_streaming_asks_for_usage_so_the_cost_is_not_a_guess() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    provider = provider_with(handler)
    _ = [chunk async for chunk in provider.stream(ChatRequest(messages=[], model="m"))]

    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


async def test_reasoning_text_is_read_from_either_field() -> None:
    """Reasoning models disagree on the name, and a gateway is as likely to use
    a vendor's as OpenAI's."""

    def handler(field: str) -> Any:
        def inner(request: httpx.Request) -> httpx.Response:
            body = f"data: {json.dumps({'choices': [{'delta': {field: 'thinking...'}}]})}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, text=body)

        return inner

    for field in ("reasoning_content", "reasoning"):
        provider = provider_with(handler(field))
        chunks = [c async for c in provider.stream(ChatRequest(messages=[], model="m"))]
        assert "".join(c.text for c in chunks if c.type == "reasoning") == "thinking..."


@pytest.mark.parametrize(
    "status,expected",
    [(401, AuthenticationError), (429, RateLimitError), (500, ProviderError), (404, ProviderError)],
)
async def test_http_failures_become_typed_errors(status: int, expected: type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "the reason from the provider"}})

    with pytest.raises(expected):
        await provider_with(handler).chat(ChatRequest(messages=[], model="m"))


async def test_the_providers_own_message_is_kept_in_the_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "model 'nope' not found"}})

    with pytest.raises(ProviderError) as excinfo:
        await provider_with(handler).chat(ChatRequest(messages=[], model="nope"))
    assert "not found" in excinfo.value.message


async def test_a_connection_failure_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(ProviderError) as excinfo:
        await provider_with(handler).chat(ChatRequest(messages=[], model="m"))
    assert excinfo.value.retryable is True


async def test_a_model_the_server_knows_and_the_studio_does_not_is_offered() -> None:
    """A gateway can serve something this project has never heard of. Hiding it
    until someone edits a config file would make the model selector a lie."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "space-bunny-free"}, {"id": "known"}]})

    provider = provider_with(handler, models=[ModelInfo(id="known", provider="openai-compatible")])
    models = await provider.list_models()
    assert {m.id for m in models} == {"known", "space-bunny-free"}


async def test_a_model_listing_failure_falls_back_to_the_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    provider = provider_with(handler, models=[ModelInfo(id="known", provider="openai-compatible")])
    assert [m.id for m in await provider.list_models()] == ["known"]


def test_a_tool_result_is_addressed_by_its_call() -> None:
    call = ToolCall(id="c1", name="t")
    wire = _to_wire(Message.tool_result(call, "42"))
    assert wire == {"role": "tool", "tool_call_id": "c1", "content": "42"}


def test_an_assistant_tool_call_keeps_its_id() -> None:
    wire = _to_wire(Message.assistant("", [ToolCall(id="c9", name="t", arguments={"a": 1})]))
    assert wire["tool_calls"][0]["id"] == "c9"
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


# --- Anthropic --------------------------------------------------------------


def anthropic_with(handler: Any) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="k", model="claude-test")
    provider._client = httpx.AsyncClient(
        base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler)
    )
    return provider


def test_the_system_prompt_is_a_top_level_field_for_anthropic() -> None:
    system, messages = anthropic_wire([Message.system("rules"), Message.user("go")])
    assert system == "rules"
    assert messages == [{"role": "user", "content": "go"}]


def test_several_tool_results_merge_into_one_user_turn() -> None:
    """The API rejects two tool_result blocks split across messages, and a model
    that asked for three tools in one turn produces exactly that case."""
    calls = [ToolCall(id="a", name="t"), ToolCall(id="b", name="t")]
    _system, messages = anthropic_wire(
        [Message.assistant("", calls), Message.tool_result(calls[0], "1"), Message.tool_result(calls[1], "2")]
    )
    assert [m["role"] for m in messages] == ["assistant", "user"]
    assert len(messages[1]["content"]) == 2
    assert messages[1]["content"][0]["tool_use_id"] == "a"


async def test_anthropic_parses_tool_use_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "Let me look"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_scene", "input": {"object_limit": 5}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 40, "output_tokens": 12},
            },
        )

    response = await anthropic_with(handler).chat(
        ChatRequest(messages=[Message.user("go")], model="claude-test")
    )
    assert response.text == "Let me look"
    assert response.tool_calls[0].id == "tu_1"
    assert response.tool_calls[0].arguments == {"object_limit": 5}
    assert response.usage.input_tokens == 40
    assert response.finish_reason == "tool_use"


async def test_anthropic_assembles_streamed_tool_arguments() -> None:
    """Anthropic streams arguments as JSON fragments; none of them parse alone."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sure"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_9", "name": "create_object"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"type"'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ': "cube"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 25}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    chunks = [c async for c in anthropic_with(handler).stream(ChatRequest(messages=[], model="claude-test"))]
    end = next(c for c in chunks if c.type == "tool_end")
    assert end.call_id == "tu_9"
    assert end.arguments == {"type": "cube"}
    usage = next(c for c in chunks if c.type == "usage")
    assert usage.usage.output_tokens == 25, "message_delta usage is cumulative"


def test_anthropic_always_sends_max_tokens() -> None:
    """The API rejects a request without one."""
    provider = AnthropicProvider(api_key="k", model="m")
    payload = provider._payload(ChatRequest(messages=[Message.user("x")], model="m"), stream=False)
    assert payload["max_tokens"] > 0


# --- Gemini -----------------------------------------------------------------


def gemini_with(handler: Any) -> GeminiProvider:
    provider = GeminiProvider(api_key="k", model="gemini-test")
    provider._client = httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        transport=httpx.MockTransport(handler),
    )
    return provider


def test_gemini_uses_system_instruction_and_two_roles() -> None:
    system, contents = gemini_wire(
        [
            Message.system("rules"),
            Message.user("go"),
            Message.assistant("", [ToolCall(id="c1", name="t", arguments={"a": 1})]),
        ]
    )
    assert system == "rules"
    assert [c["role"] for c in contents] == ["user", "model"]
    assert contents[1]["parts"][0]["functionCall"]["args"] == {"a": 1}


def test_a_gemini_tool_result_is_a_user_turn_with_the_call_id() -> None:
    call = ToolCall(id="fc-1", name="get_scene")
    _system, contents = gemini_wire([Message.assistant("", [call]), Message.tool_result(call, "ok")])
    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"][0]["functionResponse"]["id"] == "fc-1"


def test_gemini_rejects_unknown_schema_keywords() -> None:
    """A third-party MCP server's schema must not 400 the whole request."""
    from app.llm.base import _clean_schema_for_gemini

    cleaned = _clean_schema_for_gemini(
        {
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {"a": {"type": "string", "default": "x", "examples": ["y"]}},
        }
    )
    assert "$schema" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "default" not in cleaned["properties"]["a"]


def test_gemini_sends_tools_beside_the_generation_config() -> None:
    """Google answers a tool nested in generationConfig with "Unknown name",
    which reads like a schema problem and costs the model every tool it has."""
    from app.llm.providers.gemini import GeminiProvider

    payload = GeminiProvider(api_key="k")._payload(
        ChatRequest(
            messages=[Message.user("go")],
            model="m",
            tools=[ToolSpec("blender.get_scene", description="look")],
            tool_choice="any",
            temperature=0.5,
            max_tokens=64,
        )
    )
    assert "tools" in payload, "a tool the model cannot see is not a tool"
    assert "tools" not in payload.get("generationConfig", {})
    assert "tool_calling_config" not in payload.get("generationConfig", {})
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert payload["generationConfig"]["maxOutputTokens"] == 64
    names = [d["name"] for d in payload["tools"][0]["functionDeclarations"]]
    assert names == ["blender.get_scene"]


def test_a_gemini_thought_signature_survives_the_round_trip() -> None:
    """Gemini signs a call and refuses the result that comes back unsigned.

    The signature is opaque to the studio, so the test is that it is carried
    out of the response and put back on the wire untouched -- not that it is
    understood.
    """
    from app.llm.providers.gemini import GeminiProvider

    response = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"id": "c1", "name": "blender.get_scene", "args": {}},
                                # A field of the part, beside functionCall. This
                                # is where Google puts it, and reading it from
                                # inside the call yields an empty string -- which
                                # looks exactly like a model that sent none.
                                "thoughtSignature": "Cq8CAbc...",
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    )
    parsed = _parse_candidates(response.json(), "gemini-3-flash-preview")
    assert parsed.tool_calls[0].thought_signature == "Cq8CAbc..."

    payload = GeminiProvider(api_key="k")._payload(
        ChatRequest(
            messages=[
                Message.assistant("", parsed.tool_calls),
                Message.tool_result(parsed.tool_calls[0], "{}"),
            ],
            model="gemini-3-flash-preview",
        )
    )
    part = payload["contents"][0]["parts"][0]
    assert part["thoughtSignature"] == "Cq8CAbc...", "the signature Google gave, handed back"
    assert "thoughtSignature" not in part["functionCall"], "beside the call, not inside it"
    assert payload["contents"][1]["parts"][0]["functionResponse"]["name"] == "blender.get_scene"


def test_a_call_without_a_signature_is_sent_without_one() -> None:
    """A model that sent no signature gets none invented; a made-up one is a
    value the provider would reject on the next turn."""
    from app.llm.providers.gemini import GeminiProvider

    payload = GeminiProvider(api_key="k")._payload(
        ChatRequest(
            messages=[Message.assistant("", [ToolCall.new("t", {})])],
            model="gemini-flash-latest",
        )
    )
    assert "thoughtSignature" not in payload["contents"][0]["parts"][0]["functionCall"]


def test_the_default_gemini_model_is_an_alias_rather_than_a_number() -> None:
    """Google retired 2.5 to new accounts while it was the current release.

    A default pinned to a number is therefore a default that stops working for
    everyone who installs after the next release, and the failure arrives as a
    provider error rather than as anything to do with the studio.
    """
    from app.llm.providers.gemini import DEFAULT_MODEL

    assert DEFAULT_MODEL.endswith("-latest"), (
        f"the default is pinned to {DEFAULT_MODEL!r}, which goes stale on its own"
    )


def test_gemini_leaves_the_calling_mode_to_google_when_it_is_not_pinned() -> None:
    from app.llm.providers.gemini import GeminiProvider

    payload = GeminiProvider(api_key="k")._payload(
        ChatRequest(messages=[Message.user("go")], model="m", tools=[ToolSpec("t")])
    )
    assert "toolConfig" not in payload, "AUTO is the default; sending it says nothing"


async def test_gemini_parses_function_calls_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Checking"},
                                {"functionCall": {"id": "fc1", "name": "get_scene", "args": {}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 8},
            },
        )

    response = await gemini_with(handler).chat(ChatRequest(messages=[], model="gemini-test"))
    assert response.text == "Checking"
    assert response.tool_calls[0].id == "fc1"
    assert response.usage.input_tokens == 30


async def test_gemini_streaming_does_not_repeat_accumulated_text() -> None:
    events = [
        {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "Hello there"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    chunks = [c async for c in gemini_with(handler).stream(ChatRequest(messages=[], model="gemini-test"))]
    text = "".join(c.text for c in chunks if c.type == "text")
    assert text == "Hello there", "an accumulating stream must not be concatenated twice"


# --- pricing ----------------------------------------------------------------


def test_cost_is_derived_from_the_model_row() -> None:
    model = ModelInfo(id="m", provider="p", input_price=3.0, output_price=15.0)
    assert cost_of(model, 1_000_000, 0) == 3.0
    assert cost_of(model, 0, 1_000_000) == 15.0
    assert cost_of(model, 1000, 1000) == pytest.approx(0.003 + 0.015)


def test_cached_input_is_billed_at_the_cheaper_rate() -> None:
    model = ModelInfo(id="m", provider="p", input_price=10.0, output_price=0.0)
    assert cost_of(model, 1_000_000, 0, cached_tokens=900_000) == pytest.approx(1.0)


def test_an_unknown_model_costs_nothing_rather_than_a_guess() -> None:
    """A made-up price on a benchmark table is worse than a blank one."""
    provider = OpenAICompatibleProvider(api_key="k", model="m")
    assert (
        provider.price(
            "never-heard-of-it",
            type("U", (), {"input_tokens": 10, "output_tokens": 10, "cached_tokens": 0, "cost_usd": 0.0})(),
        ).cost_usd
        == 0.0
    )


def test_model_info_round_trips_through_a_dict() -> None:
    model = ModelInfo(id="m", provider="p", supports_vision=True, input_price=1.0, extra={"gateway": "x"})
    restored = ModelInfo.from_dict(model.to_dict())
    assert restored.supports_vision is True
    assert restored.extra["gateway"] == "x"
    assert restored.input_price == 1.0


def test_capabilities_are_data_not_a_name() -> None:
    """A model named like a vision model that cannot see images must report so."""
    model = ModelInfo(id="space-bunny-free", provider="p", supports_vision=False)
    assert model.supports_vision is False
    assert model.to_dict()["supports_vision"] is False
