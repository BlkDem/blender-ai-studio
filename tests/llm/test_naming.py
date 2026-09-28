"""Tool names, as the network insists on them.

An MCP tool is ``blender.create_object``. OpenAI's function names must match
``^[a-zA-Z0-9_-]+$`` and answer anything else with a 400 that says nothing about
what they wanted -- so with a real OpenAI key, every single tool call was
rejected and the studio could not drive Blender at all. Found by using the key,
not by reading the docs.

The name is translated out and back, per request, and the model still sees
something descriptive rather than ``tool_3``: a name that says nothing is a name
the model cannot choose well.
"""

from __future__ import annotations

import pytest

from app.llm.base import ChatRequest, Message, ToolSpec
from app.llm.naming import MAX_LENGTH, SAFE_NAME, ToolNameMap, needs_translation


def test_a_dotted_mcp_name_is_what_needs_translating() -> None:
    assert needs_translation("blender.create_object") is True
    assert needs_translation("generate_3d_asset") is False
    assert needs_translation("blender_create_object") is False


def test_the_wire_name_is_accepted_by_the_strict_pattern() -> None:
    names = ToolNameMap(["blender.create_object", "blender.get_scene", "generate_3d_asset"])
    for original in ("blender.create_object", "blender.get_scene", "generate_3d_asset"):
        wire = names.to_wire(original)
        assert SAFE_NAME.match(wire), f"{original} went out as {wire!r}, which would be rejected"
        assert len(wire) <= MAX_LENGTH


def test_a_name_that_needs_no_change_is_not_changed() -> None:
    """Translation the model can see is a lie about the catalogue."""
    names = ToolNameMap(["generate_3d_asset"])
    assert names.to_wire("generate_3d_asset") == "generate_3d_asset"
    assert names.to_model("generate_3d_asset") == "generate_3d_asset"


def test_the_name_comes_back_as_it_went_out() -> None:
    names = ToolNameMap(["blender.create_object"])
    wire = names.to_wire("blender.create_object")
    assert wire != "blender.create_object"
    assert names.to_model(wire) == "blender.create_object", "the studio must find its tool again"


def test_the_translation_is_stable_within_a_request() -> None:
    names = ToolNameMap()
    first = names.to_wire("blender.create_object")
    second = names.to_wire("blender.create_object")
    assert first == second, "two calls must not send the same tool under two names"


def test_two_tools_differing_only_in_a_forbidden_character_do_not_collide() -> None:
    """Sending the wrong tool because two names sanitised alike is worse than an
    ugly name."""
    names = ToolNameMap(["a.b", "a_b"])
    wires = [names.to_wire("a.b"), names.to_wire("a_b")]
    assert wires[0] != wires[1]
    assert names.to_model(wires[0]) == "a.b"
    assert names.to_model(wires[1]) == "a_b"


def test_a_very_long_name_is_shortened_but_kept_distinct() -> None:
    long_one = "blender." + "x" * 90 + ".one"
    long_two = "blender." + "x" * 90 + ".two"
    names = ToolNameMap([long_one, long_two])
    first, second = names.to_wire(long_one), names.to_wire(long_two)
    assert len(first) <= MAX_LENGTH and SAFE_NAME.match(first)
    assert first != second
    assert names.to_model(first) == long_one


def test_an_unexpected_name_from_a_model_is_left_alone() -> None:
    """A model may answer with a name we never sent. Inventing a mapping for it
    would dispatch a tool the model did not choose."""
    names = ToolNameMap(["blender.create_object"])
    assert names.to_model("something_else") == "something_else"


def test_the_payload_carries_the_translated_names() -> None:
    from app.llm.providers.openai_compatible import OpenAIProvider

    provider = OpenAIProvider(api_key="k", base_url="https://example.invalid/v1")
    request = ChatRequest(
        model="m",
        messages=[Message.user("go")],
        tools=[
            ToolSpec(
                name="blender.create_object",
                description="d",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    names = ToolNameMap([tool.name for tool in request.tools])
    payload = provider._payload(request, stream=False, names=names)  # noqa: SLF001
    sent = payload["tools"][0]["function"]["name"]
    assert sent == "blender_create_object"
    assert "blender.create_object" not in sent, "a dotted name would be rejected by OpenAI"


def test_a_completion_coming_back_is_translated_again() -> None:
    from app.llm.providers.openai_compatible import _parse_completion

    names = ToolNameMap(["blender.create_object"])
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": names.to_wire("blender.create_object"),
                                "arguments": '{"type":"cylinder"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    response = _parse_completion(payload, "m", names=names)
    assert [call.name for call in response.tool_calls] == ["blender.create_object"]
    assert response.tool_calls[0].arguments == {"type": "cylinder"}


@pytest.mark.parametrize("name", ["blender.create_object", "a.b.c", "server:tool", "tool name"])
def test_whatever_comes_in_goes_out_something_a_provider_accepts(name: str) -> None:
    names = ToolNameMap([name])
    assert SAFE_NAME.match(names.to_wire(name))
    assert names.to_model(names.to_wire(name)) == name


def test_the_anthropic_wire_carries_the_translation_too() -> None:
    """Same pattern, same rejection: Anthropic's tool names are
    ``^[a-zA-Z0-9_-]+$`` as well."""
    from app.llm.providers.anthropic import AnthropicProvider, _parse_message

    provider = AnthropicProvider(api_key="k")
    request = ChatRequest(
        model="m",
        messages=[Message.user("go")],
        tools=[
            ToolSpec(
                name="blender.create_object",
                description="d",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    names = ToolNameMap([tool.name for tool in request.tools])
    payload = provider._payload(request, stream=False, names=names)  # noqa: SLF001
    assert payload["tools"][0]["name"] == "blender_create_object"

    answer = _parse_message(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "blender_create_object",
                    "input": {"type": "cylinder"},
                }
            ],
            "usage": {},
        },
        "m",
        names,
    )
    assert [call.name for call in answer.tool_calls] == ["blender.create_object"]


def test_the_assistant_turn_is_translated_when_it_is_replayed() -> None:
    """The bug this whole thing is about, in the form it actually bites.

    Declaring the tools correctly is not enough. The assistant turn goes back out
    on the *next* request inside messages[n].tool_calls, and OpenAI validates
    the name there too -- so a dotted name there is answered with

        Invalid 'messages[2].tool_calls[0].function.name'

    and the run dies on the second step of every conversation.
    """
    from app.llm.base import ToolCall
    from app.llm.providers.openai_compatible import _to_wire

    call = ToolCall.new("blender.create_object", {"type": "cylinder"})
    assistant = Message.assistant("Creating it.", tool_calls=[call])
    names = ToolNameMap(["blender.create_object"])

    wire = _to_wire(assistant, names)
    assert wire["tool_calls"][0]["function"]["name"] == "blender_create_object"
    assert "blender.create_object" not in str(wire), "a dotted name anywhere is a 400"


def test_a_whole_second_turn_is_free_of_dotted_names() -> None:
    """The shape that produced the report: system, user, assistant-with-tool,
    tool result -- sent again."""
    from app.llm.base import Role, ToolCall
    from app.llm.providers.openai_compatible import _to_wire

    call = ToolCall.new("blender.create_object", {"type": "cylinder"})
    conversation = [
        Message.system("you drive Blender"),
        Message.user("make a cylinder"),
        Message.assistant(tool_calls=[call]),
        Message(role=Role.TOOL, content='{"name": "Cylinder"}', tool_call_id=call.id, name=call.name),
    ]
    names = ToolNameMap(["blender.create_object"])
    payload = [_to_wire(message, names) for message in conversation]
    for message in payload:
        for entry in message.get("tool_calls") or []:
            assert SAFE_NAME.match(entry["function"]["name"])


def test_the_anthropic_assistant_turn_is_translated_too() -> None:
    from app.llm.base import ToolCall
    from app.llm.providers.anthropic import _to_wire

    call = ToolCall.new("blender.create_object", {"type": "cylinder"})
    _system, wire = _to_wire([Message.assistant(tool_calls=[call])], ToolNameMap(["blender.create_object"]))
    block = wire[0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "blender_create_object"
