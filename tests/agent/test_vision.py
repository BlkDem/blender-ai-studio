"""Render, look, correct.

The loop this describes is the reason ``blender-mcp`` can render at all: the
model asks for a picture, gets a picture, and can act on what is in it. The
parts for carrying an image existed for a long time and nothing ever filled
them -- a render came back, was counted, and was never sent anywhere, so "the
model can see the result" was a claim rather than a capability.
"""

from __future__ import annotations

import base64
import json

from app.core.agent import Agent
from app.core.events import EventBus, EventType
from app.llm.base import ChatRequest, ChatResponse, ContentPart, Message, Role, ToolCall, Usage
from app.llm.models import ModelInfo
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
from app.mcp.models import ToolDescriptor, ToolOutcome

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"pixels" * 40).decode()


class PictureMCP:
    """A bridge whose one tool returns a picture, the way render_preview does."""

    def __init__(self) -> None:
        self.tools_seen: list[ChatRequest] = []

    def tools(self):  # type: ignore[override]
        return {
            "blender.render_preview": ToolDescriptor(name="blender.render_preview", description="Render"),
            "blender.get_scene": ToolDescriptor(name="blender.get_scene", description="Scene"),
        }

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [ToolSpec(name=name, description=tool.description) for name, tool in self.tools().items()]

    def tool_instructions(self) -> str:
        return ""

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        if name == "blender.render_preview":
            return ToolOutcome(
                call_id="c",
                tool=name,
                text=json.dumps({"output_path": "C:/tmp/shot.png", "render_time": 2.1}),
                images=[PNG],
                image_mime_types=["image/png"],
            )
        return ToolOutcome(call_id="c", tool=name, text=json.dumps({"objects": []}))


class SeesImages(MockLLMProvider):
    """Records what it was sent, so the test can look at the wire."""

    def __init__(self, *, turns, **kwargs) -> None:
        super().__init__(turns, **kwargs)
        self.seen: list[list[ContentPart]] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.seen.append([part for message in request.messages for part in message.parts])
        if any(m.role is Role.TOOL for m in request.messages):
            return ChatResponse(
                text="The chest is on its side.", usage=Usage(10, 5), model=self.default_model()
            )
        return ChatResponse(
            tool_calls=[ToolCall.new("blender.render_preview", {})],
            usage=Usage(20, 10),
            finish_reason="tool_calls",
            model=self.default_model(),
        )


def agent_with(provider, *, vision: bool) -> Agent:
    # stream=False: the two paths build the same request, and this way the test
    # can look at it.
    return Agent(
        provider,
        "test-model",
        PictureMCP(),  # type: ignore[arg-type]
        bus=EventBus(),
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test", supports_tools=True, supports_vision=vision),
    )


async def test_a_vision_model_is_sent_the_render() -> None:
    """The point of the whole thing: the picture reaches the model."""
    provider = SeesImages(turns=[ScriptedTurn(text="unused")], model="test-model")
    agent = agent_with(provider, vision=True)

    result = await agent.run("render the scene and tell me what is wrong with it")

    assert len(provider.seen) == 2, "one call to ask, one with the picture"
    second = provider.seen[1]
    images = [part for part in second if part.type == "image"]
    assert len(images) == 1, "the render was sent"
    assert images[0].data == PNG
    assert images[0].mime_type == "image/png", "and identified, so the server knows what it is"
    assert "The chest is on its side." in result.text


async def test_a_text_only_model_is_not_sent_a_picture() -> None:
    """Sending an image to a model that cannot see is a request it rejects, and
    the run dies on a capability the user never asked about."""
    provider = SeesImages(turns=[ScriptedTurn(text="unused")], model="test-model")
    agent = agent_with(provider, vision=False)

    result = await agent.run("render the scene")

    assert all(parts == [] for parts in provider.seen), "no picture anywhere in the conversation"
    assert result.finished, "and the run finished anyway"


async def test_the_picture_is_kept_for_the_window_even_when_the_model_cannot_see_it() -> None:
    """A person should see the render even when the model cannot."""
    bus = EventBus()
    events: list = []
    bus.subscribe(lambda event: events.append(event))
    provider = SeesImages(turns=[ScriptedTurn(text="done")], model="test-model")
    agent = Agent(
        provider,
        "test-model",
        PictureMCP(),  # type: ignore[arg-type]
        bus=bus,
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test", supports_vision=False),
    )
    await agent.run("render it")
    tool_events = [e for e in events if e.type is EventType.TOOL_FINISHED and e.payload.get("images")]
    assert tool_events and tool_events[0].payload["images"] == 1, "the card still says one picture"


async def test_a_tool_that_returns_no_picture_changes_nothing() -> None:
    provider = SeesImages(turns=[ScriptedTurn(text="fine")], model="test-model")
    agent = agent_with(provider, vision=True)
    result = await agent.run("what is in the scene")
    assert result.finished


def test_the_openai_wire_carries_a_data_uri() -> None:
    from app.llm.providers.openai_compatible import _to_wire

    message = Message(
        role=Role.TOOL,
        content="rendered in 2.1s",
        tool_call_id="c1",
        parts=[ContentPart.image_part(PNG, "image/png")],
    )
    wire = _to_wire(message)
    # A tool result stays a tool result; the picture rides with it.
    assert wire["role"] == "tool"
    assert wire["content"] == "rendered in 2.1s", "the text is unchanged for a text-only server"

    user = Message(role=Role.USER, content="what is this?", parts=[ContentPart.image_part(PNG, "image/png")])
    payload = _to_wire(user)
    assert isinstance(payload["content"], list)
    assert payload["content"][0] == {"type": "text", "text": "what is this?"}
    assert payload["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_the_anthropic_wire_carries_a_base64_source() -> None:
    from app.llm.providers.anthropic import _to_wire

    _system, wire = _to_wire(
        [Message(role=Role.USER, content="what is this?", parts=[ContentPart.image_part(PNG, "image/png")])]
    )
    block = wire[0]["content"][1]
    assert block["type"] == "image"
    assert block["source"] == {"type": "base64", "media_type": "image/png", "data": PNG}


def test_the_anthropic_wire_puts_a_render_inside_the_tool_result() -> None:
    """A tool result with a picture has to stay one message: the API rejects
    two tool_result blocks split across messages."""
    from app.llm.providers.anthropic import _to_wire

    call = ToolCall.new("blender.render_preview", {})
    _system, wire = _to_wire(
        [
            Message.assistant(tool_calls=[call]),
            Message.tool_result(call, "rendered in 2.1s", parts=[ContentPart.image_part(PNG, "image/png")]),
        ]
    )
    result_block = wire[-1]["content"][0]
    assert result_block["type"] == "tool_result"
    blocks = result_block["content"]
    assert blocks[0] == {"type": "text", "text": "rendered in 2.1s"}
    assert blocks[1]["type"] == "image"
