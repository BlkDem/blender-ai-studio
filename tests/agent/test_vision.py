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

from app.core.agent import Agent, RunResult
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


# --- pictures a person attached -------------------------------------------

#: A different payload from the render's, so a test cannot pass by mixing the two.
REFERENCE = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"reference" * 20).decode()


async def test_a_picture_the_user_attached_is_sent_with_the_turn() -> None:
    """Drop a reference, ask for it to be built: the model has to receive the
    picture, and on the *first* request -- there is no tool call in between."""
    provider = SeesImages(turns=[ScriptedTurn(text="unused")], model="test-model")
    agent = agent_with(provider, vision=True)

    await agent.run("build this", images=[ContentPart.image_part(REFERENCE, "image/png")])

    first = provider.seen[0]
    images = [part for part in first if part.type == "image"]
    assert len(images) == 1
    assert images[0].data == REFERENCE
    assert images[0].mime_type == "image/png"


async def test_an_attached_picture_is_not_sent_to_a_text_only_model() -> None:
    """The same gate a render goes through, for the same reason: the request
    would be refused, and the turn would die on a capability the user never
    asked about."""
    provider = SeesImages(turns=[ScriptedTurn(text="done")], model="test-model")
    agent = agent_with(provider, vision=False)

    result = await agent.run("build this", images=[ContentPart.image_part(REFERENCE, "image/png")])

    assert all(parts == [] for parts in provider.seen)
    assert result.finished


async def test_a_picture_with_no_words_is_still_a_turn() -> None:
    """Paste a screenshot, press Enter, type nothing. The turn carries the image
    on its own rather than being discarded as an empty message."""
    provider = SeesImages(turns=[ScriptedTurn(text="A low table.")], model="test-model")
    agent = agent_with(provider, vision=True)

    result = await agent.run("", images=[ContentPart.image_part(REFERENCE, "image/png")])

    assert provider.seen, "the turn reached the model at all"
    assert [p for p in provider.seen[0] if p.type == "image"], "carrying the picture"
    assert result.finished


async def test_an_attached_picture_and_a_render_both_arrive() -> None:
    provider = SeesImages(turns=[ScriptedTurn(text="unused")], model="test-model")
    agent = agent_with(provider, vision=True)

    await agent.run(
        "make it like this, then render it",
        images=[ContentPart.image_part(REFERENCE, "image/png")],
    )

    first = [p.data for p in provider.seen[0] if p.type == "image"]
    second = [p.data for p in provider.seen[1] if p.type == "image"]
    assert first == [REFERENCE], "the reference went out with the words"
    assert second == [REFERENCE, PNG], "and the render came back on top of it"


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


async def test_a_switched_off_tool_is_not_offered() -> None:
    """The server may have the gate open; the studio's switch is the last word.

    Without this the checkbox in Settings controlled nothing, and a server
    started with execution enabled could not be closed from the app -- a switch
    that reads as a promise and is not one.
    """
    provider = SeesImages(turns=[ScriptedTurn(text="fine")], model="test-model")
    agent = Agent(
        provider,
        "test-model",
        PictureMCP(),  # type: ignore[arg-type]
        bus=EventBus(),
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test"),
        allow_execute_python=False,
    )
    assert "blender.execute_python" not in [t.name for t in agent.tools()]


async def test_a_switched_off_tool_is_refused_with_a_reason() -> None:
    provider = SeesImages(turns=[ScriptedTurn(text="fine")], model="test-model")
    agent = Agent(
        provider,
        "test-model",
        PictureMCP(),  # type: ignore[arg-type]
        bus=EventBus(),
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test"),
        allow_execute_python=False,
    )
    outcome = await agent._run_tool(  # noqa: SLF001
        "r", ToolCall.new("blender.execute_python", {"code": "result = 1"}), RunResult(run_id="r")
    )
    assert outcome.is_error
    assert outcome.error_code == "TOOL_DISABLED"
    assert "allow blender.execute_python" in outcome.content, "and it says which switch to press"


async def test_the_tool_is_offered_when_the_switch_is_on() -> None:
    class WithPython(PictureMCP):
        def tools(self):  # type: ignore[override]
            found = super().tools()
            from app.mcp.models import ToolDescriptor

            found["blender.execute_python"] = ToolDescriptor(
                name="blender.execute_python", description="Run Python"
            )
            return found

    provider = SeesImages(turns=[ScriptedTurn(text="fine")], model="test-model")
    agent = Agent(
        provider,
        "test-model",
        WithPython(),
        bus=EventBus(),
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test"),
        allow_execute_python=True,
    )
    assert "blender.execute_python" in [t.name for t in agent.tools()]
    outcome = await agent._run_tool(  # noqa: SLF001
        "r", ToolCall.new("blender.execute_python", {"code": "result = 1"}), RunResult(run_id="r")
    )
    assert not outcome.is_error
