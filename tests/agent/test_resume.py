"""Resuming a session: what the model is handed back, and what it keeps.

The bug behind all of it: ``Agent.run`` rebuilt its message list from nothing on
every turn, so the second turn of a conversation was sent without the first, and
the transcript in the window looked continuous while the model had never heard
of any of it. Memory is now the agent's own, and an earlier session can seed it.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.core.agent import RESUME_CHARS, RESUMED_NOTE, Agent, history_block
from app.core.events import EventBus
from app.llm.base import ChatRequest, ChatResponse, Message, Role, ToolCall, Usage
from app.llm.models import ModelInfo
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn


def record(role: str, content: str) -> SimpleNamespace:
    """A stored message, as the database hands one back."""
    return SimpleNamespace(role=role, content=content, reasoning=None)


class NoTools:
    def tools(self):  # type: ignore[override]
        return {}

    def tool_specs(self):  # type: ignore[override]
        return []

    def tool_instructions(self) -> str:
        return ""


class WatchesTheWire(MockLLMProvider):
    """The mock already records every request in ``.requests``; this only says so
    in the type, so a reader does not have to go looking for it."""


def agent_for(provider, *, history=()) -> Agent:
    return Agent(
        provider,
        "test-model",
        NoTools(),  # type: ignore[arg-type]
        bus=EventBus(),
        stream=False,
        model_info=ModelInfo(id="test-model", provider="test", supports_tools=True),
        history=list(history),
    )


# --- shaping the block ------------------------------------------------------


def test_a_resumed_block_starts_on_a_user_turn() -> None:
    """Cutting mid-conversation can leave an answer as the first thing the model
    reads, and it will try to finish a reply whose question it cannot see."""
    block = history_block([record("assistant", "Here it is.")], max_chars=10_000)
    assert block == [], "an answer with no question is not a starting point"


def test_a_resumed_block_keeps_the_newest_end_and_its_order() -> None:
    records = [
        record("user", "make a table"),
        record("assistant", "made it"),
        record("user", "make it round"),
    ]
    block = history_block(records, max_chars=10_000)
    assert [m.content for m in block] == ["make a table", "made it", "make it round"]


def test_a_resumed_block_is_bounded() -> None:
    """An old session can be enormous. Sending all of it does not make the model
    better at the current task; it makes the current task the smallest thing in
    the prompt."""
    records = [record("user" if i % 2 == 0 else "assistant", f"line {i} " + "x" * 500) for i in range(200)]
    block = history_block(records)
    total = sum(len(m.content) for m in block)
    assert total <= RESUME_CHARS, total
    assert block[-1].content.startswith("line 199"), "the newest turn survives"


def test_a_resumed_block_drops_what_it_cannot_send() -> None:
    records = [
        record("user", "  "),
        record("assistant", ""),
        record("tool", "{not a message the model should re-read}"),
        record("user", "the real question"),
    ]
    block = history_block(records, max_chars=10_000)
    assert [m.content for m in block] == ["the real question"]


def test_an_empty_session_resumes_to_nothing() -> None:
    assert history_block([]) == []


# --- memory across turns ---------------------------------------------------


async def test_the_second_turn_carries_the_first() -> None:
    """The whole point: a model that cannot remember is not having a conversation."""
    provider = WatchesTheWire(script=[ScriptedTurn(text="First."), ScriptedTurn(text="Second.")], model="m")
    agent = agent_for(provider)

    await agent.run("what is in the scene?")
    await agent.run("and now?")

    contents = [m.content for m in provider.requests[-1].messages]
    assert "what is in the scene?" in contents
    assert "First." in contents
    assert "and now?" in contents


async def test_a_seeded_history_is_handed_to_the_first_request() -> None:
    provider = WatchesTheWire(script=[ScriptedTurn(text="ok")], model="m")
    agent = agent_for(provider, history=[Message.user("earlier"), Message.assistant("yes")])

    await agent.run("continue")

    contents = [m.content for m in provider.requests[-1].messages]
    assert "earlier" in contents and "yes" in contents


async def test_a_resumed_agent_is_told_the_scene_may_have_changed() -> None:
    """The block is what was said, not a record of what is there. Without saying
    so, the model treats an old claim about the scene as a fact about now."""
    provider = WatchesTheWire(script=[ScriptedTurn(text="ok")], model="m")
    agent = agent_for(provider, history=[Message.user("earlier")])

    await agent.run("continue")

    system = provider.requests[-1].messages[0].content
    assert RESUMED_NOTE in system
    assert "blender.get_scene" in system


async def test_a_fresh_agent_is_not_told_it_was_resumed() -> None:
    provider = WatchesTheWire(script=[ScriptedTurn(text="ok")], model="m")
    agent = agent_for(provider)

    await agent.run("hello")

    assert RESUMED_NOTE not in provider.requests[-1].messages[0].content


async def test_an_explicit_history_argument_wins_over_memory() -> None:
    """The parameter exists for a caller that knows better than the memory."""
    provider = WatchesTheWire(script=[ScriptedTurn(text="ok")], model="m")
    agent = agent_for(provider, history=[Message.user("remembered")])

    await agent.run("next", history=[Message.user("passed in")])

    contents = [m.content for m in provider.requests[-1].messages]
    assert "passed in" in contents
    assert "remembered" not in contents


async def test_memory_never_ends_on_an_unanswered_tool_call() -> None:
    """A provider is entitled to refuse a request whose assistant message asks
    for tools that no result follows. A run stopped between the two would leave
    exactly that, and the *next* run would be refused for it."""
    provider = WatchesTheWire(
        script=[
            ScriptedTurn(
                tool_calls=[("blender.get_scene", {})],
            )
        ],
        model="m",
    )
    agent = agent_for(provider)

    # One turn, stopped with the tool result never appended.
    original = agent._run_tool  # noqa: SLF001 - stopping the run is the point
    agent._run_tool = _raise_cancelled(original)  # type: ignore[method-assign]  # noqa: SLF001
    try:
        await agent.run("what is in the scene?")
    except BaseException:  # noqa: BLE001 - a cancelled run is the scenario
        pass

    assert not any(m.tool_calls for m in agent._memory), agent._memory  # noqa: SLF001


def _raise_cancelled(original):  # type: ignore[no-untyped-def]
    import asyncio

    async def stop(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise asyncio.CancelledError

    return stop


async def test_memory_is_bounded() -> None:
    provider = WatchesTheWire([ScriptedTurn(text="x" * 9000)], model="m")
    agent = agent_for(provider)
    for _ in range(6):
        await agent.run("go on")
    total = sum(len(m.content) for m in agent._memory)  # noqa: SLF001
    assert total <= RESUME_CHARS * 4, total


def test_replayable_drops_only_a_dangling_tail() -> None:
    messages = [
        Message.user("a"),
        Message.assistant("b"),
        Message.assistant("", [ToolCall(id="c1", name="blender.get_scene", arguments={})]),
    ]
    assert Agent._replayable(messages) == messages[:2]  # noqa: SLF001


def test_replayable_keeps_a_tail_with_no_tool_calls() -> None:
    messages = [Message.user("a"), Message.assistant("b")]
    assert Agent._replayable(messages) == messages  # noqa: SLF001


def test_the_resume_note_names_the_tool_that_answers_the_doubt() -> None:
    assert "blender.get_scene" in RESUMED_NOTE
    assert Role is not None
