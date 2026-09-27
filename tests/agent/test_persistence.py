"""The agent's half of persistence.

A run that is not written down cannot be compared, replayed or shown to a user
after the window closes, so these tests assert on rows rather than on calls.
"""

from __future__ import annotations

import pytest

from app.core.agent import Agent
from app.core.events import EventBus
from app.llm.base import Usage
from app.llm.models import ModelInfo
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
from app.mcp.manager import MCPManager
from app.mcp.models import ToolDescriptor


class StoreMCP(MCPManager):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [
            ToolSpec(
                name="blender.get_scene",
                description="Summarise the scene",
                parameters={"type": "object", "properties": {}},
            )
        ]

    def tools(self):  # type: ignore[override]
        return {"blender.get_scene": ToolDescriptor(name="blender.get_scene", description="Summarise")}

    def tool_instructions(self) -> str:
        return ""

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        from app.mcp.models import ToolOutcome

        self.calls.append((name, arguments or {}))
        return ToolOutcome(call_id="c", tool=name, text='{"objects_count": 3}')


async def agent_over(studio, mcp, *turns: ScriptedTurn) -> Agent:
    provider = MockLLMProvider(list(turns))
    return Agent(
        provider,
        "mock-model",
        mcp,
        bus=EventBus(),
        model_info=ModelInfo(id="mock-model", provider="mock", input_price=1.0, output_price=2.0),
        studio=studio,
    )


async def test_a_run_is_written_down(studio) -> None:
    mcp = StoreMCP()
    agent = await agent_over(
        studio, mcp, ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="Three objects.")
    )
    result = await agent.run("what is here?")

    messages = await studio.messages.list(agent.conversation_id)
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "assistant"]
    assert messages[0].content == "what is here?"
    assert messages[-1].content == "Three objects."
    assert all(m.run_id == result.run_id for m in messages)

    calls = await studio.messages.tool_calls_for_run(result.run_id)
    assert [c.tool for c in calls] == ["blender.get_scene"]
    assert calls[0].duration_ms is not None, "a stored call carries its duration"

    usage = await studio.usage.for_run(result.run_id)
    assert len(usage) == 2, "one row per LLM request, not one per run"
    assert usage[0].provider == "mock"
    assert usage[0].input_tokens > 0


async def test_a_failed_tool_call_is_stored_as_a_failure(studio) -> None:
    class Failing(StoreMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
            from app.mcp.models import ToolOutcome

            return ToolOutcome(call_id="c", tool=name, text="it broke", is_error=True, error_code="BOOM")

    agent = await agent_over(
        studio,
        Failing(),
        ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
        ScriptedTurn(text="It broke."),
    )
    result = await agent.run("try it")
    stored = await studio.messages.tool_calls_for_run(result.run_id)
    assert stored[0].is_error is True
    assert stored[0].error_code == "BOOM"


async def test_a_conversation_is_created_once_per_agent(studio) -> None:
    agent = await agent_over(studio, StoreMCP(), ScriptedTurn(text="one"), ScriptedTurn(text="two"))
    await agent.run("first")
    first = agent.conversation_id
    await agent.run("second")
    assert agent.conversation_id == first
    assert len(await studio.conversations.list()) == 1


async def test_an_agent_without_a_database_still_works() -> None:
    """Persistence is optional: a test of the loop should not need SQLite, and a
    chat tool that must not be a database client is a tool people use in
    throwaway contexts."""
    mcp = StoreMCP()
    provider = MockLLMProvider([ScriptedTurn(text="fine")])
    agent = Agent(provider, "mock-model", mcp, bus=EventBus())
    result = await agent.run("hello")
    assert result.text == "fine"


async def test_the_stored_cost_matches_the_run(studio) -> None:
    mcp = StoreMCP()
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    agent = await agent_over(studio, mcp, ScriptedTurn(text="done", usage=usage))
    result = await agent.run("hello")
    stored = await studio.usage.for_run(result.run_id)
    assert stored[0].cost_usd == pytest.approx(result.totals.llm_usd)
    assert stored[0].cost_usd == pytest.approx(3.0)
