"""Runtime policy, accounting, deadline and replay regressions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.agent import Agent, LocalTool, RunResult
from app.core.cost import Budget
from app.llm.base import ContentPart, Message, ToolCall, ToolResult
from app.llm.models import ModelInfo
from app.llm.providers.mock import ScriptedTurn
from tests.agent.test_agent import FakeMCP, agent_with


@pytest.mark.parametrize("arguments", [{}, {"n": "wrong"}, {"n": 0}, {"n": 2, "extra": 1}, {"__raw__": "{"}, []])
async def test_schema_rejects_before_local_dispatch(arguments):
    handler = AsyncMock(return_value="ok")
    tool = LocalTool("local", "", {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}},
                                     "required": ["n"], "additionalProperties": False}, handler)
    agent = agent_with(FakeMCP(), local_tools=[tool])
    outcome = await agent._run_tool("run", ToolCall("id", "local", arguments), RunResult("run"))
    assert outcome.is_error
    assert outcome.error_code == "INVALID_ARGUMENTS"
    handler.assert_not_awaited()


async def test_schema_rejects_before_mcp_dispatch():
    mcp = FakeMCP()
    agent = agent_with(mcp)
    outcome = await agent._run_tool("run", ToolCall.new("blender.get_scene", {"object_limit": "bad"}), RunResult("run"))
    assert outcome.error_code == "INVALID_ARGUMENTS"
    assert not mcp.calls


@pytest.mark.parametrize("payload,code", [
    ({"error": {"code": "BROKEN", "message": "no"}}, "BROKEN"),
    ({"status": "failed", "error_code": "FAILED"}, "FAILED"),
    ({"status": "error"}, "TOOL_ERROR"),
    ({"is_error": True}, "TOOL_ERROR"),
    (ToolResult(ToolCall.new("other", {}), "no", is_error=True, error_code="RESULT_ERROR"), "RESULT_ERROR"),
])
async def test_structured_local_errors(payload, code):
    agent = agent_with(FakeMCP(), local_tools=[LocalTool("local", "", {}, AsyncMock(return_value=payload))])
    result = RunResult("run")
    outcome = await agent._run_tool("run", ToolCall.new("local", {}), result)
    assert outcome.is_error and outcome.error_code == code
    assert result.totals.tool_errors == 1


async def test_refusal_is_audited_and_counted_even_with_import_capability():
    mcp = FakeMCP()
    mcp.tools = lambda: {"blender.import_glb": object()}
    agent = agent_with(mcp)
    agent.studio = SimpleNamespace(tool_calls=SimpleNamespace(
        start=AsyncMock(return_value=SimpleNamespace(id="record")), finish=AsyncMock()))
    result = RunResult("run")
    outcome = await agent._run_tool("run", ToolCall.new("blender.execute_python", {}), result)
    assert outcome.error_code == "TOOL_DISABLED"
    assert "allow blender.execute_python" in outcome.content
    assert not mcp.calls
    assert result.totals.tool_calls == result.totals.tool_errors == 1
    agent.studio.tool_calls.start.assert_awaited_once()
    assert agent.studio.tool_calls.finish.await_args.kwargs["error_code"] == "TOOL_DISABLED"


@pytest.mark.parametrize("stage", ["provider", "tool", "storage"])
async def test_real_deadline_interrupts_inflight_work(stage):
    cancelled = asyncio.Event()

    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = agent_with(FakeMCP(), ScriptedTurn(tool_calls=[("local", {})]),
                       budget=Budget(max_seconds=0.02), stream=False,
                       local_tools=[LocalTool("local", "", {}, hang)])
    if stage == "provider":
        agent.provider.chat = hang
    elif stage == "storage":
        agent.studio = object()
        agent._conversation = hang
    result = await asyncio.wait_for(agent.run("go"), timeout=1)
    assert not result.finished
    assert "wall-clock" in result.stopped_because
    assert cancelled.is_set()
    assert not agent.is_running()
    assert not any(message.tool_calls for message in agent._memory)


async def test_external_cancellation_preserves_only_complete_memory():
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    agent = agent_with(FakeMCP(), ScriptedTurn(tool_calls=[("local", {}), ("wait", {})]), stream=False,
                       local_tools=[LocalTool("local", "", {}, AsyncMock(return_value="ok")),
                                    LocalTool("wait", "", {}, hang)])
    task = asyncio.create_task(agent.run("go"))
    await entered.wait()
    agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [message.role.value for message in agent._memory] == ["user"]


def test_replay_atomic_groups_and_orphans():
    a, b = ToolCall.new("a", {}), ToolCall.new("b", {})
    group = [Message.assistant(tool_calls=[a, b]), Message.tool_result(a, "ok"), Message.tool_result(b, "ok")]
    question = Message.user("q")
    assert Agent._replayable([question, *group]) == [question, *group]
    # A block that opens on a tool call is refused by Gemini, so it never opens on one.
    assert Agent._replayable(group) == []
    assert Agent._replayable(group[:-1]) == []
    assert Agent._replayable([group[1]]) == []
    assert Agent._replayable([group[0], group[1], group[1]]) == []
    assert Agent._replayable([Message.user("old"), group[0], Message.tool_result(a, "x" * 25000), group[2],
                              Message.user("new")]) == [Message.user("new")]


def test_replay_drops_tool_pictures_but_keeps_attachments():
    call = ToolCall.new("blender.render_preview", {})
    render = ContentPart.image_part("cmVuZGVy", "image/png")
    attached = Message.user("look")
    attached.parts = [ContentPart.image_part("cGhvdG8=", "image/jpeg")]
    replayed = Agent._replayable([
        attached,
        Message.assistant(tool_calls=[call]),
        Message.tool_result(call, "rendered", parts=[render]),
    ])
    assert replayed[0].parts == attached.parts
    assert replayed[2].content == "rendered" and replayed[2].parts == []


async def test_capabilities_disable_tools_and_fallback_to_chat():
    agent = agent_with(FakeMCP(), ScriptedTurn("done"),
                       model_info=ModelInfo("mock-model", supports_tools=False, supports_streaming=False))
    agent.provider.stream = AsyncMock(side_effect=AssertionError("must not stream"))
    result = await agent.run("go")
    assert result.finished
    assert agent.provider.requests[0].tools == []
    refused = await agent._run_tool("run", ToolCall.new("blender.get_scene", {}), RunResult("run"))
    assert refused.error_code == "TOOL_UNAVAILABLE"


async def test_run_totals_are_snapshots_and_tool_budget_resets():
    turns = [ScriptedTurn(tool_calls=[("blender.get_scene", {})])] * 2
    agent = agent_with(FakeMCP(), *turns, budget=Budget(max_tool_calls=1), stream=False)
    first = await agent.run("one")
    # A final model turn may be stopped by the tool limit, but the next run gets its own allowance.
    second = await agent.run("two")
    assert first.totals.tool_calls == second.totals.tool_calls == 1
    assert first.totals is not second.totals
    assert agent.cost.totals.tool_calls == 2


async def test_session_money_limit_remains_cumulative():
    agent = agent_with(FakeMCP(), ScriptedTurn("one"), ScriptedTurn("two"), stream=False,
                       budget=Budget(max_session_cost=0.00001))
    first = await agent.run("one")
    second = await agent.run("two")
    assert first.totals.requests == 1
    assert second.totals.requests == 0
    assert "session LLM cost" in second.stopped_because
