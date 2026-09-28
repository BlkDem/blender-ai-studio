"""The agent loop.

The provider is scripted, so a test can say exactly what the model does: call a
tool, read the result, answer. What is being tested is the studio's half — does it
run what was asked, in order, tell the model what came back, and stop when it
should.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest

from app.core.agent import BASE_SYSTEM_PROMPT, Agent, LocalTool
from app.core.cost import Budget
from app.core.events import EventBus, EventType
from app.llm.base import Message, StreamChunk
from app.llm.models import ModelInfo
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
from app.mcp.manager import MCPManager
from app.mcp.models import ToolDescriptor


class FakeMCP(MCPManager):
    """An MCPManager with a catalogue and a recorded call log, no subprocess.

    The transport itself is covered against a real server in tests/mcp; here the
    subject is the loop, and a subprocess per test would only make it slower.
    """

    def __init__(self, tools: dict[str, ToolDescriptor] | None = None) -> None:
        super().__init__()
        self._catalog = tools or {
            "blender.get_scene": ToolDescriptor(
                name="blender.get_scene",
                description="Compact summary of the active scene",
                schema={"type": "object", "properties": {"object_limit": {"type": "integer"}}},
            )
        }
        self.calls: list[tuple[str, dict]] = []
        self.outcomes: dict[str, tuple[str, bool]] = {}
        self.instructions = "Prefer get_scene before changing anything."
        self.connected = True

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.schema)
            for t in self._catalog.values()
        ]

    def tools(self):  # type: ignore[override]
        return dict(self._catalog)

    def tool_instructions(self) -> str:
        return self.instructions

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        from app.mcp.models import ToolOutcome

        self.calls.append((name, arguments or {}))
        if name not in self._catalog:
            return ToolOutcome(call_id="x", tool=name, text=f"No tool named {name}", is_error=True)
        text, is_error = self.outcomes.get(name, (f"{name} ok", False))
        # The real client reads a structured code out of the result text; the stub
        # has to do the same or it hides every behaviour that keys off it.
        code = ""
        for token in re.findall(r"\b([A-Z][A-Z_]{5,})\b", text):
            code = token
        return ToolOutcome(
            call_id="x", tool=name, text=text, is_error=is_error, error_code=code if is_error else ""
        )


@pytest.fixture
def mcp() -> FakeMCP:
    return FakeMCP()


def agent_with(mcp: FakeMCP, *turns: ScriptedTurn, **kwargs) -> Agent:
    provider = MockLLMProvider(
        list(turns), models=[ModelInfo(id="mock-model", provider="mock", input_price=1.0, output_price=2.0)]
    )
    defaults: dict[str, Any] = {
        "bus": EventBus(),
        "model_info": ModelInfo(id="mock-model", provider="mock", input_price=1.0, output_price=2.0),
    }
    defaults.update(kwargs)
    return Agent(provider, "mock-model", mcp, **defaults)


class _SignedStreamProvider(MockLLMProvider):
    """A stream that carries a Gemini thought signature on its tool call.

    The signature is the sort of thing that only exists on the wire, so a test
    double built from names and arguments cannot show it going missing. This
    one can. It does its own bookkeeping rather than replaying a script,
    because the script belongs to the base class' stream, which is replaced.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.turns = 0

    async def stream(self, request):  # type: ignore[override]
        self.requests.append(request)
        self.turns += 1
        yield StreamChunk(type="start")
        if self.turns == 1:
            yield StreamChunk(
                type="tool_start",
                call_id="c1",
                name="blender.get_scene",
                thought_signature="Cq8CAbc...",
            )
            yield StreamChunk(type="tool_end", call_id="c1", name="blender.get_scene", arguments={})
        else:
            yield StreamChunk(type="text", text="There is one cube.")
        yield StreamChunk(type="end", text="", finish_reason="stop")


async def test_a_gemini_signature_survives_the_agents_own_reassembly(mcp: FakeMCP) -> None:
    """The agent rebuilds a tool call from the chunks it was given.

    It is a different function from the one that parsed the response, so
    anything it does not copy is lost by the next turn -- and Gemini refuses a
    tool result whose call came back unsigned.
    """
    provider = _SignedStreamProvider(
        models=[ModelInfo(id="mock-model", provider="mock", input_price=0.0, output_price=0.0)]
    )
    agent = Agent(
        provider,
        "mock-model",
        mcp,
        bus=EventBus(),
        model_info=ModelInfo(id="mock-model", provider="mock", input_price=0.0, output_price=0.0),
    )
    await agent.run("look around")

    assert mcp.calls == [("blender.get_scene", {})], "the tool still ran"
    history = provider.requests[1].messages
    remembered = [c for m in history for c in (m.tool_calls or [])]
    assert remembered, "the call is in the history"
    assert remembered[0].thought_signature == "Cq8CAbc..."


# --- the basics -------------------------------------------------------------


async def test_a_plain_answer_needs_no_tools(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="The scene has one cube."))
    result = await agent.run("what is in the scene?")
    assert result.text == "The scene has one cube."
    assert result.finished
    assert result.tool_calls == []
    assert mcp.calls == [], "the model did not ask for a tool, so none was run"


async def test_a_tool_call_runs_and_the_result_goes_back_to_the_model(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.get_scene", {"object_limit": 5})]),
        ScriptedTurn(text="There is one cube."),
    )
    result = await agent.run("look around")

    assert mcp.calls == [("blender.get_scene", {"object_limit": 5})]
    assert result.text == "There is one cube."
    sent = agent.provider.requests[1].messages
    assert sent[-1].role.value == "tool", "the tool result must be in the second request"
    assert "blender.get_scene ok" in sent[-1].content


async def test_several_tool_calls_in_a_row(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
        ScriptedTurn(tool_calls=[("blender.get_scene", {"object_limit": 1})]),
        ScriptedTurn(text="Done."),
    )
    result = await agent.run("look twice")
    assert len(mcp.calls) == 2
    assert result.steps == 3


async def test_several_tools_in_one_turn_run_in_order(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.get_scene", {}), ("blender.get_scene", {"object_limit": 2})]),
        ScriptedTurn(text="Both done."),
    )
    await agent.run("two at once")
    assert mcp.calls == [("blender.get_scene", {}), ("blender.get_scene", {"object_limit": 2})]


async def test_a_failed_tool_is_reported_to_the_model_not_swallowed(mcp: FakeMCP) -> None:
    mcp.outcomes["blender.get_scene"] = ("failed (OBJECT_NOT_FOUND): no such object", True)
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
        ScriptedTurn(text="That object is not there."),
    )
    result = await agent.run("look at something missing")
    assert result.tool_calls[0].is_error is True
    assert "no such object" in result.tool_calls[0].content


async def test_a_refused_transaction_is_given_a_way_out() -> None:
    """A transaction left by an earlier session is invisible to this agent, so the
    raw refusal is not something a model can act on."""
    refusing = transactional_mcp()
    refusing.outcomes["blender.begin_transaction"] = (
        "failed (TRANSACTION_ACTIVE): A transaction is already open",
        True,
    )
    agent = agent_with(
        refusing,
        ScriptedTurn(tool_calls=[("blender.begin_transaction", {})]),
        ScriptedTurn(text="I will roll it back first."),
    )
    result = await agent.run("build something")
    seen = result.tool_calls[0].content
    assert "blender.rollback_transaction" in seen
    assert "blender.commit_transaction" in seen
    assert "earlier one" in seen


async def test_an_unrelated_error_gets_no_transaction_advice(mcp: FakeMCP) -> None:
    mcp.outcomes["blender.get_scene"] = ("failed (OBJECT_NOT_FOUND)", True)
    agent = agent_with(mcp, ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="ok"))
    result = await agent.run("look")
    assert "rollback_transaction" not in result.tool_calls[0].content


async def test_an_unknown_tool_is_told_to_the_model(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.create_object", {"type": "cube"})]),
        ScriptedTurn(text="I cannot create objects here."),
    )
    result = await agent.run("add a cube")
    assert result.tool_calls[0].is_error
    assert "No tool named" in result.tool_calls[0].content


async def test_streaming_is_the_default_and_produces_tokens(mcp: FakeMCP) -> None:
    bus = EventBus()
    agent = agent_with(mcp, ScriptedTurn(text="one two three"), bus=bus)
    run_id = "run_test"
    await agent.run("hi", run_id=run_id)
    kinds = [event.type for event in bus.history(run_id)]
    assert EventType.RUN_STARTED in kinds
    assert EventType.TOKEN in kinds
    assert EventType.RUN_FINISHED in kinds


async def test_tool_calls_are_visible_as_they_happen(mcp: FakeMCP) -> None:
    bus = EventBus()
    agent = agent_with(
        mcp, ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="ok"), bus=bus
    )
    await agent.run("go", run_id="run_1")
    events = bus.history("run_1")
    finished = [e for e in events if e.type is EventType.TOOL_FINISHED]
    assert finished and finished[0].payload["tool"] == "blender.get_scene"
    assert finished[0].payload["is_error"] is False


# --- the system prompt ------------------------------------------------------


async def test_the_system_prompt_explains_how_to_use_the_tools(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="hi"))
    await agent.run("hello")
    system = agent.provider.requests[0].messages[0]
    assert system.role.value == "system"
    assert "controlling Blender" in system.content
    assert "Never claim an operation succeeded" in system.content


async def test_the_servers_own_instructions_reach_the_model(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="hi"))
    await agent.run("hello")
    assert "Prefer get_scene before changing anything" in agent.provider.requests[0].messages[0].content


async def test_a_users_prompt_is_added_not_substituted(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="hi"), system_prompt="Always use metric units.")
    await agent.run("hello")
    content = agent.provider.requests[0].messages[0].content
    assert BASE_SYSTEM_PROMPT.splitlines()[0] in content
    assert "Always use metric units." in content


async def test_the_tool_catalog_is_not_repeated_in_the_prompt(mcp: FakeMCP) -> None:
    """It already travels in the request's ``tools`` field.

    Listing it again in the system prompt cost a few hundred tokens on every
    request of a run, which on a real run was more input tokens than the whole
    conversation: 30k for five requests, most of it a duplicate catalogue.
    """
    agent = agent_with(mcp, ScriptedTurn(text="hi"))
    await agent.run("hello")
    request = agent.provider.requests[0]
    system = request.messages[0].content
    assert "Tools available now" not in system
    assert "- blender.get_scene" not in system, "no catalogue line in the prompt"
    assert [spec.name for spec in request.tools] == ["blender.get_scene"], "the tools field carries it"
    assert request.tools[0].description, "with its description and schema"
    assert "Prefer get_scene before changing anything." in system, "the server's own notes still are"


def test_a_local_tool_joins_the_mcp_ones(mcp: FakeMCP) -> None:
    async def handler(arguments, *, run_id=""):  # pragma: no cover - not called here
        return {}

    agent = agent_with(
        mcp,
        ScriptedTurn(text="x"),
        local_tools=[LocalTool("generate_3d_asset", "Generate a 3D asset", {"type": "object"}, handler)],
    )
    assert {spec.name for spec in agent.tools()} == {"blender.get_scene", "generate_3d_asset"}


async def test_a_local_tool_is_called_without_touching_mcp(mcp: FakeMCP) -> None:
    seen: list[dict] = []

    async def generate(arguments, *, run_id=""):
        seen.append(arguments)
        return {"task_id": "abc", "status": "queued"}

    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("generate_3d_asset", {"prompt": "a chest"})]),
        ScriptedTurn(text="Started the generation."),
        local_tools=[
            LocalTool(
                "generate_3d_asset",
                "Generate a 3D asset from a prompt",
                {"type": "object", "properties": {"prompt": {"type": "string"}}},
                generate,
            )
        ],
    )
    result = await agent.run("make a chest")
    assert seen == [{"prompt": "a chest"}]
    assert mcp.calls == []
    assert "abc" in result.tool_calls[0].content


# --- budgets and limits -----------------------------------------------------


async def test_the_step_limit_stops_a_loop(mcp: FakeMCP) -> None:
    """A model that keeps asking for the same thing must hit a number, not a bill."""
    agent = agent_with(
        mcp,
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(10)],
        budget=Budget(max_steps=3, max_tool_calls=50),
    )
    result = await agent.run("loop forever")
    assert result.finished is False
    assert "steps" in result.stopped_because
    assert len(mcp.calls) == 3


async def test_the_tool_call_limit_stops_a_loop(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {}), ("blender.get_scene", {})]) for _ in range(10)],
        budget=Budget(max_steps=100, max_tool_calls=3),
    )
    result = await agent.run("loop forever")
    assert "tool calls" in result.stopped_because
    assert len(mcp.calls) == 3


async def test_the_cost_limit_stops_a_run(mcp: FakeMCP) -> None:
    # 1000 input + 1000 output at $1/$2 per million is $0.003 per call, so a
    # one-cent cap is reached within a few steps.
    rich = ScriptedTurn(
        tool_calls=[("blender.get_scene", {})],
        usage=type(
            "U",
            (),
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cached_tokens": 0, "cost_usd": 0.0},
        )(),
    )
    agent = agent_with(mcp, *[rich for _ in range(5)], budget=Budget(max_steps=10, max_session_cost=0.005))
    result = await agent.run("expensive")
    assert "cost" in result.stopped_because


async def test_a_single_expensive_request_stops_the_run(mcp: FakeMCP) -> None:
    """One enormous response is enough; the run does not have to loop to hurt."""
    huge = ScriptedTurn(
        text="expensive",
        usage=type(
            "U", (), {"input_tokens": 10_000_000, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0}
        )(),
    )
    agent = agent_with(mcp, huge, budget=Budget(max_request_cost=1.0))
    result = await agent.run("too much")
    assert result.finished is False
    assert "single request" in result.stopped_because


def transactional_mcp() -> FakeMCP:
    """A bridge that actually offers the transaction tools."""
    from app.mcp.models import ToolDescriptor

    return FakeMCP(
        {
            "blender.get_scene": ToolDescriptor(name="blender.get_scene", description="Summarise"),
            "blender.begin_transaction": ToolDescriptor(
                name="blender.begin_transaction", description="Start one"
            ),
            "blender.commit_transaction": ToolDescriptor(
                name="blender.commit_transaction", description="Keep it"
            ),
            "blender.rollback_transaction": ToolDescriptor(
                name="blender.rollback_transaction", description="Undo it"
            ),
        }
    )


async def test_a_run_that_stops_mid_transaction_says_so() -> None:
    """A leftover transaction makes the *next* run's begin fail with no obvious
    cause, so the run that left it has to say so."""
    bus = EventBus()
    agent = agent_with(
        transactional_mcp(),
        ScriptedTurn(tool_calls=[("blender.begin_transaction", {})]),
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(5)],
        budget=Budget(max_steps=2),
        bus=bus,
    )
    await agent.run("build something", run_id="run_tx")

    finished = [e for e in bus.history("run_tx") if e.type is EventType.RUN_FAILED]
    assert finished
    assert finished[0].payload["transaction_open"] is True
    assert "commit it to keep the work" in finished[0].payload["hint"]
    assert agent.transaction_warning().startswith("A transaction from this run is still open")


async def test_a_closed_transaction_carries_no_warning() -> None:
    agent = agent_with(
        transactional_mcp(),
        ScriptedTurn(tool_calls=[("blender.begin_transaction", {})]),
        ScriptedTurn(tool_calls=[("blender.commit_transaction", {})]),
        ScriptedTurn(text="done"),
    )
    result = await agent.run("build and keep")
    assert result.finished
    assert agent.transaction_warning() == ""


async def test_a_failed_begin_does_not_claim_a_transaction() -> None:
    failing = transactional_mcp()
    failing.outcomes["blender.begin_transaction"] = ("failed (TRANSACTION_ACTIVE)", True)
    agent = agent_with(
        failing,
        ScriptedTurn(tool_calls=[("blender.begin_transaction", {})]),
        ScriptedTurn(text="I cannot start a transaction."),
    )
    result = await agent.run("begin")
    assert any(c.is_error for c in result.tool_calls)
    assert agent.transaction_warning() == "", "nothing of ours is open, so nothing to warn about"


async def test_the_wall_clock_limit_is_enforced(mcp: FakeMCP) -> None:
    agent = agent_with(
        mcp,
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(5)],
        budget=Budget(max_steps=100, max_tool_calls=100, max_seconds=0.0),
    )
    result = await agent.run("quick please")
    assert "seconds" in result.stopped_because


async def test_a_budget_stop_is_published_with_a_reason(mcp: FakeMCP) -> None:
    bus = EventBus()
    agent = agent_with(
        mcp,
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(5)],
        budget=Budget(max_steps=2),
        bus=bus,
    )
    await agent.run("loop", run_id="run_budget")
    failure = [e for e in bus.history("run_budget") if e.type is EventType.RUN_FAILED]
    assert failure
    assert failure[0].payload["reason"] == "BUDGET_EXCEEDED"
    assert failure[0].payload["detail"]


# --- cost -------------------------------------------------------------------


async def test_usage_is_priced_and_accumulated(mcp: FakeMCP) -> None:
    usage = type(
        "U", (), {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cached_tokens": 0, "cost_usd": 0.0}
    )()
    agent = agent_with(mcp, ScriptedTurn(text="hi", usage=usage))
    result = await agent.run("hello")
    assert result.totals.llm_usd == pytest.approx(3.0), "$1/M input + $2/M output"
    assert result.totals.input_tokens == 1_000_000


async def test_three_d_credits_are_kept_out_of_the_llm_bill(mcp: FakeMCP) -> None:
    async def generate(arguments, *, run_id=""):
        return {"task_id": "t", "credits": 20, "cost_usd": 0.4}

    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("generate_3d_asset", {"prompt": "chest"})]),
        ScriptedTurn(text="started"),
        local_tools=[LocalTool("generate_3d_asset", "gen", {"type": "object"}, generate)],
    )
    result = await agent.run("make a chest")
    assert result.totals.three_d_credits == 20
    assert result.totals.three_d_usd == pytest.approx(0.4)
    assert result.totals.llm_usd < 0.4, "a token bill is not a 3D bill"


async def test_tool_calls_are_counted(mcp: FakeMCP) -> None:
    mcp.outcomes["blender.get_scene"] = ("nope", True)
    agent = agent_with(mcp, ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="ok"))
    result = await agent.run("go")
    assert result.totals.tool_calls == 1
    assert result.totals.tool_errors == 1


# --- cancellation -----------------------------------------------------------


async def test_a_run_can_be_stopped_while_a_tool_is_in_flight(mcp: FakeMCP) -> None:
    started = asyncio.Event()

    async def slow(arguments, *, run_id=""):
        started.set()
        await asyncio.sleep(10)
        return {}

    class SlowMCP(FakeMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):
            started.set()
            await asyncio.sleep(10)
            from app.mcp.models import ToolOutcome

            return ToolOutcome(call_id="x", tool=name, text="never")

    del slow
    agent = agent_with(
        SlowMCP(),
        *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(20)],
    )
    task = asyncio.create_task(agent.run("go", run_id="run_cancel"))
    await asyncio.wait_for(started.wait(), timeout=2)

    assert agent.cancel("run_cancel") is True
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelling_a_run_that_never_started_is_false(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="hi"))
    assert agent.cancel("no-such-run") is False


async def test_cancellation_is_published(mcp: FakeMCP) -> None:
    bus = EventBus()

    class Blocking(FakeMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):
            await asyncio.sleep(10)
            raise AssertionError

    agent = agent_with(
        Blocking(), *[ScriptedTurn(tool_calls=[("blender.get_scene", {})]) for _ in range(5)], bus=bus
    )
    task = asyncio.create_task(agent.run("go", run_id="run_c"))
    await asyncio.sleep(0.05)
    agent.cancel("run_c")
    with pytest.raises(asyncio.CancelledError):
        await task
    failure = [e for e in bus.history("run_c") if e.type is EventType.RUN_FAILED]
    assert failure and failure[0].payload["reason"] == "cancelled"


# --- context hygiene --------------------------------------------------------


async def test_a_huge_tool_result_is_truncated_before_it_reaches_the_prompt(
    mcp: FakeMCP,
) -> None:
    mcp.outcomes["blender.get_scene"] = ("x" * 50_000, False)
    agent = agent_with(
        mcp,
        ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
        ScriptedTurn(text="ok"),
        max_result_chars=2000,
    )
    await agent.run("go")
    sent = agent.provider.requests[1].messages[-1].content
    assert len(sent) < 5000
    assert "characters omitted" in sent


async def test_history_is_carried_into_the_next_turn(mcp: FakeMCP) -> None:
    agent = agent_with(mcp, ScriptedTurn(text="first"), ScriptedTurn(text="second"))
    history = [Message.user("earlier question"), Message.assistant("earlier answer")]
    await agent.run("new question", history=history)
    sent = agent.provider.requests[0].messages
    assert "earlier question" in [m.content for m in sent]
    assert "new question" in [m.content for m in sent]


async def test_assistant_tool_calls_are_replayed_so_the_model_keeps_its_thread(
    mcp: FakeMCP,
) -> None:
    agent = agent_with(mcp, ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="done"))
    await agent.run("go")
    second = agent.provider.requests[1].messages
    assistant = next(m for m in second if m.role.value == "assistant" and m.tool_calls)
    assert assistant.tool_calls[0].name == "blender.get_scene"
    assert second[-1].tool_call_id == assistant.tool_calls[0].id
