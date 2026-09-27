"""Benchmark behaviour, with no Blender and no model.

What matters here is that runs are isolated, that the table reports what happened
without declaring a winner, and that an unverified scene reset is visible rather
than implied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.benchmark.models import load_suite, save_suite
from app.benchmark.runner import (
    RESET_COPY_ONLY,
    RESET_UNVERIFIED,
    RESET_VERIFIED,
    BenchmarkRunner,
    BenchmarkTask,
    ModelSpec,
)
from app.benchmark.storage import BenchmarkStorage
from app.core.cost import Budget
from app.core.events import EventBus
from app.llm.base import ChatRequest, ChatResponse, ToolCall, Usage
from app.llm.providers.mock import MockLLMProvider
from app.llm.registry import LLMRegistry, ProviderConfig
from app.mcp.manager import MCPManager
from app.mcp.models import ToolDescriptor, ToolOutcome


class CountingMCP(MCPManager):
    """Records what a run asked for, and can say the scene is busy."""

    def __init__(self, tools: dict[str, ToolDescriptor] | None = None) -> None:
        super().__init__()
        self._catalog = tools or {
            "blender.create_object": ToolDescriptor(name="blender.create_object", description="Create"),
            "blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List"),
        }
        self.calls: list[str] = []
        self.executed: list[str] = []

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [ToolSpec(name=t.name, description=t.description) for t in self._catalog.values()]

    def tools(self):  # type: ignore[override]
        return dict(self._catalog)

    def tool_instructions(self) -> str:
        return ""

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        self.calls.append(name)
        self.executed.append(str((arguments or {}).get("name", "")))
        if name == "blender.get_objects":
            return ToolOutcome(
                call_id="c", tool=name, text=json.dumps({"objects": [{"name": "Cube"}], "total": 1})
            )
        return ToolOutcome(call_id="c", tool=name, text=json.dumps({"ok": True}))


class AlwaysCreates(MockLLMProvider):
    """Calls create_object once, then answers."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._done = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if any(m.role.value == "tool" for m in request.messages):
            return ChatResponse(text="Built it.", usage=Usage(10, 5), model=self.default_model())
        return ChatResponse(
            tool_calls=[ToolCall.new("blender.create_object", {"type": "cube", "name": "Cube"})],
            usage=Usage(20, 10),
            finish_reason="tool_calls",
            model=self.default_model(),
        )


def registry_with(models: dict[str, str]) -> LLMRegistry:
    configs = []
    for name in models:
        configs.append(
            ProviderConfig(
                name=name, kind="mock", default_model=f"{name}-model", models=[{"id": f"{name}-model"}]
            )
        )
    return LLMRegistry(configs)


@pytest.fixture
def mcp() -> CountingMCP:
    return CountingMCP()


def runner(mcp: CountingMCP, models: dict[str, str], **kwargs) -> BenchmarkRunner:
    return BenchmarkRunner(registry_with(models), mcp, bus=EventBus(), **kwargs)


# --- isolation --------------------------------------------------------------


async def test_each_run_gets_its_own_copy_of_the_starting_file(mcp: CountingMCP, tmp_path: Path) -> None:
    blend = tmp_path / "start.blend"
    blend.write_bytes(b"blend")
    bench = runner(mcp, {"a": ""}, workdir=tmp_path / "work")

    first = await bench.run_task_for_model(BenchmarkTask("make a cube"), ModelSpec("a"), blend=blend)
    second = await bench.run_task_for_model(BenchmarkTask("make a cube"), ModelSpec("a"), blend=blend)

    assert first.blend_file != second.blend_file
    assert Path(first.blend_file).exists(), "the staged file is kept, so a result can be inspected"
    assert Path(second.blend_file).read_bytes() == b"blend"


async def test_a_reset_is_verified_only_when_the_file_was_actually_opened(
    mcp: CountingMCP, tmp_path: Path
) -> None:
    blend = tmp_path / "start.blend"
    blend.write_bytes(b"blend")

    # No execute_python in the catalogue: the operator has not allowed it.
    without = await runner(mcp, {"a": ""}, workdir=tmp_path / "w1").run_task_for_model(
        BenchmarkTask("x"), ModelSpec("a"), blend=blend
    )
    assert without.scene_reset == RESET_COPY_ONLY
    assert mcp.calls == ["blender.get_objects"], "no reset call was even attempted"

    # With it, and answering cleanly, the reset is verified.
    mcp_with = CountingMCP(
        {
            "blender.execute_python": ToolDescriptor(name="blender.execute_python", description="Run Python"),
            "blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List"),
        }
    )
    verified = await runner(mcp_with, {"a": ""}, workdir=tmp_path / "w2").run_task_for_model(
        BenchmarkTask("x"), ModelSpec("a"), blend=blend
    )
    assert verified.scene_reset == RESET_VERIFIED
    assert mcp_with.calls[0] == "blender.execute_python"


async def test_a_failed_reset_is_not_claimed_as_verified(mcp: CountingMCP, tmp_path: Path) -> None:
    blend = tmp_path / "start.blend"
    blend.write_bytes(b"blend")

    class Refuses(CountingMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
            if name == "blender.execute_python":
                return ToolOutcome(call_id="c", tool=name, text="blocked", is_error=True)
            return await super().call_tool(name, arguments, timeout=timeout)

    refusing = Refuses(
        {
            "blender.execute_python": ToolDescriptor(name="blender.execute_python", description="Run"),
            "blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List"),
        }
    )
    outcome = await runner(refusing, {"a": ""}, workdir=tmp_path / "w").run_task_for_model(
        BenchmarkTask("x"), ModelSpec("a"), blend=blend
    )
    assert outcome.scene_reset == RESET_COPY_ONLY


async def test_a_missing_starting_file_is_reported_as_unverified(mcp: CountingMCP) -> None:
    outcome = await runner(mcp, {"a": ""}).run_task_for_model(
        BenchmarkTask("x"), ModelSpec("a"), blend=Path("/nowhere/missing.blend")
    )
    assert outcome.scene_reset == RESET_UNVERIFIED


# --- running ----------------------------------------------------------------


async def test_a_suite_runs_every_model_against_every_task(mcp: CountingMCP) -> None:
    bench = runner(mcp, {"a": "", "b": ""})
    tasks = [BenchmarkTask("make a cube"), BenchmarkTask("make a table")]
    models = [ModelSpec("a"), ModelSpec("b")]

    comparison = await bench.run_suite(tasks, models)

    assert len(comparison.outcomes) == 4
    assert {o.model.provider for o in comparison.outcomes} == {"a", "b"}
    assert {o.task.prompt for o in comparison.outcomes} == {"make a cube", "make a table"}


async def test_runs_are_sequential_so_two_models_never_share_a_scene(mcp: CountingMCP) -> None:
    """Interleaved tool calls would produce a scene belonging to neither model."""
    bench = runner(mcp, {"a": ""})
    await bench.run_suite([BenchmarkTask("one"), BenchmarkTask("two")], [ModelSpec("a")])
    # Each run reads the scene when it finishes, so two reads per run, in order.
    assert mcp.calls.count("blender.get_objects") == 2


async def test_a_run_records_its_metrics_and_the_final_scene(mcp: CountingMCP) -> None:
    bench = runner(mcp, {"a": ""})
    bench.llm.set_provider("a", AlwaysCreates(model="a-model"))

    outcome = await bench.run_task_for_model(BenchmarkTask("make a cube"), ModelSpec("a"))

    assert outcome.status == "ok"
    assert outcome.mcp_calls == 1
    assert outcome.totals.input_tokens == 30, "20 for the call, 10 for the answer"
    assert outcome.final_scene["total"] == 1
    assert outcome.transcript[-1]["content"] == "Built it."


async def test_a_model_that_is_not_configured_fails_with_an_explanation(mcp: CountingMCP) -> None:
    outcome = await runner(mcp, {"a": ""}).run_task_for_model(BenchmarkTask("x"), ModelSpec("nope"))
    assert outcome.status == "failed"
    assert "nope" in outcome.error


async def test_a_budget_stop_is_recorded_rather_than_hidden(mcp: CountingMCP) -> None:
    bench = runner(mcp, {"a": ""}, budget=Budget(max_steps=0))
    outcome = await bench.run_task_for_model(BenchmarkTask("x"), ModelSpec("a"))
    assert outcome.status == "stopped"
    assert "steps" in outcome.error


async def test_cancelling_stops_the_suite(mcp: CountingMCP) -> None:
    bench = runner(mcp, {"a": "", "b": ""})
    bench.cancel()
    comparison = await bench.run_suite([BenchmarkTask("x")], [ModelSpec("a"), ModelSpec("b")])
    assert comparison.outcomes == []


async def test_each_run_gets_its_own_budget_counters(mcp: CountingMCP) -> None:
    """A shared budget carries its counters, so the second model would inherit
    the first model's step count and stop early."""
    bench = runner(mcp, {"a": ""}, budget=Budget(max_steps=5))
    first = bench._fresh_budget()  # noqa: SLF001
    first.steps = 4
    second = bench._fresh_budget()  # noqa: SLF001
    assert second.steps == 0


# --- the table --------------------------------------------------------------


async def test_the_table_shows_facts_and_no_verdict(mcp: CountingMCP) -> None:
    bench = runner(mcp, {"a": "", "b": ""})
    bench.llm.set_provider("a", AlwaysCreates(model="a-model"))
    bench.llm.set_provider("b", AlwaysCreates(model="b-model"))
    comparison = await bench.run_suite([BenchmarkTask("make a cube")], [ModelSpec("a"), ModelSpec("b")])

    text = comparison.to_text()
    assert "Model" in text and "Tool calls" in text
    assert "a-model" in text and "b-model" in text
    for word in ("best", "winner", "recommended", "score"):
        assert word not in text.lower(), "the benchmark reports; the person judges"


async def test_the_table_warns_about_runs_that_were_not_isolated(mcp: CountingMCP, tmp_path: Path) -> None:
    blend = tmp_path / "start.blend"
    blend.write_bytes(b"blend")
    bench = runner(mcp, {"a": ""}, workdir=tmp_path / "w")
    comparison = await bench.run_suite([BenchmarkTask("x")], [ModelSpec("a")], blend=blend)
    assert "not comparable" in comparison.to_text()


# --- storage ----------------------------------------------------------------


async def test_a_suite_and_its_rounds_are_persisted(studio) -> None:
    storage = BenchmarkStorage(studio.db)
    suite_id = await storage.create_suite("Tables", "the classic")

    bench = runner(CountingMCP(), {"a": ""})
    bench.llm.set_provider("a", AlwaysCreates(model="a-model"))
    comparison = await bench.run_suite([BenchmarkTask("make a cube")], [ModelSpec("a")])
    await storage.save_comparison(suite_id, comparison)

    runs = await storage.runs(suite_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["duration_s"] > 0
    assert runs[0]["input_tokens"] == 30
    assert runs[0]["mcp_calls"] == 1
    assert json.loads(runs[0]["final_scene"])["total"] == 1

    listed = await storage.suites()
    assert listed[0]["runs"] == 1


async def test_a_person_scores_a_run_and_the_score_is_joined_into_the_table(studio) -> None:
    storage = BenchmarkStorage(studio.db)
    suite_id = await storage.create_suite("Tables")
    bench = runner(CountingMCP(), {"a": ""})
    bench.llm.set_provider("a", AlwaysCreates(model="a-model"))
    comparison = await bench.run_suite([BenchmarkTask("x")], [ModelSpec("a")])
    run_id = (await storage.save_comparison(suite_id, comparison))[0]

    await storage.review(
        run_id, geometry=4, materials=2, instruction_following=5, overall=4, notes="legs are square"
    )

    table = await storage.table(suite_id)
    assert table[0]["geometry"] == 4
    assert table[0]["notes"] == "legs are square"
    assert len(await storage.reviews(run_id)) == 1


async def test_transcripts_survive_for_later_reading(studio) -> None:
    storage = BenchmarkStorage(studio.db)
    suite_id = await storage.create_suite("Tables")
    bench = runner(CountingMCP(), {"a": ""})
    bench.llm.set_provider("a", AlwaysCreates(model="a-model"))
    comparison = await bench.run_suite([BenchmarkTask("make a cube")], [ModelSpec("a")])
    await storage.save_comparison(suite_id, comparison)

    transcripts = await storage.transcripts(suite_id)
    assert len(transcripts) == 1
    stored = next(iter(transcripts.values()))
    assert stored[0]["content"] == "make a cube"
    assert stored[-1]["content"] == "Built it."


def test_a_suite_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    save_suite(
        path,
        [BenchmarkTask("make a table", name="table")],
        [ModelSpec("space-bunny", "space-bunny-free")],
        "Tables",
    )
    name, tasks, models = load_suite(path)
    assert name == "Tables"
    assert tasks[0].name == "table"
    assert models[0].model == "space-bunny-free"
