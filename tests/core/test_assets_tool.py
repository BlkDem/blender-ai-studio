"""The 3D tool, end to end, with a provider that behaves.

What is being tested is the shape of the whole slow half: the model gets an
answer immediately, the minutes happen somewhere it can see, and a finished
model arrives in the scene without anybody having to remember to import it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.core.context import three_d_tool
from app.core.events import Event, EventBus, EventType
from app.core.settings import ThreeDConfig
from app.core.task_manager import TaskManager
from app.mcp.models import ToolDescriptor, ToolOutcome
from app.providers3d.base import failed_task
from app.providers3d.mock import MockThreeDProvider
from app.providers3d.models import AssetRequest
from app.providers3d.registry import ThreeDRegistry


class ImportMCP:
    """Just enough bridge for the importer: the two tools it asks for."""

    def __init__(self, *, importable: bool = True, arrives: bool = True) -> None:
        self.importable = importable
        self.arrives = arrives
        self.imports: list[str] = []
        self.lookups = 0

    def tools(self):  # type: ignore[override]
        found = {"blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List")}
        if self.importable:
            found["blender.execute_python"] = ToolDescriptor(
                name="blender.execute_python", description="Run Python"
            )
        return found

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        import json

        if name == "blender.execute_python":
            code = (arguments or {}).get("code", "")
            self.imports.append(code)
            return ToolOutcome(call_id="1", tool=name, text=json.dumps({"result": {"FINISHED": ""}}))
        self.lookups += 1
        names = ["Cube"] if self.lookups == 1 else ["Cube", "Dragon_Body"]
        return ToolOutcome(
            call_id="1",
            tool=name,
            text=json.dumps({"objects": [{"name": n} for n in names], "total": len(names)}),
        )


class Watcher:
    """Collects what the Tasks panel would have seen."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.events: list[Event] = []
        self.bus.subscribe(self.events.append)
        self.tasks = TaskManager(self.bus)

    def types(self) -> list[EventType]:
        return [event.type for event in self.events]


def registry_with(provider: MockThreeDProvider) -> ThreeDRegistry:
    registry = ThreeDRegistry()
    registry.add(provider)
    return registry


async def wait_for(task, watcher: Watcher, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while task.active and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    assert not task.active, f"the task never finished: {task.state} {task.error}"
    return task


async def test_a_finished_model_arrives_in_the_scene_without_being_asked_for(tmp_path: Path) -> None:
    provider = MockThreeDProvider(polls_before_done=2)
    watcher = Watcher()
    bridge = ImportMCP()
    tool = three_d_tool(
        registry_with(provider),
        ThreeDConfig(provider="mock-3d", poll_interval=0.01),
        tasks=watcher.tasks,
        on_ready=lambda task, path: _import(bridge, path),
        download_dir=tmp_path,
    )

    answer = await tool.handler({"prompt": "a dragon"})

    # The model is told what it needs to know straight away: it is running, and
    # the minutes are somebody else's problem.
    assert answer["submitted"] is True
    assert answer["studio_task_id"].startswith("task_")
    assert answer["status"] == "running"
    assert "imported into the scene automatically" in answer["note"]

    finished = await wait_for(watcher.tasks.get(answer["studio_task_id"]), watcher)
    assert finished.state == "succeeded"
    assert finished.result["imported"] is True
    assert finished.result["objects"] == ["Dragon_Body"]
    assert finished.credits == 100, "the credits reach the same record the user watched"
    assert EventType.TASK_STARTED in watcher.types()
    assert EventType.TASK_FINISHED in watcher.types()
    assert any("import_scene.gltf" in code for code in bridge.imports), "and the import really ran"


async def test_a_refused_submission_never_becomes_a_task() -> None:
    """An account with no credit fails at submission. The model must hear that
    now, not discover an empty task in a panel five minutes later."""
    provider = MockThreeDProvider(fail_with="You don't have enough credit to create this task")
    watcher = Watcher()
    tool = three_d_tool(
        registry_with(provider),
        ThreeDConfig(provider="mock-3d"),
        tasks=watcher.tasks,
        download_dir=Path("/tmp"),
    )

    answer = await tool.handler({"prompt": "a dragon"})

    assert answer["submitted"] is False
    assert "enough credit" in answer["error"]
    assert watcher.tasks.tasks() == [], "no task for something that never started"
    assert EventType.TASK_STARTED not in watcher.types()


async def test_a_generation_that_fails_midway_fails_the_task_with_the_reason(tmp_path: Path) -> None:
    class FailsLate(MockThreeDProvider):
        async def status(self, task):  # type: ignore[override]
            self.polls += 1
            if self.polls < 2:
                return task
            return failed_task(task, "the provider ran out of memory")

    watcher = Watcher()
    tool = three_d_tool(
        registry_with(FailsLate()),
        ThreeDConfig(provider="mock-3d", poll_interval=0.01),
        tasks=watcher.tasks,
        download_dir=tmp_path,
    )
    answer = await tool.handler({"prompt": "a dragon"})
    finished = await wait_for(watcher.tasks.get(answer["studio_task_id"]), watcher)
    assert finished.state == "failed"
    assert "ran out of memory" in finished.error
    assert EventType.TASK_FAILED in watcher.types()


async def test_a_gated_blender_keeps_the_download_and_says_why(tmp_path: Path) -> None:
    """The model is generated either way. Losing it to a disabled tool would be
    the worst outcome, so the file stays and the reason names the tool."""
    bridge = ImportMCP(importable=False)
    watcher = Watcher()
    tool = three_d_tool(
        registry_with(MockThreeDProvider()),
        ThreeDConfig(provider="mock-3d", poll_interval=0.01),
        tasks=watcher.tasks,
        on_ready=lambda task, path: _import(bridge, path),
        download_dir=tmp_path,
    )
    answer = await tool.handler({"prompt": "a dragon"})
    finished = await wait_for(watcher.tasks.get(answer["studio_task_id"]), watcher)

    assert finished.state == "failed"
    assert "cannot import an asset yet" in finished.error
    assert "blender.execute_python" in finished.error, "and names the tool that would help"
    assert list(tmp_path.glob("*.glb")), "the download is kept: the model was still generated"


async def test_progress_is_read_when_the_provider_reports_it() -> None:
    from app.core.context import _progress
    from app.core.task_manager import Task
    from app.providers3d.models import ProviderTask

    task = Task(id="t", name="n")
    _progress(task, ProviderTask.new("mock", AssetRequest(prompt="x"), provider_task_id="p"))
    assert task.progress == 0.0, "a provider that has not reported yet has not progressed"

    part = ProviderTask.new("mock", AssetRequest(prompt="x"), provider_task_id="p")
    part.progress = 0.42
    _progress(task, part)
    assert task.progress == 0.42

    percent = ProviderTask.new("mock", AssetRequest(prompt="x"), provider_task_id="p")
    percent.progress = 50.0
    _progress(task, percent)
    assert task.progress == 0.5, "a percentage is not shown as a full bar"


async def test_without_a_task_manager_the_model_is_told_nothing_is_watching(tmp_path: Path) -> None:
    """The CLI runs without one. Saying so beats implying a background job."""
    tool = three_d_tool(registry_with(MockThreeDProvider()), ThreeDConfig(provider="mock-3d"))
    answer = await tool.handler({"prompt": "a dragon"})
    assert "submitted" in answer
    assert "nothing is watching it" in answer["note"]


async def _import(bridge: ImportMCP, path: Path):
    from app.providers3d.importer import import_asset

    return await import_asset(bridge, path)


def test_an_unkeyed_provider_is_not_offered_to_the_model() -> None:
    """A tool that always fails wastes a step and teaches the model it is noise."""
    from app.providers3d.tripo import TripoProvider

    registry = ThreeDRegistry()
    registry.add(TripoProvider(api_key="", base_url="https://api.tripo3d.ai"))
    assert registry.any_enabled() is False

    registry_with(MockThreeDProvider()).provider("mock-3d").enabled = False
    assert ThreeDRegistry().any_enabled() is False, "and a disabled one stays out too"


@pytest.mark.parametrize("prompt", ["", "   "])
async def test_an_empty_prompt_is_refused_before_any_request(prompt: str) -> None:
    provider = MockThreeDProvider()
    tool = three_d_tool(registry_with(provider), ThreeDConfig(provider="mock-3d"))
    answer = await tool.handler({"prompt": prompt})
    assert "prompt is required" in answer["error"]
    assert provider.requests == [], "nothing was submitted"
