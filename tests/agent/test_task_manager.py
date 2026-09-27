"""The task manager: background work a person can watch and stop."""

from __future__ import annotations

import asyncio

from app.core.errors import Cancelled
from app.core.events import EventBus, EventType
from app.core.task_manager import TaskManager, TaskState


async def test_a_task_runs_in_the_background_and_settles() -> None:
    manager = TaskManager()

    async def work(task):
        await asyncio.sleep(0.01)
        return {"answer": 42}

    task = await manager.start("generate", work, provider="tripo")
    assert task.active, "the task is returned before it finishes, so it can be shown"

    done = await manager.wait(task.id, timeout=5)
    assert done.state is TaskState.SUCCEEDED
    assert done.result == {"answer": 42}
    assert done.duration_s >= 0


async def test_progress_is_recorded_and_published() -> None:
    bus = EventBus()
    manager = TaskManager(bus)

    async def work(task):
        for percent in (25, 50, 75):
            manager.update(task.id, progress=percent / 100, detail=f"{percent}%")
            await asyncio.sleep(0.005)
        return "done"

    task = await manager.start("polling", work)
    await manager.wait(task.id, timeout=5)
    progress = [e for e in bus.history() if e.type is EventType.TASK_PROGRESS]
    assert [e.payload["progress"] for e in progress] == [0.25, 0.5, 0.75]
    assert progress[0].payload["detail"] == "25%"


async def test_a_failure_is_data_not_an_exception() -> None:
    manager = TaskManager()

    async def work(task):
        raise RuntimeError("the provider is down")

    task = await manager.start("doomed", work)
    settled = await manager.wait(task.id, timeout=5)
    assert settled.state is TaskState.FAILED
    assert "the provider is down" in settled.error


async def test_a_task_can_be_cancelled() -> None:
    manager = TaskManager()
    started = asyncio.Event()

    async def work(task):
        started.set()
        await asyncio.sleep(30)

    task = await manager.start("long", work)
    await asyncio.wait_for(started.wait(), timeout=2)
    assert await manager.cancel(task.id) is True

    settled = await manager.wait(task.id, timeout=5)
    assert settled.state is TaskState.CANCELLED


async def test_a_task_can_cancel_itself_through_the_studio_error() -> None:
    manager = TaskManager()

    async def work(task):
        raise Cancelled("the user pressed stop")

    task = await manager.start("stopped", work)
    settled = await manager.wait(task.id, timeout=5)
    assert settled.state is TaskState.CANCELLED
    assert settled.error == "the user pressed stop"


async def test_cancelling_a_finished_task_is_false() -> None:
    manager = TaskManager()

    async def work(task):
        return "quick"

    task = await manager.start("quick", work)
    await manager.wait(task.id, timeout=5)
    assert await manager.cancel(task.id) is False


async def test_tasks_are_listed_newest_first_and_filterable() -> None:
    manager = TaskManager()

    async def quick(task):
        return 1

    async def slow(task):
        await asyncio.sleep(30)

    first = await manager.start("first", quick)
    second = await manager.start("second", slow)
    await manager.wait(first.id, timeout=5)

    names = [task.name for task in manager.tasks()]
    assert names == ["second", "first"]
    assert [task.name for task in manager.tasks(active_only=True)] == ["second"]
    assert manager.get(second.id) is not None
    assert manager.get("nope") is None


async def test_the_history_is_bounded() -> None:
    manager = TaskManager(keep=5)

    async def work(task):
        return None

    for index in range(12):
        task = await manager.start(f"task{index}", work)
        await manager.wait(task.id, timeout=5)
    assert len(manager) == 5
    # tasks() is newest first, so the last entry is the oldest survivor.
    assert manager.tasks()[-1].name == "task7", "task0..task6 were dropped"
    assert manager.get("task0") is None


async def test_cancel_all_stops_everything_running() -> None:
    manager = TaskManager()
    started = asyncio.Event()

    async def work(task):
        started.set()
        await asyncio.sleep(30)

    tasks = [await manager.start(f"t{index}", work) for index in range(3)]
    await asyncio.wait_for(started.wait(), timeout=2)

    await manager.cancel_all()
    for task in tasks:
        assert (await manager.wait(task.id, timeout=5)).state is TaskState.CANCELLED


async def test_a_task_dict_carries_what_the_table_shows() -> None:
    manager = TaskManager()

    async def work(task):
        return None

    task = await manager.start("tripo generation", work, provider="tripo", run_id="run_1")
    await manager.wait(task.id, timeout=5)
    payload = manager.get(task.id).to_dict()
    assert payload["name"] == "tripo generation"
    assert payload["provider"] == "tripo"
    assert payload["run_id"] == "run_1"
    assert payload["state"] == "succeeded"
    assert isinstance(payload["duration_s"], float)
