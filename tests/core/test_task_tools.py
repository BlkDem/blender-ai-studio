import asyncio

import pytest

from app.core.context import AppContext
from app.core.settings import Settings
from app.core.task_manager import TaskManager, TaskState


@pytest.mark.asyncio
async def test_task_lifecycle():
    manager = TaskManager()
    context = AppContext(Settings(), tasks=manager)
    tools = {tool.name: tool.handler for tool in context.task_tools()}
    release = asyncio.Event()

    async def work(task):
        await release.wait()
        return {"path": "/asset.glb", "imported": True}

    task = await manager.start("asset", work)
    found = await tools["list_tasks"]({"active_only": True})
    assert found["tasks"][0]["id"] == task.id
    waiting = await tools["wait_task"]({"task_id": task.id, "timeout_s": 0})
    assert not waiting["completed"]
    assert task.active
    release.set()
    await manager.wait(task.id)
    found = await tools["get_task"]({"task_id": task.id})
    assert found["task"]["result"]["path"] == "/asset.glb"
    assert not (await tools["cancel_task"]({"task_id": task.id}))["cancel_requested"]


@pytest.mark.asyncio
async def test_waiter_cancellation_does_not_cancel_work():
    manager = TaskManager()
    gate = asyncio.Event()
    task = await manager.start("blocked", lambda task: gate.wait())
    waiter = asyncio.create_task(manager.wait(task.id))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert task.active
    gate.set()
    await manager.wait(task.id)
    assert task.state == TaskState.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_before_start_and_invalid_requests():
    manager = TaskManager()
    tools = {t.name: t.handler for t in AppContext(Settings(), tasks=manager).task_tools()}
    task = await manager.start("queued", lambda task: asyncio.sleep(100))
    assert (await tools["cancel_task"]({"task_id": task.id}))["cancel_requested"]
    await manager.wait(task.id)
    assert task.state == TaskState.CANCELLED
    for name in ("get_task", "wait_task", "cancel_task"):
        assert not (await tools[name]({"task_id": "missing"}))["ok"]
    for timeout in (True, -1, float("inf"), "10"):
        assert not (await tools["wait_task"]({"task_id": task.id, "timeout_s": timeout}))["ok"]
    assert not (await tools["list_tasks"]({"active_only": "yes"}))["ok"]
