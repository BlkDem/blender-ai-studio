"""Tasks that outlive the process that started them.

A 3D generation costs money and takes minutes. If the only record of it is a
table in memory, then closing the window loses the bill as well as the model --
and the docstrings in three different modules used to promise otherwise.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.core.events import EventBus
from app.core.task_manager import TaskManager, TaskState
from app.storage.repositories import Studio


class Store:
    """Just the 3D repository, so the test can read back what was written."""

    def __init__(self, studio: Studio) -> None:
        self.three_d = studio.three_d


@pytest.fixture
async def studio(database_path: Path):
    instance = await Studio.open(database_path)
    try:
        yield instance
    finally:
        instance.close()


async def test_a_finished_3d_task_is_in_the_database(studio: Studio) -> None:
    tasks = TaskManager(EventBus(), studio=studio)

    async def work(task):
        task.progress = 0.5
        task.credits = 100
        return {"path": "/tmp/a.glb", "url": "https://example/a.glb", "imported": True}

    task = await tasks.start(
        "3D: a chest",
        work,
        provider="tripo",
        run_id="run_1",
        payload={"prompt": "a chest", "kind": "text_to_3d", "provider_task_id": "tp_1"},
    )
    await tasks.wait(task.id, timeout=5)

    records = await studio.three_d.list()
    assert len(records) == 1, "the task that cost money left a row"
    row = records[0]
    assert row.provider == "tripo"
    assert row.provider_task_id == "tp_1"
    assert row.status == str(TaskState.SUCCEEDED)
    assert row.credits == 100
    assert row.local_path == "/tmp/a.glb"
    assert row.prompt == "a chest"
    assert row.finished_at is not None


async def test_a_3d_task_is_filed_under_the_project_that_asked_for_it(studio: Studio) -> None:
    """The panel, the README and AppContext's own comment all said 3D
    generations were filed under a project, and the table had no column to file
    them in. A generation that costs money should be findable next to the work
    that caused it."""
    project = await studio.projects.create("Kitchen")
    open_project = project.id
    tasks = TaskManager(EventBus(), studio=studio, project_id=lambda: open_project)

    async def work(task):
        return {"path": "/tmp/a.glb", "url": "https://example/a.glb"}

    task = await tasks.start(
        "3D: a chest", work, provider="tripo", run_id="run_1",
        payload={"prompt": "a chest", "kind": "text_to_3d"},
    )
    await tasks.wait(task.id, timeout=5)

    filed = await studio.three_d.list(project_id=project.id)
    assert [r.id for r in filed] == [f"tdt_{task.id}"], "the generation is under the project"
    assert await studio.three_d.list() == filed, "and still in the unfiltered list"

    # Closing the project mid-session must not retroactively re-file it.
    open_project = None
    later = await tasks.start(
        "3D: a lamp", work, provider="tripo", run_id="run_2",
        payload={"prompt": "a lamp", "kind": "text_to_3d"},
    )
    await tasks.wait(later.id, timeout=5)
    assert [r.id for r in await studio.three_d.list(project_id=project.id)] == [f"tdt_{task.id}"]
    assert next(r for r in await studio.three_d.list() if r.id == f"tdt_{later.id}").project_id is None


async def test_a_failed_3d_task_records_why(studio: Studio) -> None:
    tasks = TaskManager(EventBus(), studio=studio)

    async def work(task):
        raise RuntimeError("the account has no credit")

    task = await tasks.start(
        "3D: a chest",
        work,
        provider="tripo",
        payload={"prompt": "a chest", "kind": "text_to_3d"},
    )
    await tasks.wait(task.id, timeout=5)

    row = (await studio.three_d.list())[0]
    assert row.status == "failed"
    assert "no credit" in (row.error or "")


async def test_a_task_is_updated_rather_than_duplicated(studio: Studio) -> None:
    """A 3D task is polled for minutes; one row has to follow it, not grow."""
    tasks = TaskManager(EventBus(), studio=studio)

    async def work(task):
        await asyncio.sleep(0.05)
        task.progress = 0.5
        await asyncio.sleep(0.05)
        task.credits = 20
        return {"path": "/tmp/a.glb"}

    task = await tasks.start(
        "3D: a chest", work, provider="tripo", payload={"prompt": "a", "kind": "text_to_3d"}
    )
    await tasks.wait(task.id, timeout=5)

    records = await studio.three_d.list()
    assert len(records) == 1
    assert records[0].credits == 20, "the final numbers, not the first ones"
    assert records[0].progress == 1.0


async def test_a_task_that_is_not_3d_is_not_written_to_the_3d_table(studio: Studio) -> None:
    """A benchmark run is a task, and it is not a 3D task.

    Writing it into the 3D table would make the table claim to hold things it
    does not hold.
    """
    tasks = TaskManager(EventBus(), studio=studio)

    async def work(task):
        return "done"

    task = await tasks.start("benchmark: a-model", work, provider="", payload={"prompt": "x"})
    await tasks.wait(task.id, timeout=5)
    assert await studio.three_d.list() == []


async def test_a_database_that_cannot_be_written_does_not_fail_the_task(studio: Studio) -> None:
    """The work happened and cost money. A record that could not be written is
    a problem to log, not a reason to report the generation as failed."""

    class Broken(Store):
        async def _no_three_d(self):  # pragma: no cover - never called
            return None

    broken = Store(studio)
    broken.three_d = type(
        "Broken",
        (),
        {"create": lambda self, record: _raise(record), "update": lambda self, *a, **k: _raise(a)},
    )()
    tasks = TaskManager(EventBus(), studio=broken)

    async def work(task):
        return "the model arrived"

    task = await tasks.start("3D: a", work, provider="tripo", payload={"kind": "text_to_3d"})
    finished = await tasks.wait(task.id, timeout=5)
    assert finished.state == TaskState.SUCCEEDED


async def _raise(*args, **kwargs):
    raise RuntimeError("the database is gone")
