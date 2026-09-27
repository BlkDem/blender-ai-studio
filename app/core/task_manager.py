"""Background work, tracked so a person can see it.

A 3D generation is a network job with a provider's own state machine, and a
benchmark run is a long sequence of those. Both are tasks: they have an id, a
status, a start time, a duration and a cost, and both can be cancelled. The GUI's
Tasks panel is a view of this, and the database is fed from it, so a run's cost
and duration in the table came from the same numbers the user watched appear.

Deliberately not a thread pool. Everything here is asyncio, because every job is
I/O: a poll, a download, an MCP call.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.core.errors import Cancelled
from app.core.events import EventBus, EventType

logger = logging.getLogger(__name__)


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """One unit of background work."""

    id: str
    name: str
    provider: str = ""
    state: TaskState = TaskState.QUEUED
    progress: float = 0.0
    detail: str = ""
    result: Any = None
    error: str = ""
    credits: float = 0.0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    run_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def active(self) -> bool:
        return self.state in (TaskState.QUEUED, TaskState.RUNNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "state": str(self.state),
            "progress": self.progress,
            "detail": self.detail,
            "error": self.error,
            "credits": self.credits,
            "cost_usd": self.cost_usd,
            "duration_s": round(self.duration_s, 1),
            "run_id": self.run_id,
            "started_at": self.started_at,
        }


class TaskManager:
    """Starts tasks, tracks them, and publishes their progress."""

    def __init__(self, bus: EventBus | None = None, *, keep: int = 200) -> None:
        self._bus = bus or EventBus()
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._handles: dict[str, asyncio.Task[Any]] = {}
        self._keep = keep

    def __len__(self) -> int:
        return len(self._tasks)

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def tasks(self, *, active_only: bool = False) -> list[Task]:
        found = [self._tasks[key] for key in self._order if key in self._tasks]
        if active_only:
            found = [task for task in found if task.active]
        return list(reversed(found))

    async def start(
        self,
        name: str,
        work: Callable[[Task], Awaitable[Any]],
        *,
        provider: str = "",
        run_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Run ``work`` in the background and return its :class:`Task` at once.

        Returning the task rather than awaiting it is the point: the caller gets
        something to show immediately, and the work continues on its own.
        """
        task = Task(
            id=f"task_{uuid.uuid4().hex[:10]}",
            name=name,
            provider=provider,
            run_id=run_id,
            payload=payload or {},
            state=TaskState.QUEUED,
        )
        self._remember(task)
        self._publish(EventType.TASK_STARTED, **task.to_dict())
        handle = asyncio.create_task(self._execute(task, work), name=task.id)
        self._handles[task.id] = handle
        return task

    async def _execute(self, task: Task, work: Callable[[Task], Awaitable[Any]]) -> None:
        task.state = TaskState.RUNNING
        self._publish(EventType.TASK_STARTED, **task.to_dict())
        try:
            task.result = await work(task)
        except asyncio.CancelledError:
            task.state = TaskState.CANCELLED
            task.error = "cancelled"
            self._publish(EventType.TASK_FAILED, **task.to_dict())
            raise
        except Cancelled as exc:
            task.state = TaskState.CANCELLED
            task.error = exc.message
            self._publish(EventType.TASK_FAILED, **task.to_dict())
        except Exception as exc:  # noqa: BLE001 - a task's failure is data
            task.state = TaskState.FAILED
            # The hint is the actionable half -- which tool to enable, which
            # account to top up -- and a background task has nowhere else to
            # show it. The Tasks panel only ever renders this one string.
            hint = getattr(exc, "hint", "")
            task.error = f"{exc} -- {hint}" if hint else str(exc)
            logger.exception("task %s failed", task.name)
            self._publish(EventType.TASK_FAILED, **task.to_dict())
        else:
            task.state = TaskState.SUCCEEDED
            task.progress = 1.0
            task.finished_at = time.time()
            self._publish(EventType.TASK_FINISHED, **task.to_dict())
        finally:
            self._handles.pop(task.id, None)
            if task.state in (TaskState.CANCELLED, TaskState.FAILED) and task.finished_at is None:
                task.finished_at = time.time()

    def update(self, task_id: str, **fields: Any) -> Task | None:
        """Progress from inside a running task."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, value)
        self._publish(EventType.TASK_PROGRESS, **task.to_dict())
        return task

    async def wait(self, task_id: str, timeout: float | None = None) -> Task:
        """Block until a task settles. For the CLI and for tests."""
        handle = self._handles.get(task_id)
        task = self._tasks.get(task_id)
        if handle is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(handle), timeout=timeout)
        if task is None:
            raise KeyError(task_id)
        return task

    async def cancel(self, task_id: str) -> bool:
        handle = self._handles.get(task_id)
        if handle is None or handle.done():
            return False
        handle.cancel()
        return True

    async def cancel_all(self) -> None:
        for handle in list(self._handles.values()):
            handle.cancel()
        for handle in list(self._handles.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handle

    def _remember(self, task: Task) -> None:
        self._tasks[task.id] = task
        self._order.append(task.id)
        while len(self._order) > self._keep:
            dropped = self._order.pop(0)
            self._tasks.pop(dropped, None)

    def _publish(self, event_type: EventType, **payload: Any) -> None:
        self._bus.emit(event_type, **payload)
