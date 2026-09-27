"""One event bus, shared by everything.

The agent, the MCP manager and the task manager all run concurrently and all
report progress. The GUI must never poll for that, and a run's log must be
reproducible after the fact, so progress is events: every run carries a
``run_id`` and every event carries the ids needed to place it in a sequence.

Two consumers, one mechanism: the GUI subscribes through the same bus the
database logger uses, which is why the transcript a person sees and the transcript
stored for a benchmark cannot drift apart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """The kinds of progress a run can report.

    Names are what the GUI switches on. Adding one is cheap; renaming one is a
    change to every consumer.
    """

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    TEXT = "text"
    THINKING = "thinking"
    TOKEN = "token"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    MESSAGE_COMPLETE = "message_complete"
    USAGE = "usage"
    STATUS = "status"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    COST = "cost"
    ERROR = "error"
    INFO = "info"


@dataclass(slots=True)
class Event:
    """One thing that happened, in order, within a run."""

    type: EventType
    run_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


Subscriber = Callable[[Event], Any]


class EventBus:
    """Fan-out with a bounded replay buffer.

    The buffer exists for two reasons. A GUI panel that attaches late (the
    Benchmark view opening after a run finished) still wants the run, and the
    database logger attaches after the agent has started. Both need history, and
    a queue of everything would be a leak in a long session.
    """

    def __init__(self, history: int = 2000) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._history_limit = history
        self._seq = 0
        self._lock = asyncio.Lock()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register a callback; call the returned function to unsubscribe."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(subscriber)

        return unsubscribe

    def publish_now(self, event: Event) -> list[Any]:
        """Record the event and hand it to subscribers, synchronously.

        History is updated here rather than in a task, because a consumer that
        asks "what happened in this run" straight after an emit must see it. Only
        the coroutine results are deferred; a plain callback runs inline, so the
        GUI's signal emission is not a scheduling race.
        """
        self._seq += 1
        event.seq = self._seq
        self._history.append(event)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        pending: list[Any] = []
        for subscriber in list(self._subscribers):
            try:
                result = subscriber(event)
            except Exception:
                # One bad subscriber must not stop the run it is watching.
                logger.exception("event subscriber failed for %s", event.type)
                continue
            if asyncio.iscoroutine(result):
                pending.append(result)
        return pending

    async def publish(self, event: Event) -> None:
        for pending in self.publish_now(event):
            try:
                await pending
            except Exception:
                logger.exception("async event subscriber failed for %s", event.type)

    def emit(self, event_type: EventType, *, run_id: str = "", **payload: Any) -> Event:
        """Publish without awaiting. For the synchronous paths that cannot await."""
        event = Event(type=event_type, run_id=run_id, payload=payload)
        pending = self.publish_now(event)
        if pending:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - only from a sync entry point
                for coroutine in pending:
                    coroutine.close()
                return event
            for coroutine in pending:
                loop.create_task(coroutine)
        return event

    def history(self, run_id: str | None = None, limit: int | None = None) -> list[Event]:
        events = self._history if run_id is None else [e for e in self._history if e.run_id == run_id]
        return events[-limit:] if limit else list(events)


def new_run_id(prefix: str = "run") -> str:
    """Short, sortable, and readable in a log line."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def collect(bus: EventBus, run_id: str, count: int, timeout: float) -> list[Event]:
    """Wait for ``count`` events of one run. For tests and the CLI."""
    seen: list[Event] = []
    pending = asyncio.get_running_loop().create_future()

    def subscriber(event: Event) -> None:
        if event.run_id != run_id:
            return
        seen.append(event)
        if len(seen) >= count and not pending.done():
            pending.set_result(None)

    unsubscribe = bus.subscribe(subscriber)
    try:
        await asyncio.wait_for(pending, timeout)
    except TimeoutError:
        pass
    finally:
        unsubscribe()
    return seen


async def stream(bus: EventBus, run_id: str) -> AsyncIterator[Event]:
    """Follow a run that is already in flight."""
    queue: asyncio.Queue[Event] = asyncio.Queue()
    unsubscribe = bus.subscribe(queue.put_nowait)
    try:
        for event in bus.history(run_id):
            yield event
        while True:
            yield await queue.get()
    finally:
        unsubscribe()
