"""Running the core on its own thread, and getting its events into Qt.

The studio's core is asyncio: an MCP session is a subprocess with pipes, an LLM
call is a stream, a 3D generation is a poll. Qt is a synchronous event loop with
its own thread. So the core runs on a dedicated thread with its own loop, and the
two halves talk through exactly two things: a queue in one direction and Qt
signals in the other.

``qasync`` would do this too. It would also make the core untestable without a
QApplication and put a Qt dependency in the middle of the agent, which is why
this is thirty lines of thread instead: the core keeps working headlessly, which
is what ``--check``, the acceptance run and every test in this project rely on.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CoreSignals(QObject):
    """What the GUI thread is allowed to hear about.

    These signals are emitted *in the GUI thread* — the drain timer in
    :class:`CoreThread` does it — so ordinary auto connections work and a slot
    can touch widgets. Emitting them straight from a worker thread does not: Qt
    cannot classify a plain Python thread, and the emission is dropped without a
    word. Hence the queue.
    """

    event = Signal(object)  # app.core.events.Event
    status = Signal(str)  # human-readable one-liner
    error = Signal(str, str)  # code, message
    finished = Signal()


class CoreThread:
    """The studio's event loop, on its own thread."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.signals = CoreSignals()
        self._ready = threading.Event()
        self._unsubscribe: Callable[[], None] | None = None
        self._stopping = False
        #: Core thread -> GUI thread. A plain queue, because both ends are real
        #: threads and neither Qt nor asyncio can hand over for us.
        self._inbox: queue.SimpleQueue[tuple[str, tuple[Any, ...]]] = queue.SimpleQueue()
        self._timer: QTimer | None = None

    # --- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 10.0) -> None:
        """Start the core thread and the GUI-side drain timer.

        Must be called from the GUI thread: the timer belongs to whichever thread
        creates it, and that is the one allowed to touch widgets.
        """
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="studio-core", daemon=True)
        self.thread.start()
        if not self._ready.wait(timeout):  # pragma: no cover - only a hung loop
            raise RuntimeError("the studio's core thread did not start")
        self._timer = QTimer()
        self._timer.setInterval(20)
        self._timer.timeout.connect(self.drain)
        self._timer.start()

    def drain(self) -> None:
        """Hand everything the core produced to the GUI, in order."""
        while True:
            try:
                kind, arguments = self._inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "event":
                self.signals.event.emit(arguments[0])
            elif kind == "error":
                self.signals.error.emit(*arguments)
            elif kind == "status":
                self.signals.status.emit(*arguments)
            elif kind == "callback":
                # on_done belongs to the GUI thread, like everything else that
                # ends up touching a widget.
                try:
                    arguments[0](*arguments[1:])
                except Exception:  # noqa: BLE001 - a UI callback must not kill the loop
                    logger.exception("a GUI callback failed")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self.context.close())
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("closing the context failed", exc_info=True)
            loop.close()

    def stop(self, timeout: float = 10.0) -> None:
        if self.loop is None or self.thread is None:
            return
        self._stopping = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=timeout)

    # --- calling into the core --------------------------------------------

    def submit(
        self,
        coroutine: Coroutine[Any, Any, T],
        on_done: Callable[..., None] | None = None,
        *extra: Any,
    ) -> None:
        """Run a coroutine on the core loop. Returns immediately.

        Every GUI action that touches the network goes through here. A blocking
        call in the Qt thread is how a window stops repainting, and the MCP
        handshake alone takes a second.
        """
        if self.loop is None:  # pragma: no cover - start() always runs first
            raise RuntimeError("the core thread is not running")
        loop = self.loop

        def run() -> None:
            try:
                result = asyncio.run_coroutine_threadsafe(coroutine, loop)
            except RuntimeError as exc:  # loop already closed
                self._inbox.put(("error", ("CORE_CLOSED", str(exc))))
                return
            if on_done is None:
                result.add_done_callback(self._report_failure)
            else:
                result.add_done_callback(lambda future: self._deliver(future, on_done, *extra))

        self.loop.call_soon_threadsafe(run)

    def _deliver(self, future: Any, on_done: Callable[..., None], *args: Any) -> None:
        """Hand a finished coroutine's value to the GUI thread."""
        try:
            value = future.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - reported, not raised, into Qt
            logger.exception("a core task failed")
            self._inbox.put(("error", (getattr(exc, "code", "ERROR"), str(exc))))
            return
        self._inbox.put(("callback", (on_done, value, *args)))

    def _report_failure(self, future: Any) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            self.signals.error.emit(getattr(error, "code", "ERROR"), str(error))

    # --- events into the GUI ---------------------------------------------

    def _unsubscribe_bus(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def attach_bus(self) -> None:
        """Forward every core event to the GUI thread.

        The bus is the studio's only channel for progress, so this one hookup is
        what makes tool calls, costs and task updates appear in the window. The
        subscriber does nothing but enqueue: touching a widget from the core
        thread would be a crash that happens at random and reads as a Qt bug.
        """
        bus = getattr(self.context, "bus", None)
        if bus is None:
            return
        self._unsubscribe_bus()
        self._unsubscribe = bus.subscribe(lambda event: self._inbox.put(("event", (event,))))
        logger.debug("event bus attached to the GUI")

    def post_error(self, code: str, message: str) -> None:
        self._inbox.put(("error", (code, message)))

    def post_status(self, text: str) -> None:
        self._inbox.put(("status", (text,)))


def qt_object_name(prefix: str, value: str) -> str:
    """A stable object name, so a screenshot is comparable between runs."""
    return f"{prefix}-{value}"


def in_main_thread(fn: Callable[[], None]) -> Callable[[], None]:
    """Mark a callable as one that must run on the GUI thread.

    Used as documentation and as a hook for assertions in tests; the queued
    connection is what actually enforces it.
    """
    fn.__studio_main_thread__ = True  # type: ignore[attr-defined]
    return fn


def elide(text: str, limit: int = 120) -> str:
    """One line, for a label that must not grow."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def format_money(value: float) -> str:
    """Small numbers need more digits than large ones."""
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_clock(timestamp: float) -> str:
    import time

    return time.strftime("%H:%M:%S", time.localtime(timestamp))
