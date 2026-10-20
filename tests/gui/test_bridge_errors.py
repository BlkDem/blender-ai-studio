from concurrent.futures import Future

from app.gui.bridge import CoreThread


def test_background_failure_is_delivered_only_when_gui_drains(qapp):
    core = CoreThread(None)
    errors = []
    core.signals.error.connect(lambda code, message: errors.append((code, message)))
    future = Future()
    future.set_exception(RuntimeError("network unavailable"))
    core._report_failure(future)
    assert errors == []
    core.drain()
    assert errors == [("ERROR", "network unavailable")]


def test_callback_failure_is_visible_and_drain_continues(qapp):
    core = CoreThread(None)
    errors = []
    completed = []
    core.signals.error.connect(lambda code, message: errors.append((code, message)))

    def broken():
        raise ValueError("cannot display result")

    core._inbox.put(("callback", (broken,)))
    core._inbox.put(("callback", (lambda: completed.append(True),)))
    core.drain()
    assert errors == [("GUI_CALLBACK_ERROR", "cannot display result")]
    assert completed == [True]


def test_cancelled_future_does_not_report_an_error(qapp):
    core = CoreThread(None)
    future = Future()
    future.cancel()
    core._deliver(future, lambda result: None)
    core._report_failure(future)
    assert core._inbox.empty()
