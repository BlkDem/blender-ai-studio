from types import SimpleNamespace

import pytest

from app.core.events import Event, EventType

pytestmark = pytest.mark.gui


async def test_retry_countdown_is_monotonic_and_scoped(window, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.gui.main_window.time.monotonic", lambda: clock[0])
    window.chat.set_busy(True)
    window._on_event(Event(EventType.RUN_STARTED, run_id="our-run"))
    window._on_event(Event(EventType.REQUEST_RETRY, run_id="our-run", payload={
        "attempt": 2, "max_attempts": 3, "delay": 4, "reason": "rate limited",
    }))
    assert "Retry 2/3 in 4s" in window.run_status.text()
    assert "rate limited" in window.run_status.text()
    assert window.chat.busy()
    clock[0] += 1.2
    window._update_retry_countdown()
    assert "in 3s" in window.run_status.text()
    window._on_event(Event(EventType.REQUEST_STARTED, run_id="another-run"))
    assert "Retry" in window.run_status.text()
    window._on_event(Event(EventType.REQUEST_STARTED, run_id="our-run"))
    assert "Waiting" in window.run_status.text()
    assert not window._retry_timer.isActive()
    window._on_event(Event(EventType.RUN_FAILED, run_id="our-run", payload={"reason": "cancelled"}))
    assert "Cancelled" in window.run_status.text()
    assert not window.chat.busy()


async def test_text_answer_rendered_once_at_completion(window, monkeypatch):
    messages = []
    monkeypatch.setattr(window.chat, "add_message", lambda role, text: messages.append((role, text)))
    monkeypatch.setattr(window, "refresh_scene", lambda: None)
    window._on_event(Event(EventType.TEXT, payload={"text": "A non-streaming answer"}))
    result = SimpleNamespace(text="A non-streaming answer", stopped_because="", tool_calls=[],
                             totals=SimpleNamespace(tokens=0, llm_usd=0), elapsed_s=0)
    window._run_finished(result)
    assert messages == [("assistant", "A non-streaming answer")]
    assert "Completed" in window.run_status.text()


async def test_run_error_cleans_stream_and_retry_without_unrelated_error_interrupt(window):
    window.current_run_id = "run"
    window.chat.set_busy(True)
    window.chat.start_stream()
    window.chat.append_stream("Partial answer")
    window._on_error("SCENE_ERROR", "Scene unavailable")
    assert window.chat.busy()
    assert window.current_run_id == "run"
    window._retry_timer.start()
    window._run_finished(RuntimeError("Provider unavailable"))
    assert not window.chat.busy()
    assert window.chat._streaming is None
    assert window.current_run_id == ""
    assert not window._retry_timer.isActive()
    assert "Failed: Provider unavailable" in window.run_status.text()
