"""Tool inputs and complete outputs remain inspectable in the transcript."""

import pytest
from PySide6.QtTest import QSignalSpy

from app.gui.chat.widget import ChatView, ToolCallCard

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("arguments", [{"code": "<b>hello</b>"}, '{"code":"<b>hello</b>"}'])
def test_arguments_stay_visible_after_completion(qapp, arguments):
    chat = ChatView()
    card = chat.add_tool("call", "execute", arguments)
    assert '"code": "<b>hello</b>"' in card.arguments.text()
    assert not card.arguments.isHidden()
    chat.finish_tool("call", is_error=False, text="ok", duration_ms=1)
    assert not card.arguments.isHidden()
    assert "<b>hello</b>" in card.arguments.text()


def test_full_result_is_expandable_and_updates(qapp):
    card = ToolCallCard("execute")
    result = "<b>literal output</b>\n" + "x" * 1000
    card.set_result(is_error=True, text=result, duration_ms=2)
    assert card.full_text() == result
    assert not card.expand.isHidden()
    assert card.result_text.isHidden()
    card.expand.click()
    assert not card.result_text.isHidden()
    assert card.result_text.toPlainText() == result
    card.set_result(is_error=False, text="short", duration_ms=3)
    assert card.full_text() == "short"
    assert card.expand.isHidden()
    assert card.result_text.isHidden()


def test_legacy_tool_creation(qapp):
    chat = ChatView()
    card = chat.add_tool("call", "execute")
    assert card.detail.text() == "running…"
    assert card.arguments.isHidden()


def test_busy_submit_preserves_draft_and_attachments(qapp):
    chat = ChatView()
    submitted = QSignalSpy(chat.submitted)
    chat.input.setText("draft")
    chat._attachments = ["staged"]
    chat.set_busy(True)
    chat._submit()
    assert submitted.count() == 0
    assert chat.input.text() == "draft"
    assert chat.attachments() == ["staged"]
    chat.set_busy(False)
    chat._submit()
    assert submitted.count() == 1
    assert submitted.at(0) == ["draft", ["staged"]]


def test_prompt_history_stops_at_oldest_and_restores_draft(qapp):
    chat = ChatView()
    chat.set_prompt_history(["newest", "oldest"])
    chat.input.setText("draft")
    assert chat.step_prompt_history(-1) == "newest"
    assert chat.step_prompt_history(-1) == "oldest"
    assert chat.step_prompt_history(-1) == "oldest"
    assert chat.step_prompt_history(1) == "newest"
    assert chat.step_prompt_history(1) == "draft"
