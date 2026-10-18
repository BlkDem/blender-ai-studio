"""Chat composer and the shared window footer, rendered offscreen."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QStatusBar

from app.gui.chat.widget import ChatView

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("width,height", [(480, 360), (900, 640)])
def test_composer_has_room_for_input_and_actions(qapp, width, height):
    chat = ChatView()
    chat.resize(width, height)
    chat.show()
    qapp.processEvents()

    assert chat.input.width() >= chat.composer.width() - 24
    assert chat.scroll.geometry().bottom() < chat.composer.geometry().top()
    for button in (chat.history, chat.new_chat, chat.stop, chat.send):
        assert button.geometry().top() > chat.input.geometry().bottom()
        assert chat.composer.rect().contains(button.geometry())
        assert button.minimumWidth() < button.maximumWidth()
    assert chat.composer.height() - chat.send.geometry().bottom() >= 16


def test_composer_preserves_submission_and_busy_state(qapp):
    chat = ChatView()
    submitted = QSignalSpy(chat.submitted)
    chat.input.setText("Build a chair")
    chat.send.click()
    assert submitted.count() == 1
    assert submitted.at(0) == ["Build a chair", []]
    chat.set_busy(True)
    assert not chat.input.isEnabled()
    assert not chat.send.isEnabled()
    assert chat.stop.isEnabled()


async def test_statusbar_spans_window_below_composer(window, qapp):
    bar = window.statusBar()
    assert isinstance(bar, QStatusBar)
    assert window.findChildren(QStatusBar) == [bar]
    window.nav.setCurrentRow(0)
    for width, height in ((1000, 700), (1280, 820)):
        window.resize(width, height)
        qapp.processEvents()
        assert bar.geometry().left() == 0
        assert bar.width() == window.width()
        assert bar.geometry().top() > window.centralWidget().geometry().bottom()
        composer_bottom = window.chat.composer.mapTo(window, QPoint(0, window.chat.composer.height()))
        assert composer_bottom.y() <= bar.geometry().top()
        for label in (
            window.project_status, window.mcp_status, window.llm_status,
            window.three_d_status, window.cost_status,
        ):
            assert bar.isAncestorOf(label)
    window.nav.setCurrentRow(1)
    qapp.processEvents()
    assert bar.isVisible()
    assert window.statusBar() is bar


def settle(qapp):
    for _ in range(12):
        qapp.processEvents()


def overflowing_chat(qapp):
    chat = ChatView()
    chat.resize(480, 360)
    chat.show()
    for index in range(20):
        chat.add_message("user", f"Message {index}\n" + "More text\n" * 5)
    settle(qapp)
    assert chat.scroll.verticalScrollBar().maximum() > 0
    return chat


def assert_following(chat, qapp):
    settle(qapp)
    bar = chat.scroll.verticalScrollBar()
    assert bar.value() == bar.maximum()


def test_new_content_follows_after_deferred_layout(qapp):
    chat = overflowing_chat(qapp)
    bar = chat.scroll.verticalScrollBar()
    for update in (
        lambda: chat.add_message("assistant", "Answer\n" * 12),
        lambda: chat.add_tool("call", "blender.get_scene"),
        lambda: chat.finish_tool("call", is_error=False, text="Done", duration_ms=10),
        lambda: chat.add_cost("100 tokens"),
    ):
        bar.setValue(0)
        update()
        assert_following(chat, qapp)


def test_stream_deltas_and_final_follow_bottom(qapp):
    chat = overflowing_chat(qapp)
    chat.start_stream()
    for _ in range(5):
        chat.scroll.verticalScrollBar().setValue(0)
        chat.append_stream("More streamed content\n" * 10)
        assert_following(chat, qapp)
    chat.scroll.verticalScrollBar().setValue(0)
    chat.end_stream()
    assert_following(chat, qapp)


def test_typing_and_submitting_follow_bottom(qapp):
    chat = overflowing_chat(qapp)
    chat.scroll.verticalScrollBar().setValue(0)
    QTest.keyClicks(chat.input, "Build a chair")
    assert_following(chat, qapp)
    chat.scroll.verticalScrollBar().setValue(0)
    QTest.keyClick(chat.input, Qt.Key.Key_Return)
    assert_following(chat, qapp)
