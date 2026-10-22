"""Confirmation before enabling image input for a saved model."""

from __future__ import annotations

import base64

import pytest
from PySide6.QtCore import QBuffer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMessageBox

pytestmark = pytest.mark.gui


def stage(window):
    images = []
    for color in (0x4499CC, 0xCC9944):
        image = QImage(20, 10, QImage.Format.Format_RGB32)
        image.fill(color)
        buffer = QBuffer()
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        assert image.save(buffer, "PNG")
        data = base64.b64encode(bytes(buffer.data())).decode()
        assert window.chat.add_attachment(data)
        images.append(data)
    window.chat.input.setText("Build these references")
    return images


def capture_submissions(window, monkeypatch):
    pending = []
    monkeypatch.setattr(window.core, "submit", lambda coro, callback: pending.append((coro, callback)))
    return pending


async def test_cancel_preserves_draft_and_images_without_saving_or_running(window, monkeypatch):
    pictures = stage(window)
    pending = capture_submissions(window, monkeypatch)
    questions = []

    def cancel(*args):
        questions.append(args)
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    messages = []
    monkeypatch.setattr(window.chat, "add_message", lambda *args, **kwargs: messages.append((args, kwargs)))
    window.chat.send.click()
    assert len(questions) == 1
    assert questions[0][-1] == QMessageBox.StandardButton.Cancel
    assert window.chat.input.text() == "Build these references"
    assert window.chat.attachments() == pictures
    assert not window.chat.busy()
    assert not pending
    assert not messages
    assert not window._model_sees_images()
    assert await window.context.studio.settings.get("llm_providers") is None


async def test_yes_persists_vision_before_submitting_all_images(window, monkeypatch):
    pictures = stage(window)
    pending = capture_submissions(window, monkeypatch)
    questions = []

    def yes(*args):
        questions.append(args)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", yes)
    messages = []
    monkeypatch.setattr(window.chat, "add_message", lambda *args, **kwargs: messages.append((args, kwargs)))

    class Agent:
        async def run(self, text, *, images):
            assert text == "Build these references"
            assert [part.data for part in images] == pictures
            return "submitted"

    built = []
    monkeypatch.setattr(
        window, "_agent_for_turn", lambda provider="", model="": built.append((provider, model)) or Agent()
    )
    window._agent_key = ("stale",)
    window.chat.send.click()
    assert window.chat.busy()
    assert not messages
    assert len(pending) == 1
    coroutine, callback = pending.pop(0)
    callback(await coroutine)
    # The agent read the model's capabilities when it was built, so the next
    # turn gets a new one -- built once, for the turn that was held back.
    assert window._agent_key is None
    assert built == [(window.current_provider(), window.current_model())]
    assert len(questions) == 1
    assert window._model_sees_images()
    stored = await window.context.studio.settings.get("llm_providers")
    config = next(config for config in stored if config["name"] == "scripted")
    assert config["models"][0]["supports_vision"] is True
    assert config["models"][0]["supports_tools"] is True
    assert messages == [(("user", "Build these references"), {"images": pictures})]
    assert not window.chat.attachments()
    assert window.chat.input.text() == ""
    assert window.chat.busy()
    results = []
    for coroutine, callback in pending:
        results.append(await coroutine)
    assert "submitted" in results


async def test_save_failure_restores_draft_and_images(window, monkeypatch):
    pictures = stage(window)
    pending = capture_submissions(window, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    async def fail(payload):
        raise RuntimeError("disk full")

    monkeypatch.setattr(window, "_persist_model_editor", fail)
    window.chat.send.click()
    coroutine, callback = pending.pop()
    result = await coroutine
    assert isinstance(result, RuntimeError)
    callback(result)
    assert window.chat.input.text() == "Build these references"
    assert window.chat.attachments() == pictures
    assert not window.chat.busy()
    assert not window._model_sees_images()
    assert not pending


async def test_model_change_during_save_restores_instead_of_sending_to_another_model(window, monkeypatch):
    pictures = stage(window)
    pending = capture_submissions(window, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.chat.send.click()
    coroutine, callback = pending.pop()
    window.model_selector.blockSignals(True)
    window.model_selector.addItem("Another model", "scripted:another")
    window.model_selector.setCurrentIndex(window.model_selector.count() - 1)
    window.model_selector.blockSignals(False)
    callback(await coroutine)
    assert window.chat.input.text() == "Build these references"
    assert window.chat.attachments() == pictures
    assert not window.chat.busy()
    assert not pending
