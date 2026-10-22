"""Reference pictures selected through the visible composer action."""

from __future__ import annotations

import base64

import pytest
from PySide6.QtCore import QBuffer
from PySide6.QtGui import QImage
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QFileDialog, QLabel

from app.gui.chat.widget import MAX_ATTACHMENTS, MAX_IMAGE_SIDE, ChatView, ImageLabel

pytestmark = pytest.mark.gui


def picture(tmp_path, name, width=80, height=40):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0x4499CC)
    path = tmp_path / name
    assert image.save(str(path))
    return str(path)


def picker(monkeypatch, paths):
    calls = []

    def select(*args):
        calls.append(args)
        return paths, "Pictures"

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", select)
    return calls


@pytest.mark.parametrize("width", [480, 900])
def test_attach_is_visible_and_selects_multiple_images(qapp, tmp_path, monkeypatch, width):
    paths = [picture(tmp_path, "reference.png"), picture(tmp_path, "photo.jpg")]
    calls = picker(monkeypatch, paths)
    chat = ChatView()
    chat.resize(width, 360)
    chat.show()
    qapp.processEvents()
    assert chat.attach.text() == "Attach"
    assert chat.attach.isVisible()
    assert chat.composer.rect().contains(chat.attach.geometry())
    chat.input.setText("Build this")
    chat.attach.click()
    assert len(calls) == 1
    assert len(chat.attachments()) == 2
    assert chat._attachment_bar.isVisible()
    assert chat.input.text() == "Build this"
    submitted = QSignalSpy(chat.submitted)
    expected = chat.attachments()
    chat.send.click()
    assert submitted.at(0) == ["Build this", expected]
    assert not chat.attachments()


def test_picker_cancel_preserves_existing_draft(qapp, tmp_path, monkeypatch):
    picker(monkeypatch, [picture(tmp_path, "reference.png")])
    chat = ChatView()
    chat.attach.click()
    expected = chat.attachments()
    chat.input.setText("draft")
    picker(monkeypatch, [])
    rejected = QSignalSpy(chat.rejected_attachment)
    chat.attach.click()
    assert chat.attachments() == expected
    assert chat.input.text() == "draft"
    assert rejected.count() == 0


def test_picker_skips_invalid_and_missing_images(qapp, tmp_path, monkeypatch):
    bad = tmp_path / "broken.png"
    bad.write_text("not an image")
    picker(monkeypatch, [str(bad), str(tmp_path / "missing.jpg"), picture(tmp_path, "good.png")])
    chat = ChatView()
    rejected = QSignalSpy(chat.rejected_attachment)
    chat.attach.click()
    assert len(chat.attachments()) == 1
    assert rejected.count() == 2
    assert "broken.png" in rejected.at(0)[0]
    assert "missing.jpg" in rejected.at(1)[0]


def test_picker_respects_existing_attachment_and_dimension_limits(qapp, tmp_path, monkeypatch):
    path = picture(tmp_path, "large.png", MAX_IMAGE_SIDE + 100, 100)
    picker(monkeypatch, [path] * (MAX_ATTACHMENTS - 1))
    chat = ChatView()
    chat.attach.click()
    rejected = QSignalSpy(chat.rejected_attachment)
    picker(monkeypatch, [path, path])
    chat.attach.click()
    assert len(chat.attachments()) == MAX_ATTACHMENTS
    assert rejected.count() == 1
    assert "Only" in rejected.at(0)[0]
    image = QImage.fromData(base64.b64decode(chat.attachments()[0]))
    assert image.width() == MAX_IMAGE_SIDE


def test_busy_picker_is_disabled_and_does_not_open(qapp, monkeypatch):
    calls = picker(monkeypatch, [])
    chat = ChatView()
    chat.set_busy(True)
    assert not chat.attach.isEnabled()
    chat.attach.click()
    chat.choose_attachments()
    assert not calls
    chat.set_busy(False)
    assert chat.attach.isEnabled()
    chat.attach.click()
    assert len(calls) == 1


def test_picker_does_not_stage_if_run_starts_during_dialog(qapp, tmp_path, monkeypatch):
    chat = ChatView()
    path = picture(tmp_path, "reference.png")

    def select(*args):
        chat.set_busy(True)
        return [path], "Pictures"

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", select)
    chat.attach.click()
    assert not chat.attachments()


def test_jpeg_payload_renders_in_staging_and_transcript(qapp):
    image = QImage(80, 40, QImage.Format.Format_RGB32)
    image.fill(0x4499CC)
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    assert image.save(buffer, "JPEG")
    data = base64.b64encode(bytes(buffer.data())).decode()
    chat = ChatView()
    assert chat.add_attachment(data)
    thumb = chat._attachment_bar.findChild(QLabel)
    assert not thumb.pixmap().isNull()
    bubble = chat.add_message("user", "", images=[data])
    label = bubble.findChild(ImageLabel)
    assert not label.pixmap().isNull()
    assert label.suggested_name().endswith(".jpg")


def test_an_attachment_declares_the_format_its_bytes_are_in() -> None:
    """A large attachment is written as JPEG, and must not be sent as a PNG."""
    import base64

    from app.gui.chat.widget import image_mime_type

    assert image_mime_type(base64.b64encode(b"\xff\xd8\xff\xe0" + b"x" * 80).decode()) == "image/jpeg"
    assert image_mime_type(base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 80).decode()) == "image/png"
