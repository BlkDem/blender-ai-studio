"""The chat transcript.

Tool calls are the point of this view, not a side effect of it. A person who
asked a model to build them a table wants to see *what it did* — which tool, what
arguments, how long, and whether it worked — laid out as a sequence they can
follow, distinct from the prose. So each step is its own collapsed card, and the
prose stays readable.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QBuffer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import elide
from app.gui.chat.history import history_shortcut_help

logger = logging.getLogger(__name__)

#: Enough of a tool result to recognise it. A whole scene's object list in a chat
#: bubble is unreadable, and the full text is one click away in the card.
PREVIEW_CHARS = 400

#: Longest side an attached picture is sent at. A 4K screenshot is several
#: megabytes of base64 in a prompt, and the detail that buys is not what anyone
#: asks a model to look at a reference image *for* -- a shape, a layout, a
#: material. Asked for as a cost, this is a context window spent on a picture the
#: user cannot see the size of.
MAX_IMAGE_SIDE = 1600

#: A ceiling on the encoded picture, whatever the dimensions were. PNG of a
#: photographic screenshot gets here first, and JPEG is the escape hatch.
MAX_IMAGE_BYTES = 4 * 1024 * 1024

#: How many pictures one turn may carry. Every extra one is tokens the model
#: spends looking rather than working, and a person who pastes fifteen
#: screenshots has mistyped something.
MAX_ATTACHMENTS = 8

ROLE_COLOURS = {
    "user": QColor("#1f6feb"),
    "assistant": QColor("#2da44e"),
    "error": QColor("#cf222e"),
    "tool": QColor("#8250df"),
    "system": QColor("#6e7781"),
}


def pretty_json(text: str) -> str:
    """Make a JSON payload readable without pretending it is not JSON."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


def encode_image(image: QImage) -> str:
    """A QImage as base64 PNG, small enough to be worth sending.

    Returns "" for anything that will not encode, so a caller has one thing to
    check rather than an exception to catch: a drop of a text file, or an image
    format Qt cannot read, should leave the composer untouched rather than take
    the window down.
    """
    if image.isNull():
        return ""
    if max(image.width(), image.height()) > MAX_IMAGE_SIDE:
        image = image.scaled(
            MAX_IMAGE_SIDE, MAX_IMAGE_SIDE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        return ""
    payload = bytes(buffer.data())
    if len(payload) > MAX_IMAGE_BYTES:
        # Photographic content barely compresses as PNG. Step the quality down
        # rather than giving up: a soft 92 is a better reference than no picture.
        for quality in (90, 75, 60):
            buffer.open(QBuffer.OpenModeFlag.WriteOnly)
            buffer.truncate(0)
            if not image.save(buffer, "JPEG", quality):
                break
            payload = bytes(buffer.data())
            if len(payload) <= MAX_IMAGE_BYTES:
                break
    buffer.close()
    return base64.b64encode(payload).decode()


class ChatInput(QLineEdit):
    """The one-line composer: pastes pictures, and walks back through prompts.

    A QLineEdit quietly drops a pasted image: the paste path only knows about
    text, so a screenshot pasted from the clipboard used to vanish with no sign it
    had been offered at all. ``paste`` is the one method Ctrl+V actually reaches,
    and intercepting it is the whole fix. (``insertFromMimeData`` is the C++
    virtual underneath, but PySide6 does not expose it to override, and
    ``insert`` is the text slot, not the paste path.)
    """

    image_pasted = Signal(str)  # base64, ready to send
    #: Ctrl+Up / Ctrl+Down, like a shell.
    history_step = Signal(int)  # -1 older, +1 newer

    def paste(self) -> None:
        from PySide6.QtWidgets import QApplication

        mime = QApplication.clipboard().mimeData()
        if mime is not None and mime.hasImage():
            data = encode_image(mime.imageData())
            if data:
                self.image_pasted.emit(data)
                return
        super().paste()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        up = event.key() == Qt.Key.Key_Up and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        down = event.key() == Qt.Key.Key_Down and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        if up or down:
            self.history_step.emit(-1 if up else 1)
            return
        super().keyPressEvent(event)


#: The formats a picture in the transcript can arrive as. Sniffed rather than
#: assumed: an attachment is written as JPEG when PNG would be too large, so the
#: extension a save dialog offers has to come from the bytes, not from the path
#: Qt happened to hand us.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
)


def image_suffix(raw: bytes) -> str:
    """The file extension the bytes in this payload really are."""
    for magic, suffix in _MAGIC:
        if raw.startswith(magic):
            return suffix
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return ".png"


class ImageLabel(QLabel):
    """A picture in the transcript, which the user can keep.

    Read-only is the wrong answer for a render. The person looking at it usually
    wants the file -- to keep it, to send it, to look at it full size -- and
    until now the only copy lived inside a chat card where nothing could reach
    it. The bytes are already in hand, so saving is a write.
    """

    def __init__(self, data: str, *, width: int, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = data
        self.setObjectName("chat-image")
        pixmap = QPixmap()
        if not pixmap.loadFromData(base64.b64decode(data), "PNG"):
            return
        self.setPixmap(pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation))
        self.setToolTip(f"{tooltip} — right-click to save")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        action = QAction("Save picture as…", self)
        action.triggered.connect(self.choose_save_path)
        self.addAction(action)

    def raw(self) -> bytes:
        return base64.b64decode(self._data)

    def suggested_name(self) -> str:
        raw = self.raw()
        return f"blender-ai-studio-{datetime.now():%Y%m%d-%H%M%S}{image_suffix(raw)}"

    def save_to(self, path: str) -> bool:
        """Write the picture to ``path``. False if it could not be written.

        Kept apart from the dialog so it can be exercised without a modal
        window, which a test cannot answer.
        """
        try:
            target = Path(path)
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.raw())
        except OSError as exc:
            logger.warning("could not save a picture to %s: %s", path, exc)
            return False
        return True

    def choose_save_path(self) -> str:
        """Ask where to put it, and write it. "" when the user cancels."""
        suggested = self.suggested_name()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save picture", suggested, f"Pictures (*{Path(suggested).suffix})"
        )
        if not path:
            return ""
        return path if self.save_to(path) else ""


class ToolCallCard(QFrame):
    """One step of the agent's work."""

    def __init__(self, tool: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tool = tool
        self.setObjectName("tool-card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Maximum, not Preferred: a Qt layout hands leftover space to items that
        # can grow before it honours a stretch, so a single tool card in an empty
        # transcript would otherwise fill the whole panel.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setStyleSheet(
            """
            #tool-card {
                background: palette(alternate-base);
                border: 1px solid palette(mid);
                border-radius: 6px;
                margin: 2px 0px 2px 0px;
            }
            """
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 6, 10, 6)
        self._layout.setSpacing(2)

        self.header = QLabel()
        self.header.setFont(QFont("", 9, QFont.Weight.DemiBold))
        self.header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._layout.addWidget(self.header)

        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setFont(QFont("monospace", 9))
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._layout.addWidget(self.detail)

    def set_running(self) -> None:
        self.header.setText(f"🔧 {self.tool}")
        self.header.setStyleSheet(f"color: {ROLE_COLOURS['tool'].name()};")
        self.detail.setText("running…")

    def set_result(
        self,
        *,
        is_error: bool,
        text: str,
        duration_ms: float,
        images: int = 0,
        image_data: Sequence[str] = (),
    ) -> None:
        mark = "✗" if is_error else "✓"
        colour = ROLE_COLOURS["error"] if is_error else ROLE_COLOURS["tool"]
        suffix = f" · {images} image(s)" if images else ""
        # Sub-second calls differ by a few tens of milliseconds, which rounding
        # to whole milliseconds hides: two different calls showing "213 ms" reads
        # as a bug in the measurement.
        shown = f"{duration_ms / 1000:.2f} s" if duration_ms >= 1000 else f"{duration_ms:.0f} ms"
        self.header.setText(f"{mark} {self.tool} · {shown}{suffix}")
        self.header.setStyleSheet(f"color: {colour.name()};")
        body = pretty_json(text)
        self.detail.setText(elide(body, PREVIEW_CHARS))
        # A card can be finished more than once, and each finishing brings its own
        # picture. Without this the second render stacks under the first and the
        # card claims to show a result it no longer has.
        for stale in self.findChildren(QLabel):
            if stale.objectName() == "chat-image":
                stale.setParent(None)
                stale.deleteLater()
        for data in image_data[:1]:
            self._layout.addWidget(self._image_label(data))

    def _image_label(self, data: str) -> QLabel:
        """Show the picture the tool returned.

        A card that says "1 image" tells the user there is something to look at
        and then makes them go and find it. A render is the whole point of
        asking for one.
        """
        return ImageLabel(data, width=360, tooltip="the render this tool returned")

    def full_text(self) -> str:
        return self.detail.toolTip() or self.detail.text()


class MessageBubble(QFrame):
    """One turn of the conversation."""

    def __init__(
        self,
        role: str,
        text: str = "",
        parent: QWidget | None = None,
        images: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.role = role
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        colour = ROLE_COLOURS.get(role, ROLE_COLOURS["system"])
        self.setStyleSheet(
            f"""
            QFrame {{
                background: palette(base);
                border-left: 3px solid {colour.name()};
                border-radius: 4px;
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        label = QLabel(role)
        label.setFont(QFont("", 8, QFont.Weight.DemiBold))
        label.setStyleSheet(f"color: {colour.name()}; border: none;")
        layout.addWidget(label)
        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setFont(QFont())
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Hug the text. Left to its own policy a QTextEdit stretches to fill the
        # transcript, so a one-line answer occupies a screen.
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body.setText(text)
        self._fit()
        layout.addWidget(self.body)
        #: What the person attached, kept so the transcript shows the turn as it
        #: was sent rather than as bare words. A reference picture that is not in
        #: the transcript is a picture the next turn has to be told about again.
        self.attached: list[str] = []
        for data in images:
            self.attached.append(data)
            layout.addWidget(ImageLabel(data, width=240, tooltip="attached to this message"))

    def _fit(self) -> None:
        """Size the body to its content, up to a readable maximum."""
        height = int(self.body.document().size().height()) + 12
        self.body.setFixedHeight(max(24, min(240, height)))

    def append(self, text: str) -> None:
        """Streaming in. The cursor follows, because a stream that scrolls away
        from the caret looks frozen."""
        cursor = self.body.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.body.setTextCursor(cursor)
        self._fit()
        self.body.ensureCursorVisible()


class ChatView(QWidget):
    """The transcript, with the input under it."""

    submitted = Signal(str, list)
    stop_requested = Signal()
    rejected_attachment = Signal(str)
    history_requested = Signal()
    new_chat_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, ToolCallCard] = {}
        self._streaming: MessageBubble | None = None
        #: base64 pictures waiting to go out with the next turn, oldest first.
        self._attachments: list[str] = []
        #: Prompts walked by Ctrl+Up, newest first, and where the cursor is in
        #: them. -1 means "the draft", which is where forward stops.
        self._prompt_history: list[str] = []
        self._history_index = -1
        self._history_draft = ""
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.transcript = QWidget()
        self.transcript.setObjectName("transcript")
        self._transcript_layout = QVBoxLayout(self.transcript)
        self._transcript_layout.setContentsMargins(12, 12, 12, 12)
        self._transcript_layout.setSpacing(8)
        self._transcript_layout.addStretch(1)

        from PySide6.QtWidgets import QScrollArea

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.transcript)
        layout.addWidget(self.scroll, 1)

        self._attachment_bar = QWidget()
        self._attachment_bar.setObjectName("attachment-bar")
        self._attachment_layout = QHBoxLayout(self._attachment_bar)
        self._attachment_layout.setContentsMargins(8, 4, 8, 0)
        self._attachment_layout.setSpacing(6)
        self._attachment_layout.addStretch(1)
        self._attachment_bar.setVisible(False)
        layout.addWidget(self._attachment_bar)

        input_row = QHBoxLayout()
        self.input = ChatInput()
        self.input.setPlaceholderText("Create a table with four legs…  (drop or paste a picture)")
        self.input.setObjectName("chat-input")
        self.input.returnPressed.connect(self._submit)
        self.input.image_pasted.connect(self.add_attachment)
        self.input.history_step.connect(self.step_prompt_history)
        self.send = QPushButton("Send")
        self.send.clicked.connect(self._submit)
        self.history = QPushButton("History")
        self.history.setToolTip(history_shortcut_help())
        self.history.clicked.connect(self.history_requested.emit)
        self.new_chat = QPushButton("New")
        self.new_chat.setToolTip("Start a fresh conversation, forgetting the last one")
        self.new_chat.clicked.connect(self.new_chat_requested.emit)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_requested.emit)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.history)
        input_row.addWidget(self.new_chat)
        input_row.addWidget(self.send)
        input_row.addWidget(self.stop)
        layout.addLayout(input_row)

    # --- attachments -------------------------------------------------------

    def add_attachment(self, data: str) -> bool:
        """Stage one base64 picture for the next turn. True if it was kept.

        Refusing is reported rather than ignored: a person who dropped four
        pictures and sees two of them has to be told, or they will ask why the
        model ignored the rest.
        """
        if not data:
            return False
        if len(self._attachments) >= MAX_ATTACHMENTS:
            self.rejected_attachment.emit(
                f"Only {MAX_ATTACHMENTS} pictures per turn; the last one was not added."
            )
            return False
        self._attachments.append(data)
        self._rebuild_attachment_bar()
        return True

    def attachments(self) -> list[str]:
        return list(self._attachments)

    def take_attachments(self) -> list[str]:
        """The staged pictures, and the composer is emptied of them.

        Taken rather than read, so a turn that is composed and then abandoned
        cannot smuggle a picture into a later, unrelated turn.
        """
        taken, self._attachments = self._attachments, []
        self._rebuild_attachment_bar()
        return taken

    def _rebuild_attachment_bar(self) -> None:
        while self._attachment_layout.count():
            item = self._attachment_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for index, data in enumerate(self._attachments):
            thumb = QLabel()
            pixmap = QPixmap()
            if pixmap.loadFromData(base64.b64decode(data), "PNG"):
                thumb.setPixmap(pixmap.scaledToWidth(56, Qt.TransformationMode.SmoothTransformation))
            thumb.setToolTip(f"picture {index + 1} of {len(self._attachments)} — sent with the next message")
            self._attachment_layout.addWidget(thumb)
            drop = QPushButton("×")
            drop.setFixedWidth(22)
            drop.setToolTip("remove this picture")
            drop.clicked.connect(lambda _=False, i=index: self._remove_attachment(i))
            self._attachment_layout.addWidget(drop)
        self._attachment_bar.setVisible(bool(self._attachments))

    def _remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self._attachments):
            del self._attachments[index]
            self._rebuild_attachment_bar()

    # --- drops and pastes --------------------------------------------------

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        mime = event.mimeData()
        added = 0
        if mime.hasImage() and not mime.hasUrls():
            if self.add_attachment(encode_image(mime.imageData())):
                added += 1
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            if not self.add_attachment(encode_image(QImage(url.toLocalFile()))):
                self.rejected_attachment.emit(f"Not a picture I could read: {Path(url.toLocalFile()).name}")
                continue
            added += 1
        if added:
            event.acceptProposedAction()

    # --- composing ---------------------------------------------------------

    def _submit(self) -> None:
        text = self.input.text().strip()
        images = self.take_attachments()
        # A picture on its own is a message: "build this" is a turn with no
        # words, and refusing to send it would make the paste look broken.
        if not text and not images:
            return
        self.input.clear()
        # Walking the history leaves a draft in the input; sending it should not
        # keep the cursor stuck two prompts back.
        self._history_index = -1
        self.submitted.emit(text, images)

    # --- prompt history ----------------------------------------------------

    def set_prompt_history(self, prompts: Sequence[str]) -> None:
        """What the arrow keys walk, newest first."""
        self._prompt_history = list(prompts)
        self._history_index = -1

    def prompt_history(self) -> list[str]:
        return list(self._prompt_history)

    def step_prompt_history(self, direction: int) -> str:
        """Move through the history. "" at the end, where the draft returns.

        The draft is put back rather than dropped, so walking past the oldest
        prompt and then coming forward again lands on what was being typed
        instead of on nothing.
        """
        if not self._prompt_history:
            return ""
        if self._history_index == -1 and direction < 0:
            self._history_draft = self.input.text()
        if direction < 0:
            self._history_index = min(self._history_index + 1, len(self._prompt_history))
        else:
            self._history_index = max(self._history_index - 1, -1)
        if self._history_index < 0:
            self.input.setText(self._history_draft)
            return self._history_draft
        prompt = self._prompt_history[self._history_index]
        self.input.setText(prompt)
        return prompt

    def put_prompt(self, text: str) -> None:
        """Drop a prompt into the composer, from the popup or from history."""
        self.input.setText(text)
        self.input.setFocus()
        self._history_index = -1

    def set_busy(self, busy: bool) -> None:
        self.send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.stop.setEnabled(busy)

    def busy(self) -> bool:
        """Whether a run is in progress."""
        return self.stop.isEnabled()

    # --- transcript --------------------------------------------------------

    def add_message(self, role: str, text: str, images: Sequence[str] = ()) -> MessageBubble:
        bubble = MessageBubble(role, text, images=images)
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, bubble)
        self._scroll_to_end()
        return bubble

    def start_stream(self, role: str = "assistant") -> MessageBubble:
        self._streaming = self.add_message(role, "")
        return self._streaming

    def append_stream(self, text: str) -> None:
        if self._streaming is not None:
            self._streaming.append(text)

    def end_stream(self) -> None:
        self._streaming = None

    def add_tool(self, call_id: str, tool: str) -> ToolCallCard:
        card = ToolCallCard(tool)
        card.set_running()
        self._cards[call_id] = card
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, card)
        return card

    def finish_tool(
        self,
        call_id: str,
        *,
        is_error: bool,
        text: str,
        duration_ms: float,
        images: int = 0,
        image_data: Sequence[str] = (),
    ) -> None:
        card = self._cards.get(call_id)
        if card is None:
            return
        card.set_result(
            is_error=is_error, text=text, duration_ms=duration_ms, images=images, image_data=image_data
        )
        card.detail.setToolTip(pretty_json(text))
        self._scroll_to_end()

    def add_note(self, text: str, role: str = "system") -> None:
        self.add_message(role, text)

    def add_cost(self, line: str) -> None:
        label = QLabel(line)
        label.setStyleSheet(f"color: {ROLE_COLOURS['system'].name()};")
        label.setFont(QFont("", 8))
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, label)

    def clear(self) -> None:
        while self._transcript_layout.count() > 1:
            item = self._transcript_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()
        self._streaming = None

    def transcript_text(self) -> str:
        """What a person sees, as text. For copying out, and for the tests."""
        lines: list[str] = []
        for index in range(self._transcript_layout.count()):
            widget = self._transcript_layout.itemAt(index).widget()
            if isinstance(widget, MessageBubble):
                lines.append(f"{widget.role}: {widget.body.toPlainText()}")
            elif isinstance(widget, ToolCallCard):
                lines.append(f"tool: {widget.header.text()}")
        return "\n".join(lines)

    def _scroll_to_end(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def summary(self) -> dict[str, Any]:
        bubbles = 0
        characters = 0
        for index in range(self._transcript_layout.count()):
            widget = self._transcript_layout.itemAt(index).widget()
            if isinstance(widget, MessageBubble):
                bubbles += 1
                characters += len(widget.body.toPlainText())
        return {"messages": bubbles, "tools": len(self._cards), "text_chars": characters}
