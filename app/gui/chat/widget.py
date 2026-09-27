"""The chat transcript.

Tool calls are the point of this view, not a side effect of it. A person who
asked a model to build them a table wants to see *what it did* — which tool, what
arguments, how long, and whether it worked — laid out as a sequence they can
follow, distinct from the prose. So each step is its own collapsed card, and the
prose stays readable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import elide

#: Enough of a tool result to recognise it. A whole scene's object list in a chat
#: bubble is unreadable, and the full text is one click away in the card.
PREVIEW_CHARS = 400

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
        for stale in self.findChildren(QLabel):
            if stale.objectName() == "tool-image":
                stale.setParent(None)
                stale.deleteLater()
        for data in image_data[:1]:
            self._layout.addWidget(self._image_label(data))

    @staticmethod
    def _image_label(data: str) -> QLabel:
        """Show the picture the tool returned.

        A card that says "1 image" tells the user there is something to look at
        and then makes them go and find it. A render is the whole point of
        asking for one.
        """
        import base64

        from PySide6.QtGui import QPixmap

        label = QLabel()
        label.setObjectName("tool-image")
        pixmap = QPixmap()
        if pixmap.loadFromData(base64.b64decode(data), "PNG"):
            label.setPixmap(pixmap.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation))
            label.setToolTip("the render this tool returned")
        return label

    def full_text(self) -> str:
        return self.detail.toolTip() or self.detail.text()


class MessageBubble(QFrame):
    """One turn of the conversation."""

    def __init__(self, role: str, text: str = "", parent: QWidget | None = None) -> None:
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

    submitted = Signal(str)
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, ToolCallCard] = {}
        self._streaming: MessageBubble | None = None

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

        from PySide6.QtWidgets import QLineEdit, QPushButton

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Create a table with four legs…")
        self.input.setObjectName("chat-input")
        self.input.returnPressed.connect(self._submit)
        self.send = QPushButton("Send")
        self.send.clicked.connect(self._submit)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_requested.emit)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send)
        input_row.addWidget(self.stop)
        layout.addLayout(input_row)

    # --- composing ---------------------------------------------------------

    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.stop.setEnabled(busy)

    # --- transcript --------------------------------------------------------

    def add_message(self, role: str, text: str) -> MessageBubble:
        bubble = MessageBubble(role, text)
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
