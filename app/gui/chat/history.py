"""The prompt history popup.

Asking for something is the most repeated act in this window, and the prompts
that worked are the ones nobody wants to retype. A history of them is already in
the database -- every prompt is a stored user message -- so this reads them back
rather than keeping a second list that could disagree with the first.

A popup rather than a panel, because it is not a place to live: it is opened,
something is taken from it, and it closes. Enter takes the highlighted prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import elide

#: What a row shows. The full prompt is in the tooltip and goes into the input;
#: the row only has to be recognisable.
ROW_CHARS = 90

#: One prompt, and when it was asked. ``(text, created_at)``, newest first.
Prompt = tuple[str, float]


def history_shortcut_help() -> str:
    """One line, for a button tooltip."""
    return "Ctrl+Up walks back through what you have asked before"


class HistoryPopup(QDialog):
    """Recent prompts. Enter takes the selected one into the composer."""

    chosen = Signal(str)

    def __init__(self, prompts: Sequence[Prompt], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt history")
        self.resize(640, 460)
        self._prompts: list[Prompt] = list(prompts)

        layout = QVBoxLayout(self)
        hint = QLabel("Enter takes it into the input, Escape closes.")
        hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(hint)

        self.filter = QLineEdit()
        self.filter.setObjectName("history-filter")
        self.filter.setPlaceholderText("Type to filter…")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)

        self.list = QListWidget()
        self.list.setObjectName("history-list")
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemActivated.connect(lambda _item: self._take())
        self.list.itemDoubleClicked.connect(lambda _item: self._take())
        layout.addWidget(self.list, 1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.status)

        self._fill()
        if self.list.count():
            self.list.setCurrentRow(0)
        self.filter.setFocus()

    # --- content -----------------------------------------------------------

    def _fill(self) -> None:
        self.list.clear()
        for text, when in self._prompts:
            stamp = datetime.fromtimestamp(when).strftime("%d %b %H:%M") if when else ""
            item = QListWidgetItem(f"{stamp:<12}  {elide(text, ROW_CHARS)}")
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            self.list.addItem(item)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        """Hide what does not match rather than rebuilding: the rows stay, so the
        selection survives a keystroke and the arrow keys keep their place."""
        needle = text.strip().lower()
        shown = 0
        for row in range(self.list.count()):
            item = self.list.item(row)
            prompt = str(item.data(Qt.ItemDataRole.UserRole) or "")
            visible = not needle or needle in prompt.lower()
            item.setHidden(not visible)
            shown += visible
        total = len(self._prompts)
        self.status.setText(f"{shown} of {total}" if shown != total else f"{total} prompts")

    def matching(self) -> list[str]:
        """What the popup is currently offering, newest first.

        The list is filtered, which is what Enter will hand over: taking a prompt
        the filter has hidden is the one way the visible rows and the taken row
        could disagree. The composer's arrow keys deliberately walk the whole
        history rather than this -- see the comment on the window's call.
        """
        result: list[str] = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None and not item.isHidden():
                result.append(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        return result

    def _take(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        prompt = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if prompt:
            self.chosen.emit(prompt)
        self.accept()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._take()
            return
        super().keyPressEvent(event)
