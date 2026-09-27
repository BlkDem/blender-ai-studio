"""The Tasks panel: background work, with its cost.

3D generation takes minutes, so the user needs to see it exist, see what it is
costing, and be able to stop it. The numbers come from the task manager that ran
the work, so this table and the database row agree by construction.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import format_duration, format_money

COLUMNS = ["Task", "Provider", "Status", "Progress", "Started", "Duration", "Credits", "Cost", "Error"]


class TasksPanel(QWidget):
    """One row per task, newest first."""

    cancel_requested = Signal(str)  # task id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel selected")
        self.cancel_button.clicked.connect(self._cancel_selected)
        self.clear_button = QPushButton("Clear finished")
        self.clear_button.clicked.connect(self.clear_finished)
        row.addWidget(self.cancel_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("tasks-table")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self._rows: dict[str, int] = {}

    def add_or_update(self, task: dict[str, Any]) -> None:
        """Insert or update one row. Called for every TASK_* event."""
        task_id = str(task.get("id", ""))
        if not task_id:
            return
        cells = [
            str(task.get("name", "")),
            str(task.get("provider", "")),
            str(task.get("state", "")),
            f"{float(task.get('progress', 0.0)) * 100:.0f}%",
            str(task.get("started_at", "")),
            format_duration(float(task.get("duration_s", 0.0))),
            f"{float(task.get('credits', 0.0)):.0f}",
            format_money(float(task.get("cost_usd", 0.0))),
            str(task.get("error", "")),
        ]
        row = self._rows.get(task_id)
        if row is None:
            row = self.table.rowCount()
            self._rows[task_id] = row
            self.table.insertRow(row)
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self.table.setItem(row, column, item)

    def _cancel_selected(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            return
        task_id = selected[0].data(Qt.ItemDataRole.UserRole)
        if task_id:
            self.cancel_requested.emit(str(task_id))

    def clear_finished(self) -> None:
        for task_id, row in sorted(self._rows.items(), key=lambda item: item[1], reverse=True):
            widget = self.table.item(row, 2)
            if widget is not None and widget.text() in ("succeeded", "failed", "cancelled"):
                self.table.removeRow(row)
                self._rows.pop(task_id, None)
        self._rows = {
            task_id: row for task_id, row in self._rows.items() if row < self.table.rowCount()
        }

    def totals(self) -> dict[str, float]:
        """Credits and cost in the table, summed."""
        credits = cost = 0.0
        for row in range(self.table.rowCount()):
            try:
                credits += float(self.table.item(row, 6).text() or 0)
            except (AttributeError, ValueError):
                pass
            try:
                cost += float((self.table.item(row, 7).text() or "$0").lstrip("$"))
            except (AttributeError, ValueError):
                pass
        return {"credits": credits, "cost_usd": cost}
