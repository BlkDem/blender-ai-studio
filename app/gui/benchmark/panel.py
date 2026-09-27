"""The Benchmark panel: a suite, a run, and a table of facts.

Two things it deliberately does not do. It does not rank the models, and it does
not hide a run that was not scene-isolated: the "Scene reset" column is there
because a comparison against a run that started from someone else's scene is not
a comparison, and the person reading the table has to be able to see that.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import format_duration, format_money

RESULT_COLUMNS = ["Model", "Status", "Time", "Tokens", "Cost", "Tools", "Errors", "Scene reset", "Score"]
REVIEW_FIELDS = ("Geometry", "Materials", "Following", "Composition", "Overall")


class BenchmarkPanel(QWidget):
    """Build a suite, run it, read the results, score them by hand."""

    run_requested = Signal(str, str)  # blend path, models csv
    cancel_requested = Signal()
    review_requested = Signal(str, dict)  # run id, scores

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        suite = QGroupBox("Suite")
        form = QFormLayout(suite)
        self.name = QLineEdit("Table benchmark")
        self.prompts = QPlainTextEdit("Create a table with four legs.")
        self.prompts.setMaximumHeight(80)
        self.blend = QLineEdit()
        self.blend.setPlaceholderText("/path/to/starting.blend (one copy per run)")
        self.models = QLineEdit()
        self.models.setPlaceholderText("space-bunny:space-bunny-free, openai:gpt-4o-mini")
        self.run_button = QPushButton("Run benchmark")
        self.run_button.clicked.connect(
            lambda: self.run_requested.emit(self.blend.text().strip(), self.models.text().strip())
        )
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        form.addRow("Name", self.name)
        form.addRow("Prompts", self.prompts)
        form.addRow("Starting .blend", self.blend)
        form.addRow("Models", self.models)
        row = QHBoxLayout()
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        form.addRow("", row)
        layout.addWidget(suite)

        self.table = QTableWidget(0, len(RESULT_COLUMNS))
        self.table.setObjectName("benchmark-table")
        self.table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        review = QGroupBox("Your score for the selected run")
        review_layout = QHBoxLayout(review)
        self.spins: dict[str, QLineEdit] = {}
        for field in REVIEW_FIELDS:
            spin = QLineEdit()
            spin.setPlaceholderText("0-5")
            spin.setMaximumWidth(60)
            self.spins[field] = spin
            review_layout.addWidget(QLabel(field))
            review_layout.addWidget(spin)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes")
        review_layout.addWidget(self.notes, 1)
        self.save_review = QPushButton("Save score")
        self.save_review.clicked.connect(self._save_review)
        review_layout.addWidget(self.save_review)
        layout.addWidget(review)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumHeight(140)
        self.transcript.setPlaceholderText("The selected run's conversation")
        layout.addWidget(self.transcript)

    # --- data --------------------------------------------------------------

    def show_comparison(self, rows: list[dict[str, Any]]) -> None:
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [
                str(row.get("model", "")),
                str(row.get("status", "")),
                format_duration(float(row.get("duration_s", 0.0))),
                str(int(row.get("tokens", 0) or 0)),
                format_money(float(row.get("llm_cost_usd", 0.0))),
                str(row.get("mcp_calls", 0)),
                str(row.get("tool_errors", 0)),
                str(row.get("scene_reset", "")),
                "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(row.get("id", "")))
                self.table.setItem(index, column, item)
        isolated = sum(1 for row in rows if row.get("scene_reset") == "verified")
        self.summary.setText(
            f"{len(rows)} run(s); {isolated} with a verified starting scene. "
            "Scores are yours — the studio does not rank models."
        )

    def show_transcript(self, lines: list[dict[str, Any]]) -> None:
        self.transcript.setPlainText(
            "\n\n".join(f"{entry.get('role', '')}: {entry.get('content', '')}" for entry in lines)
        )

    def _save_review(self) -> None:
        items = self.table.selectedItems()
        if not items:
            self.summary.setText("Select a run first.")
            return
        row = items[0].row()
        run_id_item = self.table.item(row, 0)
        run_id = run_id_item.data(Qt.ItemDataRole.UserRole) if run_id_item else ""
        if not run_id:
            self.summary.setText("That row has no stored run id.")
            return
        scores: dict[str, int | None] = {}
        for field, widget in self.spins.items():
            text = widget.text().strip()
            try:
                scores[field] = int(text) if text else None
            except ValueError:
                scores[field] = None
        self.review_requested.emit(str(run_id), {"notes": self.notes.text(), **scores})
        self.summary.setText("Score saved.")
