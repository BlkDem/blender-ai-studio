"""Projects: the thing a studio session is about.

A project is a named piece of work with a starting ``.blend`` and a model, and
everything the studio records for it -- conversations, 3D generations, what they
cost -- is filed underneath. The storage for this was complete from the start and
nothing in the application ever called it, so a studio session was a stream of
unrelated turns that could not be told apart six weeks later.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("Name", "Starting .blend", "Default model", "Created")


class ProjectsPanel(QWidget):
    """The projects in this studio, and which one is open."""

    create_requested = Signal(str, str)  # name, starting .blend
    opened = Signal(str)  # project id
    closed = Signal()
    delete_requested = Signal(str)  # project id
    blend_requested = Signal(str, str)  # project id, path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.current = QLabel("No project — this session is not filed anywhere.")
        self.current.setObjectName("projects-current")
        self.current.setWordWrap(True)
        layout.addWidget(self.current)

        create_row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("New project name")
        self.blend = QLineEdit()
        self.blend.setPlaceholderText("starting .blend, if the project has one")
        create_row.addWidget(self.name)
        create_row.addWidget(self.blend)
        layout.addLayout(create_row)

        buttons = QHBoxLayout()
        self.create_button = QPushButton("Create")
        self.create_button.clicked.connect(self._create_requested)
        self.open_button = QPushButton("Open selected")
        self.open_button.clicked.connect(self._open_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected)
        self.close_button = QPushButton("Close project")
        self.close_button.clicked.connect(self.closed.emit)
        self.load_button = QPushButton("Load starting .blend")
        self.load_button.setToolTip(
            "Open the project's file in Blender, from a copy -- the original is never touched"
        )
        self.load_button.clicked.connect(self._load_selected_blend)
        for button in (
            self.create_button,
            self.open_button,
            self.delete_button,
            self.close_button,
            self.load_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.list = QListWidget()
        self.list.setObjectName("projects-list")
        self.list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.list, 1)

        self._rows: dict[str, int] = {}
        self._blends: dict[str, str] = {}

    # --- filling in --------------------------------------------------------

    def show_projects(self, projects: list[Any], current: str | None) -> None:
        """The list, with the open project marked.

        Files are the reason a project exists, so the list says which ``.blend``
        each one starts from rather than showing a name and nothing else.
        """
        self.list.clear()
        self._rows = {}
        self._blends = {p.id: (p.initial_blend or "") for p in projects}
        for index, project in enumerate(projects):
            item = QListWidgetItem(
                f"{project.name}"
                f"    · {Path(project.initial_blend).name if project.initial_blend else 'no starting file'}"
                f"    · {project.default_model or 'any model'}"
            )
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            if project.id == current:
                item.setText(f"● {item.text()}")
            self.list.addItem(item)
            self._rows[project.id] = index
        self._say_current(current, projects)
        self.load_button.setEnabled(self.selected_blend() != "")

    def _say_current(self, current: str | None, projects: list[Any]) -> None:
        if not current:
            self.current.setText("No project — this session is not filed anywhere.")
            return
        project = next((p for p in projects if p.id == current), None)
        if project is None:
            self.current.setText("No project — this session is not filed anywhere.")
            return
        self.current.setText(
            f"Open: {project.name}"
            + (f" · starts from {project.initial_blend}" if project.initial_blend else "")
        )

    def selected_id(self) -> str:
        item = self.list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else ""

    def requested_name(self) -> str:
        return self.name.text().strip()

    def requested_blend(self) -> str:
        return self.blend.text().strip()

    def clear_inputs(self) -> None:
        self.name.clear()
        self.blend.clear()

    # --- signals -----------------------------------------------------------

    def _create_requested(self) -> None:
        if self.requested_name():
            self.create_requested.emit(self.requested_name(), self.requested_blend())

    def _open_selected(self) -> None:
        project_id = self.selected_id()
        if project_id:
            self.opened.emit(project_id)

    def _delete_selected(self) -> None:
        project_id = self.selected_id()
        if project_id:
            self.delete_requested.emit(project_id)

    def _selection_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        has = current is not None
        self.open_button.setEnabled(has)
        self.delete_button.setEnabled(has)
        self.load_button.setEnabled(has and self.selected_blend() != "")

    def selected_blend(self) -> str:
        """The starting file of the highlighted project, if it has one."""
        project_id = self.selected_id()
        return str(self._blends.get(project_id, ""))

    def _load_selected_blend(self) -> None:
        project_id = self.selected_id()
        blend = self.selected_blend()
        if project_id and blend:
            self.blend_requested.emit(project_id, blend)
