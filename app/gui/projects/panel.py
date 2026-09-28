"""Projects: the thing a studio session is about.

A project is a named piece of work, and it is a workspace rather than a label:
it has the file the work happens in, the model the work is done with, and the
conversation it is carried on in. Opening one gives you all three, and everything
after that is filed under it.

The storage for this was complete from the start and the window never called it,
so a studio session was a stream of unrelated turns that could not be told apart
six weeks later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProjectsPanel(QWidget):
    """The projects in this studio, and which one is open."""

    create_requested = Signal(str, str)  # name, starting .blend
    opened = Signal(str)  # project id
    closed = Signal()
    delete_requested = Signal(str)  # project id
    blend_requested = Signal(str, str)  # project id, path
    rename_requested = Signal(str, str)  # project id, new name

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
        self.blend.returnPressed.connect(self._create_requested)
        create_row.addWidget(self.name)
        create_row.addWidget(self.blend)
        layout.addLayout(create_row)

        buttons = QHBoxLayout()
        self.create_button = QPushButton("Create")
        self.create_button.setToolTip(
            "Start a project. It takes over the model, the file and the transcript."
        )
        self.create_button.clicked.connect(self._create_requested)
        self.open_button = QPushButton("Open selected")
        self.open_button.setToolTip("Make this the project: its file, its model, its conversation")
        self.open_button.clicked.connect(self._open_selected)
        self.rename_button = QPushButton("Rename")
        self.rename_button.setToolTip("Give the project a different name")
        self.rename_button.clicked.connect(self._rename_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete the project and every conversation filed under it")
        self.delete_button.clicked.connect(self._delete_selected)
        self.close_button = QPushButton("Close project")
        self.close_button.clicked.connect(self.closed.emit)
        self.load_button = QPushButton("Open working file")
        self.load_button.setToolTip(
            "Put Blender on the project's own copy of the file. The starting file is "
            "never opened directly and never modified."
        )
        self.load_button.clicked.connect(self._load_selected_blend)
        for button in (
            self.create_button,
            self.open_button,
            self.rename_button,
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
        self.list.itemDoubleClicked.connect(lambda _item: self._open_selected())
        layout.addWidget(self.list, 1)

        self._names: dict[str, str] = {}
        self._blends: dict[str, str] = {}

    # --- filling in --------------------------------------------------------

    def show_projects(self, projects: list[Any], current: str | None) -> None:
        """The list, with the open project marked.

        Files are the reason a project exists, so the list says which ``.blend``
        each one starts from rather than showing a name and nothing else.

        The highlight survives the rebuild. Refilling the list clears it, and
        this panel is refilled after every action that touches a project -- so
        without this, Delete and Rename stop working the moment anything else
        happens, which reads as a button that is broken rather than a selection
        that went away.
        """
        highlighted = self.selected_id()
        self.list.clear()
        self._names = {p.id: p.name for p in projects}
        self._blends = {p.id: (p.initial_blend or "") for p in projects}
        for project in projects:
            item = QListWidgetItem(self._row_text(project))
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            if project.id == current:
                item.setText(f"● {item.text()}")
            self.list.addItem(item)
        if highlighted:
            for row in range(self.list.count()):
                if self.list.item(row).data(Qt.ItemDataRole.UserRole) == highlighted:
                    self.list.setCurrentRow(row)
                    break
        self._say_current(current, projects)
        self._selection_changed(self.list.currentItem(), None)

    @staticmethod
    def _row_text(project: Any) -> str:
        where = project.workspace or project.initial_blend
        return (
            f"{project.name}"
            f"    · {Path(where).name if where else 'no file'}"
            f"    · {project.default_model or 'any model'}"
        )

    def _say_current(self, current: str | None, projects: list[Any]) -> None:
        if not current:
            self.current.setText("No project — this session is not filed anywhere.")
            return
        project = next((p for p in projects if p.id == current), None)
        if project is None:
            self.current.setText("No project — this session is not filed anywhere.")
            return
        parts = [f"Open: {project.name}"]
        if project.workspace:
            parts.append(f"working on {project.workspace}")
        elif project.initial_blend:
            parts.append(f"starts from {project.initial_blend}")
        self.current.setText(" · ".join(parts))

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
        if project_id and self.confirm_delete(self._names.get(project_id, "")):
            self.delete_requested.emit(project_id)

    def _rename_selected(self) -> None:
        """A name is the one thing about a project a person is likely to regret."""
        project_id = self.selected_id()
        if not project_id:
            return
        renamed, accepted = QInputDialog.getText(
            self, "Rename project", "Name:", text=self._names.get(project_id, "")
        )
        renamed = renamed.strip()
        if accepted and renamed and renamed != self._names.get(project_id):
            self.rename_requested.emit(project_id, renamed)

    def confirm_delete(self, name: str) -> bool:
        """Ask before the cascade, which cannot be undone.

        Deleting a project deletes every conversation filed under it, and the
        window says so only afterwards. The question has to come first, or the
        warning is a footnote to something that already happened.
        """
        answer = QMessageBox.question(
            self,
            "Delete project",
            f"Delete '{name}'?\n\nEvery conversation filed under it goes with it, "
            "and they cannot be brought back.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _selection_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        has = current is not None
        self.open_button.setEnabled(has)
        self.rename_button.setEnabled(has)
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
