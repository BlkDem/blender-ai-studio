"""The Scene panel: what Blender is showing, read through MCP.

Every field here comes from a tool or a resource, never from a Blender API. That is
the whole reason the panel can be trusted to describe what the agent is looking
at: it is asking the same server the agent asks.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.bridge import elide, format_clock


class ScenePanel(QWidget):
    """Connection, scene summary, and the object list."""

    #: Declared on the class, not assigned in __init__: PySide6 binds a signal
    #: when it is read off the instance, and an instance attribute would shadow
    #: that with the raw Signal object, which has no ``connect``.
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene: dict[str, Any] = {}
        self._objects: list[dict[str, Any]] = []

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.refresh = QPushButton("Refresh")
        self.refresh.setObjectName("scene-refresh")
        self.refresh.clicked.connect(self.refresh_requested)
        row.addWidget(self.refresh)
        row.addStretch(1)
        layout.addLayout(row)

        summary = QGroupBox("Scene")
        form = QFormLayout(summary)
        self.fields: dict[str, QLabel] = {}
        for key, label in (
            ("connection", "Connection"),
            ("scene", "Scene"),
            ("objects", "Objects"),
            ("active_object", "Active object"),
            ("camera", "Camera"),
            ("engine", "Render engine"),
            ("frame", "Frame"),
            ("updated", "Updated"),
        ):
            value = QLabel("—")
            value.setObjectName(f"scene-{key}")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.fields[key] = value
            form.addRow(label, value)
        layout.addWidget(summary)

        self.tree = QTreeWidget()
        self.tree.setObjectName("scene-objects")
        self.tree.setHeaderLabels(["Object", "Type", "Location", "Dimensions"])
        self.tree.setRootIsDecorated(False)
        layout.addWidget(self.tree, 1)

    # --- data --------------------------------------------------------------

    def show_scene(self, payload: dict[str, Any], connection: str = "connected") -> None:
        self._scene = payload or {}
        objects = self._scene.get("objects", [])
        self._objects = objects if isinstance(objects, list) else []
        self.fields["connection"].setText(connection)
        self.fields["scene"].setText(str(self._scene.get("scene", "—")))
        self.fields["objects"].setText(
            f"{self._scene.get('objects_count', len(self._objects))}"
            + (
                f" (showing {self._scene.get('objects_shown', len(self._objects))},"
                f" truncated={self._scene.get('objects_truncated', False)})"
                if self._scene.get("objects_truncated")
                else ""
            )
        )
        self.fields["active_object"].setText(str(self._scene.get("active_object") or "—"))
        self.fields["camera"].setText(str(self._scene.get("active_camera") or "—"))
        self.fields["engine"].setText(str(self._scene.get("render_engine") or "—"))
        self.fields["frame"].setText(str(self._scene.get("frame", "—")))
        self.fields["updated"].setText(format_clock(self._scene.get("_read_at", 0.0)) if self._scene.get("_read_at") else "—")

        self.tree.clear()
        for entry in self._objects:
            location = entry.get("location") or []
            item = QTreeWidgetItem(
                [
                    str(entry.get("name", "")),
                    str(entry.get("type", "")),
                    ", ".join(f"{float(v):.2f}" for v in location) if location else "—",
                    elide(json.dumps(entry.get("dimensions") or []), 28),
                ]
            )
            item.setToolTip(0, json.dumps(entry, indent=2, default=str)[:2000])
            self.tree.addTopLevelItem(item)

    def show_error(self, message: str) -> None:
        self.fields["connection"].setText(f"error: {elide(message, 60)}")

    def object_count(self) -> int:
        return len(self._objects)
