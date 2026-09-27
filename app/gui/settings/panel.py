"""Settings: the things a person changes, and the ones they rarely do.

General and Blender settings are plain fields because they are the ones people
touch. The agent's limits are here too, next to a plain statement of what each
one is for, because a limit that stops a run should be one the user chose and can
find again.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsPanel(QWidget):
    """Writes dotted settings keys, which is what the store reads back."""

    setting_changed = Signal(str, object)  # "agent.max_steps", 30
    provider_changed = Signal(str, dict)  # name, fields
    mcp_changed = Signal(dict)  # command, args, cwd, port

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        general = QGroupBox("General")
        general_form = QFormLayout(general)
        self.theme = QLineEdit("system")
        self.language = QLineEdit("en")
        self.log_level = QLineEdit("INFO")
        for label, widget in (
            ("Theme", self.theme),
            ("Language", self.language),
            ("Log level", self.log_level),
        ):
            widget.editingFinished.connect(
                lambda w=widget, key=label.lower().replace(" ", "_"): self.setting_changed.emit(key, w.text())
            )
            general_form.addRow(label, widget)
        layout.addWidget(general)

        blender = QGroupBox("Blender (the MCP server)")
        blender_form = QFormLayout(blender)
        self.mcp_command = QLineEdit()
        self.mcp_command.setPlaceholderText("/path/to/blender-mcp/.venv/bin/python")
        self.mcp_args = QLineEdit("-m server.main")
        self.mcp_cwd = QLineEdit()
        self.mcp_cwd.setPlaceholderText("/path/to/blender-mcp")
        self.mcp_port = QSpinBox()
        self.mcp_port.setRange(1, 65535)
        self.mcp_port.setValue(8765)
        self.mcp_timeout = QDoubleSpinBox()
        self.mcp_timeout.setRange(1.0, 600.0)
        self.mcp_timeout.setValue(30.0)
        for label, widget in (
            ("Python", self.mcp_command),
            ("Arguments", self.mcp_args),
            ("Working directory", self.mcp_cwd),
            ("Bridge port", self.mcp_port),
            ("Connect timeout (s)", self.mcp_timeout),
        ):
            blender_form.addRow(label, widget)
        self.save_mcp = QPushButton("Save Blender settings")
        self.save_mcp.clicked.connect(self._save_mcp)
        blender_form.addRow("", self.save_mcp)
        layout.addWidget(blender)

        agent = QGroupBox("Agent")
        agent_form = QFormLayout(agent)
        self.system_prompt = QLineEdit()
        self.system_prompt.setPlaceholderText("Extra instructions for the model (optional)")
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 200)
        self.max_steps.setValue(30)
        self.max_tool_calls = QSpinBox()
        self.max_tool_calls.setRange(1, 500)
        self.max_tool_calls.setValue(50)
        self.max_seconds = QDoubleSpinBox()
        self.max_seconds.setRange(10.0, 7200.0)
        self.max_seconds.setValue(900.0)
        self.max_session_cost = QDoubleSpinBox()
        self.max_session_cost.setRange(0.01, 1000.0)
        self.max_session_cost.setDecimals(2)
        self.max_session_cost.setValue(5.0)
        self.max_3d_credits = QSpinBox()
        self.max_3d_credits.setRange(0, 100000)
        self.max_3d_credits.setValue(200)
        self.allow_execute_python = QCheckBox("allow blender.execute_python")
        self.allow_execute_python.setToolTip(
            "Off by default. It is also how a generated 3D asset gets imported."
        )
        for label, widget, key in (
            ("System prompt", self.system_prompt, "agent.system_prompt"),
            ("Max agent steps", self.max_steps, "agent.max_steps"),
            ("Max tool calls", self.max_tool_calls, "agent.max_tool_calls"),
            ("Max seconds", self.max_seconds, "agent.max_seconds"),
            ("Max session cost ($)", self.max_session_cost, "agent.max_session_cost"),
            ("Max 3D credits", self.max_3d_credits, "agent.max_3d_credits"),
        ):
            agent_form.addRow(label, widget)
            connect = widget.editingFinished if isinstance(widget, QLineEdit) else widget.valueChanged
            connect.connect(
                lambda _=None, w=widget, k=key: self.setting_changed.emit(
                    k, w.value() if not isinstance(w, QLineEdit) else w.text()
                )
            )
        self.allow_execute_python.toggled.connect(
            lambda checked: self.setting_changed.emit("agent.allow_execute_python", checked)
        )
        agent_form.addRow("", self.allow_execute_python)
        layout.addWidget(agent)

        note = QLabel(
            "A run that reaches a limit stops with a reason in the chat, and the usage is still recorded."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)
        layout.addStretch(1)

    def _save_mcp(self) -> None:
        self.mcp_changed.emit(
            {
                "command": self.mcp_command.text().strip(),
                "args": self.mcp_args.text().split(),
                "cwd": self.mcp_cwd.text().strip() or None,
                "blender_port": self.mcp_port.value(),
                "connect_timeout": self.mcp_timeout.value(),
            }
        )

    def load(self, settings: Any) -> None:
        """Fill the fields from the settings object."""
        self.theme.setText(settings.theme)
        self.language.setText(settings.language)
        self.log_level.setText(settings.log_level)
        server = settings.mcp_servers[0] if settings.mcp_servers else None
        if server is not None:
            self.mcp_command.setText(server.command)
            self.mcp_args.setText(" ".join(server.args))
            self.mcp_cwd.setText(server.cwd or "")
            self.mcp_port.setValue(server.blender_port)
            self.mcp_timeout.setValue(server.connect_timeout)
        agent = settings.agent
        self.system_prompt.setText(agent.system_prompt)
        self.max_steps.setValue(agent.max_steps)
        self.max_tool_calls.setValue(agent.max_tool_calls)
        self.max_seconds.setValue(agent.max_seconds)
        self.max_session_cost.setValue(agent.max_session_cost)
        self.max_3d_credits.setValue(agent.max_3d_credits)
        self.allow_execute_python.setChecked(agent.allow_execute_python)
