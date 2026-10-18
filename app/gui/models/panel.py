"""The Models panel: providers, keys, capabilities, and the model selector.

A model is a row, so this panel is a table of rows rather than a set of fields
per vendor. Adding an OpenAI-compatible endpoint, or a model this project has
never heard of, is typing a row — which is what makes "swap the model" a
configuration change rather than a code change.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox, QComboBox, QPlainTextEdit, QScrollArea, QTabWidget, QMessageBox,
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _paste_into(field: QLineEdit) -> None:
    """Put the clipboard in the field.

    A key field cannot rely on the keyboard or the platform's own menu: Qt
    drops the standard context menu for a password echo mode, and a shortcut
    that is swallowed once leaves no way in at all. Reading the clipboard is
    the one route that does not depend on how the field was drawn.
    """
    text = QApplication.clipboard().text().strip()
    if not text:
        return
    field.setText(text)
    field.setCursorPosition(len(text))


def _key_context_menu(field: QLineEdit, position: Any) -> None:
    """Right-click on a secret: paste in, but never copy out."""
    menu = QMenu(field)
    menu.addAction("Paste", lambda: _paste_into(field))
    menu.addSeparator()
    menu.addAction("Select all", field.selectAll)
    return menu.exec(field.mapToGlobal(position))


def _make_key_field(placeholder: str = "paste the key, then Save") -> QLineEdit:
    field = QLineEdit()
    field.setEchoMode(QLineEdit.EchoMode.Password)
    field.setPlaceholderText(placeholder)
    # The default menu is withheld for a password field, which is a sensible
    # default that leaves a secret with no way in but typing it out.
    field.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    field.customContextMenuRequested.connect(
        lambda position, target=field: _key_context_menu(target, position)
    )
    return field


MODEL_COLUMNS = ["Model", "Provider", "Tools", "Vision", "Context", "$/M in", "$/M out"]
PROVIDER_COLUMNS = ["Provider", "Kind", "Base URL", "Key", "Models", "Ready"]


class ModelsPanel(QWidget):
    model_selected = Signal(str)
    save_provider = Signal(str, str, str, str, str)
    save_model = Signal(str, str)
    save_3d = Signal(str)
    set_api_key = Signal(str, str)
    edit_provider = Signal(dict)
    edit_model = Signal(dict)
    remove_provider = Signal(str)
    remove_model = Signal(str, str)
    configure_3d = Signal(dict, str)

    def __init__(self, secrets_backend: str = "file", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._providers = {}
        self._models = {}
        self._provider_original = ""
        self._model_original = ""
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        model_page = self._page("Models")
        provider_page = self._page("Providers")
        three_page = self._page("3D providers")
        self.models_table = self._table(MODEL_COLUMNS, "models-table", model_page)
        self.models_table.itemSelectionChanged.connect(self._emit_selection)
        form = QFormLayout()
        model_page.addLayout(form)
        self.model_provider = QComboBox()
        self.existing_model = QLineEdit()
        self.model_tools = QCheckBox("Tool calling")
        self.model_vision = QCheckBox("Image input")
        self.model_context = QLineEdit("128000")
        self.model_input_price = QLineEdit("0")
        self.model_output_price = QLineEdit("0")
        self.model_default = QCheckBox("Default for this provider")
        self.model_default.setChecked(True)
        for label, field in [("Provider", self.model_provider), ("Exact model ID", self.existing_model), ("Capabilities", self.model_tools), ("", self.model_vision), ("Context tokens", self.model_context), ("Input USD / million tokens", self.model_input_price), ("Output USD / million tokens", self.model_output_price), ("", self.model_default)]:
            form.addRow(label, field)
        self.save_existing_model = QPushButton("Save model")
        self.save_existing_model.clicked.connect(self._save_model_editor)
        self.new_model = QPushButton("New model")
        self.new_model.clicked.connect(self._new_model)
        self.delete_model = QPushButton("Remove model")
        self.delete_model.clicked.connect(self._remove_model)
        self._buttons(model_page, self.new_model, self.save_existing_model, self.delete_model)
        self.providers_table = self._table(PROVIDER_COLUMNS, "providers-table", provider_page)
        self.providers_table.itemSelectionChanged.connect(self._provider_row_selected)
        form = QFormLayout()
        provider_page.addLayout(form)
        self.provider_name = QLineEdit()
        self.provider_kind = QLineEdit("openai-compatible")
        self.provider_kind.setToolTip("openai-compatible / openai / anthropic / gemini / mock")
        self.provider_base_url = QLineEdit()
        self.provider_base_url.setPlaceholderText("https://gateway.example/v1")
        self.provider_model = QLineEdit()
        self.provider_model.setPlaceholderText("Optional initial model; edit models on the Models tab")
        self.provider_enabled = QCheckBox("Enabled")
        self.provider_enabled.setChecked(True)
        self.provider_default = QComboBox()
        self.provider_options = QPlainTextEdit("{}")
        self.provider_options.setMaximumHeight(100)
        self.provider_key = _make_key_field("Blank keeps the saved key")
        for label, field in [("Name", self.provider_name), ("Kind", self.provider_kind), ("Base URL", self.provider_base_url), ("", self.provider_enabled), ("Default model", self.provider_default), ("Initial model ID (new provider)", self.provider_model), ("Options (JSON object)", self.provider_options), ("New API key", self.provider_key)]:
            form.addRow(label, field)
        self.new_provider = QPushButton("New provider")
        self.new_provider.clicked.connect(self._new_provider)
        self.add_provider = QPushButton("Save provider")
        self.add_provider.clicked.connect(self._save_provider_editor)
        self.delete_provider = QPushButton("Remove provider")
        self.delete_provider.clicked.connect(self._remove_provider)
        self._buttons(provider_page, self.new_provider, self.add_provider, self.delete_provider)
        self.key_target = QLabel("Select a provider to manage its saved key.")
        provider_page.addWidget(self.key_target)
        self.existing_key = _make_key_field("Replacement key; never displays the saved secret")
        provider_page.addWidget(self.existing_key)
        self.paste_existing_key = QPushButton("Paste key")
        self.paste_existing_key.clicked.connect(lambda: _paste_into(self.existing_key))
        self.save_existing_key = QPushButton("Save key")
        self.save_existing_key.clicked.connect(self._save_key)
        self.clear_existing_key = QPushButton("Delete saved key")
        self.clear_existing_key.clicked.connect(self._delete_key)
        self._buttons(provider_page, self.paste_existing_key, self.save_existing_key, self.clear_existing_key)
        form = QFormLayout()
        three_page.addLayout(form)
        self.three_d_enabled = QCheckBox("Offer 3D generation to the agent")
        self.three_d_provider = QLineEdit()
        self.three_d_options = QPlainTextEdit("{}")
        self.three_d_options.setMinimumHeight(180)
        self.three_d_key = _make_key_field("Blank keeps the saved key")
        for label, field in [("Enabled", self.three_d_enabled), ("Provider", self.three_d_provider), ("Options (JSON object)", self.three_d_options), ("New API key", self.three_d_key)]:
            form.addRow(label, field)
        self.save_three_d = QPushButton("Save 3D settings")
        self.save_three_d.clicked.connect(self._save_three_d)
        self.clear_three_d_key = QPushButton("Delete saved 3D key")
        self.clear_three_d_key.clicked.connect(self._delete_three_d_key)
        self._buttons(three_page, self.save_three_d, self.clear_three_d_key)
        three_page.addWidget(QLabel("Options include model, quality, texture, max_credits, poll_interval, poll_timeout and download_dir."))
        self.hint = QLabel(f"Keys are stored separately in {secrets_backend}. Select a row to edit; use New to add.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self._provider_row_selected()

    def _page(self, title):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return layout

    def _table(self, columns, name, layout):
        table = QTableWidget(0, len(columns))
        table.setObjectName(name)
        table.setHorizontalHeaderLabels(columns)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(150)
        layout.addWidget(table)
        return table

    def _buttons(self, layout, *buttons):
        row = QHBoxLayout()
        for button in buttons:
            row.addWidget(button)
        layout.addLayout(row)

    def _json(self, field):
        import json
        value = json.loads(field.toPlainText())
        if not isinstance(value, dict):
            raise ValueError("Options must be a JSON object.")
        return value

    def _confirm(self, text):
        return QMessageBox.question(self, "Confirm deletion", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _selected_provider(self):
        return self._provider_original

    def _new_provider(self):
        self.providers_table.clearSelection()
        self._provider_original = ""
        self.provider_name.setReadOnly(False)
        for field in (self.provider_name, self.provider_base_url, self.provider_model, self.provider_key, self.existing_key):
            field.clear()
        self.provider_kind.setText("openai-compatible")
        self.provider_enabled.setChecked(True)
        self.provider_default.clear()
        self.provider_options.setPlainText("{}")

    def _provider_row_selected(self):
        import json
        rows = self.providers_table.selectionModel().selectedRows()
        name = self.providers_table.item(rows[0].row(), 0).text() if rows else ""
        self._provider_original = name
        self.provider_key.clear()
        self.existing_key.clear()
        self.key_target.setText(name or "Select a provider to manage its saved key.")
        for button in (self.save_existing_key, self.clear_existing_key, self.paste_existing_key, self.delete_provider):
            button.setEnabled(bool(name))
        if not name:
            return
        entry = self._providers[name]
        self.provider_name.setText(name)
        self.provider_name.setReadOnly(True)
        self.provider_kind.setText(entry.get("kind", ""))
        self.provider_base_url.setText(entry.get("base_url", ""))
        self.provider_enabled.setChecked(entry.get("enabled", True))
        self.provider_model.clear()
        self.model_provider.setCurrentText(name)
        self.provider_default.clear()
        self.provider_default.addItems([m["id"] for m in entry.get("models", [])])
        self.provider_default.setCurrentText(entry.get("default_model", ""))
        self.provider_options.setPlainText(json.dumps(entry.get("options", {}), indent=2))

    def _save_provider_editor(self):
        try:
            payload = dict(original=self._provider_original, name=self.provider_name.text().strip(), kind=self.provider_kind.text().strip(), base_url=self.provider_base_url.text().strip(), enabled=self.provider_enabled.isChecked(), default_model=self.provider_default.currentText(), options=self._json(self.provider_options), api_key=self.provider_key.text().strip(), model_id=self.provider_model.text().strip())
            if not payload["name"]:
                raise ValueError("Enter a provider name.")
            self.edit_provider.emit(payload)
        except (ValueError, TypeError) as exc:
            self.set_hint(str(exc))

    def _new_model(self):
        self.models_table.clearSelection()
        self._model_original = ""
        self.existing_model.clear()
        self.model_provider.setEnabled(True)
        self.model_tools.setChecked(True)
        self.model_vision.setChecked(False)
        self.model_default.setChecked(True)
        self.model_context.setText("128000")
        self.model_input_price.setText("0")
        self.model_output_price.setText("0")
        self.model_default.setChecked(True)

    def _save_model_editor(self):
        try:
            payload = dict(original=self._model_original, provider=self.model_provider.currentText(), id=self.existing_model.text().strip(), supports_tools=self.model_tools.isChecked(), supports_vision=self.model_vision.isChecked(), context_window=int(self.model_context.text()), input_price=float(self.model_input_price.text()), output_price=float(self.model_output_price.text()), default=self.model_default.isChecked())
            if not payload["id"] or not payload["provider"]:
                raise ValueError("Choose a provider and enter the exact model ID.")
            if payload["context_window"] <= 0 or min(payload["input_price"], payload["output_price"]) < 0:
                raise ValueError("Context must be positive; prices must be nonnegative.")
            self.edit_model.emit(payload)
        except (ValueError, TypeError) as exc:
            self.set_hint(str(exc))

    def _remove_provider(self):
        if self._provider_original and self._confirm("Remove this provider and all its models? Its saved key is kept; delete the key separately if needed."):
            self.remove_provider.emit(self._provider_original)

    def _remove_model(self):
        if self._model_original and self._confirm("Remove the selected model? The provider default will be adjusted if necessary."):
            self.remove_model.emit(self.model_provider.currentText(), self._model_original)

    def _save_key(self):
        if self._selected_provider() and self.existing_key.text().strip():
            self.set_api_key.emit(self._selected_provider(), self.existing_key.text().strip())

    def _delete_key(self):
        if self._selected_provider() and self._confirm("Delete the saved API key? Environment credentials, if any, remain available."):
            self.set_api_key.emit(self._selected_provider(), "")

    def _delete_three_d_key(self):
        if self.three_d_provider.text().strip() and self._confirm("Delete the saved key for this 3D provider?"):
            self.set_api_key.emit(self.three_d_provider.text().strip(), "")

    def _save_three_d(self):
        try:
            config = self._json(self.three_d_options)
            config.update(enabled=self.three_d_enabled.isChecked(), provider=self.three_d_provider.text().strip())
            self.configure_3d.emit(config, self.three_d_key.text().strip())
        except (ValueError, TypeError) as exc:
            self.set_hint(str(exc))

    def show_three_d(self, config):
        import json
        self.three_d_enabled.setChecked(config.get("enabled", False))
        self.three_d_provider.setText(config.get("provider", ""))
        self.three_d_options.setPlainText(json.dumps({k: v for k, v in config.items() if k not in {"enabled", "provider"}}, indent=2))
        self.three_d_key.clear()

    def show_providers(self, providers):
        selected = self._provider_original
        self._providers = {entry["name"]: entry for entry in providers}
        self.providers_table.blockSignals(True)
        self.providers_table.setRowCount(len(providers))
        current = self.model_provider.currentText()
        self.model_provider.clear()
        self.model_provider.addItems(list(self._providers))
        self.model_provider.setCurrentText(current)
        for row, entry in enumerate(providers):
            values = [entry["name"], entry.get("kind", ""), entry.get("base_url", ""), "saved" if entry.get("key") else "—", len(entry.get("models", [])), "yes" if entry.get("ready") else "disabled / no key"]
            for column, value in enumerate(values):
                self.providers_table.setItem(row, column, QTableWidgetItem(str(value)))
            if entry["name"] == selected:
                self.providers_table.selectRow(row)
        self.providers_table.blockSignals(False)
        self._provider_row_selected()

    def show_models(self, models):
        self._models = {f"{m['provider']}:{m['id']}": m for m in models}
        self.models_table.blockSignals(True)
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            values = [model.get(k, "") for k in ("id", "provider", "supports_tools", "supports_vision", "context_window", "input_price", "output_price")]
            values[2:4] = ["yes" if model.get(k) else "no" for k in ("supports_tools", "supports_vision")]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, f"{model['provider']}:{model['id']}")
                self.models_table.setItem(row, column, item)
        self.models_table.blockSignals(False)

    def set_hint(self, text):
        self.hint.setText(text)

    def _emit_selection(self):
        rows = self.models_table.selectionModel().selectedRows()
        if not rows:
            return
        chosen = self.models_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        model = self._models[chosen]
        self._model_original = model["id"]
        self.model_provider.setCurrentText(model["provider"])
        self.model_provider.setEnabled(False)
        self.existing_model.setText(model["id"])
        self.model_tools.setChecked(model.get("supports_tools", False))
        self.model_vision.setChecked(model.get("supports_vision", False))
        self.model_context.setText(str(model.get("context_window", 128000)))
        self.model_input_price.setText(str(model.get("input_price", 0)))
        self.model_output_price.setText(str(model.get("output_price", 0)))
        self.model_default.setChecked(self._providers.get(model["provider"], {}).get("default_model") == model["id"])
        self.model_selected.emit(chosen)

    def select_first(self):
        if self.models_table.rowCount():
            self.models_table.selectRow(0)
