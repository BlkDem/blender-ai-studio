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
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MODEL_COLUMNS = ["Model", "Provider", "Tools", "Vision", "Context", "$/M in", "$/M out"]

PROVIDER_COLUMNS = ["Provider", "Kind", "Base URL", "Key", "Models", "Ready"]


class ModelsPanel(QWidget):
    """Everything about which model the agent will use."""

    model_selected = Signal(str)  # "provider:model"
    save_provider = Signal(str, str, str, str)  # name, kind, base_url, api key
    save_3d = Signal(str)  # provider name
    set_api_key = Signal(str, str)  # provider, key

    def __init__(self, secrets_backend: str = "file", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._secrets_backend = secrets_backend

        layout = QVBoxLayout(self)

        providers = QGroupBox("LLM providers")
        providers_form = QFormLayout(providers)
        self.provider_name = QLineEdit()
        self.provider_name.setPlaceholderText("space-bunny")
        self.provider_kind = QLineEdit("openai-compatible")
        self.provider_base_url = QLineEdit()
        self.provider_base_url.setPlaceholderText("https://gateway.example/v1")
        self.provider_key = QLineEdit()
        self.provider_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.provider_key.setPlaceholderText(f"stored in: {secrets_backend}")
        self.add_provider = QPushButton("Add provider")
        self.add_provider.clicked.connect(
            lambda: self.save_provider.emit(
                self.provider_name.text().strip(),
                self.provider_kind.text().strip() or "openai-compatible",
                self.provider_base_url.text().strip(),
                self.provider_key.text(),
            )
        )
        providers_form.addRow("Name", self.provider_name)
        providers_form.addRow("Kind", self.provider_kind)
        providers_form.addRow("Base URL", self.provider_base_url)
        providers_form.addRow("API key", self.provider_key)
        providers_form.addRow("", self.add_provider)
        layout.addWidget(providers)

        self.providers_table = QTableWidget(0, len(PROVIDER_COLUMNS))
        self.providers_table.setObjectName("providers-table")
        self.providers_table.setHorizontalHeaderLabels(PROVIDER_COLUMNS)
        self.providers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.providers_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.providers_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.providers_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.providers_table.itemSelectionChanged.connect(self._provider_row_selected)
        layout.addWidget(self.providers_table)

        # A key for a provider that is already configured. "Add provider" is the
        # other way to set one, and it replaces the whole configuration -- the
        # model list, the default model -- with a placeholder. So adding a key
        # used to mean risking the provider it was meant to enable.
        #
        # The provider is whichever row is selected in the table above, so the
        # key lands on the row the window is pointing at and needs no second
        # list of names to keep in step with the first.
        existing = QGroupBox("API key for the selected provider")
        existing_form = QFormLayout(existing)
        self.key_target = QLabel("Select a provider in the table above.")
        self.existing_key = QLineEdit()
        self.existing_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.existing_key.setPlaceholderText("paste the key, then Save")
        self.save_existing_key = QPushButton("Save key")
        self.save_existing_key.setEnabled(False)
        self.save_existing_key.clicked.connect(
            lambda: self.set_api_key.emit(self._selected_provider(), self.existing_key.text())
        )
        self.clear_existing_key = QPushButton("Clear key")
        self.clear_existing_key.setEnabled(False)
        self.clear_existing_key.clicked.connect(lambda: self.set_api_key.emit(self._selected_provider(), ""))
        existing_form.addRow("Saving for", self.key_target)
        existing_form.addRow("API key", self.existing_key)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_existing_key)
        buttons.addWidget(self.clear_existing_key)
        existing_form.addRow("", buttons)
        layout.addWidget(existing)

        three_d = QGroupBox("3D provider")
        three_d_form = QFormLayout(three_d)
        self.three_d_provider = QLineEdit("tripo")
        self.three_d_key = QLineEdit()
        self.three_d_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.save_three_d = QPushButton("Save 3D settings")
        self.save_three_d.clicked.connect(
            lambda: self.set_api_key.emit(self.three_d_provider.text().strip(), self.three_d_key.text())
        )
        three_d_form.addRow("Provider", self.three_d_provider)
        three_d_form.addRow("API key", self.three_d_key)
        three_d_form.addRow("", self.save_three_d)
        layout.addWidget(three_d)

        models_box = QGroupBox("Models")
        models_layout = QVBoxLayout(models_box)
        self.models_table = QTableWidget(0, len(MODEL_COLUMNS))
        self.models_table.setObjectName("models-table")
        self.models_table.setHorizontalHeaderLabels(MODEL_COLUMNS)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.models_table.itemSelectionChanged.connect(self._emit_selection)
        models_layout.addWidget(self.models_table)
        layout.addWidget(models_box, 1)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.hint)

    # --- data --------------------------------------------------------------

    def _selected_provider(self) -> str:
        rows = self.providers_table.selectionModel().selectedRows()
        if not rows:
            return ""
        item = self.providers_table.item(rows[0].row(), 0)
        return item.text().strip() if item else ""

    def _provider_row_selected(self) -> None:
        """Name the provider the key will go to, before it goes there.

        A key saved against the wrong provider is a secret in the wrong place,
        and the field is empty on every selection so an old key cannot be
        re-saved over a new one.
        """
        name = self._selected_provider()
        self.existing_key.clear()
        self.key_target.setText(name or "Select a provider in the table above.")
        self.save_existing_key.setEnabled(bool(name))
        self.clear_existing_key.setEnabled(bool(name))

    def show_providers(self, providers: list[dict[str, Any]]) -> None:
        self.providers_table.setRowCount(len(providers))
        for row, entry in enumerate(providers):
            values = [
                entry.get("name", ""),
                entry.get("kind", ""),
                entry.get("base_url", "") or "(default)",
                entry.get("key", "") or "—",
                str(len(entry.get("models") or [])),
                "yes" if entry.get("ready") else "no key",
            ]
            for column, value in enumerate(values):
                self.providers_table.setItem(row, column, QTableWidgetItem(str(value)))

    def show_models(self, models: list[dict[str, Any]]) -> None:
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            values = [
                model.get("id", ""),
                model.get("provider", ""),
                "yes" if model.get("supports_tools") else "no",
                "yes" if model.get("supports_vision") else "no",
                f"{int(model.get('context_window', 0)) // 1000}k",
                f"{float(model.get('input_price', 0)):.2f}",
                f"{float(model.get('output_price', 0)):.2f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, f"{model.get('provider')}:{model.get('id')}")
                self.models_table.setItem(row, column, item)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)

    def _emit_selection(self) -> None:
        items = self.models_table.selectedItems()
        if not items:
            return
        column = items[0].column()
        if column != 0:
            items = [item for item in items if item.column() == 0]
            if not items:
                return
        chosen = items[0].data(Qt.ItemDataRole.UserRole)
        if chosen:
            self.model_selected.emit(str(chosen))

    def select_first(self) -> None:
        if self.models_table.rowCount():
            self.models_table.selectRow(0)
