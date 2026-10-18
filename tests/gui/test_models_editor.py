"""Editable model catalogue: selection, secrets and destructive actions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QScrollArea
from app.gui.models.panel import ModelsPanel

pytestmark = pytest.mark.gui


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = ModelsPanel()
    widget.show_providers([dict(name="gateway", kind="openai-compatible", base_url="https://example.org/v1", enabled=False, default_model="exact", models=[{"id": "exact"}], options={"custom": True}, key="masked")])
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


async def test_backend_model_update_preserves_metadata_and_removal_default(window):
    registry = window.context.llm
    config = registry.config("scripted")
    original = config.models[0]["id"]
    config.models[0]["custom_metadata"] = "keep"
    payload = dict(provider="scripted", original=original, id="edited", supports_tools=True, supports_vision=True, context_window=32768, input_price=0.5, output_price=1.5, default=True)
    assert await window._persist_model_editor(payload) == "scripted:edited"
    saved = registry.config("scripted")
    assert saved.models[0]["custom_metadata"] == "keep"
    assert saved.default_model == "edited"
    await window._persist_model_editor(dict(provider="scripted", original="edited", id="edited"), remove=True)
    assert registry.config("scripted").default_model == ""
    stored = await window.context.studio.settings.get("llm_providers")
    assert next(c for c in stored if c["name"] == "scripted")["models"] == []


async def test_backend_three_d_persists_without_forcing_tripo(window):
    await window._persist_three_d(dict(enabled=True, provider="mock", quality="high", poll_interval=1, poll_timeout=30), "")
    assert window.context.settings.three_d.provider == "mock"
    assert window.context.three_d.config.provider == "mock"
    stored = await window.context.studio.settings.get("three_d")
    assert stored["enabled"] and stored["quality"] == "high"
    assert "api_key" not in stored


def test_tabs_are_scrollable_and_selection_populates_provider(panel):
    assert [panel.tabs.tabText(i) for i in range(3)] == ["Models", "Providers", "3D providers"]
    assert all(isinstance(panel.tabs.widget(i), QScrollArea) for i in range(3))
    panel.providers_table.selectRow(0)
    assert panel.provider_name.text() == "gateway"
    assert not panel.provider_enabled.isChecked()
    assert panel.provider_default.currentText() == "exact"
    assert panel.provider_key.text() == panel.existing_key.text() == ""
    captured = []
    panel.edit_provider.connect(captured.append)
    panel.add_provider.click()
    assert captured[0]["original"] == "gateway"
    assert captured[0]["options"] == {"custom": True}
    assert captured[0]["api_key"] == ""


def test_model_selection_edit_and_new_are_distinct(panel):
    panel.show_models([dict(provider="gateway", id="exact", supports_tools=True, supports_vision=True, context_window=64000, input_price=1.25, output_price=3)])
    panel.models_table.selectRow(0)
    assert panel.existing_model.text() == "exact"
    assert panel.model_tools.isChecked() and panel.model_vision.isChecked()
    assert panel.model_default.isChecked()
    captured = []
    panel.edit_model.connect(captured.append)
    panel.existing_model.setText("renamed")
    panel.save_existing_model.click()
    assert captured[0]["original"] == "exact"
    panel.new_model.click()
    assert panel._model_original == ""
    assert panel.model_provider.isEnabled()


def test_blank_key_is_not_deletion_and_delete_requires_confirmation(panel, monkeypatch):
    panel.providers_table.selectRow(0)
    captured = []
    panel.set_api_key.connect(lambda *args: captured.append(args))
    panel.save_existing_key.click()
    assert not captured
    monkeypatch.setattr(panel, "_confirm", lambda text: False)
    panel.clear_existing_key.click()
    assert not captured
    monkeypatch.setattr(panel, "_confirm", lambda text: True)
    panel.clear_existing_key.click()
    assert captured == [("gateway", "")]


def test_three_d_uses_configured_provider_and_preserves_options(panel):
    panel.show_three_d(dict(enabled=True, provider="mock", model="custom", quality="high", poll_timeout=45))
    captured = []
    panel.configure_3d.connect(lambda *args: captured.append(args))
    panel.save_three_d.click()
    assert captured == [({"enabled": True, "provider": "mock", "model": "custom", "quality": "high", "poll_timeout": 45}, "")]
