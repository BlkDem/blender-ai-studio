"""Remembering which model was in use.

A preference, not configuration: it belongs in the settings table rather than in
the .env, because a file the studio rewrites on its own is a file the user stops
trusting. And it has to be *reported* when it cannot be honoured, because landing
quietly on a different model makes a working preference look broken.
"""

from __future__ import annotations

import asyncio

import pytest

from app.storage.repositories import Studio


async def _drain(qapp, predicate, rounds: int = 120) -> bool:
    for _ in range(rounds):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if predicate():
            return True
    return False


# --- the repository --------------------------------------------------------


async def test_a_setting_round_trips(studio: Studio) -> None:
    await studio.settings.set("ui.last_model", "openai:gpt-5-mini")
    assert await studio.settings.get("ui.last_model") == "openai:gpt-5-mini"


async def test_an_unset_setting_is_the_default_not_an_error(studio: Studio) -> None:
    assert await studio.settings.get("ui.last_model") is None
    assert await studio.settings.get("ui.last_model", "fallback") == "fallback"


async def test_setting_a_value_twice_keeps_the_last(studio: Studio) -> None:
    await studio.settings.set("ui.last_model", "a:one")
    await studio.settings.set("ui.last_model", "b:two")
    assert await studio.settings.get("ui.last_model") == "b:two"


# --- the window ------------------------------------------------------------


def _settings(database_path, models: list[str]) -> object:
    from app.core.settings import MCPServerConfig, Settings
    from app.llm.registry import ProviderConfig

    settings = Settings(data_dir=database_path.parent)
    settings.mcp_servers = [MCPServerConfig(name="Blender MCP")]
    settings.llm_providers = [
        ProviderConfig(
            name="scripted",
            kind="mock",
            default_model=models[0],
            models=[{"id": model, "supports_tools": True} for model in models],
        )
    ]
    return settings


async def _launch(qapp, database_path, *, remembered: str = "", models=None):
    """A started window, with ``remembered`` already in the settings table.

    Written as a function rather than a fixture because a test has to be able to
    plant the preference *before* start-up reads it -- which is the only order
    that exercises the real path.
    """
    from app.core.context import AppContext
    from app.gui.bridge import CoreThread
    from app.gui.main_window import MainWindow

    settings = _settings(database_path, models or ["alpha", "beta"])
    context = await AppContext(settings=settings).open()
    if remembered:
        await context.studio.settings.set("ui.last_model", remembered)
    core = CoreThread(context)
    core.start()
    window = MainWindow(context, core)
    window.show()
    window.start()
    await _drain(qapp, lambda: window.model_selector.count() > 0)
    return window, context, core


async def _close(window, context, core, qapp) -> None:
    await context.close()
    window.close()
    window.deleteLater()
    core.stop()
    qapp.processEvents()


@pytest.fixture
async def two_model_window(qapp, database_path):
    """A window with two models on one provider, so switching is possible."""
    window, context, core = await _launch(qapp, database_path)
    try:
        yield window
    finally:
        await _close(window, context, core, qapp)


async def test_the_model_in_use_is_remembered(two_model_window, qapp) -> None:
    window = two_model_window
    assert window.model_selector.count() == 2
    window.model_selector.setCurrentIndex(1)
    assert await _drain(qapp, lambda: "beta" in window.current_model())
    stored = await window.context.studio.settings.get("ui.last_model")
    assert stored == "scripted:beta", stored


async def test_a_reopened_window_starts_on_the_remembered_model(qapp, database_path) -> None:
    window, context, core = await _launch(qapp, database_path)
    window.model_selector.setCurrentIndex(1)
    assert await _drain(qapp, lambda: "beta" in window.current_model())
    await _close(window, context, core, qapp)

    # A second launch on the same data directory: the model should still be the
    # one that was in use, not the first in the list.
    second, second_context, second_core = await _launch(qapp, database_path)
    try:
        assert second.current_model() == "beta", second.current_model()
        assert "beta" in second.llm_status.text(), second.llm_status.text()
    finally:
        await _close(second, second_context, second_core, qapp)


async def test_a_remembered_model_that_is_gone_is_said_out_loud(qapp, database_path) -> None:
    """Silently landing on another model is how a preference comes to look broken
    rather than gone."""
    window, context, core = await _launch(qapp, database_path, remembered="scripted:removed")
    try:
        assert window.current_model() == "alpha", "a usable model is still chosen"
        assert "scripted · removed" in window.chat.transcript_text()
    finally:
        await _close(window, context, core, qapp)


async def test_a_first_run_never_invents_a_last_used_model(qapp, database_path) -> None:
    """On a first run there is no last-used model. Writing "the first one in the
    list" would fabricate a preference and then faithfully restore it forever."""
    window, context, core = await _launch(qapp, database_path)
    try:
        await _drain(qapp, lambda: False, rounds=30)
        stored = await window.context.studio.settings.get("ui.last_model")
        assert stored is None, stored
        assert window.current_model() == "alpha", "a model is chosen, just not remembered yet"
    finally:
        await _close(window, context, core, qapp)


async def test_the_model_a_run_went_to_is_the_one_remembered(two_model_window, qapp) -> None:
    """"Last used" means used, not merely pointed at. A model restored at
    start-up has to be confirmed by a run before it counts."""
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    window = two_model_window
    window.context.llm.set_provider(
        "scripted", MockLLMProvider([ScriptedTurn(text="ok")], model="alpha")
    )
    window.model_selector.setCurrentIndex(1)
    await _drain(qapp, lambda: "beta" in window.current_model())
    window.send("hello")
    assert await _drain(qapp, lambda: "ok" in window.chat.transcript_text())
    await _drain(qapp, lambda: window.chat.send.isEnabled(), rounds=60)
    assert await window.context.studio.settings.get("ui.last_model") == "scripted:beta"


async def test_the_first_model_in_the_list_can_be_remembered(two_model_window, qapp) -> None:
    """currentIndexChanged passes the new index, and index 0 is falsy. Wired
    directly, moving back to the first model would land 0 in `remember` and it
    could never be stored.

    The move has to be a real one: Qt does not emit when the index does not
    change, so going 1 -> 0 is the only way to reach the branch at all.
    """
    window = two_model_window
    window.model_selector.setCurrentIndex(1)
    await _drain(qapp, lambda: "beta" in window.current_model())
    window.model_selector.setCurrentIndex(0)
    await _drain(qapp, lambda: "alpha" in window.current_model())
    assert await window.context.studio.settings.get("ui.last_model") == "scripted:alpha"


async def test_a_remembered_model_from_a_deleted_provider_is_not_fatal(qapp, database_path) -> None:
    """Configuration drifts. A window that will not open because yesterday's
    provider is gone cannot be repaired by the person who needs it."""
    window, context, core = await _launch(qapp, database_path, remembered="vanished:model")
    try:
        assert window.current_model() == "alpha"
        assert "vanished · model" in window.chat.transcript_text()
    finally:
        await _close(window, context, core, qapp)


async def test_the_status_bar_follows_the_selection_not_the_first_provider(two_model_window) -> None:
    window = two_model_window
    window.model_selector.setCurrentIndex(1)
    assert "beta" in window.llm_status.text(), window.llm_status.text()


async def test_picking_a_model_in_the_table_is_remembered_too(two_model_window, qapp) -> None:
    """The table and the combo are two doors into the same room; only the combo
    remembered would leave half the ways of choosing a model unremembered."""
    window = two_model_window
    window.models.models_table.setCurrentCell(1, 0)
    window.models.models_table.selectRow(1)
    await _drain(qapp, lambda: "beta" in window.current_model())
    assert await window.context.studio.settings.get("ui.last_model") == "scripted:beta"
