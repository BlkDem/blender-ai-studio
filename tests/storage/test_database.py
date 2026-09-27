"""Migrations, the database wrapper and the settings override store."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.core.errors import StorageError
from app.core.settings import MCPServerConfig, Settings
from app.storage.database import Database, json_dumps, json_loads
from app.storage.migrations import LATEST_VERSION, migrate
from app.storage.repositories import Studio


async def test_migrations_apply_and_record_their_version(raw_database: Database) -> None:
    version = await migrate(raw_database)
    assert version == LATEST_VERSION

    tables = {
        row["name"]
        for row in await raw_database.all("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"projects", "conversations", "messages", "tool_calls", "settings"} <= tables
    assert {"llm_requests", "three_d_tasks"} <= tables
    assert {"benchmark_suites", "benchmark_runs", "benchmark_reviews"} <= tables


async def test_migrations_are_idempotent(raw_database: Database) -> None:
    """Running twice must not fail and must not re-create anything."""
    await migrate(raw_database)
    version = await migrate(raw_database)
    assert version == LATEST_VERSION


async def test_foreign_keys_are_on(raw_database: Database) -> None:
    """SQLite leaves them off by default, and an orphaned message is a bad bug
    to have in a transcript store."""
    row = await raw_database.one("PRAGMA foreign_keys")
    assert int(next(iter(row.values()))) == 1


async def test_a_conversation_does_not_survive_its_project(studio: Studio) -> None:
    project = await studio.projects.create("Demo")
    conversation = await studio.conversations.create(project.id, "First")
    await studio.messages.add(conversation.id, "user", "hello")

    await studio.projects.delete(project.id)

    assert await studio.conversations.get(conversation.id) is None


async def test_json_columns_round_trip(studio: Studio) -> None:
    project = await studio.projects.create(
        "With metadata", metadata={"viewport": [1, 2, 3], "tags": ["a"], "unicode": "грань"}
    )
    stored = await studio.projects.get(project.id)
    assert stored is not None
    assert stored.metadata["viewport"] == [1, 2, 3]
    assert stored.metadata["unicode"] == "грань"


def test_json_helpers_are_forgiving() -> None:
    assert json_loads(None, {"a": 1}) == {"a": 1}
    assert json_loads("not json", []) == []
    assert json_loads('{"a": 1}') == {"a": 1}
    assert json_dumps({"b": 1, "a": 2}) == '{"a": 2, "b": 1}', "sorted keys keep rows diffable"


async def test_settings_overrides_persist_and_reload(data_dir: Path) -> None:
    studio = await Studio.open(data_dir / "settings.db")
    try:
        await studio.settings.set("agent.max_steps", 12)
        await studio.settings.set("theme", "dark")
        assert await studio.settings.all() == {"agent.max_steps": 12, "theme": "dark"}

        settings = Settings()
        settings.apply_overrides(await studio.settings.all())
        assert settings.agent.max_steps == 12
        assert settings.theme == "dark"
    finally:
        studio.close()


async def test_a_stored_override_that_no_longer_parses_is_ignored(data_dir: Path) -> None:
    """A hand-edited or outdated value must not stop the app from starting."""
    studio = await Studio.open(data_dir / "broken.db")
    try:
        await studio.settings.set("agent.max_steps", "not a number")
        settings = Settings()
        settings.apply_overrides(await studio.settings.all())
        assert settings.agent.max_steps == settings.agent.max_steps  # unchanged default
    finally:
        studio.close()


async def test_opening_an_impossible_path_is_a_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    with pytest.raises(StorageError) as excinfo:
        Database(blocker / "nested" / "studio.db").connect()
    assert "database" in excinfo.value.message.lower()


def test_a_child_process_environment_is_explicit(tmp_path: Path) -> None:
    """The GUI's port must win over a stale shell variable, or the user will be
    talking to a Blender that is not the one they are looking at."""
    config = MCPServerConfig(
        command="python", blender_port=9999, env={"BLENDER_PORT": "1234", "EXTRA": "1"}
    )
    env = config.child_env()
    assert env["BLENDER_PORT"] == "9999"
    assert env["EXTRA"] == "1"
    assert "PATH" in env


def test_settings_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("STUDIO_THEME", "dark")
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.theme == "dark"


async def test_a_backup_is_a_usable_database(data_dir: Path, studio: Studio) -> None:
    project = await studio.projects.create("Kept")
    destination = await studio.db.backup_to(data_dir / "backup" / "studio.db")

    copy = await Studio.open(destination)
    try:
        stored = await copy.projects.get(project.id)
        assert stored is not None and stored.name == "Kept"
    finally:
        copy.close()


def test_wal_mode_is_on(database_path: Path) -> None:
    database = Database(database_path)
    database.connect()
    try:
        mode = database.query_one("PRAGMA journal_mode")
        assert str(next(iter(mode.values())).lower()) == "wal"
    finally:
        database.close()


async def test_concurrent_writes_do_not_raise(tmp_path: Path) -> None:
    """The GUI writes usage while an agent streams; that is the whole reason for
    WAL and the lock, so it is worth asserting rather than assuming."""
    studio = await Studio.open(tmp_path / "concurrent.db")
    try:
        conversation = await studio.conversations.create(None, "Busy")
        await asyncio.gather(
            *(
                studio.messages.add(conversation.id, "user", f"message {index}")
                for index in range(20)
            )
        )
        stored = await studio.messages.list(conversation.id)
        assert len(stored) == 20
    finally:
        studio.close()


def test_json_dumps_handles_paths() -> None:
    assert "tmp" in json_dumps({"path": Path("/tmp/x")})
    assert json.loads(json_dumps({"a": 1})) == {"a": 1}
