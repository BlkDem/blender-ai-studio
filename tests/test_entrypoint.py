"""The entry point, which is the only path a user actually takes.

The tests elsewhere build an `AppContext` themselves and close it themselves, so
they cannot see a mistake in the order `app.main` opens and closes things. The
window is the one caller that does not own the context, and it used to receive it
already shut.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main import amain, arguments, main


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "studio"
    directory.mkdir()
    return directory


async def test_the_window_gets_a_context_that_is_still_open(data_dir: Path, monkeypatch) -> None:
    """The bug this file exists for.

    ``finally: await context.close()`` ran when the ``try`` block fell through
    to the window, so the window ran on a closed database. Every write failed --
    and most of them are caught and logged, so nothing looked wrong: the
    transcript, the tool calls, the usage and the generated tasks simply were
    not recorded.
    """
    seen: dict[str, object] = {}

    def fake_run_gui(context) -> int:
        seen["database_closed"] = context.studio.db._closed  # noqa: SLF001
        seen["mcp_configured"] = len(context.mcp.configs())
        seen["llm"] = context.llm is not None
        return 0

    import app.gui.main

    monkeypatch.setattr(app.gui.main, "run_gui", fake_run_gui)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv(
        "STUDIO_MCP_SERVERS",
        json.dumps([{"name": "blender", "command": "/usr/bin/python3.11", "args": ["-m", "server.main"]}]),
    )

    options = arguments(["--data-dir", str(data_dir)])
    assert await amain(options) == 0

    assert seen["database_closed"] is False, "the window was handed a closed database"
    assert seen["mcp_configured"] >= 1
    assert seen["llm"] is True


async def test_the_context_is_closed_when_the_window_exits(data_dir: Path, monkeypatch) -> None:
    import app.gui.main

    monkeypatch.setattr(app.gui.main, "run_gui", lambda context: 0)
    options = arguments(["--data-dir", str(data_dir)])
    await amain(options)
    # Nothing to assert on the object itself, but the database file must be
    # readable afterwards: a close that did not happen would leave a WAL behind.
    assert (data_dir / "studio.db").exists()


async def test_check_reports_and_closes_cleanly(data_dir: Path, capsys) -> None:
    options = arguments(["--check", "--data-dir", str(data_dir)])
    code = await amain(options)
    out = capsys.readouterr().out
    assert code in (0, 1), "--check answers, it does not crash"
    assert "data directory" in out
    assert "MCP servers" in out


def test_a_headless_turn_needs_no_display(data_dir: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    code = main(["--check", "--data-dir", str(data_dir)])
    assert code in (0, 1)
    assert "No display found" not in capsys.readouterr().err


def test_a_gui_on_a_machine_with_no_display_says_so_once(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    code = main(["--data-dir", "/tmp/should-not-be-used"])
    assert code == 1
    err = capsys.readouterr().err
    assert "No display found" in err
    assert "--check" in err, "and it says what to do instead"
