"""The Tasks panel, including what it knows from earlier sessions.

Restarting the window used to show an empty table, which reads as "nothing was
ever generated" -- and for a 3D generation, which cost money, that is the worst
possible thing to lose.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui.tasks.panel import TasksPanel  # noqa: E402
from app.storage.repositories import Studio, ThreeDTaskRecord  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_widget_left_behind(qapp: QApplication) -> None:
    yield
    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
async def studio(database_path: Path):
    instance = await Studio.open(database_path)
    try:
        yield instance
    finally:
        instance.close()


async def test_the_tasks_panel_shows_what_earlier_sessions_stored(studio: Studio, qapp) -> None:
    """Restarting the window used to show an empty table, which reads as
    'nothing was ever generated'."""

    await studio.three_d.create(
        ThreeDTaskRecord(
            id="tdt_1",
            run_id="run_1",
            provider="tripo",
            kind="text_to_3d",
            prompt="a medieval wooden chest",
            model="",
            status="succeeded",
            started_at=1000.0,
            finished_at=1042.0,
            credits=20,
            local_path="/tmp/Chest.glb",
        )
    )

    panel = TasksPanel()
    count = await panel.load_stored(studio)
    assert count == 1
    assert panel.table.rowCount() == 1
    assert "a medieval wooden chest" in panel.table.item(0, 0).text()
    assert panel.table.item(0, 2).text() == "succeeded"
    assert panel.table.item(0, 6).text() == "20", "the credits are on the row"
    assert "20 3D credits" in panel.summary.text()
    assert panel.totals()["credits"] == 20.0


async def _raise(*args, **kwargs):
    raise RuntimeError("the database is gone")
