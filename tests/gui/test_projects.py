"""Projects, and the fact that a turn belongs to one.

A project is a named piece of work with a starting file. What makes it worth
having is that everything under it -- conversations, generations, what they cost
-- can be told apart from another project's six weeks later. The storage for that
was written first and never called, so every session was one undifferentiated
stream of turns.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui.projects.panel import ProjectsPanel  # noqa: E402
from app.storage.repositories import Studio  # noqa: E402


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


async def test_the_panel_lists_projects_with_what_they_start_from(studio: Studio, qapp) -> None:
    """A name and nothing else is not enough: the file is the point."""
    await studio.projects.create("Kitchen", initial_blend="/work/kitchen.blend", default_model="gpt-5")
    await studio.projects.create("Balcony")

    panel = ProjectsPanel()
    panel.show_projects(await studio.projects.list(), None)

    assert panel.list.count() == 2
    texts = [panel.list.item(row).text() for row in range(2)]
    kitchen = next(text for text in texts if "Kitchen" in text)
    assert "kitchen.blend" in kitchen and "gpt-5" in kitchen
    assert "no starting file" in next(text for text in texts if "Balcony" in text)
    assert "not filed anywhere" in panel.current.text(), "and it says when nothing is open"


async def test_the_open_project_is_marked(studio: Studio, qapp) -> None:
    await studio.projects.create("One")
    second = await studio.projects.create("Two", initial_blend="/work/two.blend")
    panel = ProjectsPanel()
    panel.show_projects(await studio.projects.list(), second.id)
    marked = [panel.list.item(row).text() for row in range(2) if panel.list.item(row).text().startswith("●")]
    assert len(marked) == 1 and "Two" in marked[0], "the open project is visible as open"
    assert "Two" in panel.current.text()
    assert panel.selected_blend() == "", "nothing is selected yet, so there is nothing to load"


async def test_a_conversation_is_filed_under_the_project(studio: Studio, qapp) -> None:
    project = await studio.projects.create("Kitchen")
    first = await studio.conversations.create(project.id, "first")
    await studio.conversations.create(None, "loose")
    second = await studio.conversations.create(project.id, "second")

    filed = await studio.conversations.list(project.id)
    assert [c.id for c in filed] == [second.id, first.id]
    assert all(c.project_id == project.id for c in filed)

    everything = await studio.conversations.list(None)
    assert len(everything) == 3, "listing without a project still means all of them"


async def test_deleting_a_project_takes_its_conversations_with_it(studio: Studio, qapp) -> None:
    """The cascade exists in the schema and has to work, or deleting a project
    leaves orphaned turns that no project can claim."""
    project = await studio.projects.create("Kitchen")
    conversation = await studio.conversations.create(project.id, "work")
    await studio.messages.add(conversation.id, "user", "make me a table")
    loose = await studio.conversations.create(None, "loose")

    await studio.projects.delete(project.id)

    assert await studio.projects.get(project.id) is None
    remaining = await studio.conversations.list(None)
    assert [c.id for c in remaining] == [loose.id], "and the loose conversation is untouched"
    assert await studio.messages.list(conversation.id) == [], "its messages went with it"


async def test_a_window_files_its_turns_under_the_open_project(window, qapp) -> None:
    """The end of the story: a turn in the window lands in the project."""
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    assert window.context.current_project is None
    await window._create_project("Kitchen", "/work/kitchen.blend")  # noqa: SLF001
    project_id = window.context.current_project
    assert project_id, "the project is open"
    assert "Kitchen" in window.projects.current.text()

    window.context.llm.set_provider(
        "scripted",
        MockLLMProvider([ScriptedTurn(text="A table.")], model="scripted-model"),
    )
    window.model_selector.setCurrentIndex(0)
    window.send("make me a table")
    for _ in range(80):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "A table." in window.chat.transcript_text():
            break

    filed = await window.context.studio.conversations.list(project_id)  # noqa: SLF001
    assert len(filed) == 1, "the turn is under the project"
    messages = await window.context.studio.messages.list(filed[0].id)  # noqa: SLF001
    assert "make me a table" in messages[0].content


async def test_opening_a_project_brings_its_transcript_back(window, qapp) -> None:
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    await window._create_project("Kitchen", "")  # noqa: SLF001
    project_id = window.context.current_project
    window.context.llm.set_provider(
        "scripted",
        MockLLMProvider([ScriptedTurn(text="A table.")], model="scripted-model"),
    )
    window.model_selector.setCurrentIndex(0)
    window.send("make me a table")
    for _ in range(80):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "A table." in window.chat.transcript_text():
            break

    await window._close_project()  # noqa: SLF001
    assert window.context.current_project is None
    window.chat.clear()
    assert window.chat.transcript_text() == ""

    await window._open_project(project_id)  # noqa: SLF001
    assert "make me a table" in window.chat.transcript_text()
    assert "A table." in window.chat.transcript_text()


async def test_deleting_the_open_project_leaves_the_window_in_no_project(window, qapp) -> None:
    await window._create_project("Kitchen", "")  # noqa: SLF001
    project_id = window.context.current_project
    await window._delete_project(project_id)  # noqa: SLF001
    assert window.context.current_project is None
    assert "not filed anywhere" in window.projects.current.text()


def test_a_projects_blend_is_never_opened_directly(window, qapp, tmp_path: Path) -> None:
    """Blender would then be editing the file the project starts from, and the
    next run would not start where this one did. So a copy is opened, and it is
    a copy the user can point at afterwards."""
    from app.benchmark.runner import BenchmarkRunner, ModelSpec

    source = tmp_path / "kitchen.blend"
    source.write_bytes(b"BLENDER-v300")
    runner = BenchmarkRunner(window.context.llm, window.context.mcp, workdir=tmp_path / "work")
    staged = runner.stage_scene(source, ModelSpec(provider="", model="project"), 0)
    assert staged != source
    assert staged.read_bytes() == source.read_bytes()
    assert staged.parent == tmp_path / "work"
