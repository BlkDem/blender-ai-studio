"""Projects, and the fact that a turn belongs to one.

A project is a named piece of work, and it is a workspace rather than a label:
it has the file the work happens in, the model the work is done with, and the
conversation it is carried on in. Opening one gives you all three, and everything
after that is filed under it.

The window-level tests here press the buttons. They used to call the handlers
directly, which is how the panel spent its whole life not creating anything: the
handlers were ``async def`` and connected straight to Qt signals, so Qt called
them, was handed a coroutine and dropped it. A test that awaits the handler
misses that completely, because the handler is the part that worked.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui.projects.panel import ProjectsPanel  # noqa: E402
from app.storage.repositories import Studio  # noqa: E402

#: How long a driven window is given to catch up. Generous, because these are
#: the slowest tests in the suite and a flaky timeout here would read as a real
#: failure and be ignored.
PUMP = 200


async def pump(qapp, until: Callable[[], bool]) -> None:
    """Let the window do what it was asked, the way a person would see it.

    The window hands work to the core thread and gets an answer back as a queued
    signal, so a test that returns before the queue drains is testing the state
    before the click, not after it.
    """
    for _ in range(PUMP):
        qapp.processEvents()
        await asyncio.sleep(0.01)
        if until():
            qapp.processEvents()
            return
    raise AssertionError("the window never got there")


async def create_through_the_panel(window, qapp, name: str, blend: str = "") -> str:
    """Press Create, and wait for the window to be showing what it made.

    Both halves matter. ``current_project`` is set before the panel is refilled,
    so waiting on it alone leaves the label a turn behind -- and a test that
    asserts on the label right after would be reading the window before the
    click, which is the mistake this file already made once.
    """
    panel = window.projects
    panel.name.setText(name)
    panel.blend.setText(blend)
    panel.create_button.click()
    await pump(
        qapp,
        lambda: window.context.current_project is not None and name in panel.current.text(),
    )
    project_id = window.context.current_project
    assert project_id, "pressing Create made no project"
    return str(project_id)


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
    assert "no file" in next(text for text in texts if "Balcony" in text)
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
    project_id = await create_through_the_panel(window, qapp, "Kitchen", "/work/kitchen.blend")
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


async def test_pressing_create_is_what_makes_a_project(window, qapp) -> None:
    """The regression this panel spent its life failing.

    Every handler was an ``async def`` wired straight to a Qt signal. Qt called
    the function, got back a coroutine and discarded it, so the body never ran
    and Create did nothing at all -- while the tests, which awaited the handler
    directly, passed the whole time.
    """
    studio = window.context.studio
    assert await studio.projects.list() == []
    await create_through_the_panel(window, qapp, "Kitchen")
    stored = await studio.projects.list()
    assert [p.name for p in stored] == ["Kitchen"]
    assert window.projects.list.count() == 1, "and it is on screen"


async def test_opening_a_project_gives_you_its_model(window, qapp) -> None:
    """A project's model was written at creation, shown in the list, and applied
    to nothing: opening the project ran on whatever the last-used setting said.

    The project is made through the panel on purpose. Built in the repository
    instead, the stored model is whatever the test typed, and the window's own
    path -- which stored the bare model id while the selector is keyed by
    "provider:model", so every project reported its own model as unavailable --
    would not be exercised at all.
    """
    studio = window.context.studio
    second = await studio.projects.create("Balcony", default_model="scripted:scripted-model")
    await create_through_the_panel(window, qapp, "Kitchen")
    created = next(p for p in await studio.projects.list() if p.name == "Kitchen")
    assert created.default_model == "scripted:scripted-model", (
        "stored the way the selector keys it, or opening it cannot find it again"
    )
    assert "not available" not in window.chat.transcript_text()

    window.projects.list.setCurrentRow(_row_of(window, second.id))
    window.projects.open_button.click()
    await pump(qapp, lambda: window.context.current_project == second.id)
    assert window.model_selector.currentData() == "scripted:scripted-model"
    assert "not available" not in window.chat.transcript_text()


async def test_opening_a_project_says_so_when_its_model_is_gone(window, qapp) -> None:
    studio = window.context.studio
    gone = await studio.projects.create("Retired", default_model="retired:model-that-left")
    window.projects.opened.emit(gone.id)
    await pump(qapp, lambda: window.context.current_project == gone.id)
    assert "not available" in window.chat.transcript_text()
    assert "retired · model-that-left" in window.chat.transcript_text()


async def test_a_project_gets_its_own_working_file_and_keeps_it(window, qapp, tmp_path: Path) -> None:
    """The file a project is worked on, made once and then left alone.

    The starting ``.blend`` is never opened directly -- Blender would be
    editing the file the next run begins from -- and the copy is not thrown away
    either, or a project would be no better than re-reading its start every time.
    """
    source = tmp_path / "kitchen.blend"
    source.write_bytes(b"BLENDER-v300")
    window.context.settings.default_project_dir = tmp_path / "projects"

    project_id = await create_through_the_panel(window, qapp, "Kitchen", str(source))
    studio = window.context.studio

    # The working file is staged after the project is open, so wait for that
    # rather than reading the row the moment the project appears.
    async def staged() -> bool:
        return bool(str((await studio.projects.get(project_id)).workspace))

    for _ in range(PUMP):
        qapp.processEvents()
        await asyncio.sleep(0.01)
        if await staged():
            break
    project = await studio.projects.get(project_id)
    workspace = Path(str(project.workspace))
    assert workspace.exists(), "the project has a file of its own"
    assert workspace != source, "and it is not the file it started from"
    assert workspace.read_bytes() == b"BLENDER-v300"
    assert workspace == tmp_path / "projects" / project_id / "workspace.blend"

    # The work done on it, which is the thing that has to survive.
    workspace.write_bytes(b"EDITED")
    window.projects.close_button.click()
    await pump(qapp, lambda: window.context.current_project is None)
    window.projects.opened.emit(project_id)
    await pump(qapp, lambda: window.context.current_project == project_id)

    assert workspace.read_bytes() == b"EDITED", "reopening a project did not start over"
    assert source.read_bytes() == b"BLENDER-v300", "and the starting file is untouched"


async def test_a_project_with_no_starting_file_opens_without_pretending(window, qapp) -> None:
    """Path("") is Path("."), and "." exists -- so a project with no file used to
    try to copy the studio's own working directory and fail with a raw errno in
    the transcript. It has nothing to work on, and should say nothing about it.
    """
    project_id = await create_through_the_panel(window, qapp, "Nothing yet")
    for _ in range(PUMP):
        qapp.processEvents()
        await asyncio.sleep(0.01)
        if "is open" in window.chat.transcript_text():
            break
    project = await window.context.studio.projects.get(project_id)
    assert project.workspace is None
    for complaint in ("Errno", "no such file", "is a directory", "error"):
        assert complaint not in window.chat.transcript_text(), (
            f"a project with no file has nothing to complain about, but said {complaint!r}"
        )


async def test_the_model_is_told_which_project_it_is_in(window, qapp) -> None:
    """A project is a workspace, and none of that is visible from a prompt."""
    from app.core.agent import PROJECT_NOTE

    await create_through_the_panel(window, qapp, "Kitchen")
    agent = window.context.agent("scripted", "scripted-model")
    assert "Kitchen" in agent.build_system_prompt()
    assert agent.project_name == "Kitchen"

    window.projects.close_button.click()
    await pump(qapp, lambda: window.context.current_project is None)
    outside = window.context.agent("scripted", "scripted-model")
    assert "Kitchen" not in outside.build_system_prompt()
    assert PROJECT_NOTE.format(name="Kitchen") not in outside.build_system_prompt()


async def test_opening_a_project_brings_its_transcript_back(window, qapp) -> None:
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    project_id = await create_through_the_panel(window, qapp, "Kitchen")
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

    window.projects.close_button.click()
    await pump(qapp, lambda: window.context.current_project is None)
    window.chat.clear()
    assert window.chat.transcript_text() == ""

    window.projects.opened.emit(project_id)
    await pump(qapp, lambda: "A table." in window.chat.transcript_text())
    assert "make me a table" in window.chat.transcript_text()
    assert "A table." in window.chat.transcript_text()


async def test_the_highlight_survives_the_list_being_refilled(window, qapp) -> None:
    """Refilling the list used to clear the highlight, and this panel is
    refilled after every action that touches a project -- so Delete and Rename
    quietly stopped working the moment anything else happened. A button that
    does nothing reads as a broken button, not a lost selection."""
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    window.projects.list.setCurrentRow(_row_of(window, project_id))
    window.projects.rename_requested.emit(project_id, "Kitchen v2")
    await pump(qapp, lambda: window.context.project_name == "Kitchen v2")

    assert window.projects.selected_id() == project_id, "still highlighted after the refresh"
    assert window.projects.delete_button.isEnabled()


async def test_the_open_project_is_said_on_every_page(window, qapp) -> None:
    """The Projects page is not where the work happens.

    Creating a project and being left on the Projects page with nothing
    elsewhere saying so is how a finished setup reads as a finished session --
    the user cannot tell whether a turn would be filed, and does not think to
    ask. The status bar is on every page and the title is what the window
    manager shows, so both carry it.
    """
    assert window.project_status.text() == "Project: none"
    assert window.windowTitle() == "Blender AI Studio"

    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    assert window.project_status.text() == "Project: Kitchen"
    assert "Kitchen" in window.windowTitle()
    assert "filed under it" in window.project_status.toolTip()

    window.projects.close_button.click()
    await pump(qapp, lambda: window.project_status.text() == "Project: none")
    assert window.windowTitle() == "Blender AI Studio"
    assert "Nothing is being filed" in window.project_status.toolTip()

    window.projects.opened.emit(project_id)
    await pump(qapp, lambda: window.project_status.text() == "Project: Kitchen")
    assert "Kitchen" in window.windowTitle()


async def test_renaming_moves_the_title_and_the_bar_together(window, qapp) -> None:
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    window.projects.rename_requested.emit(project_id, "Kitchen v2")
    await pump(qapp, lambda: window.context.project_name == "Kitchen v2")
    assert window.project_status.text() == "Project: Kitchen v2"
    assert "Kitchen v2" in window.windowTitle()


async def test_creating_a_project_says_what_to_do_next(window, qapp) -> None:
    """A confirmation is not a next step. The user is left on the Projects page
    and the work happens in the chat, so that is what the note has to say."""
    await create_through_the_panel(window, qapp, "Kitchen")
    text = window.chat.transcript_text()
    assert "Go to Chat" in text, f"the note does not say where the work happens: {text[-300:]}"
    assert "filed" in text


async def test_the_note_names_a_button_that_is_there(window, qapp) -> None:
    """Saying "Go to Chat" and having nothing called that is a dead end. The
    window does not move on its own, so the way out has to be pressable."""
    await create_through_the_panel(window, qapp, "Kitchen")
    assert window.projects.chat_button.text() == "Go to Chat"


async def test_the_go_to_chat_button_leaves_the_projects_page(window, qapp) -> None:
    """The button and the sidebar row have to agree, or the window is showing
    one page while the list beside it says you are on another."""
    from app.gui.main_window import PAGES

    await create_through_the_panel(window, qapp, "Kitchen")
    window.nav.setCurrentRow(PAGES.index("Projects"))
    qapp.processEvents()

    window.projects.chat_button.click()
    qapp.processEvents()
    assert window.nav.currentRow() == PAGES.index("Chat")
    assert window.pages.currentWidget() is window.chat


async def test_double_clicking_a_project_opens_it_and_goes_to_the_chat(window, qapp) -> None:
    """The double-click is the gesture people use on a file to start working.

    It has to be driven as a double-click rather than by calling the handler:
    a wiring mistake in the list is exactly the bug being looked for, and a
    test that emits the signal itself would pass with the connection removed.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    balcony = await create_through_the_panel(window, qapp, "Balcony")
    await create_through_the_panel(window, qapp, "Kitchen")
    # The page has to be on screen: Qt drops mouse events for a widget that is
    # not visible, and the double-click would simply never arrive.
    window.nav.setCurrentRow(3)
    row = _row_of(window, balcony)
    window.projects.list.setCurrentRow(row)
    qapp.processEvents()

    list_ = window.projects.list
    point = list_.visualItemRect(list_.item(row)).center()
    # Click, then double-click, because that is the gesture: a double-click
    # with no click in front of it is not one to Qt, which is why the press it
    # needs arrives by itself.
    QTest.mouseClick(
        list_.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point
    )
    QTest.mouseDClick(
        list_.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point
    )
    await pump(qapp, lambda: window.context.current_project == balcony)

    assert window.context.project_name == "Balcony"
    assert window.nav.currentRow() == 0
    assert window.pages.currentWidget() is window.chat


async def test_the_open_button_still_keeps_you_on_the_projects_page(window, qapp) -> None:
    """Two ways to open, two meanings: the button picks the project, the
    double-click starts working in it. If the button also moved the page, there
    would be no way back to the list without the sidebar."""
    balcony = await create_through_the_panel(window, qapp, "Balcony")
    await create_through_the_panel(window, qapp, "Kitchen")
    window.nav.setCurrentRow(3)
    qapp.processEvents()

    window.projects.list.setCurrentRow(_row_of(window, balcony))
    window.projects.open_button.click()
    await pump(qapp, lambda: window.context.current_project == balcony)

    assert window.nav.currentRow() == 3
    assert window.pages.currentWidget() is window.projects


async def test_creating_a_project_without_a_model_says_so(window, qapp) -> None:
    """A project you cannot run in is a dead end, and the first sign of it is
    Send doing nothing. Said at the moment the project is set up instead."""
    window.model_selector.clear()
    await create_through_the_panel(window, qapp, "Kitchen")
    assert "No model is selected" in window.chat.transcript_text()


async def test_a_deleted_project_takes_its_turns_with_it_and_asks_first(window, qapp) -> None:
    """Two things: the cascade is irreversible, so the question comes before it,
    and the window must not be left claiming a project that is gone."""
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    window.projects.list.setCurrentRow(_row_of(window, project_id))

    # Refusing the question must change nothing.
    window.projects.confirm_delete = lambda name: False  # type: ignore[method-assign]
    window.projects.delete_button.click()
    qapp.processEvents()
    assert await window.context.studio.projects.get(project_id) is not None
    assert window.context.current_project == project_id

    window.projects.confirm_delete = lambda name: True  # type: ignore[method-assign]
    window.projects.delete_button.click()
    await pump(qapp, lambda: window.context.current_project is None)
    assert await window.context.studio.projects.get(project_id) is None
    assert "not filed anywhere" in window.projects.current.text()


async def test_a_project_can_be_renamed(window, qapp) -> None:
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    window.projects.list.setCurrentRow(_row_of(window, project_id))
    window.projects.rename_requested.emit(project_id, "Kitchen v2")
    await pump(qapp, lambda: window.context.project_name == "Kitchen v2")
    stored = await window.context.studio.projects.get(project_id)
    assert stored.name == "Kitchen v2", "the name is what the model is told, so it has to stick"

    agent = window.context.agent("scripted", "scripted-model")
    assert "Kitchen v2" in agent.build_system_prompt()


async def test_the_project_you_were_in_is_the_one_you_get(window, qapp, tmp_path: Path) -> None:
    """A window that reopens with no project while a project's transcript is on
    screen contradicts itself, and the next turn is then filed nowhere."""
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    await pump(qapp, lambda: window.context.current_project == project_id)

    stored = await window.context.studio.settings.get("ui.last_project")
    assert stored == project_id, "it is remembered the way the model is"


def _row_of(window, project_id: str) -> int:
    """The row showing a project, so a test can select it the way a person does."""
    from PySide6.QtCore import Qt

    for row in range(window.projects.list.count()):
        if window.projects.list.item(row).data(Qt.ItemDataRole.UserRole) == project_id:
            return row
    raise AssertionError(f"no row for {project_id}")


async def test_deleting_the_open_project_leaves_the_window_in_no_project(window, qapp) -> None:
    project_id = await create_through_the_panel(window, qapp, "Kitchen")
    window.projects.delete_requested.emit(project_id)
    await pump(qapp, lambda: window.context.current_project is None)
    assert "not filed anywhere" in window.projects.current.text()


def test_a_staged_scene_is_a_copy_in_the_benchmarks_workdir(window, qapp, tmp_path: Path) -> None:
    """The benchmark's isolation, which is where a copy is still made per run.

    Projects do not go through this any more: they keep one working file of
    their own, made once, because re-copying on every open would throw away the
    work between two runs. The guarantee it gave -- a run never edits the file
    the next one starts from -- is what the project working file keeps, and
    ``test_a_project_gets_its_own_working_file_and_keeps_it`` is the test for it.
    """
    from app.benchmark.runner import BenchmarkRunner, ModelSpec

    source = tmp_path / "kitchen.blend"
    source.write_bytes(b"BLENDER-v300")
    runner = BenchmarkRunner(window.context.llm, window.context.mcp, workdir=tmp_path / "work")
    staged = runner.stage_scene(source, ModelSpec(provider="", model="project"), 0)
    assert staged != source
    assert staged.read_bytes() == source.read_bytes()
    assert staged.parent == tmp_path / "work"
