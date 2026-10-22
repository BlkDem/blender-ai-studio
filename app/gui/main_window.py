"""The main window.

Six panels and a status bar, and the wiring between them: a chat message produces
tool calls in the transcript, a cost lands in the footer, a task lands in the Tasks
panel, a scene refresh updates the Scene panel. All of it comes from one event
bus, so the window shows what actually happened rather than what a panel assumed.

The rule that shapes this file: the GUI thread starts coroutines and renders
events. It never waits. Every network operation goes to the core thread through
:meth:`CoreThread.submit`, and every update arrives as a queued Qt signal.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.benchmark.runner import BenchmarkRunner, BenchmarkTask, ModelSpec
from app.benchmark.storage import BenchmarkStorage
from app.core.agent import history_block
from app.core.events import EventType
from app.gui.benchmark.panel import BenchmarkPanel
from app.gui.bridge import CoreThread, format_money
from app.gui.chat.history import HistoryPopup
from app.gui.chat.widget import ChatView, image_mime_type
from app.gui.models.panel import ModelsPanel
from app.gui.projects.panel import ProjectsPanel
from app.gui.scene.panel import ScenePanel
from app.gui.settings.panel import SettingsPanel
from app.gui.tasks.panel import TasksPanel
from app.llm.base import ContentPart, Message

logger = logging.getLogger(__name__)

PAGES = ("Chat", "Scene", "Tasks", "Projects", "Benchmark", "Models", "Settings")

#: Which model was in use last, as "provider:model". A preference, not
#: configuration, so it lives in the settings table and not in the .env: a file
#: the studio rewrites is a file the user stops trusting, and one whose value the
#: window has to agree with by hand.
LAST_MODEL_KEY = "ui.last_model"

#: The project that was open when the window closed, as a project id. Same
#: reasoning as the model: a preference, not configuration, and the window
#: reopening in no project while a project's transcript is on screen is a
#: contradiction the user is in a poor position to explain.
LAST_PROJECT_KEY = "ui.last_project"


class MainWindow(QMainWindow):
    """The studio's one window."""

    def __init__(self, context: Any, core: CoreThread) -> None:
        super().__init__()
        self.context = context
        self.core = core
        self.setWindowTitle("Blender AI Studio")
        self.resize(1280, 820)
        self.setObjectName("main-window")

        self._saving = False
        self._workspace_busy = False
        self.current_run_id = ""
        self._answer_seen = False
        self._run_failure = ""
        self._retry_deadline = 0.0
        self._retry_description = ""
        self._retry_timer = QTimer(self)
        self._retry_timer.setInterval(200)
        self._retry_timer.timeout.connect(self._update_retry_countdown)
        self._active_agent: Any = None
        self._agent: Any = None
        #: Everything the agent was built from. A turn whose key differs gets a
        #: new agent, so a changed model or setting reaches the next request.
        self._agent_key: tuple[Any, ...] | None = None
        self._conversation_id = ""
        #: The restored block, handed to the agent when it is built. Kept here
        #: rather than read from the database per turn so that the agent is still
        #: the single owner of what the model has been told.
        self._history: list[Message] = []
        #: Filled by _load_models, applied by _models_loaded.
        self._last_model = ""
        #: What is already in the settings table, so moving the selector back and
        #: forth does not write on every step.
        self._last_saved_model = ""
        #: Whether the remembered model was found, for the message when it was not.
        self._remembered = False
        self._cards_by_tool: dict[str, str] = {}
        self._benchmarks = BenchmarkStorage(context.studio.db) if context.studio else None

        # --- widgets --------------------------------------------------------
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("model-selector")
        self.model_selector.setMinimumWidth(240)
        # Through a lambda, not directly: currentIndexChanged passes the new
        # index, and an index of 0 would land in `remember` and read as False --
        # so the first model in the list could never be remembered.
        self.model_selector.currentIndexChanged.connect(lambda _index: self._model_changed())

        self.connection_dot = QLabel("●")
        self.connection_dot.setObjectName("connection-dot")
        self.connection_dot.setToolTip("MCP servers")

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        title = QLabel("<b>Blender AI Studio</b>")
        header_layout.addWidget(title)
        header_layout.addSpacing(16)
        header_layout.addWidget(self.connection_dot)
        header_layout.addWidget(self.connection_dot_label())
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("Model:"))
        header_layout.addWidget(self.model_selector)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setMaximumWidth(150)
        for name in PAGES:
            self.nav.addItem(QListWidgetItem(name))
        self.nav.currentRowChanged.connect(self._page_changed)

        self.pages = QStackedWidget()
        self.chat = ChatView()
        self.scene = ScenePanel()
        self.tasks = TasksPanel()
        self.projects = ProjectsPanel()
        self.benchmark = BenchmarkPanel()
        self.models = ModelsPanel(secrets_backend=context.secrets.backend if context.secrets else "file")
        self.settings = SettingsPanel()
        for widget in (
            self.chat,
            self.scene,
            self.tasks,
            self.projects,
            self.benchmark,
            self.models,
            self.settings,
        ):
            self.pages.addWidget(widget)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self.nav)
        body.addWidget(self.pages)
        body.setStretchFactor(1, 1)
        body.setSizes([150, 1000])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addWidget(body, 1)
        self.setCentralWidget(central)

        status_bar = self.statusBar()
        status_bar.setObjectName("studio-statusbar")
        status_bar.setStyleSheet(
            "#studio-statusbar { border-top: 1px solid palette(mid); }"
            "#studio-statusbar::item { border: none; }"
        )
        self.mcp_status = QLabel("MCP: …")
        self.llm_status = QLabel("LLM: …")
        self.three_d_status = QLabel("3D: …")
        self.project_status = QLabel("Project: none")
        self.project_status.setObjectName("project-status")
        #: Which project this is is not a fact the user should have to go and
        #: check. The Projects page says it, and the Projects page is not where
        #: the work happens -- so the status bar, which is on every page, says
        #: it too. First in the bar, because it is the one that decides where the
        #: next turn is filed.
        self.project_status.setToolTip(
            "Nothing is being filed. Open or create a project to keep this work together."
        )
        self.cost_status = QLabel("")
        for label in (self.project_status, self.mcp_status, self.llm_status, self.three_d_status):
            self.statusBar().addWidget(label)
        self.run_status = QLabel("Run: idle")
        self.run_status.setObjectName("run-status")
        self.run_status.setTextFormat(Qt.TextFormat.PlainText)
        self.statusBar().addPermanentWidget(self.run_status)
        self.statusBar().addPermanentWidget(self.cost_status)

        # --- signals --------------------------------------------------------
        self.core.signals.event.connect(self._on_event)
        self.core.signals.error.connect(self._on_error)
        self.chat.submitted.connect(self.send)
        self.chat.stop_requested.connect(self.stop_run)
        self.chat.rejected_attachment.connect(
            lambda why: self.chat.add_note(why, role="error")
        )
        self.chat.history_requested.connect(self._open_history)
        self.chat.new_chat_requested.connect(self._start_new_chat)
        self.scene.refresh_requested.connect(self.refresh_scene)
        self.scene.commit_transaction.connect(
            lambda: self._close_transaction("blender.commit_transaction", "Transaction committed.")
        )
        self.scene.rollback_transaction.connect(
            lambda: self._close_transaction("blender.rollback_transaction", "Transaction rolled back.")
        )
        self.models.model_selected.connect(self._model_selected)
        self.models.save_provider.connect(self._save_provider)
        self.models.set_api_key.connect(self._set_api_key)
        self.models.save_model.connect(self._save_provider_model)
        self.models.edit_provider.connect(self._edit_provider)
        self.models.edit_model.connect(self._edit_model)
        self.models.remove_provider.connect(self._remove_provider_config)
        self.models.remove_model.connect(self._remove_model_config)
        self.models.configure_3d.connect(self._configure_three_d)
        self.settings.setting_changed.connect(self._setting_changed)
        self.settings.mcp_changed.connect(self._mcp_changed)
        self.tasks.cancel_requested.connect(self._cancel_task)
        self.benchmark.run_requested.connect(self._run_benchmark)
        self.benchmark.cancel_requested.connect(self._cancel_benchmark)
        self.benchmark.review_requested.connect(self._save_review)
        self.projects.create_requested.connect(self._create_project)
        self.projects.opened.connect(self._open_project)
        self.projects.opened_in_chat.connect(self._open_project_in_chat)
        self.projects.chat_requested.connect(self._go_to_chat)
        self.projects.closed.connect(self._close_project)
        self.projects.delete_requested.connect(self._delete_project)
        self.projects.blend_requested.connect(self._load_project_blend)
        self.projects.rename_requested.connect(self._rename_project)
        self.models.select_first()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("File")
        self.save_action = file_menu.addAction("Save .blend")
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(lambda: self._save_blend(False))
        self.save_as_action = file_menu.addAction("Save as…")
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(lambda: self._save_blend(True))
        self.pack_resources_action = file_menu.addAction("Pack resources (textures) before saving")
        self.pack_resources_action.setCheckable(True)
        self.pack_resources_action.setChecked(True)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _file_busy(self) -> bool:
        return bool(self._saving or self._workspace_busy or self.chat.busy() or self._active_agent)

    def _save_blend(self, save_as: bool = False) -> None:
        if self._file_busy():
            self.chat.add_note("Wait for the current run or file operation before saving.", role="error")
            return
        self._saving = True
        self.projects.setEnabled(False)
        self.save_action.setEnabled(False)
        self.save_as_action.setEnabled(False)
        self.core.submit(self._save_target(), lambda project: self._choose_save(project, save_as))

    async def _save_target(self) -> Any:
        return await self._fetch_project(self.context.current_project) if self.context.current_project else None

    def _choose_save(self, project: Any, save_as: bool) -> None:
        path = str(project.workspace or "") if project else ""
        if save_as or not path:
            dialog = QFileDialog(self, "Save Blender file")
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setNameFilter("Blender files (*.blend)")
            dialog.setDefaultSuffix("blend")
            dialog.selectFile(path or (str(self.context.workspace_path(project)) if project else "untitled.blend"))
            if not dialog.exec():
                self._save_finished(("", ""))
                return
            path = dialog.selectedFiles()[0]
            if not path.lower().endswith(".blend"):
                path += ".blend"
                if Path(path).exists() and QMessageBox.question(
                    self, "Replace Blender file?", f"Replace {path}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                ) != QMessageBox.StandardButton.Yes:
                    self._save_finished(("", ""))
                    return
        self.core.submit(
            self._do_save_blend(project.id if project else None, path, self.pack_resources_action.isChecked()),
            self._save_finished,
        )

    async def _do_save_blend(self, project_id: str | None, path: str, pack: bool) -> tuple[str, str]:
        from app.core.blend_save import save_blend

        try:
            if not self.context.settings.agent.allow_execute_python:
                raise RuntimeError("Saving requires blender.execute_python permission in Settings. No settings were changed.")
            await save_blend(self.context.mcp, path, pack=pack)
            if project_id:
                await self.context.studio.projects.update(project_id, workspace=path)
            return path, ""
        except Exception as exc:
            return "", str(exc)

    def _save_finished(self, outcome: tuple[str, str]) -> None:
        self._saving = False
        self.projects.setEnabled(True)
        self.save_action.setEnabled(True)
        self.save_as_action.setEnabled(True)
        path, error = outcome
        if error:
            self.chat.add_note(f"Save failed: {error}", role="error")
        elif path:
            self.chat.add_note(f"Saved .blend: {path}", role="system")
            self._load_projects()
            self._show_project_state()
            self.refresh_scene()

    def connection_dot_label(self) -> QLabel:
        self._connection_label = QLabel("Blender")
        return self._connection_label

    # --- start-up ----------------------------------------------------------

    def start(self) -> None:
        """Connect the servers and fill the panels. Never blocks the window."""
        self.core.attach_bus()
        self.core.submit(self.context.mcp.connect_all(), self._servers_connected)
        self.core.submit(self._load_models(), self._models_loaded)
        self.core.submit(self.tasks.load_stored(self.context.studio), self._tasks_loaded)
        self.core.submit(self._read_last_project(), self._reopen_project)
        self.settings.load(self.context.settings)
        self._show_three_d_status()
        self._show_providers()

    async def _read_last_project(self) -> Any:
        """The project that was open last, if it is still there."""
        if self.context.studio is None:
            return None
        wanted = str(await self.context.studio.settings.get(LAST_PROJECT_KEY) or "")
        if not wanted:
            return None
        return await self.context.studio.projects.get(wanted)

    def _reopen_project(self, project: Any) -> None:
        """Put the window back in the project it was left in.

        The project is restored before the conversation, and the order is the
        whole point. The transcript used to be restored first, with nothing
        knowing which project was open -- and ``list(None)`` means *every*
        conversation, so the window came back showing another project's turns
        while the panel said this session was filed nowhere, and the next turn
        was written with a null project, splitting one project's history in two.
        """
        if project is None:
            # No project: restore the newest conversation of any, which is what
            # there is to restore. The next turn is not filed anywhere, and the
            # panel says so rather than implying otherwise.
            self._load_projects()
            self._restore_conversation()
            return
        self._adopt_project(project)
        self._restore_conversation()

    # --- prompt history ----------------------------------------------------

    def _open_history(self) -> None:
        """Ask the core thread for the prompts, then show them.

        The read goes through the core thread like everything else: a database
        query on the GUI thread is a frozen window, and the window freezing for
        the length of a query is how people decide an app has hung.
        """
        self.core.submit(self._load_prompt_history(), self._show_history)

    async def _load_prompt_history(self) -> list[tuple[str, float]]:
        if self.context.studio is None:
            return []
        return list(await self.context.studio.messages.recent_user_texts())

    def _show_history(self, prompts: list[tuple[str, float]]) -> None:
        # The arrow keys walk the whole list, the unfiltered one: what the popup's
        # filter narrows is what it shows, and narrowing the arrows as a side
        # effect of browsing would leave them narrow after the popup closed.
        self.chat.set_prompt_history([text for text, _when in prompts])
        popup = HistoryPopup(prompts, self)
        popup.chosen.connect(self.chat.put_prompt)
        popup.exec()

    # --- projects ----------------------------------------------------------
    #
    # Every one of these was an `async def` connected straight to a Qt signal,
    # which is the one thing in this file that never worked: Qt calls the
    # function, gets a coroutine back, and discards it. The body never ran, so
    # Create did nothing at all -- and the tests missed it by calling the
    # handlers directly instead of pressing the button. The split below is the
    # shape the rest of the window already uses: a coroutine for the work, a
    # callback for the widgets.

    def _load_projects(self) -> None:
        """Refill the panel. Reading is core-thread work; the widgets are ours."""
        self.core.submit(self._fetch_projects(), self._show_projects)

    async def _fetch_projects(self) -> list[Any]:
        if self.context.studio is None:
            return []
        return list(await self.context.studio.projects.list())

    def _show_projects(self, projects: list[Any]) -> None:
        self.projects.show_projects(projects, self.context.current_project)

    def _create_project(self, name: str, blend: str) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        # Read here, on the GUI thread, and pass it in: the model selector is a
        # widget, and the coroutine below does not run here.
        #
        # currentData(), not current_model(): the project is looked up again by
        # the selector when it is opened, and the selector's key is
        # "provider:model". Storing the bare model id would make every project
        # report its own model as unavailable, forever.
        model = self.model_selector.currentData() or None
        self.core.submit(self._do_create_project(name, blend, model), self._project_created)

    async def _do_create_project(self, name: str, blend: str, model: str | None) -> Any:
        assert self.context.studio is not None
        return await self.context.studio.projects.create(
            name, initial_blend=blend or None, default_model=model
        )

    def _project_created(self, project: Any) -> None:
        if project is None:
            self.chat.add_note("Could not create the project.", role="error")
            return
        self.projects.clear_inputs()
        self._adopt_project(project, created=True)

    def _open_project(self, project_id: str) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        self.core.submit(self._fetch_project(project_id), self._project_fetched)

    def _open_project_in_chat(self, project_id: str) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        """A double-clicked project: opened, and then followed to the chat.

        The page changes when the project is actually open, not when it was
        asked for. Switching on the click would put the previous project's
        transcript on screen for as long as the read takes, and the user would
        have started typing into it.
        """
        self.core.submit(
            self._fetch_project(project_id),
            lambda project: self._project_fetched(project, to_chat=True),
        )

    async def _fetch_project(self, project_id: str) -> Any:
        if self.context.studio is None:
            return None
        return await self.context.studio.projects.get(project_id)

    def _project_fetched(self, project: Any, *, opening: bool = True, to_chat: bool = False) -> None:
        if project is None:
            self.chat.add_note("That project is not there any more.", role="error")
            self._load_projects()
            return
        self._adopt_project(project)
        if opening:
            self._restore_conversation()
        if to_chat:
            self._go_to_chat()

    def _go_to_chat(self) -> None:
        """Leave the Projects page, keeping the sidebar's row with it.

        Not a switch on the stacked widget: that would show the chat while the
        list beside it still says you are on Projects, and the two disagreeing
        about where you are is worse than either being on its own.
        """
        self.nav.setCurrentRow(PAGES.index("Chat"))

    def _close_project(self) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        self.context.current_project = None
        self.context.project_name = ""
        self._remember_project(None)
        self._new_conversation()
        self._show_project_state()
        self._load_projects()
        self.chat.add_note("Project closed. New turns are not filed anywhere.", role="system")

    def _delete_project(self, project_id: str) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        self.core.submit(self._do_delete_project(project_id), self._project_deleted)

    async def _do_delete_project(self, project_id: str) -> bool:
        assert self.context.studio is not None
        await self.context.studio.projects.delete(project_id)
        return self.context.current_project == project_id

    def _project_deleted(self, was_open: bool) -> None:
        if was_open:
            self.context.current_project = None
            self.context.project_name = ""
            self._new_conversation()
            self._show_project_state()
        self._load_projects()
        self.chat.add_note("Project deleted, with the conversations filed under it.", role="system")

    def _rename_project(self, project_id: str, name: str) -> None:
        self.core.submit(self._do_rename_project(project_id, name), lambda _n: self._load_projects())

    async def _do_rename_project(self, project_id: str, name: str) -> None:
        assert self.context.studio is not None
        await self.context.studio.projects.update(project_id, name=name)
        if project_id == self.context.current_project:
            # The name is what the model is told it is working on, and what the
            # status bar and the title show, so all three have to move together.
            self.context.project_name = name
            self._show_project_state()

    def _load_project_blend(self, project_id: str, _blend: str) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        """The panel's manual "open the file" button.

        Opening a project already puts Blender on its file, so this is for the
        times that is not enough: the file was closed, replaced, or deleted, and
        the project still has a starting file to fall back on.
        """
        self.core.submit(self._fetch_project(project_id), self._workspace_requested)

    def _workspace_requested(self, project: Any) -> None:
        if project is None:
            self.chat.add_note("That project is not there any more.", role="error")
            return
        self._open_workspace(project)

    def _adopt_project(self, project: Any, *, created: bool = False) -> None:
        if self._saving or self.chat.busy() or self._active_agent:
            return
        """Make this the project the window is working in."""
        self.context.current_project = project.id
        self.context.project_name = project.name
        self._show_project_state()
        self._apply_project_model(project)
        self._new_conversation()
        self._remember_project(project.id)
        self._load_projects()
        detail = f" on {project.workspace}" if project.workspace else ""
        if created:
            # What to do next, not a statement of what just happened. Creating a
            # project and being left staring at the Projects page, with nothing
            # saying that the work happens somewhere else, is how a finished
            # setup reads as a finished session. The note names the way out --
            # and there is a button on the Projects page called exactly that, so
            # the sentence is an instruction someone can follow rather than a
            # place they have to go and look for.
            self.chat.add_note(
                f"Project '{project.name}' is open{detail} — it is shown at the bottom of the "
                "window. Go to Chat and say what you want built; every turn from here is filed "
                "under it.",
                role="system",
            )
            if not self.current_provider():
                self.chat.add_note(
                    "No model is selected, so a turn would not go anywhere. Choose one in the "
                    "selector at the top first.",
                    role="error",
                )
        else:
            self.chat.add_note(f"Project '{project.name}' is open{detail}.", role="system")
        self._open_workspace(project)

    def _show_project_state(self) -> None:
        """Keep the status bar and the title in step with the open project.

        Two places, one function: they answer the same question and a window that
        says one thing in the title and another in the bar is worse than a
        window that says nothing. The title is what the window manager shows
        when several are open, and the bar is what is on screen when the Projects
        page is not.
        """
        name = self.context.project_name if self.context.current_project else ""
        self.project_status.setText(f"Project: {name}" if name else "Project: none")
        self.project_status.setToolTip(
            "Turns, 3D generations and tool calls are filed under it."
            if name
            else "Nothing is being filed. Open or create a project to keep this work together."
        )
        self.setWindowTitle(f"Blender AI Studio · {name}" if name else "Blender AI Studio")

    def _apply_project_model(self, project: Any) -> None:
        """Select the model the project was made with.

        A project's model was written at creation, shown in the list, and never
        applied to anything: opening a project that needs a different model ran
        on whatever the last-used setting said. Said out loud when the model is
        gone, because a project that quietly runs on another model looks like the
        project is at fault.
        """
        wanted = str(project.default_model or "")
        if not wanted or wanted == (self.model_selector.currentData() or ""):
            return
        if self._select_model(wanted):
            # remember=False: the run has not happened yet, and a preference that
            # records a project being opened would overwrite the user's own.
            self._model_changed(remember=False)
            return
        self.chat.add_note(
            f"Project '{project.name}' asks for {wanted.replace(':', ' · ')}, which is not "
            f"available now; staying on {self.current_model() or 'nothing'}.",
            role="error",
        )

    def _remember_project(self, project_id: str | None) -> None:
        async def store() -> None:
            if self.context.studio is None:
                return
            await self.context.studio.settings.set(LAST_PROJECT_KEY, project_id or "")

        self.core.submit(store(), lambda _v: None)

    def _open_workspace(self, project: Any) -> None:
        """Put Blender on the project's own file, and report what actually happened.

        The starting ``.blend`` is never opened directly: Blender would then be
        editing the file the next run begins from. The first time, it is copied
        once into the project's own directory; after that the copy *is* the
        project, which is the point of having a workspace at all.
        """
        if self._saving:
            return
        self._workspace_busy = True
        self.core.submit(self._stage_workspace(project), self._workspace_staged)

    async def _stage_workspace(self, project: Any) -> tuple[str, str]:
        """Copy the starting file once, then ask Blender to open the copy.

        Returns (path, state), where state is one of the benchmark's three scene
        outcomes. Which one is exactly the thing that must not be assumed: the
        copy is made whether or not Blender can be told to open it, and a user
        whose execute_python is off still gets a file to open by hand.
        """
        from app.benchmark.runner import RESET_UNVERIFIED, BenchmarkRunner

        assert self.context.studio is not None and self.context.mcp is not None
        # Path("") is Path("."), and "." always exists -- so an unset workspace
        # would read as "the file is already there" and the project's working
        # file would become the directory the studio happens to be running in.
        # Same trap one line below for a project with no starting file.
        recorded = str(project.workspace or "").strip()
        existing = Path(recorded) if recorded else None
        if existing is not None and existing.exists():
            staged = existing
        else:
            source_text = str(project.initial_blend or "").strip()
            if not source_text:
                return "", RESET_UNVERIFIED
            source = Path(source_text)
            if not source.exists():
                return str(source), RESET_UNVERIFIED
            staged = self.context.workspace_path(project)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            await self.context.studio.projects.update(project.id, workspace=str(staged))
        runner = BenchmarkRunner(
            self.context.llm, self.context.mcp, bus=self.context.bus, workdir=staged.parent
        )
        state = await runner.reset_scene(staged)
        return str(staged), state

    def _workspace_staged(self, outcome: tuple[str, str]) -> None:
        from app.benchmark.runner import RESET_COPY_ONLY, RESET_UNVERIFIED

        self._workspace_busy = False
        path, state = outcome
        if state == RESET_UNVERIFIED:
            if path:
                self.chat.add_note(f"There is no such file: {path}", role="error")
            return
        if state == RESET_COPY_ONLY:
            self.chat.add_note(
                f"Staged a copy at {path} but Blender could not open it: enable "
                "blender.execute_python in Settings, or open the file yourself.",
                role="error",
            )
            return
        self.chat.add_note(f"Blender is on the project's own file: {path}", role="system")
        self._load_projects()

    def _new_conversation(self) -> None:
        """Start a fresh conversation, in whatever project is now open."""
        self._agent = None
        self._agent_key = None
        self._conversation_id = ""
        # The model has to be told, or the first turn of the "new" conversation
        # still carries everything the old one said.
        self._history = []

    def _start_new_chat(self) -> None:
        """The New button: a blank window and a model that has forgotten.

        Offered because a window that can only be emptied by closing it is not
        much of a workspace. The previous conversation stays in the database — it
        is a record, not something to be destroyed by a second thought.
        """
        if self.chat.busy():
            self.chat.add_note("A run is still going — Stop it first.", role="error")
            return
        self._new_conversation()
        self.chat.clear()
        self.chat.add_note("New conversation. The last one is still on disk.")

    def _restore_conversation(self) -> None:
        """Put the last conversation back on screen *and* back in the model's head.

        Closing the window used to throw the transcript away even though every
        message was in the database. Reopening it is what makes the window a
        workspace rather than a slot machine.

        Drawing it and remembering it are two different jobs, and only doing the
        first is the worse half: the transcript looks continuous while the model
        has never heard of any of it, so "it forgot what I asked for" is a
        correct observation about a bug. The block handed to the model is capped
        and starts on a user turn -- see :func:`app.core.agent.history_block`.

        The read and the drawing are split for the same reason the project
        handlers are: this used to be one coroutine that rebuilt the transcript
        from the core thread, which is the thread violation that stopped the
        project panel doing anything at all.
        """
        self.core.submit(self._fetch_last_conversation(), self._show_conversation)

    async def _fetch_last_conversation(self) -> tuple[str, list[Any]]:
        """The newest conversation of the open project, and its messages."""
        if self.context.studio is None:
            return "", []
        conversations = await self.context.studio.conversations.list(self.context.current_project)
        if not conversations:
            return "", []
        latest = conversations[0]
        return latest.id, list(await self.context.studio.messages.list(latest.id))

    def _show_conversation(self, found: tuple[str, list[Any]]) -> int:
        conversation_id, messages = found
        if not messages:
            return 0
        self._conversation_id = conversation_id
        self._history = history_block(messages)
        self.chat.clear()
        for message in messages:
            self.chat.add_message(message.role, message.content or "")
        if self._history:
            self.chat.add_note(
                f"Resumed — the model has the last {len(self._history)} messages of this session.",
            )
        return len(messages)

    def _tasks_loaded(self, count: int) -> None:
        """Say where the rows came from, so an empty table is not a mystery."""
        if count:
            self.tasks._refresh_summary(f"{count} from earlier sessions · ")  # noqa: SLF001

    def _servers_connected(self, statuses: list[Any]) -> None:
        total = sum(status.tools for status in statuses)
        failed = [status for status in statuses if not status.ready]
        if failed:
            self.mcp_status.setText(f"MCP: {failed[0].name} — {failed[0].last_error or 'failed'}")
            self.connection_dot.setStyleSheet("color: #cf222e;")
            self._connection_label.setText("not connected")
        else:
            self.mcp_status.setText(f"MCP: connected · {total} tools")
            self.connection_dot.setStyleSheet("color: #2da44e;")
            self._connection_label.setText("Blender")
        if any(status.ready for status in statuses):
            self.refresh_scene()

    async def _load_models(self) -> list[dict[str, Any]]:
        assert self.context.llm is not None
        # Read together with the catalogue rather than in a second submit: the
        # two callbacks would race, and the one that arrived second would find
        # an empty selector and silently do nothing.
        self._last_model = await self._read_last_model()
        return [model.to_dict() for model in self.context.llm.models()]

    async def _read_last_model(self) -> str:
        if self.context.studio is None:
            return ""
        stored = await self.context.studio.settings.get(LAST_MODEL_KEY)
        return str(stored or "")

    async def _store_last_model(self, chosen: str) -> str:
        if not chosen or self.context.studio is None:
            return ""
        await self.context.studio.settings.set(LAST_MODEL_KEY, chosen)
        return chosen

    def _show_providers(self) -> None:
        """Fill the providers table.

        The method existed and nothing called it, so the table stayed empty: a
        panel with a heading, six columns and no rows, which reads as "no
        providers" even when three are configured.
        """
        if self.context.llm is None:
            return
        entries = []
        for config in self.context.llm.configs():
            key = ""
            if self.context.secrets is not None:
                from app.storage.secrets import key_name

                key = self.context.secrets.masked(key_name(config.name))
            entries.append(
                {
                    "name": config.name,
                    "kind": config.kind,
                    "base_url": config.base_url,
                    "key": key,
                    "models": config.models,
                    "ready": self.context.llm.is_configured(config.name),
                    "enabled": config.enabled,
                    "default_model": config.default_model,
                    "options": config.options,
                }
            )
        self.models.show_providers(entries)
        self.models.show_three_d(self.context.settings.three_d.model_dump())

    def _models_loaded(self, models: list[dict[str, Any]]) -> None:
        editor_models = models
        if self.context.llm is not None:
            editor_models = [dict(entry, provider=config.name) for config in self.context.llm.configs() for entry in config.models]
        self.models.show_models(editor_models)
        block = self.model_selector.blockSignals(True)
        self.model_selector.clear()
        for model in models:
            self.model_selector.addItem(
                f"{model['id']}  ·  {model['provider']}", f"{model['provider']}:{model['id']}"
            )
        self.model_selector.blockSignals(block)
        # The model that was in use last, if it is still here. Applied with the
        # signals blocked so remembering a choice does not write it again.
        restored = self._select_model(self._last_model)
        self._remembered = restored
        if self.model_selector.count():
            self._model_changed(remember=False)
        if self._last_model and not restored:
            # Said out loud, because silently landing on another model is how a
            # preference comes to look broken rather than gone.
            self.chat.add_note(
                f"The model you used last ({self._last_model.replace(':', ' · ')}) is not available now, "
                f"so this session starts on {self.current_model() or 'nothing'}.",
                role="error",
            )

    def _select_model(self, chosen: str) -> bool:
        """Point the selector at "provider:model". False if it is not there."""
        if not chosen:
            return False
        index = self.model_selector.findData(chosen)
        if index < 0:
            return False
        block = self.model_selector.blockSignals(True)
        self.model_selector.setCurrentIndex(index)
        self.model_selector.blockSignals(block)
        return True

    def _show_three_d_status(self) -> None:
        three_d = self.context.three_d
        if three_d is None:
            self.three_d_status.setText("3D: not configured")
            return
        described = three_d.describe()
        if not described:
            self.three_d_status.setText("3D: not configured")
        elif described[0]["ready"]:
            self.three_d_status.setText(f"3D: {described[0]['name']} ready")
        else:
            self.three_d_status.setText(f"3D: {described[0]['name']} — no API key")

    # --- chat --------------------------------------------------------------

    def send(self, text: str, images: list[str] | None = None) -> None:
        """Start a run. The window stays responsive; the transcript fills in."""
        if not self.current_provider():
            self.chat.add_note("Choose a model first — the selector at the top is empty.", role="error")
            return
        if self._saving or self.chat.busy() or self._active_agent:
            return
        pictures = list(images or [])
        # The gate is the catalogue's, not a guess from the model's name, and it is
        # asked here as well as inside the agent so a picture the model will never
        # see is reported to the person who attached it. Silently dropping it is
        # how "it ignored my image" gets filed as a model problem.
        if pictures and not self._model_sees_images():
            answer = QMessageBox.question(
                self,
                "Enable image input?",
                f"Image input is disabled in the saved settings for {self.current_model()}. "
                "This does not necessarily mean the model cannot see images.\n\n"
                "Enable image input for this model and send the attachments? "
                "Only enable this if your provider supports images.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.chat.input.setText(text)
                for picture in pictures:
                    self.chat.add_attachment(picture)
                return
            registry = self.context.llm
            assert registry is not None
            info = registry.find(self.current_provider(), self.current_model())
            if info is None:
                self.chat.add_note("Model metadata is unavailable. Refresh the model list first.", role="error")
                self.chat.input.setText(text)
                for picture in pictures:
                    self.chat.add_attachment(picture)
                return
            payload = dict(info.to_dict(), original=info.id, supports_vision=True)
            chosen_model = self.model_selector.currentData()
            self.chat.set_busy(True)

            def enabled(result: Any) -> None:
                self.chat.set_busy(False)
                if isinstance(result, Exception):
                    self.chat.add_note(f"Could not enable image input: {result}", role="error")
                    self.chat.input.setText(text)
                    for picture in pictures:
                        self.chat.add_attachment(picture)
                    return
                # Rebuilt on the next turn, with the memory carried over: the
                # model now declares vision, and the agent read that at birth.
                self._agent_key = None
                if self.model_selector.currentData() != chosen_model:
                    self.chat.input.setText(text)
                    for picture in pictures:
                        self.chat.add_attachment(picture)
                    return
                self.send(text, pictures)

            async def enable_images() -> Any:
                try:
                    return await self._persist_model_editor(payload)
                except Exception as exc:
                    return exc

            self.core.submit(enable_images(), enabled)
            return
        # Built here, on the GUI thread, from the selector as it is *now*: the
        # core thread must not read a widget, and a selector moved between this
        # click and the coroutine starting would send the turn to another model.
        try:
            agent = self._agent_for_turn(self.current_provider(), self.current_model())
        except Exception as exc:  # noqa: BLE001 - shown, not raised, into Qt
            self.chat.add_note(f"Could not start the run: {exc}", role="error")
            return
        self._active_agent = agent
        self.chat.add_message("user", text, images=pictures)
        self._answer_seen = False
        self._run_failure = ""
        self._set_run_status("Waiting for model · Stop available")
        self.chat.set_busy(True)
        self._cards_by_tool.clear()
        # "The last used model" means the one a run actually went to, not the one
        # the selector happened to be pointing at. Recorded here rather than only
        # on a deliberate change, so a model restored at start-up is confirmed by
        # being used rather than merely by having been shown.
        chosen = self.model_selector.currentData() or ""
        if chosen and chosen != self._last_saved_model:
            self._last_saved_model = chosen
            self.core.submit(self._store_last_model(chosen), lambda _s: None)
        self.core.submit(self._run_agent(agent, text, pictures), self._run_finished)

    def _model_sees_images(self) -> bool:
        """Whether the selected model is declared multimodal."""
        assert self.context.llm is not None
        info = self.context.llm.find(self.current_provider(), self.current_model())
        return bool(getattr(info, "supports_vision", False))

    def _agent_for_turn(self, provider: str = "", model: str = "") -> Any:
        """One agent per conversation, not one per turn.

        A new agent every turn meant a new conversation every turn: the model was
        stateless, and the transcript in the window looked continuous anyway,
        which is worse than an obvious break -- the user reads a memory that is
        not there.

        The agent is rebuilt when anything it was built from changes -- the
        model, the gates, the prompt, the limits, the model's declared
        capabilities -- and the rebuilt one inherits the conversation: its id,
        and what the model has been told so far. A provider switch keeps only
        the text turns, because tool-call ids and signatures are one vendor's.
        """
        provider = provider or self.current_provider()
        model = model or self.current_model()
        settings = self.context.settings
        registry = self.context.llm
        key = (
            provider,
            model,
            settings.agent.allow_execute_python,
            settings.three_d.enabled,
            repr(settings.agent),
            repr(getattr(settings, "budgets", None)),
            repr(registry.find(provider, model) if registry is not None else None),
        )
        if self._agent is not None and self._agent_key == key:
            return self._agent
        previous, previous_key = self._agent, self._agent_key
        if previous is not None:
            conversation_id = previous.conversation_id or self._conversation_id
            memory = list(previous._memory)  # noqa: SLF001
            # No key with an agent still here means "rebuild this one" -- the
            # vision switch -- which is never a change of provider.
            same_provider = previous_key is None or previous_key[0] == provider
            history = memory if same_provider else history_block(memory)
        else:
            conversation_id, history = self._conversation_id, self._history
        self._agent = self.context.agent(
            provider,
            model,
            conversation_id=conversation_id,
            history=history,
        )
        self._agent_key = key
        return self._agent

    async def _run_agent(self, agent: Any, text: str, images: list[str] | None = None) -> Any:
        parts = [ContentPart.image_part(data, image_mime_type(data)) for data in images or []]
        try:
            return await agent.run(text, images=parts)
        except Exception as exc:
            return exc
        finally:
            # The run is over whichever way it ended; Stop has nothing left to stop.
            if self._active_agent is agent:
                self._active_agent = None

    def _set_run_status(self, text: str) -> None:
        self._retry_timer.stop()
        self.run_status.setText(f"Run: {text}")
        self.run_status.setToolTip(text)

    def _update_retry_countdown(self) -> None:
        remaining = max(0, math.ceil(self._retry_deadline - time.monotonic()))
        text = f"Retry {self._retry_description.format(seconds=remaining)} · Stop available"
        self.run_status.setText(f"Run: {text}")
        self.run_status.setToolTip(text)

    def _run_finished(self, result: Any) -> None:
        if isinstance(result, Exception):
            self._on_error("RUN_FAILED", str(result))
            return
        self.current_run_id = ""
        self.chat.set_busy(False)
        self.chat.end_stream()
        reason = result.stopped_because or self._run_failure
        if reason:
            self._set_run_status("Cancelled" if reason == "cancelled" else f"Failed: {reason}")
        else:
            self._set_run_status("Completed")
        if not self._answer_seen:
            text = getattr(result, "text", "")
            if text:
                self.chat.add_message("assistant", text)
                self._answer_seen = True
            elif not reason:
                self.chat.add_note("Run completed without a text answer.", role="system")
        self.cost_status.setText(
            f"{len(result.tool_calls)} tool call(s) · {result.totals.tokens} tokens · "
            f"{format_money(result.totals.llm_usd)} · {result.elapsed_s:.1f}s"
        )
        if result.stopped_because:
            self.chat.add_note(result.stopped_because, role="error")
        self.refresh_scene()

    def stop_run(self) -> None:
        """Stop the run in flight, here and now.

        Cancellation is a task cancel inside the core, so it has to be asked for
        from the core thread -- but waiting for a round trip through the queue
        before the UI unlocks would let a run that is already stopping look like
        one that is still going.
        """
        agent = self._active_agent
        if agent is not None and self.current_run_id:
            self.core.submit(self._cancel_agent(agent), self._cancelled)
        elif agent is None:
            self.chat.set_busy(False)

    async def _cancel_agent(self, agent: Any) -> bool:
        return agent.cancel(self.current_run_id)

    def _cancelled(self, ok: bool) -> None:
        if ok:
            self.chat.add_note("Stopped.", role="system")
        elif self._active_agent is not None:
            self.chat.add_note("Nothing to stop -- the run had already finished.", role="system")

    def current_provider(self) -> str:
        data = self.model_selector.currentData() or ""
        return data.split(":", 1)[0] if data else ""

    def current_model(self) -> str:
        data = self.model_selector.currentData() or ""
        return data.split(":", 1)[1] if ":" in data else ""

    def _model_changed(self, remember: bool = True) -> None:
        """The selector moved: say which model, and remember it.

        The status follows the *selected* model. It used to name the first
        configured provider, which is a different thing whenever the person
        picked anything else -- a status bar that contradicts the selector is
        worse than no status bar.

        ``remember`` is off while the selector is being filled: on a first run
        there is no last-used model, and writing "the first one in the list" would
        invent a preference and then faithfully restore it forever after.
        """
        provider = self.current_provider()
        if not provider:
            self.llm_status.setText("LLM: no model")
            return
        keyed = self.context.llm is not None and self.context.llm.is_configured(provider)
        self.llm_status.setText(f"LLM: {self.current_model()} · {provider}" + ("" if keyed else " · no key"))
        chosen = self.model_selector.currentData() or ""
        if remember and chosen and chosen != self._last_saved_model:
            self._last_saved_model = chosen
            self.core.submit(self._store_last_model(chosen), lambda _s: None)

    def _model_selected(self, chosen: str) -> None:
        for index in range(self.model_selector.count()):
            if self.model_selector.itemData(index) == chosen:
                self.model_selector.setCurrentIndex(index)
                return

    # --- events ------------------------------------------------------------

    def _on_event(self, event: Any) -> None:
        """Every core event, in one place."""
        kind, payload = event.type, event.payload
        value = getattr(kind, "value", str(kind))
        run_id = payload.get("run_id") or getattr(event, "run_id", "")
        if run_id and self.current_run_id and run_id != self.current_run_id:
            if value not in ("status", "task_started", "task_progress", "task_finished", "task_failed"):
                return
        if value in ("request_retry", "request_started"):
            if not self.current_run_id or (run_id and run_id != self.current_run_id):
                return
            if value == "request_started":
                self._set_run_status("Waiting for model · Stop available")
            else:
                self._retry_deadline = time.monotonic() + max(0.0, float(payload.get("delay", 0)))
                reason = str(payload.get("reason", "temporary provider error"))
                self._retry_description = (
                    f"{payload.get('attempt', '?')}/{payload.get('max_attempts', '?')} in "
                    "{seconds}s — " + reason.replace("{", "{{").replace("}", "}}")
                )
                self._update_retry_countdown()
                self._retry_timer.start()
            return
        if kind is EventType.TEXT:
            text = payload.get("text", "")
            if text:
                self.chat.end_stream()
                self.chat.add_message("assistant", text)
                self._answer_seen = True
                self._set_run_status("Receiving answer · Stop available")
        elif kind is EventType.TOKEN:
            self._answer_seen = self._answer_seen or bool(payload.get("text"))
            self._set_run_status("Receiving answer · Stop available")
            if not self.chat._streaming:
                self.chat.start_stream()
            self.chat.append_stream(payload.get("text", ""))
        elif kind is EventType.MESSAGE_COMPLETE:
            self.chat.end_stream()
        elif kind is EventType.TOOL_STARTED and not payload.get("streaming"):
            self._set_run_status(f"Running tool: {payload.get('tool', '')} · Stop available")
            call_id = payload.get("call_id", "")
            if call_id and call_id not in self._cards_by_tool:
                self._cards_by_tool[call_id] = payload.get("tool", "")
                self.chat.end_stream()
                self.chat.add_tool(call_id, payload.get("tool", ""), arguments=payload.get("arguments", {}))
        elif kind in (EventType.TOOL_FINISHED, EventType.TOOL_FAILED):
            call_id = payload.get("call_id", "")
            self.chat.finish_tool(
                call_id,
                is_error=kind is EventType.TOOL_FAILED,
                text=payload.get("text", ""),
                duration_ms=float(payload.get("duration_ms", 0.0) or 0.0),
                images=int(payload.get("images", 0) or 0),
                image_data=list(payload.get("image_data", []) or []),
            )
        elif kind is EventType.RUN_STARTED:
            self.current_run_id = payload.get("run_id", event.run_id)
            self._set_run_status("Waiting for model · Stop available")
        elif kind in (EventType.RUN_FINISHED, EventType.RUN_FAILED):
            self.current_run_id = ""
            self.chat.end_stream()
            self.chat.set_busy(False)
            if kind is EventType.RUN_FAILED:
                reason = str(payload.get("detail") or payload.get("reason") or "Unknown error")
                self._run_failure = reason
                self._set_run_status("Cancelled" if reason == "cancelled" else f"Failed: {reason}")
            else:
                self._set_run_status("Completed")
        elif kind is EventType.USAGE:
            self.cost_status.setText(
                f"{payload.get('input_tokens', 0)} in / {payload.get('output_tokens', 0)} out · "
                f"{format_money(float(payload.get('cost_usd', 0.0)))}"
            )
        elif kind in (
            EventType.TASK_STARTED,
            EventType.TASK_PROGRESS,
            EventType.TASK_FINISHED,
            EventType.TASK_FAILED,
        ):
            self.tasks.add_or_update(payload)
        elif kind is EventType.STATUS:
            self.mcp_status.setText(f"MCP: {payload.get('server', '?')} {payload.get('state', '')}")

    def _on_error(self, code: str, message: str) -> None:
        if self._saving:
            self._save_finished(("", message))
        self._workspace_busy = False
        self.chat.add_note(f"{code}: {message}", role="error")
        # Generic bridge errors can belong to scene refresh or settings, not the run.
        if code == "RUN_FAILED" or self._run_failure:
            self.current_run_id = ""
            self._active_agent = None
            self.chat.end_stream()
            self.chat.set_busy(False)
            self._run_failure = message
            self._set_run_status(f"Failed: {message}")

    # --- scene -------------------------------------------------------------

    def refresh_scene(self) -> None:
        self.core.submit(self._read_scene(), self._scene_read)

    async def _read_scene(self) -> dict[str, Any]:
        assert self.context.mcp is not None
        if not self.context.mcp.any_connected():
            return {"error": "no MCP server is connected"}
        outcome = await self.context.mcp.call_tool("blender.get_scene", {})
        if outcome.is_error:
            return {"error": outcome.error_code or outcome.text[:120]}
        try:
            payload = json.loads(outcome.text)
        except json.JSONDecodeError:
            return {"error": "unreadable scene payload"}
        payload["_read_at"] = time.time()
        return payload

    def _scene_read(self, payload: dict[str, Any]) -> None:
        if "error" in payload:
            self.scene.show_error(str(payload["error"]))
            return
        connected = "connected" if self.context.mcp and self.context.mcp.any_connected() else "not connected"
        self.scene.show_scene(payload, connected)

    def _close_transaction(self, tool: str, message: str) -> None:
        """Commit or roll back a transaction the agent left open.

        Explicit, and on a button, because both outcomes destroy something: a
        commit keeps work the user may not want, and a rollback discards work they
        may. The studio is not going to choose for them.
        """
        self.core.submit(self._call(tool), lambda text: self._transaction_closed(text or message))

    async def _call(self, tool: str) -> str:
        assert self.context.mcp is not None
        outcome = await self.context.mcp.call_tool(tool, {})
        return outcome.text if outcome.is_error else ""

    def _transaction_closed(self, text: str) -> None:
        if text.startswith("blender."):
            self.chat.add_note(text, role="error")
            return
        self.chat.add_note(text, role="system")
        self.refresh_scene()

    # --- models and settings ----------------------------------------------

    def _edit_provider(self, payload: dict[str, Any]) -> None:
        self.core.submit(self._persist_provider_editor(payload), self._provider_saved)

    async def _persist_provider_editor(self, payload: dict[str, Any]) -> str:
        from dataclasses import replace
        from urllib.parse import urlparse
        from app.llm.registry import PROVIDER_TYPES, ProviderConfig
        from app.storage.secrets import key_name

        registry = self.context.llm
        assert registry is not None
        name = payload["name"].strip()
        original = payload.get("original", "")
        existing = registry.config(original) if original else None
        if not name or (original and (existing is None or name != original)):
            raise ValueError("Provider name is required; existing names cannot be changed.")
        if not original and registry.config(name):
            raise ValueError("This provider already exists. Select its row to update it.")
        kind, url = payload["kind"], payload["base_url"]
        if kind not in PROVIDER_TYPES:
            raise ValueError(f"Unknown provider kind. Choose: {', '.join(PROVIDER_TYPES)}")
        if kind == "openai-compatible" and not url:
            raise ValueError("An OpenAI-compatible provider needs a Base URL.")
        if url and kind != "mock":
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Base URL must be an HTTP or HTTPS URL.")
        config = replace(existing) if existing else ProviderConfig(name=name)
        config.kind, config.base_url = kind, url
        config.enabled = bool(payload["enabled"])
        config.options = dict(payload["options"])
        if not existing and payload.get("model_id"):
            config.models = [{"id": payload["model_id"], "supports_tools": True}]
        ids = [m["id"] for m in config.models]
        default = payload.get("default_model", "") or config.default_model
        if default and default not in ids:
            raise ValueError("The default model must be in this provider's model list.")
        config.default_model = default or (ids[0] if ids else "")
        configs = [config if c.name == name else c for c in registry.configs()]
        if not existing:
            configs.append(config)
        await self._save_setting("llm_providers", [c.to_dict() for c in configs])
        if payload.get("api_key") and self.context.secrets:
            self.context.secrets.set(key_name(name), payload["api_key"])
        await registry.invalidate(name)
        registry.add(config)
        return name

    def _edit_model(self, payload: dict[str, Any]) -> None:
        self.core.submit(self._persist_model_editor(payload), self._provider_model_saved)

    async def _persist_model_editor(self, payload: dict[str, Any], remove: bool = False) -> str:
        import math
        from dataclasses import replace

        registry = self.context.llm
        assert registry is not None
        provider, model_id = payload["provider"], payload["id"].strip()
        config = registry.config(provider)
        if config is None or not model_id:
            raise ValueError("Select a provider and enter an exact model ID.")
        original = payload.get("original", "")
        models = [dict(m) for m in config.models]
        if original and not any(m["id"] == original for m in models):
            raise ValueError("The original model no longer exists. Refresh and select it again.")
        if not remove:
            if any(m["id"] == model_id and m["id"] != original for m in models):
                raise ValueError("This model ID already exists for this provider.")
            if payload["context_window"] <= 0 or any(not math.isfinite(payload[k]) or payload[k] < 0 for k in ("input_price", "output_price")):
                raise ValueError("Context must be positive and prices finite and nonnegative.")
            entry = next((dict(m) for m in models if m["id"] == original), {})
            entry.update({k: payload[k] for k in ("id", "supports_tools", "supports_vision", "context_window", "input_price", "output_price")})
            if original:
                models = [entry if m["id"] == original else m for m in models]
            else:
                models.append(entry)
        else:
            models = [m for m in models if m["id"] != model_id]
        default = config.default_model
        if not remove and (payload.get("default") or default == original):
            default = model_id
        if default not in [m["id"] for m in models]:
            default = models[0]["id"] if models else ""
        updated = replace(config, models=models, default_model=default)
        await self._save_setting("llm_providers", [(updated if c.name == provider else c).to_dict() for c in registry.configs()])
        await registry.invalidate(provider)
        registry.add(updated)
        chosen = f"{provider}:{default if remove else model_id}" if (default if remove else model_id) and config.enabled else ""
        if not chosen:
            available = registry.models()
            chosen = f"{available[0].provider}:{available[0].id}" if available else ""
        if self.context.studio is not None:
            await self.context.studio.settings.set(LAST_MODEL_KEY, chosen)
        return chosen

    def _remove_model_config(self, provider: str, model_id: str) -> None:
        self.core.submit(self._persist_model_editor({"provider": provider, "id": model_id, "original": model_id}, remove=True), self._provider_model_saved)

    def _remove_provider_config(self, name: str) -> None:
        self.core.submit(self._delete_provider_config(name), self._provider_model_saved)

    async def _delete_provider_config(self, name: str) -> str:
        registry = self.context.llm
        assert registry is not None
        await self._save_setting("llm_providers", [c.to_dict() for c in registry.configs() if c.name != name])
        await registry.invalidate(name)
        registry.remove(name)
        chosen = await self._read_last_model()
        if chosen.startswith(name + ":"):
            models = registry.models()
            chosen = f"{models[0].provider}:{models[0].id}" if models else ""
            if self.context.studio is not None:
                await self.context.studio.settings.set(LAST_MODEL_KEY, chosen)
        return chosen

    def _configure_three_d(self, config: dict[str, Any], key: str) -> None:
        self.core.submit(self._persist_three_d(config, key), self._three_d_saved)

    async def _persist_three_d(self, values: dict[str, Any], key: str) -> str:
        from app.core.settings import ThreeDConfig
        from app.providers3d.registry import PROVIDER_TYPES
        from app.storage.secrets import key_name

        unknown = set(values) - set(ThreeDConfig.model_fields)
        if unknown:
            raise ValueError(f"Unknown 3D options: {', '.join(sorted(unknown))}")
        config = ThreeDConfig.model_validate(values)
        if config.provider not in PROVIDER_TYPES:
            raise ValueError(f"Unknown 3D provider. Choose: {', '.join(PROVIDER_TYPES)}")
        if config.poll_interval <= 0 or config.poll_timeout <= 0 or config.max_credits < 0:
            raise ValueError("Polling times must be positive; credit limit must be nonnegative.")
        await self._save_setting("three_d", config.model_dump())
        if key and self.context.secrets:
            self.context.secrets.set(key_name(config.provider), key)
        if self.context.three_d:
            await self.context.three_d.close_all()
            self.context.three_d.configure(config)
            if self.context.secrets:
                self.context.three_d.use_secrets(self.context.secrets)
            else:
                self.context.three_d.provider()
        return config.provider

    def _three_d_saved(self, name: str) -> None:
        self.models.show_three_d(self.context.settings.three_d.model_dump())
        self.models.set_hint(f"3D settings for '{name}' saved. Keys are stored separately.")
        self._show_three_d_status()

    def _save_provider(self, name: str, kind: str, base_url: str, api_key: str, model_id: str = "") -> None:
        if not name:
            self.chat.add_note("A provider needs a name.", role="error")
            return
        self.core.submit(self._add_provider(name, kind, base_url, api_key, model_id), self._provider_saved)

    async def _add_provider(
        self, name: str, kind: str, base_url: str, api_key: str, model_id: str = ""
    ) -> str:
        from urllib.parse import urlparse

        from app.llm.registry import PROVIDER_TYPES, ProviderConfig

        assert self.context.llm is not None
        name, kind, base_url, model_id = name.strip(), kind.strip(), base_url.strip(), model_id.strip()
        if not name:
            raise ValueError("A provider needs a name.")
        if self.context.llm.config(name) is not None:
            raise ValueError(f"Provider '{name}' already exists. Use Save key to update its API key.")
        if kind not in PROVIDER_TYPES:
            raise ValueError(f"Unknown provider kind '{kind}'. Use: {', '.join(PROVIDER_TYPES)}")
        if kind == "openai-compatible" and not base_url:
            raise ValueError("An OpenAI-compatible provider needs a Base URL.")
        if base_url:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Base URL must be an HTTP or HTTPS URL.")
        if not model_id:
            raise ValueError("Enter the exact model ID supplied by your provider.")
        config = ProviderConfig(
            name=name,
            kind=kind,
            base_url=base_url,
            default_model=model_id,
            models=[{"id": model_id, "supports_tools": True}],
        )
        if api_key.strip() and self.context.secrets is not None:
            from app.storage.secrets import key_name

            self.context.secrets.set(key_name(name), api_key.strip())
        configs = [entry.to_dict() for entry in self.context.llm.configs()] + [config.to_dict()]
        await self._save_setting("llm_providers", configs)
        self.context.llm.add(config)
        return name

    def _provider_saved(self, name: str) -> None:
        self.models.provider_key.clear()
        self.models.set_hint(f"Provider '{name}' saved.")
        self.chat.add_note(f"Provider '{name}' saved. Select its model to start chatting.", role="system")
        self._show_providers()
        self.core.submit(self._load_models(), self._models_loaded)

    def _save_provider_model(self, provider: str, model_id: str) -> None:
        if not provider or not model_id.strip():
            self.models.set_hint("Select a provider and enter its exact Model ID.")
            return
        self.core.submit(self._update_provider_model(provider, model_id.strip()), self._provider_model_saved)

    async def _update_provider_model(self, provider: str, model_id: str) -> str:
        from dataclasses import replace

        assert self.context.llm is not None
        config = self.context.llm.config(provider)
        if config is None:
            raise ValueError(f"No provider named '{provider}'.")
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("Enter the exact Model ID.")
        models = list(config.models)
        if not any(entry["id"] == model_id for entry in models):
            models.append({"id": model_id, "supports_tools": True})
        updated = replace(config, default_model=model_id, models=models)
        configs = [
            (updated if entry.name == provider else entry).to_dict()
            for entry in self.context.llm.configs()
        ]
        await self._save_setting("llm_providers", configs)
        await self.context.llm.invalidate(provider)
        self.context.llm.add(updated)
        chosen = f"{provider}:{model_id}"
        await self._store_last_model(chosen)
        return chosen

    def _provider_model_saved(self, chosen: str) -> None:
        self._last_model = chosen
        self.models.set_hint(f"Model '{chosen}' saved and selected.")
        self._show_providers()
        self.core.submit(self._load_models(), self._models_loaded)

    def _set_api_key(self, provider: str, api_key: str) -> None:
        if not provider:
            return
        from app.providers3d.registry import PROVIDER_TYPES as THREE_D_TYPES

        if self.context.llm is not None and self.context.llm.config(provider) is None and provider not in THREE_D_TYPES:
            # A key stored under a name nothing is configured for is a trap: it
            # looks saved, and the provider it was meant for stays "no key".
            self.chat.add_note(f"There is no provider called '{provider}'.", role="error")
            return
        self.core.submit(self._store_key(provider, api_key), self._key_stored)

    async def _store_key(self, provider: str, api_key: str) -> str:
        from app.storage.secrets import key_name

        if self.context.secrets is None:
            return provider
        if api_key:
            self.context.secrets.set(key_name(provider), api_key)
        else:
            self.context.secrets.delete(key_name(provider))
        if self.context.llm is not None:
            await self.context.llm.invalidate(provider)
        if self.context.three_d is not None:
            self.context.three_d.use_secrets(self.context.secrets)
            # An unknown provider name raises, and the caller is told so by the
            # check that follows; here it only means "nothing to rebuild".
            with contextlib.suppress(Exception):
                self.context.three_d.provider(provider)
        return provider

    def _key_stored(self, provider: str) -> None:
        self.models.existing_key.clear()
        self.models.provider_key.clear()
        self.models.three_d_key.clear()
        self.chat.add_note(f"Key for '{provider}' saved.", role="system")
        # The table still shows the old state, so the column that says "no key"
        # would contradict the note above it until something else refreshed it.
        self._show_providers()
        self._show_three_d_status()

    def _setting_changed(self, key: str, value: Any) -> None:
        self.core.submit(self._save_setting(key, value), lambda _: None)

    async def _save_setting(self, key: str, value: Any) -> None:
        if self.context.studio is not None:
            await self.context.studio.settings.set(key, value)
        self.context.settings.apply_overrides({key: value})

    def _mcp_changed(self, fields: dict[str, Any]) -> None:
        self.core.submit(self._save_mcp(fields), self._mcp_saved)

    async def _save_mcp(self, fields: dict[str, Any]) -> None:
        """Upsert one MCP server, by name, and leave the others alone.

        This used to assign a one-element list, which meant that saving the
        Blender command in the window quietly deleted every other configured
        server -- from the settings, from the database, and from the manager.
        A person with a filesystem server configured lost it by clicking Save.
        """
        from app.core.context import mcp_server_info
        from app.core.settings import MCPServerConfig

        assert self.context.studio is not None and self.context.mcp is not None
        config = MCPServerConfig(
            name=fields.get("name") or "Blender MCP",
            command=fields["command"],
            args=list(fields["args"]),
            cwd=fields["cwd"],
            blender_port=int(fields["blender_port"]),
            connect_timeout=float(fields["connect_timeout"]),
        )
        servers = [
            config if server.name == config.name else server for server in self.context.settings.mcp_servers
        ]
        if all(server.name != config.name for server in servers):
            servers.append(config)
        self.context.settings.mcp_servers = servers
        await self.context.studio.settings.set("mcp_servers", [s.model_dump() for s in servers])
        await self.context.mcp.disconnect_all()
        self.context.mcp.configure([mcp_server_info(s) for s in servers])
        return await self.context.mcp.connect_all()

    def _mcp_saved(self, statuses: list[Any]) -> None:
        self._servers_connected(statuses)

    def _page_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if PAGES[index] == "Scene":
            self.refresh_scene()

    # --- tasks -------------------------------------------------------------

    def _cancel_task(self, task_id: str) -> None:
        self.core.submit(self._cancel_one_task(task_id), lambda ok: None)

    async def _cancel_one_task(self, task_id: str) -> bool:
        if self.context.tasks is None:
            return False
        return await self.context.tasks.cancel(task_id)

    # --- benchmark ---------------------------------------------------------

    def _run_benchmark(self, blend: str, models: str) -> None:
        specs = [
            ModelSpec(provider=item.split(":", 1)[0], model=item.split(":", 1)[1] if ":" in item else "")
            for item in (part.strip() for part in models.split(","))
            if item
        ]
        if not specs:
            self.chat.add_note("Name at least one model as provider:model.", role="error")
            return
        prompts = [line.strip() for line in self.benchmark.prompts.toPlainText().splitlines() if line.strip()]
        tasks = [BenchmarkTask(prompt=prompt) for prompt in prompts]
        self.benchmark.run_button.setEnabled(False)
        self.benchmark.cancel_button.setEnabled(True)
        self.core.submit(self._benchmark(tasks, specs, blend), self._benchmark_done)

    async def _benchmark(self, tasks: list[BenchmarkTask], models: list[ModelSpec], blend: str) -> Any:
        assert self.context.mcp is not None
        await self.context.mcp.connect_all()
        runner = BenchmarkRunner(
            self.context.llm,
            self.context.mcp,
            bus=self.context.bus,
            system_prompt=self.context.settings.agent.system_prompt,
        )
        # Saved as each run finishes, not at the end: a benchmark cancelled
        # half-way is still three runs somebody may want to look at.
        run_ids: list[str] = []

        async def persist(comparison: Any, outcome: Any) -> None:
            if self._benchmarks is None or not suite_id:
                return
            run_ids.append(await self._benchmarks.save_run(suite_id, outcome, task_index=len(run_ids)))

        suite_id = ""
        if self._benchmarks is not None:
            prompts = ", ".join(task.prompt for task in tasks)[:120]
            models_label = ", ".join(spec.label() for spec in models)
            suite_id = await self._benchmarks.create_suite(
                name=f"{len(models)} model(s) x {len(tasks)} task(s)",
                description=f"{models_label} -- {prompts}",
            )
        comparison = await runner.run_suite(
            tasks, models, blend=Path(blend) if blend else None, persist=persist
        )
        return comparison, run_ids, suite_id

    def _benchmark_done(self, result: Any) -> None:
        if not isinstance(result, tuple):  # pragma: no cover - defensive
            result = (result, [], "")
        comparison, run_ids, suite_id = result
        self.benchmark.run_button.setEnabled(True)
        self.benchmark.cancel_button.setEnabled(False)
        rows = []
        for index, outcome in enumerate(comparison.outcomes):
            run_id = run_ids[index] if index < len(run_ids) else ""
            rows.append(outcome.row() | {"id": run_id})
        self.benchmark.show_comparison(rows)
        if not run_ids:
            self.benchmark.summary.setText(
                self.benchmark.summary.text() + " Nothing was saved, so these runs cannot be scored."
            )
        else:
            self.benchmark.summary.setText(
                self.benchmark.summary.text() + f" Saved as suite {suite_id}; score a row to keep a review."
            )
        if comparison.outcomes and comparison.fastest() is not None:
            # Reported as a fact about the run, not as a recommendation.
            self.benchmark.summary.setText(
                self.benchmark.summary.text() + f" Shortest run: {comparison.fastest().model.label()}."
            )

    def _cancel_benchmark(self) -> None:
        self.chat.add_note("The benchmark will stop after the current run.", role="system")

    def _save_review(self, run_id: str, scores: dict[str, Any]) -> None:
        if not run_id or self._benchmarks is None:
            return
        self.core.submit(
            self._store_review(run_id, scores), lambda _: self.benchmark.summary.setText("Score saved.")
        )

    async def _store_review(self, run_id: str, scores: dict[str, Any]) -> None:
        assert self._benchmarks is not None
        await self._benchmarks.review(
            run_id,
            geometry=scores.get("Geometry"),
            materials=scores.get("Materials"),
            instruction_following=scores.get("Following"),
            composition=scores.get("Composition"),
            overall=scores.get("Overall"),
            notes=str(scores.get("notes", "")),
        )

    # --- shutdown ----------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        self.core.stop()
        super().closeEvent(event)


def run_gui(context: Any) -> int:
    """Start the window. Returns the process exit code."""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Blender AI Studio")

    core = CoreThread(context)
    core.start()
    window = MainWindow(context, core)
    window.show()
    window.start()
    return app.exec()
