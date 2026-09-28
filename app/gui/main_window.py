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
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.benchmark.runner import BenchmarkRunner, BenchmarkTask, ModelSpec
from app.benchmark.storage import BenchmarkStorage
from app.core.events import EventType
from app.gui.benchmark.panel import BenchmarkPanel
from app.gui.bridge import CoreThread, format_money
from app.gui.chat.widget import ChatView
from app.gui.models.panel import ModelsPanel
from app.gui.projects.panel import ProjectsPanel
from app.gui.scene.panel import ScenePanel
from app.gui.settings.panel import SettingsPanel
from app.gui.tasks.panel import TasksPanel

logger = logging.getLogger(__name__)

PAGES = ("Chat", "Scene", "Tasks", "Projects", "Benchmark", "Models", "Settings")


class MainWindow(QMainWindow):
    """The studio's one window."""

    def __init__(self, context: Any, core: CoreThread) -> None:
        super().__init__()
        self.context = context
        self.core = core
        self.setWindowTitle("Blender AI Studio")
        self.resize(1280, 820)
        self.setObjectName("main-window")

        self.current_run_id = ""
        self._active_agent: Any = None
        self._agent: Any = None
        self._agent_provider = ""
        self._agent_gate: bool | None = None
        self._conversation_id = ""
        self._cards_by_tool: dict[str, str] = {}
        self._benchmarks = BenchmarkStorage(context.studio.db) if context.studio else None

        # --- widgets --------------------------------------------------------
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("model-selector")
        self.model_selector.setMinimumWidth(240)
        self.model_selector.currentIndexChanged.connect(self._model_changed)

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

        self.setStatusBar(QStatusBar())
        self.mcp_status = QLabel("MCP: …")
        self.llm_status = QLabel("LLM: …")
        self.three_d_status = QLabel("3D: …")
        self.cost_status = QLabel("")
        for label in (self.mcp_status, self.llm_status, self.three_d_status):
            self.statusBar().addWidget(label)
        self.statusBar().addPermanentWidget(self.cost_status)

        # --- signals --------------------------------------------------------
        self.core.signals.event.connect(self._on_event)
        self.core.signals.error.connect(self._on_error)
        self.chat.submitted.connect(self.send)
        self.chat.stop_requested.connect(self.stop_run)
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
        self.settings.setting_changed.connect(self._setting_changed)
        self.settings.mcp_changed.connect(self._mcp_changed)
        self.tasks.cancel_requested.connect(self._cancel_task)
        self.benchmark.run_requested.connect(self._run_benchmark)
        self.benchmark.cancel_requested.connect(self._cancel_benchmark)
        self.benchmark.review_requested.connect(self._save_review)
        self.projects.create_requested.connect(self._create_project)
        self.projects.opened.connect(self._open_project)
        self.projects.closed.connect(self._close_project)
        self.projects.delete_requested.connect(self._delete_project)
        self.projects.blend_requested.connect(self._load_project_blend)
        self.models.select_first()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.menuBar().addAction(quit_action)

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
        self.core.submit(self._restore_conversation(), lambda _n: None)
        self.core.submit(self._load_projects(), lambda _p: None)
        self.settings.load(self.context.settings)
        self._show_three_d_status()
        self._show_providers()

    # --- projects ----------------------------------------------------------

    async def _load_projects(self) -> list[Any]:
        if self.context.studio is None:
            return []
        projects = await self.context.studio.projects.list()
        self.projects.show_projects(projects, self.context.current_project)
        return projects

    async def _create_project(self, name: str, blend: str) -> None:
        assert self.context.studio is not None
        project = await self.context.studio.projects.create(
            name, initial_blend=blend or None, default_model=self.current_model() or None
        )
        self.context.current_project = project.id
        self.projects.clear_inputs()
        await self._load_projects()
        self._new_conversation()
        self.chat.add_note(f"Project '{project.name}' is open. Turns are filed under it.", role="system")

    async def _open_project(self, project_id: str) -> None:
        assert self.context.studio is not None
        project = await self.context.studio.projects.get(project_id)
        if project is None:
            return
        self.context.current_project = project.id
        await self._load_projects()
        await self._restore_conversation()
        self.chat.add_note(
            f"Project '{project.name}' is open"
            + (f", starting from {project.initial_blend}" if project.initial_blend else "")
            + ".",
            role="system",
        )

    async def _close_project(self) -> None:
        self.context.current_project = None
        self._new_conversation()
        await self._load_projects()
        self.chat.add_note("Project closed. New turns are not filed anywhere.", role="system")

    async def _delete_project(self, project_id: str) -> None:
        assert self.context.studio is not None
        await self.context.studio.projects.delete(project_id)
        if self.context.current_project == project_id:
            self.context.current_project = None
            self._new_conversation()
        await self._load_projects()
        self.chat.add_note("Project deleted, with the conversations filed under it.", role="system")

    async def _load_project_blend(self, project_id: str, blend: str) -> None:
        """Open a project's starting file in Blender, from a copy.

        The original is never opened directly: Blender would then be editing the
        file the project starts from, and the next run would not start where this
        one did. The copy is what gets opened, and which of the three things
        actually happened is reported rather than assumed.
        """
        assert self.context.studio is not None and self.context.mcp is not None
        from app.benchmark.runner import RESET_COPY_ONLY, BenchmarkRunner, ModelSpec

        source = Path(blend)
        if not source.exists():
            self.chat.add_note(f"There is no such file: {blend}", role="error")
            return
        workdir = self.context.asset_dir().parent / "scenes"
        runner = BenchmarkRunner(self.context.llm, self.context.mcp, bus=self.context.bus, workdir=workdir)
        staged = runner.stage_scene(source, ModelSpec(provider="", model="project"), 0)
        state = await runner.reset_scene(staged)
        if state == RESET_COPY_ONLY:
            self.chat.add_note(
                f"Staged a copy at {staged} but Blender could not open it: "
                "enable blender.execute_python in Settings, or open the file yourself.",
                role="error",
            )
        else:
            self.chat.add_note(f"Blender is now on a copy of {source.name}: {staged}", role="system")

    def _new_conversation(self) -> None:
        """Start a fresh conversation, in whatever project is now open."""
        self._agent = None
        self._conversation_id = ""

    async def _restore_conversation(self) -> int:
        """Put the last conversation back on screen.

        Closing the window used to throw the transcript away even though every
        message was in the database. Reopening it is what makes the window a
        workspace rather than a slot machine.
        """
        if self.context.studio is None:
            return 0
        conversations = await self.context.studio.conversations.list(self.context.current_project)
        if not conversations:
            return 0
        latest = conversations[0]
        messages = await self.context.studio.messages.list(latest.id)
        if not messages:
            return 0
        self._conversation_id = latest.id
        self.chat.clear()
        for message in messages:
            self.chat.add_message(message.role, message.content or "")
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
        return [model.to_dict() for model in self.context.llm.models()]

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
                }
            )
        self.models.show_providers(entries)

    def _models_loaded(self, models: list[dict[str, Any]]) -> None:
        self.models.show_models(models)
        block = self.model_selector.blockSignals(True)
        self.model_selector.clear()
        for model in models:
            self.model_selector.addItem(
                f"{model['id']}  ·  {model['provider']}", f"{model['provider']}:{model['id']}"
            )
        self.model_selector.blockSignals(block)
        configured = [
            m for m in models if m.get("provider") and self.context.llm.is_configured(m["provider"])
        ]
        if configured:
            self.llm_status.setText(f"LLM: {configured[0]['provider']} ready")
        else:
            self.llm_status.setText("LLM: no API key")

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

    def send(self, text: str) -> None:
        """Start a run. The window stays responsive; the transcript fills in."""
        if not self.current_provider():
            self.chat.add_note("Choose a model first — the selector at the top is empty.", role="error")
            return
        self.chat.add_message("user", text)
        self.chat.set_busy(True)
        self._cards_by_tool.clear()
        self.core.submit(self._run_agent(text), self._run_finished)

    def _agent_for_turn(self) -> Any:
        # The gate is read at agent construction, so a change in Settings has to
        # make the next turn build a new agent.
        """One agent per conversation, not one per turn.

        A new agent every turn meant a new conversation every turn: the model was
        stateless, and the transcript in the window looked continuous anyway,
        which is worse than an obvious break -- the user reads a memory that is
        not there.
        """
        gate = self.context.settings.agent.allow_execute_python
        if self._agent is None or self._agent_provider != self.current_provider() or self._agent_gate != gate:
            self._agent = self.context.agent(
                self.current_provider(), self.current_model(), conversation_id=self._conversation_id
            )
            self._agent_provider = self.current_provider()
            self._agent_gate = gate
        return self._agent

    async def _run_agent(self, text: str) -> Any:
        agent = self._agent_for_turn()
        # Kept so Stop has something to stop. The agent was built and dropped
        # inside this coroutine, which made the button a no-op that still looked
        # like it worked: the run finished, the UI had already said otherwise.
        self._active_agent = agent
        try:
            return await agent.run(text)
        finally:
            self._active_agent = None

    def _run_finished(self, result: Any) -> None:
        self.chat.set_busy(False)
        self.chat.end_stream()
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

    def _model_changed(self) -> None:
        if self.current_provider():
            self.llm_status.setText(f"LLM: {self.current_provider()}")

    def _model_selected(self, chosen: str) -> None:
        provider, _, model = chosen.partition(":")
        for index in range(self.model_selector.count()):
            if self.model_selector.itemData(index) == chosen:
                self.model_selector.setCurrentIndex(index)
                return
        del provider, model

    # --- events ------------------------------------------------------------

    def _on_event(self, event: Any) -> None:
        """Every core event, in one place."""
        kind, payload = event.type, event.payload
        if kind is EventType.TOKEN:
            if not self.chat._streaming:
                self.chat.start_stream()
            self.chat.append_stream(payload.get("text", ""))
        elif kind is EventType.MESSAGE_COMPLETE:
            self.chat.end_stream()
        elif kind is EventType.TOOL_STARTED and not payload.get("streaming"):
            call_id = payload.get("call_id", "")
            if call_id and call_id not in self._cards_by_tool:
                self._cards_by_tool[call_id] = payload.get("tool", "")
                self.chat.add_tool(call_id, payload.get("tool", ""))
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
        elif kind in (EventType.RUN_FINISHED, EventType.RUN_FAILED):
            self.current_run_id = ""
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
        self.chat.add_note(f"{code}: {message}", role="error")
        self.chat.set_busy(False)

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

    def _save_provider(self, name: str, kind: str, base_url: str, api_key: str) -> None:
        if not name:
            self.chat.add_note("A provider needs a name.", role="error")
            return
        self.core.submit(self._add_provider(name, kind, base_url, api_key), self._provider_saved)

    async def _add_provider(self, name: str, kind: str, base_url: str, api_key: str) -> str:
        from app.llm.registry import ProviderConfig

        assert self.context.llm is not None
        if api_key and self.context.secrets is not None:
            from app.storage.secrets import key_name

            self.context.secrets.set(key_name(name), api_key)
        self.context.llm.add(
            ProviderConfig(
                name=name,
                kind=kind,
                base_url=base_url,
                default_model="",
                models=[{"id": "model-id", "supports_tools": True}],
            )
        )
        return name

    def _provider_saved(self, name: str) -> None:
        self.chat.add_note(f"Provider '{name}' saved. Add its models in the table below.", role="system")
        self.core.submit(self._load_models(), self._models_loaded)

    def _set_api_key(self, provider: str, api_key: str) -> None:
        if not provider:
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
        if self.context.three_d is not None:
            self.context.three_d.use_secrets(self.context.secrets)
            # An unknown provider name raises, and the caller is told so by the
            # check that follows; here it only means "nothing to rebuild".
            with contextlib.suppress(Exception):
                self.context.three_d.provider(provider)
        return provider

    def _key_stored(self, provider: str) -> None:
        self.chat.add_note(f"Key for '{provider}' saved.", role="system")
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
