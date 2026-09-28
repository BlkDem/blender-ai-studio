"""The studio's own wiring, for the app, the CLI and the acceptance run.

One place that knows how a :class:`~app.core.settings.Settings` becomes the
objects that do the work. The GUI, ``--check`` and the acceptance script all build
the studio the same way, so a configuration that works in one works in all three.

``--check`` and the acceptance run both start from here, which is the point: a
diagnostic that builds a different application than the one being used is a
diagnostic that lies.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.agent import Agent, LocalTool
from app.core.cost import Budget
from app.core.errors import ConfigurationError, ThreeDError
from app.core.events import EventBus
from app.core.settings import AgentConfig, BudgetConfig, MCPServerConfig, Settings, ThreeDConfig
from app.core.task_manager import TaskManager
from app.llm.base import LLMProvider, Message, ToolSpec
from app.llm.models import ModelInfo
from app.llm.registry import LLMRegistry, registry_from_settings
from app.mcp.manager import MCPManager
from app.mcp.models import ServerInfo
from app.providers3d.models import ProviderTask, TaskStatus
from app.providers3d.registry import ThreeDRegistry
from app.storage.repositories import Studio
from app.storage.secrets import SecretStore

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Console logging, without a handler being added twice.

    The GUI runs the core on its own thread, so logging is configured once, here,
    and never from a window.
    """
    root = logging.getLogger()
    if any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        root.setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    root.setLevel(level.upper())


def mcp_server_info(config: MCPServerConfig | dict) -> ServerInfo:
    """The settings form of a server, as the client wants it.

    The bridge address lives in the *child's* environment, not in the client's
    configuration, so it is resolved here. Getting this wrong is quiet and
    expensive: the MCP session connects, every tool answers, and each one reports
    that no Blender is attached — to a bridge the server never told the add-on
    about.
    """
    if not isinstance(config, MCPServerConfig):
        # A hand-edited STUDIO_MCP_SERVERS, or a value from an older release. The
        # AttributeError that would otherwise surface -- "'str' object has no
        # attribute 'env'" -- says nothing about which setting is wrong.
        try:
            config = MCPServerConfig.model_validate(config)
        except Exception as exc:
            raise ConfigurationError(
                f"The MCP server configuration is not valid: {exc}",
                hint="Check STUDIO_MCP_SERVERS: it must be a JSON list of objects "
                "with at least a name and a command.",
            ) from exc
    env = dict(config.env)
    if config.kind == "stdio" and config.command:
        for key, value in config.child_env().items():
            if key in ("BLENDER_HOST", "BLENDER_PORT"):
                env[key] = value
    return ServerInfo(
        name=config.name,
        kind=config.kind,
        command=config.command,
        args=list(config.args),
        env=env,
        cwd=config.cwd,
        enabled=config.enabled,
        connect_timeout=config.connect_timeout,
        tool_timeout=config.tool_timeout,
    )


def budget_from(config: AgentConfig, budgets: BudgetConfig | None = None) -> Budget:
    limits = budgets or BudgetConfig()
    return Budget(
        max_steps=config.max_steps,
        max_tool_calls=config.max_tool_calls,
        max_seconds=config.max_seconds,
        max_request_cost=min(config.max_request_cost, limits.llm_budget_per_request),
        max_session_cost=min(config.max_session_cost, limits.llm_budget_per_session),
        max_3d_credits=min(config.max_3d_credits, limits.tripo_credit_limit),
    )


@dataclass
class AppContext:
    """Everything the studio is made of, assembled once."""

    settings: Settings
    bus: EventBus = field(default_factory=EventBus)
    studio: Studio | None = None
    secrets: SecretStore | None = None
    llm: LLMRegistry | None = None
    mcp: MCPManager | None = None
    three_d: ThreeDRegistry | None = None
    tasks: TaskManager | None = None
    #: The project the window is working in, or None for a session that is not
    #: in one. Every conversation, and every 3D task, is filed under it.
    current_project: str | None = None
    #: That project's name, kept beside the id so the model can be told what it
    #: is working on. The id alone reaches the system prompt as a bare string a
    #: model cannot use, and asking the window for a name on every turn would be
    #: a database read per step to learn something that changes rarely.
    project_name: str = ""

    async def open(self) -> AppContext:
        """Migrate the database, load overrides and build the registries."""
        settings = self.settings
        self.secrets = self.secrets or SecretStore(settings.resolved_secrets_path)
        self.studio = await Studio.open(settings.resolved_database_path)
        settings.apply_overrides(await self.studio.settings.all())
        self.llm = registry_from_settings(settings, self.secrets)
        self.mcp = MCPManager(self.bus)
        self.mcp.configure(mcp_server_info(config) for config in settings.mcp_servers)
        self.three_d = ThreeDRegistry(self.bus, settings.three_d)
        self.three_d.use_secrets(self.secrets)
        # Built eagerly so the Models panel and ``--check`` can show a provider
        # that exists but has no key, which is the state a user starts in.
        try:
            self.three_d.provider()
        except ConfigurationError:
            logger.info("no 3D provider selected; the capability will not be offered")
        # project_id is a callable, not a value: a project can be opened or
        # closed while the window is up, and a task manager built with the
        # project that happened to be open at start-up would file every later
        # generation under it.
        self.tasks = TaskManager(self.bus, studio=self.studio, project_id=lambda: self.current_project)
        return self

    async def close(self) -> None:
        # Each registry is a different type with the same method; they share no
        # base class, and inventing one for a shutdown path would be worse.
        for name, close in (
            ("mcp", self.mcp.disconnect_all() if self.mcp else None),
            ("llm", self.llm.close_all() if self.llm else None),
            ("3d", self.three_d.close_all() if self.three_d else None),
        ):
            if close is None:
                continue
            try:
                await close
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("closing %s failed", name, exc_info=True)
        if self.tasks is not None:
            await self.tasks.cancel_all()
        if self.studio is not None:
            self.studio.close()

    # --- building an agent -------------------------------------------------

    def agent(
        self,
        provider_name: str,
        model: str = "",
        *,
        stream: bool = True,
        extra_tools: Sequence[LocalTool] = (),
        budget: Budget | None = None,
        conversation_id: str = "",
        project_id: str | None = None,
        history: Sequence[Message] = (),
    ) -> Agent:
        """An agent wired to this studio's MCP servers, tools and budget.

        The model is optional: an empty one means the provider's configured
        default, which is what a user who has set a default expects.

        ``history`` seeds the agent's memory with an earlier session, so the
        first turn after reopening the window continues it. The agent keeps
        growing that memory itself from there.

        ``project_id`` defaults to the open project. Passing ``""`` explicitly
        means no project, which the old ``is not None`` default could not
        express: a caller that wanted a run filed nowhere had no way to say so
        while one was open.
        """
        assert self.llm is not None and self.mcp is not None, "open() first"
        provider, resolved_model = self.llm.resolve(provider_name, model)
        model_info = self.llm.find(provider_name, resolved_model)
        chosen = project_id if project_id is not None else (self.current_project or "")
        return Agent(
            provider,
            resolved_model,
            self.mcp,
            bus=self.bus,
            model_info=model_info,
            system_prompt=self.settings.agent.system_prompt,
            budget=budget or budget_from(self.settings.agent, self.settings.budgets),
            local_tools=[*self.assets_tool(), *extra_tools],
            stream=stream,
            studio=self.studio,
            conversation_id=conversation_id,
            allow_execute_python=self.settings.agent.allow_execute_python,
            project_id=chosen or None,
            project_name=self.project_name if chosen else "",
            history=history,
        )

    def assets_tool(self) -> list[LocalTool]:
        """``generate_3d_asset``, when a 3D provider is enabled.

        Registered as a local tool rather than an MCP one because it is a
        capability of the studio: the agent asks for an asset, and which service
        makes it is the registry's business.
        """
        if self.three_d is None or not self.three_d.any_enabled():
            return []
        return [
            three_d_tool(
                self.three_d,
                self.settings.three_d,
                tasks=self.tasks,
                on_ready=self._import_generated,
                download_dir=self.asset_dir(),
            )
        ]

    def asset_dir(self) -> Path:
        """Where generated models are written, and the boundary to watch.

        The studio may well be on a different side of a filesystem from Blender,
        in which case a download that lands here cannot be opened there. The
        import reports that clearly, but naming the directory here means the
        failure arrives with a path in it instead of as a mystery.
        """
        # str(), not .strip(): a caller that assigned a Path here is not wrong,
        # and the failure should be about the directory, not its type.
        configured = str(self.settings.three_d.download_dir or "").strip()
        return Path(configured) if configured else self.settings.data_dir / "assets"

    def projects_dir(self) -> Path:
        """Where projects keep the files they are worked on.

        ``default_project_dir`` has been in the settings since the beginning and
        nothing has ever read it. A project's working file has to live somewhere
        anyway, and a setting nobody sets is not a place to put it: Blender may
        be on the other side of a filesystem, and the same reasoning as
        :meth:`asset_dir` applies.
        """
        configured = self.settings.default_project_dir
        return Path(configured) if configured else self.settings.data_dir / "projects"

    def workspace_path(self, project: Any) -> Path:
        """The file a project is worked on: its own copy of the starting .blend.

        Keyed on the id rather than the name so a project can be renamed without
        losing its file, and so a name with a slash or a colon in it is not a
        path.
        """
        return self.projects_dir() / str(project.id) / "workspace.blend"

    async def _import_generated(self, task: Any, path: Path) -> Any:
        """Hand a finished model to Blender, when there is a way in."""
        from app.providers3d.importer import can_import, import_asset

        if self.mcp is None or not can_import(self.mcp):
            from app.providers3d.importer import EXECUTE_PYTHON

            raise ThreeDError(
                "The model is ready, but there is no way into Blender",
                hint=(
                    f"{EXECUTE_PYTHON} is disabled in Settings, so the asset was "
                    f"downloaded to {path} and left there."
                ),
                path=str(path),
            )
        return await import_asset(self.mcp, path)


def three_d_tool(
    registry: ThreeDRegistry,
    config: ThreeDConfig,
    tasks: TaskManager | None = None,
    on_ready: Callable[[ProviderTask, Path], Awaitable[Any]] | None = None,
    download_dir: Path | None = None,
) -> LocalTool:
    """The one tool the model sees for 3D generation.

    The submission is awaited -- one request, and a refusal such as an empty
    account has to reach the model in the same breath, not minutes later. What
    follows is the part that takes minutes, and that runs as a tracked task, so
    the Tasks panel, the database and the agent's own costs all see the same
    numbers.
    """
    from app.providers3d.base import AssetRequest

    async def generate(arguments: dict[str, Any], *, run_id: str = "") -> dict[str, Any]:
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return {"error": "prompt is required"}
        image = arguments.get("image_url") or arguments.get("image_base64")
        kind = "image_to_3d" if image else "text_to_3d"
        provider_name = str(arguments.get("provider") or config.provider)
        provider = registry.provider(provider_name)
        request = AssetRequest(
            prompt=prompt,
            image_url=image,
            model=str(arguments.get("model") or config.model or ""),
            quality=str(arguments.get("quality") or config.quality),
            texture=bool(arguments.get("texture", config.texture)),
        )
        try:
            task = await provider.create(request, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
            return {"error": str(exc), "provider": provider_name}
        if str(task.status) == str(TaskStatus.FAILED):
            return {
                "provider": provider_name,
                "status": str(task.status),
                "error": task.error,
                "credits": task.credits,
                "submitted": False,
            }

        answer: dict[str, Any] = {
            "provider": provider_name,
            "task_id": task.id,
            "provider_task_id": task.provider_task_id,
            "status": str(task.status),
            "kind": kind,
            "credits": task.credits,
            "submitted": True,
        }
        if tasks is None:
            answer["note"] = (
                "Generation is asynchronous and nothing is watching it: check the "
                "Tasks panel, then import it when it is ready."
            )
            return answer

        studio_task = await tasks.start(
            f"3D: {prompt[:48]}",
            _finish_asset(provider, task, config, on_ready, download_dir),
            provider=provider_name,
            run_id=run_id,
            payload={"prompt": prompt, "provider_task_id": task.provider_task_id, "kind": kind},
        )
        answer["studio_task_id"] = studio_task.id
        answer["note"] = (
            "Submitted. It is being generated now, and the finished model is "
            "imported into the scene automatically; watch it in the Tasks panel."
        )
        return answer

    return LocalTool(
        name="generate_3d_asset",
        # The description is the model's only guidance about when to reach for
        # this, so it names the boundary explicitly. A small model reading
        # "create a medieval wooden chest" and reaching for blender.create_object
        # is not a reasoning failure so much as a missing distinction, and the
        # difference between a specific modelled object and an arrangement of
        # primitives is the whole reason this tool exists.
        description=(
            "Generate a real 3D model of a specific object with a 3D generation "
            "service, from a text description or an image URL.\n"
            "Use it for any object a studio would have modelled: furniture, "
            "vehicles, animals, plants, weapons, buildings, props, characters.\n"
            "Examples: 'a medieval wooden chest', 'a rusty samurai helmet', "
            "'a dragon' -> this tool.\n"
            "Use blender.create_object instead for arrangement of primitives: "
            "'a table with four legs', 'a room', 'a stack of boxes'.\n"
            "Returns a task id: generation takes minutes, runs in the background, "
            "and the asset is imported into the scene afterwards."
        ),
        schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to generate, in detail."},
                "image_url": {"type": "string", "description": "Optional reference image URL."},
                "provider": {"type": "string", "description": "Which 3D provider to use."},
                "model": {"type": "string", "description": "Provider model name."},
                "quality": {"type": "string", "description": "draft, medium or high."},
                "texture": {"type": "boolean", "description": "Generate textures too."},
            },
            "required": ["prompt"],
        },
        handler=generate,
    )


def _finish_asset(
    provider: Any,
    task: Any,
    config: ThreeDConfig,
    on_ready: Callable[[Any, Path], Awaitable[Any]] | None,
    download_dir: Path | None,
) -> Callable[[Any], Awaitable[Any]]:
    """The slow half of a 3D request: wait, download, import.

    Written as a closure so the tracked task carries the provider task it is
    about rather than reaching for a registry of its own.
    """

    async def work(studio_task: Any) -> dict[str, Any]:
        finished = await provider.wait_for(
            task,
            interval=config.poll_interval,
            timeout=config.poll_timeout,
            on_progress=lambda part: _progress(studio_task, part),
        )
        # Two ways to come back unfinished, and neither may be downloaded: a
        # failure, and a timeout that leaves the task still running with the
        # reason in `error`. Fetching either would hand back whatever the
        # provider has so far and call it the finished model.
        if finished.status == TaskStatus.FAILED:
            raise ThreeDError(finished.error or "the provider reported a failure")
        if not finished.status.terminal:
            raise ThreeDError(finished.error or "the generation never finished")
        # The credits are spent whether or not the import works. Recording them
        # only at the end loses the bill for a run whose download or import
        # failed -- which is exactly the run a person needs to see the cost of.
        studio_task.credits = finished.credits
        destination = (download_dir or Path(".")) / f"{finished.provider_task_id or finished.id}.glb"
        studio_task.detail = "downloading"
        path = await provider.download(finished, destination)
        studio_task.detail = "importing"
        if on_ready is None:
            studio_task.detail = f"ready at {path}"
            return {"path": str(path), "imported": False}
        report = await on_ready(finished, path)
        studio_task.detail = f"{len(report.imported)} object(s) in the scene"
        return {"path": str(path), "imported": True, "objects": report.imported}

    return work


def _progress(studio_task: Any, part: Any) -> None:
    """Copy a provider's progress onto the task the user is watching.

    The providers disagree about units -- some count 0 to 1, some 0 to 100 -- and
    neither reports anything before the first poll. A bar that jumps to full
    because the field was empty is worse than one that sits still, so unknown
    stays unknown.
    """
    value = getattr(part, "progress", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return
    studio_task.progress = min(1.0, float(value) / 100.0 if value > 1 else float(value))


def describe_provider(provider: LLMProvider, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=provider.name)


def tools_summary(agent: Agent) -> str:
    specs: list[ToolSpec] = agent.tools()
    return ", ".join(spec.name for spec in specs)


def three_d_summary(registry: ThreeDRegistry) -> list[str]:
    return [f"{name}: {provider.name}" for name, provider in registry.all().items()]


def provider_summary(registry: ThreeDRegistry) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": type(provider).__name__,
            "enabled": provider.enabled,
            "ready": provider.is_configured(),
        }
        for name, provider in registry.all().items()
    ]
