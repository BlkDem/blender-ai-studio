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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.agent import Agent, LocalTool
from app.core.cost import Budget
from app.core.errors import ConfigurationError
from app.core.events import EventBus
from app.core.settings import AgentConfig, BudgetConfig, MCPServerConfig, Settings, ThreeDConfig
from app.core.task_manager import TaskManager
from app.llm.base import LLMProvider, ToolSpec
from app.llm.models import ModelInfo
from app.llm.registry import LLMRegistry, registry_from_settings
from app.mcp.manager import MCPManager
from app.mcp.models import ServerInfo
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
        self.tasks = TaskManager(self.bus)
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
    ) -> Agent:
        """An agent wired to this studio's MCP servers, tools and budget.

        The model is optional: an empty one means the provider's configured
        default, which is what a user who has set a default expects.
        """
        assert self.llm is not None and self.mcp is not None, "open() first"
        provider, resolved_model = self.llm.resolve(provider_name, model)
        model_info = self.llm.find(provider_name, resolved_model)
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
        )

    def assets_tool(self) -> list[LocalTool]:
        """``generate_3d_asset``, when a 3D provider is enabled.

        Registered as a local tool rather than an MCP one because it is a
        capability of the studio: the agent asks for an asset, and which service
        makes it is the registry's business.
        """
        if self.three_d is None or not self.three_d.any_enabled():
            return []
        return [three_d_tool(self.three_d, self.settings.three_d)]


def three_d_tool(registry: ThreeDRegistry, config: ThreeDConfig) -> LocalTool:
    """The one tool the model sees for 3D generation."""
    from app.providers3d.base import AssetRequest
    from app.providers3d.models import TaskStatus

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
        return {
            "provider": provider_name,
            "task_id": task.id,
            "provider_task_id": task.provider_task_id,
            "status": str(task.status),
            "kind": kind,
            "credits": task.credits,
            "note": (
                "Generation is asynchronous. The asset is not in the scene yet; "
                "check the Tasks panel, then import it when it is ready."
            ),
            "status_enum": str(TaskStatus.RUNNING),
        }

    return LocalTool(
        name="generate_3d_asset",
        description=(
            "Generate a 3D asset from a text description, or from an image URL. "
            "Use it when a real model of the object is wanted rather than an "
            "approximation built from primitives. Returns a task id: generation "
            "is asynchronous, and the asset is imported separately."
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
