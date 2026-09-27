"""Configuration.

Precedence, lowest to highest: built-in defaults, then ``.env``/environment
(``STUDIO_*``), then the ``settings`` table in the database, which is where the
GUI writes what a person changed. Secrets are deliberately *not* part of this:
they live in :mod:`app.storage.secrets` so that a settings dump, a log line or a
project export can never contain one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce(annotation: Any, value: Any) -> Any:
    """Validate one value against one annotation.

    A union of models, or a bare ``list``, is what pydantic's TypeAdapter handles
    best; anything it cannot make sense of is returned unchanged and rejected by
    the field's own validation instead.
    """
    if annotation is None:
        return value
    try:
        return TypeAdapter(annotation).validate_python(value)
    except Exception:
        return value


#: XDG-ish default. One file to delete, one path to document.
DEFAULT_DATA_DIR = Path(
    os.environ.get("STUDIO_DATA_DIR", Path.home() / ".local" / "share" / "blender-ai-studio")
).expanduser()


class MCPServerConfig(BaseModel):
    """How to reach one MCP server.

    Enough for a stdio child process, which is what ``blender-mcp`` is: a command,
    its arguments, an environment and a working directory. The fields are a
    superset of that on purpose, so a future HTTP or socket transport is a new
    ``kind`` and not a new configuration format.
    """

    name: str = "Blender MCP"
    kind: Literal["stdio"] = "stdio"
    enabled: bool = True
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # blender-mcp's own bridge settings, passed through to the child.
    blender_host: str = "127.0.0.1"
    blender_port: int = 8765
    connect_timeout: float = 30.0
    tool_timeout: float = 120.0

    def child_env(self) -> dict[str, str]:
        """The environment for the child, with this server's settings applied.

        Explicit values win over the inherited environment so a user who set
        ``BLENDER_PORT`` in their shell is not surprised by the GUI's value.
        """
        env = {**os.environ, **self.env}
        env["BLENDER_HOST"] = self.blender_host
        env["BLENDER_PORT"] = str(self.blender_port)
        return env


class ThreeDConfig(BaseModel):
    """3D generation settings. The API key is a secret, and lives elsewhere."""

    provider: str = "tripo"
    model: str = ""
    quality: str = "medium"
    texture: bool = True
    max_credits: int = 200
    poll_interval: float = 5.0
    poll_timeout: float = 900.0


class AgentConfig(BaseModel):
    """The agent's limits and its instructions.

    The limits are not decoration. A model that misreads a tool result can loop
    until it costs something, and the cap is what makes that a visible event
    rather than a bill.
    """

    system_prompt: str = ""
    max_steps: int = 30
    max_tool_calls: int = 50
    max_seconds: float = 900.0
    max_session_cost: float = 5.0
    max_request_cost: float = 2.0
    max_3d_credits: int = 200
    # Off by default: execute_python is the one MCP tool that can do anything,
    # and it is also how a 3D asset gets imported. A user who has not turned it
    # on gets asset generation without the import step, said plainly.
    allow_execute_python: bool = False


class BudgetConfig(BaseModel):
    llm_budget_per_request: float = 2.0
    llm_budget_per_session: float = 5.0
    tripo_credit_limit: int = 200


class Settings(BaseSettings):
    """Everything that is not a secret."""

    model_config = SettingsConfigDict(
        env_prefix="STUDIO_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = DEFAULT_DATA_DIR
    database_path: Path | None = None
    log_level: str = "INFO"
    theme: str = "system"
    language: str = "en"
    default_project_dir: Path | None = None
    # Where blender-mcp lives. Set by the installer or by hand; the studio never
    # writes into that directory.
    blender_mcp_path: Path | None = None
    blender_mcp_python: str = ""
    llm_providers: list[dict[str, Any]] = Field(default_factory=list)
    three_d: ThreeDConfig = Field(default_factory=ThreeDConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or (self.data_dir / "studio.db")

    @property
    def resolved_secrets_path(self) -> Path:
        return self.data_dir / "secrets.json"

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge values loaded from the database over the environment.

        Keys are dotted paths, so the GUI can store one number without rewriting
        its neighbours: ``agent.max_steps`` sets that field and leaves the rest of
        the agent configuration alone. Each value is validated on its own against
        its own field's annotation, and one that no longer parses is dropped: an
        outdated database must not be the reason the app will not start.
        """
        for path, value in overrides.items():
            if value is None:
                continue
            owner, attribute = self._resolve(path)
            if owner is None:
                continue
            field = type(owner).model_fields.get(attribute) if isinstance(owner, BaseModel) else None
            annotation = field.annotation if field is not None else None
            try:
                setattr(owner, attribute, _coerce(annotation, value))
            except (TypeError, ValueError):
                continue

    def _resolve(self, path: str) -> tuple[Any, str]:
        """Follow a dotted path to the object that owns the last segment."""
        parts = path.split(".")
        owner: Any = self
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if not isinstance(owner, BaseModel):
                return None, ""
        if not isinstance(owner, BaseModel) or parts[-1] not in type(owner).model_fields:
            return None, ""
        return owner, parts[-1]

    def mcp_server(self, name: str) -> MCPServerConfig | None:
        return next((s for s in self.mcp_servers if s.name == name), None)
