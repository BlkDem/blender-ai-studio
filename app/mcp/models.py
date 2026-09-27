"""What the studio knows about an MCP server.

The point of these types is that nothing above this layer names a Blender tool.
A tool arrives as a :class:`ToolDescriptor` with the server's own JSON Schema,
and the GUI, the agent and the model selector all work from that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConnectionState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass(slots=True)
class ToolDescriptor:
    """One tool, exactly as the server described it.

    ``schema`` is kept verbatim. It is the server's contract with the model, and
    a studio that "tidied" it would be the reason a tool stopped working.
    """

    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict[str, Any] | None = None
    server: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)

    def required_arguments(self) -> list[str]:
        return list(self.schema.get("required") or [])

    def argument_names(self) -> list[str]:
        return list((self.schema.get("properties") or {}).keys())

    def summary(self) -> str:
        """One line for a tool card in the GUI."""
        first = self.description.strip().splitlines()[0] if self.description.strip() else ""
        return first or self.name

    def estimated_size(self) -> str:
        """How much of the model's context this tool costs.

        A rough character count, shown in the GUI so a user can see why a 15-tool
        server is a large system prompt before they switch it on.
        """
        size = len(self.name) + len(self.description) + len(str(self.schema))
        return f"{size // 1000}k" if size >= 1000 else str(size)


@dataclass(slots=True)
class ResourceDescriptor:
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""
    server: str = ""


@dataclass(slots=True)
class ToolOutcome:
    """The result of one tool call, in the studio's terms.

    ``text`` is what goes back to the model. ``blocks`` keeps the richer content
    (an image from a render, for instance) so the GUI can show what actually came
    back instead of a stringified placeholder.
    """

    call_id: str
    tool: str
    text: str
    is_error: bool = False
    error_code: str = ""
    duration_ms: float = 0.0
    blocks: list[Any] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def size(self) -> str:
        length = len(self.text)
        return f"{length // 1000}k chars" if length >= 1000 else f"{length} chars"


class ServerInfo(BaseModel):
    """A server as configured, before it is connected."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    connect_timeout: float = 30.0
    tool_timeout: float = 120.0

    def describe(self) -> str:
        if self.kind != "stdio":
            return f"{self.name} ({self.kind})"
        return f"{self.name}: {' '.join([self.command, *self.args])[:80]}"


@dataclass(slots=True)
class ServerStatus:
    """Live state, for the status bar and for ``--check``."""

    name: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    server_name: str = ""
    version: str = ""
    tools: int = 0
    resources: int = 0
    connected_since: float | None = None
    last_error: str = ""
    latency_ms: float = 0.0

    @property
    def ready(self) -> bool:
        return self.state is ConnectionState.CONNECTED

    def label(self) -> str:
        if self.state is ConnectionState.CONNECTED:
            return f"{self.name}: {self.tools} tools"
        if self.state is ConnectionState.CONNECTING:
            return f"{self.name}: connecting…"
        if self.state is ConnectionState.FAILED:
            return f"{self.name}: {self.last_error or 'failed'}"
        return f"{self.name}: disconnected"
