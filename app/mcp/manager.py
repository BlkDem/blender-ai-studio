"""Every configured MCP server, as one thing the agent can ask.

The studio is built for Blender but not *only* for Blender: a filesystem server, a
git server and the Blender server are three entries in the same list, and the
agent sees one flat tool namespace built from all of them. That is why nothing in
this module says "blender".

A name clash is a real possibility with several servers, so it is reported rather
than silently resolved — a tool that quietly means something else is worse than a
tool that is missing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from app.core.errors import ConfigurationError, MCPError
from app.core.events import Event, EventBus, EventType
from app.llm.base import ToolSpec
from app.mcp.client import MCPSession
from app.mcp.models import (
    ConnectionState,
    ResourceDescriptor,
    ServerInfo,
    ServerStatus,
    ToolDescriptor,
    ToolOutcome,
)

logger = logging.getLogger(__name__)


class MCPManager:
    """Owns the sessions and the merged catalog."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self._sessions: dict[str, MCPSession] = {}
        self._configs: dict[str, ServerInfo] = {}
        self._bus = bus

    # --- configuration -----------------------------------------------------

    def configure(self, configs: Iterable[ServerInfo]) -> None:
        self._configs = {config.name: config for config in configs}

    def add(self, config: ServerInfo) -> None:
        self._configs[config.name] = config

    def configs(self) -> list[ServerInfo]:
        return list(self._configs.values())

    def session(self, name: str) -> MCPSession | None:
        return self._sessions.get(name)

    # --- lifecycle ---------------------------------------------------------

    async def connect(self, name: str) -> ServerStatus:
        config = self._configs.get(name)
        if config is None:
            raise ConfigurationError(f"No MCP server named '{name}' is configured")
        session = self._sessions.get(name)
        if session is None:
            session = MCPSession(config)
            self._sessions[name] = session
        self._publish(EventType.STATUS, server=name, state="connecting")
        status = await session.connect()
        self._publish(
            EventType.STATUS,
            server=name,
            state=str(status.state),
            tools=status.tools,
            resources=status.resources,
            error=status.last_error,
        )
        return status

    async def connect_all(self) -> list[ServerStatus]:
        """Connect every enabled server, in parallel, and never raise.

        One server being down is a state the GUI shows, not a reason the others
        stay disconnected.
        """
        enabled = [config for config in self._configs.values() if config.enabled and config.command]
        if not enabled:
            return []
        results = await asyncio.gather(
            *(self.connect(config.name) for config in enabled), return_exceptions=True
        )
        statuses: list[ServerStatus] = []
        for config, result in zip(enabled, results, strict=True):
            if isinstance(result, BaseException):
                session = self._sessions.get(config.name)
                status = session.status if session else ServerStatus(name=config.name)
                status.state = ConnectionState.FAILED
                status.last_error = str(result)
                statuses.append(status)
            else:
                statuses.append(result)
        return statuses

    async def disconnect(self, name: str) -> None:
        session = self._sessions.get(name)
        if session is not None:
            await session.disconnect()
            self._publish(EventType.STATUS, server=name, state="disconnected")
            self._sessions.pop(name, None)

    async def disconnect_all(self) -> None:
        for name in list(self._sessions):
            with contextlib.suppress(Exception):
                await self.disconnect(name)

    async def __aenter__(self) -> MCPManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.disconnect_all()

    # --- discovery ---------------------------------------------------------

    def statuses(self) -> list[ServerStatus]:
        return [session.status for session in self._sessions.values()]

    def status(self, name: str) -> ServerStatus | None:
        session = self._sessions.get(name)
        return session.status if session else None

    def any_connected(self) -> bool:
        return any(status.ready for status in self.statuses())

    def tools(self) -> dict[str, ToolDescriptor]:
        """Every tool from every connected server, by name."""
        catalog: dict[str, ToolDescriptor] = {}
        clashes: list[str] = []
        for session in self._sessions.values():
            for name, tool in session.tools.items():
                if name in catalog and catalog[name].server != tool.server:
                    clashes.append(name)
                catalog[name] = tool
        if clashes:
            # A duplicate name means one of the two tools is unreachable, and
            # which one depends on iteration order. Say so instead of picking.
            raise MCPError(
                f"Two MCP servers define the same tool: {', '.join(sorted(set(clashes)))}",
                hint="Rename one of them, or disable one of the servers.",
            )
        return catalog

    def tool_specs(self) -> list[ToolSpec]:
        """The catalog in the shape a model wants."""
        return [
            ToolSpec(name=tool.name, description=tool.description, parameters=tool.schema)
            for tool in self.tools().values()
        ]

    def resources(self) -> list[ResourceDescriptor]:
        found: list[ResourceDescriptor] = []
        for session in self._sessions.values():
            found.extend(session.resources)
        return found

    def tool_instructions(self) -> str:
        """The servers' own instructions, for the agent's system prompt.

        A server that documents how it wants to be used says so here, and the
        agent should be told: ignoring a server's instructions would be a choice,
        not a simplification.
        """
        parts = [
            f"From the MCP server '{name}':\n{session.instructions}"
            for name, session in self._sessions.items()
            if session.instructions
        ]
        return "\n\n".join(parts)

    # --- calls -------------------------------------------------------------

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> ToolOutcome:
        """Find the server that owns the tool and run it there."""
        session = self._session_for(name)
        if session is None:
            known = ", ".join(sorted(self.tools())) or "none"
            return ToolOutcome(
                call_id=f"call_unknown_{name}",
                tool=name,
                text=(
                    f"No connected MCP server provides '{name}'. "
                    f"Available tools: {known}."
                ),
                is_error=True,
                error_code="UNKNOWN_TOOL",
            )
        return await session.call_tool(name, arguments, timeout=timeout)

    async def read_resource(self, uri: str) -> tuple[str, str]:
        for session in self._sessions.values():
            if any(resource.uri == uri for resource in session.resources):
                return await session.read_resource(uri)
        raise MCPError(
            f"No connected MCP server publishes {uri}",
            hint="Check the server's resource list; the URI may have changed.",
        )

    def _session_for(self, name: str) -> MCPSession | None:
        for session in self._sessions.values():
            if name in session.tools and session.connected:
                return session
        return None

    def tool_names(self) -> list[str]:
        return sorted(self.tools())

    async def status_stream(self, interval: float = 1.0) -> AsyncIterator[ServerStatus]:
        """Every status change from every server. For the GUI's status bar."""
        seen: dict[str, ConnectionState] = {}
        while True:
            for status in self.statuses():
                if seen.get(status.name) is not status.state:
                    seen[status.name] = status.state
                    yield status
            await asyncio.sleep(interval)

    def _publish(self, event_type: EventType, **payload: Any) -> None:
        if self._bus is not None:
            self._bus.emit(event_type, **payload)
