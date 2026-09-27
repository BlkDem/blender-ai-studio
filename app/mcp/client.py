"""One MCP server, connected.

A thin, honest wrapper over the MCP SDK: it owns the stdio subprocess, the client
session, and the tool catalog discovered from the server. It does not implement
MCP, does not cache schemas it could have stale, and does not know what a Blender
is.

Lifecycle note: the SDK's ``stdio_client`` and ``ClientSession`` are async context
managers, so the whole connection lives inside one task. :meth:`MCPSession.serve`
is that task; :meth:`connect` starts it and waits until the handshake is done, and
:meth:`disconnect` cancels it. Holding the session any other way means holding a
closed pipe, which is the classic way to get a dead server that looks connected.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import MCPError, MCPToolError
from app.mcp.models import (
    ConnectionState,
    ResourceDescriptor,
    ServerInfo,
    ServerStatus,
    ToolDescriptor,
    ToolOutcome,
)

logger = logging.getLogger(__name__)

#: A tool result bigger than this is truncated before it goes back to the model.
#: A get_scene on a large scene is the case that matters, and a model that has
#: read 200k characters of JSON stops reasoning about the request.
MAX_RESULT_CHARS = 24_000


@dataclass
class _State:
    status: ServerStatus
    tools: dict[str, ToolDescriptor] = field(default_factory=dict)
    resources: list[ResourceDescriptor] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    session: Any = None
    failure: str = ""


class MCPSession:
    """A connected MCP server."""

    def __init__(self, info: ServerInfo) -> None:
        self.info = info
        self._state = _State(status=ServerStatus(name=info.name))
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        #: Set by :meth:`disconnect`. The serving task waits on it, because the
        #: stdio session only exists inside its context managers: a task that
        #: returned after the handshake would take the subprocess and the client
        #: session down with it, and every later call would report a closed
        #: connection.
        self._stop = asyncio.Event()

    # --- properties --------------------------------------------------------

    @property
    def status(self) -> ServerStatus:
        return self._state.status

    @property
    def tools(self) -> dict[str, ToolDescriptor]:
        return dict(self._state.tools)

    @property
    def resources(self) -> list[ResourceDescriptor]:
        return list(self._state.resources)

    @property
    def connected(self) -> bool:
        return self._state.status.ready and self._state.session is not None

    def tool(self, name: str) -> ToolDescriptor | None:
        return self._state.tools.get(name)

    # --- lifecycle ---------------------------------------------------------

    async def connect(self) -> ServerStatus:
        """Start the server and block until the handshake has finished.

        Returns the status instead of raising, because a failed server is a state
        the GUI shows next to the others rather than an error that stops startup.
        """
        async with self._lock:
            if self.connected:
                return self.status
            self._state = _State(status=ServerStatus(name=self.info.name, state=ConnectionState.CONNECTING))
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._serve(), name=f"mcp:{self.info.name}")

        try:
            await asyncio.wait_for(self._state.ready.wait(), timeout=self.info.connect_timeout)
        except TimeoutError:
            self._state.status.state = ConnectionState.FAILED
            self._state.status.last_error = "the server did not complete its handshake in time"
            await self._cancel()
        return self.status

    async def disconnect(self) -> None:
        await self._cancel()
        self._state.status = ServerStatus(name=self.info.name)
        self._state.tools.clear()
        self._state.resources.clear()

    async def _cancel(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            # Give the serving task a moment to unwind its context managers on its
            # own, so the child process is reaped rather than killed.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _serve(self) -> None:
        """Own the connection for as long as the studio is running."""
        from mcp import ClientSession as SdkSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if not self.info.command:
            self._fail("no command configured for this server")
            return

        parameters = StdioServerParameters(
            command=self.info.command,
            args=list(self.info.args),
            env={**self.info.env} or None,
            cwd=self.info.cwd,
        )
        try:
            # Both managers are needed for the session to exist at all: the
            # subprocess and the client live exactly as long as this block, which
            # is why the task then waits here instead of returning.
            async with stdio_client(parameters) as (read, write), SdkSession(read, write) as session:
                self._state.session = session
                await self._handshake(session)
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure is a connection failure
            logger.warning("MCP server %s failed: %s", self.info.name, exc)
            self._fail(str(exc))
            return
        finally:
            self._state.session = None
            if self._state.status.state is not ConnectionState.FAILED:
                self._state.status.state = ConnectionState.DISCONNECTED

    async def _handshake(self, session: Any) -> None:
        """Initialize, then discover. Both are needed before the GUI is useful."""
        started = time.perf_counter()
        result = await session.initialize()
        self._state.status.server_name = getattr(result.server_info, "name", "") or ""
        self._state.status.version = getattr(result.server_info, "version", "") or ""
        #: The server's own instructions go to the agent's system prompt. A
        #: server that knows how it wants to be used says so here, and ignoring
        #: it would be a choice, not a simplification.
        self.instructions = getattr(result, "instructions", "") or ""

        tools = await session.list_tools()
        for tool in tools.tools:
            self._state.tools[tool.name] = ToolDescriptor(
                name=tool.name,
                description=tool.description or "",
                schema=dict(tool.input_schema or {"type": "object", "properties": {}}),
                output_schema=dict(tool.output_schema) if tool.output_schema else None,
                server=self.info.name,
                annotations=dict(tool.annotations or {}) if tool.annotations else {},
            )
        self._state.status.tools = len(self._state.tools)

        try:
            resources = await session.list_resources()
        except Exception:  # noqa: BLE001 - resources are optional in MCP
            resources = None
        if resources is not None:
            for resource in resources.resources:
                self._state.resources.append(
                    ResourceDescriptor(
                        uri=str(resource.uri),
                        name=getattr(resource, "name", "") or "",
                        description=getattr(resource, "description", "") or "",
                        mime_type=getattr(resource, "mimeType", "") or "",
                        server=self.info.name,
                    )
                )
        self._state.status.resources = len(self._state.resources)
        self._state.status.state = ConnectionState.CONNECTED
        self._state.status.connected_since = time.time()
        self._state.status.latency_ms = (time.perf_counter() - started) * 1000
        self._state.status.last_error = ""
        self._state.ready.set()
        logger.info(
            "MCP %s connected: %d tools, %d resources in %.0f ms",
            self.info.name,
            self._state.status.tools,
            self._state.status.resources,
            self._state.status.latency_ms,
        )

    def _fail(self, reason: str) -> None:
        self._state.status.state = ConnectionState.FAILED
        self._state.status.last_error = reason
        self._state.failure = reason
        self._state.ready.set()

    #: Set by :meth:`_handshake`; declared here so the attribute always exists.
    instructions: str = ""

    # --- calls -------------------------------------------------------------

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> ToolOutcome:
        """Run one tool.

        A tool that fails is a result, not an exception: the model has to be told
        what went wrong so it can correct itself, and the studio's own failure to
        reach the server is a different thing entirely.
        """
        session = self._require_session()
        call_id = f"call_{int(time.time() * 1000):x}"
        started = time.perf_counter()
        try:
            result = await session.call_tool(
                name, arguments or {}, read_timeout_seconds=timeout or self.info.tool_timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport or protocol failure
            raise MCPError(
                f"{self.info.name} could not run {name}: {exc}",
                hint="The server may have gone away. Reconnect in the status bar.",
                tool=name,
            ) from exc

        duration = (time.perf_counter() - started) * 1000
        text, blocks, images, mime_types = _render_result(result)
        is_error = bool(getattr(result, "is_error", False))
        code = _error_code(text) if is_error else ""
        if is_error:
            text = _explain_failure(self.info.name, name, code, text)
        return ToolOutcome(
            call_id=call_id,
            tool=name,
            text=_truncate(text),
            is_error=is_error,
            error_code=code,
            duration_ms=duration,
            blocks=blocks,
            images=images,
            image_mime_types=mime_types,
        )

    async def read_resource(self, uri: str) -> tuple[str, str]:
        """``(text, mime type)``. Images come back as base64 in the text."""
        session = self._require_session()
        try:
            result = await session.read_resource(uri)
        except Exception as exc:  # noqa: BLE001
            raise MCPError(f"Could not read {uri}: {exc}", uri=uri) from exc
        contents = result.contents[0] if result.contents else None
        if contents is None:
            return "", ""
        payload = getattr(contents, "text", None)
        if payload is None:
            payload = getattr(contents, "blob", "") or ""
        return payload, getattr(contents, "mime_type", "") or getattr(contents, "mimeType", "")

    def _require_session(self) -> Any:
        session = self._state.session
        if session is None or not self.connected:
            raise MCPError(
                f"{self.info.name} is not connected",
                hint="Connect it in the status bar, or check the command in Settings → Blender.",
            )
        return session

    async def __aenter__(self) -> MCPSession:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.disconnect()

    async def watch(self) -> AsyncIterator[ServerStatus]:
        """Yield whenever the connection state changes. For the status bar."""
        seen = self.status.state
        while True:
            await asyncio.sleep(0.5)
            if self.status.state is not seen:
                seen = self.status.state
                yield self.status


def _render_result(result: Any) -> tuple[str, list[Any], list[str], list[str]]:
    """Result content into text, plus the blocks and images worth showing.

    An MCP server may answer with a structured payload, text, or both;
    ``blender.render_preview`` answers with a JSON summary and a PNG. The text is
    what the model needs; the image is what a person needs to see.
    """
    structured = getattr(result, "structured_content", None)
    blocks = list(getattr(result, "content", []) or [])
    texts: list[str] = []
    images: list[str] = []
    mime_types: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(getattr(block, "text", ""))
        elif kind == "image":
            data = getattr(block, "data", "")
            if data:
                images.append(data)
                mime_types.append(str(getattr(block, "mimeType", "") or "image/png"))
    if structured is not None and not texts:
        texts.append(json.dumps(structured, ensure_ascii=False, indent=2))
    elif structured is not None and texts and not texts[0].strip().startswith("{"):
        # Both present and the text is prose: keep the structured payload, it is
        # the part a model can act on.
        texts.append(json.dumps(structured, ensure_ascii=False, indent=2))
    return "\n".join(text for text in texts if text), blocks, images, mime_types


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    half = MAX_RESULT_CHARS // 2
    return (
        text[:half]
        + f"\n\n… {len(text) - MAX_RESULT_CHARS} characters omitted by Blender AI Studio …\n\n"
        + text[-half:]
    )


def _error_code(text: str) -> str:
    """The server's structured error code, when it sent one.

    Tried from the left, because a JSON payload contains braces of its own and the
    last one is the innermost object rather than the message.
    """
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"])
    return ""


def _explain_failure(server: str, tool: str, code: str, text: str) -> str:
    """Say which server, which tool and which code, even when the server said
    nothing useful.

    The SDK reduces an unexpected server-side exception to "Error executing tool
    X" and puts the detail in the child's stderr, which a client cannot read. A
    model told only that cannot correct itself, so the context is added here
    rather than leaving the bare string to travel back.
    """
    reason = text.strip()
    bare = reason in {"", f"Error executing tool {tool}"}
    if bare:
        return (
            f"{tool} on {server} failed ({code or 'no code reported'}). "
            "The server did not say why; its log has the detail."
        )
    prefix = f"{tool} on {server} failed"
    return f"{prefix} ({code}): {reason}" if code else f"{prefix}: {reason}"


def tool_error(tool: str, message: str, code: str = "") -> MCPToolError:
    return MCPToolError(tool, message, tool_code=code)
