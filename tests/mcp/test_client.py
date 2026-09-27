"""MCP client behaviour, against a real MCP server over stdio.

A fake server would test the fake. These tests spawn an actual MCP server
(``app.mcp.support.mcp_server``, the same one the tests use) over a real pipe, so
the handshake, the tool discovery and the stdio lifecycle are exercised for real.
"""

from __future__ import annotations

import json
import sys

import pytest

from app.core.errors import MCPError
from app.mcp.client import MAX_RESULT_CHARS, MCPSession, _error_code, _render_result, _truncate
from app.mcp.manager import MCPManager
from app.mcp.models import ConnectionState, ServerInfo, ToolDescriptor
from tests.conftest import REPO_ROOT  # noqa: E402 - the repo root, already on sys.path

pytestmark = pytest.mark.anyio


@pytest.fixture
def mcp_config() -> ServerInfo:
    """The studio's own MCP server, as a child process."""
    return ServerInfo(
        name="test-mcp",
        command=sys.executable,
        args=["-m", "app.mcp.support"],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "LOG_LEVEL": "WARNING"},
        connect_timeout=30.0,
        tool_timeout=30.0,
    )


@pytest.fixture
def second_mcp_config(mcp_config: ServerInfo) -> ServerInfo:
    """A second server with a different catalog, for testing the merge."""
    return mcp_config.model_copy(update={"name": "test-mcp-two", "args": ["-m", "app.mcp.support_two"]})


async def test_connecting_discovers_tools_and_resources(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        assert session.status.ready
        assert session.status.state is ConnectionState.CONNECTED
        assert "echo" in session.tools
        assert "add" in session.tools
        assert session.tool("echo").description
        assert any(resource.uri.startswith("test://") for resource in session.resources)


async def test_a_tool_schema_arrives_verbatim(mcp_config: ServerInfo) -> None:
    """A studio that tidied a server's schema would be the reason a tool stopped
    working, so the schema is passed through as received."""
    async with MCPSession(mcp_config) as session:
        schema = session.tool("add").schema
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"a", "b"}
        assert schema["properties"]["a"]["type"] == "number"


async def test_calling_a_tool_returns_text_and_a_duration(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        outcome = await session.call_tool("add", {"a": 2, "b": 3})
    assert outcome.is_error is False
    assert json.loads(outcome.text)["result"] == 5
    assert outcome.duration_ms >= 0
    assert outcome.tool == "add"


async def test_a_tool_failure_is_a_result_not_an_exception(mcp_config: ServerInfo) -> None:
    """The model has to be told what went wrong so it can correct itself.

    The SDK reduces an unexpected server-side exception to a bare "Error
    executing tool X" and puts the detail in the child's stderr, which a client
    cannot read — so the studio adds which server and which tool failed.
    """
    async with MCPSession(mcp_config) as session:
        outcome = await session.call_tool("fail", {"reason": "the disk is full"})
    assert outcome.is_error is True
    assert "fail on test-mcp" in outcome.text
    assert "the disk is full" not in outcome.text  # the SDK really does drop it


async def test_a_structured_error_from_a_server_keeps_its_code(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        outcome = await session.call_tool("refuse", {"code": "OBJECT_NOT_FOUND", "why": "no such object"})
    assert outcome.is_error is True
    assert outcome.error_code == "OBJECT_NOT_FOUND"
    assert "no such object" in outcome.text


async def test_a_structured_result_is_used_when_there_is_no_text(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        outcome = await session.call_tool("structured", {"value": 7})
    assert json.loads(outcome.text)["structured"] == 7


async def test_calling_an_unknown_tool_reports_an_error(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        outcome = await session.call_tool("does_not_exist", {})
    assert outcome.is_error is True


async def test_reading_a_resource(mcp_config: ServerInfo) -> None:
    async with MCPSession(mcp_config) as session:
        text, mime = await session.read_resource("test://greeting")
    assert "hello" in text
    assert mime == "text/plain"


async def test_a_disconnected_session_refuses_to_call(mcp_config: ServerInfo) -> None:
    session = MCPSession(mcp_config)
    with pytest.raises(MCPError) as excinfo:
        await session.call_tool("echo", {"text": "x"})
    assert "not connected" in excinfo.value.message


async def test_a_server_that_never_starts_fails_fast_and_says_so() -> None:
    session = MCPSession(
        ServerInfo(
            name="broken",
            command=sys.executable,
            args=["-c", "import nonexistent_module_xyz"],
            connect_timeout=10.0,
        )
    )
    status = await session.connect()
    assert status.state is ConnectionState.FAILED
    assert status.last_error
    await session.disconnect()


async def test_a_server_with_no_command_is_refused() -> None:
    status = await MCPSession(ServerInfo(name="empty")).connect()
    assert status.state is ConnectionState.FAILED
    assert "no command" in status.last_error


async def test_reconnecting_after_a_disconnect(mcp_config: ServerInfo) -> None:
    session = MCPSession(mcp_config)
    await session.connect()
    assert session.tools
    await session.disconnect()
    assert session.tools == {}
    assert session.status.state is ConnectionState.DISCONNECTED
    await session.connect()
    assert session.tools, "a reconnect must discover the tools again"
    await session.disconnect()


async def test_the_manager_connects_several_servers_and_merges_their_tools(
    mcp_config: ServerInfo, second_mcp_config: ServerInfo
) -> None:
    manager = MCPManager()
    manager.configure([mcp_config, second_mcp_config])
    try:
        statuses = await manager.connect_all()
        assert len(statuses) == 2
        assert all(status.ready for status in statuses)
        assert manager.any_connected()
        catalog = manager.tools()
        assert {"echo", "add"} <= set(catalog), "the first server's tools"
        assert {"read_note", "list_notes"} <= set(catalog), "the second server's tools"
        assert manager.tool_names() == sorted(catalog)
        assert {spec.name for spec in manager.tool_specs()} == set(catalog)
        assert catalog["echo"].server == "test-mcp"
        assert catalog["read_note"].server == "test-mcp-two"
        assert {resource.uri for resource in manager.resources()} >= {
            "test://greeting",
            "test://notes",
        }
    finally:
        await manager.disconnect_all()


async def test_a_tool_call_goes_to_the_server_that_owns_it(
    mcp_config: ServerInfo, second_mcp_config: ServerInfo
) -> None:
    manager = MCPManager()
    manager.configure([mcp_config, second_mcp_config])
    try:
        await manager.connect_all()
        outcome = await manager.call_tool("list_notes", {})
        assert outcome.is_error is False
        assert "one" in outcome.text
    finally:
        await manager.disconnect_all()


async def test_two_servers_offering_the_same_tool_is_reported(
    mcp_config: ServerInfo,
) -> None:
    """A tool that quietly means something else is worse than a missing one, and
    the same server reached under two names is the simplest way to hit it."""
    manager = MCPManager()
    manager.configure([mcp_config, mcp_config.model_copy(update={"name": "same-again"})])
    try:
        await manager.connect_all()
        with pytest.raises(MCPError) as excinfo:
            manager.tools()
        assert "echo" in excinfo.value.message
    finally:
        await manager.disconnect_all()


async def test_one_dead_server_does_not_stop_the_others(mcp_config: ServerInfo) -> None:
    manager = MCPManager()
    manager.configure([mcp_config, ServerInfo(name="dead", command="definitely-not-a-real-binary-xyz")])
    try:
        statuses = {status.name: status for status in await manager.connect_all()}
        assert statuses["test-mcp"].ready
        assert statuses["dead"].state is ConnectionState.FAILED
    finally:
        await manager.disconnect_all()


async def test_calling_a_tool_that_nobody_provides_says_which_ones_exist(
    mcp_config: ServerInfo,
) -> None:
    manager = MCPManager()
    manager.configure([mcp_config])
    try:
        await manager.connect_all()
        outcome = await manager.call_tool("blender.create_object", {"type": "cube"})
        assert outcome.is_error
        assert "echo" in outcome.text, "the model needs to know what it can call instead"
    finally:
        await manager.disconnect_all()


def test_a_clash_between_two_servers_is_reported() -> None:
    """A tool that quietly means something else is worse than a missing one."""
    manager = MCPManager()
    for name in ("one", "two"):
        session = MCPSession(ServerInfo(name=name))
        session._state.tools["same_tool"] = ToolDescriptor(name="same_tool", server=name)
        session._state.status.state = ConnectionState.CONNECTED
        manager._sessions[name] = session
    with pytest.raises(MCPError) as excinfo:
        manager.tools()
    assert "same_tool" in excinfo.value.message
    assert "Rename one of them" in (excinfo.value.hint or "")


def test_a_long_result_is_truncated_from_the_middle() -> None:
    text = "a" * (MAX_RESULT_CHARS * 2)
    short = _truncate(text)
    assert len(short) < len(text)
    assert "omitted by Blender AI Studio" in short
    assert short.startswith("aaa") and short.endswith("aaa")


def test_a_short_result_is_untouched() -> None:
    assert _truncate("short") == "short"


def test_the_servers_error_code_is_read_out_of_its_json() -> None:
    text = 'Error executing tool blender.get_object: {"error": {"code": "OBJECT_NOT_FOUND"}}'
    assert _error_code(text) == "OBJECT_NOT_FOUND"
    assert _error_code("just prose") == ""


def test_an_image_block_is_kept_for_the_gui() -> None:
    class Block:
        type = "image"
        data = "base64data"

    class Result:
        content = [Block()]
        structured_content = None

    text, _blocks, images, _mime = _render_result(Result())
    assert images == ["base64data"]
    assert text == "", "an image-only result has no text to send to the model"


def test_structured_content_is_preferred_over_prose() -> None:
    class Block:
        type = "text"
        text = "Here is what I found."

    class Result:
        content = [Block()]
        structured_content = {"objects_count": 3}

    text, _blocks, _images, _mime = _render_result(Result())
    assert "Here is what I found." in text
    assert '"objects_count": 3' in text, "the actionable part must reach the model too"


async def test_the_tools_server_instructions_reach_the_agent(mcp_config: ServerInfo) -> None:
    """A server that documents how it wants to be used says so in its
    instructions, and ignoring that would be a choice rather than a simplification."""
    manager = MCPManager()
    manager.configure([mcp_config])
    try:
        await manager.connect_all()
        assert "test server" in manager.tool_instructions().lower()
    finally:
        await manager.disconnect_all()


def test_a_tool_card_has_a_summary_and_a_size() -> None:
    tool = ToolDescriptor(
        name="blender.create_object",
        description="Create a primitive mesh object.\nMore detail below.",
        schema={"type": "object", "properties": {"type": {"type": "string"}}},
    )
    assert tool.summary() == "Create a primitive mesh object."
    assert tool.argument_names() == ["type"]
    assert tool.required_arguments() == []
    assert isinstance(tool.estimated_size(), str)


def test_a_tool_with_no_description_still_summarises() -> None:
    assert ToolDescriptor(name="mystery").summary() == "mystery"
