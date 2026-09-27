"""The MCP client layer: connect, discover, call.

A standard MCP client, so any MCP server works. Nothing here knows what Blender
is; the tools arrive from the server and the schemas are the server's own.
"""

from __future__ import annotations

from app.mcp.client import MCPSession
from app.mcp.manager import MCPManager
from app.mcp.models import (
    ConnectionState,
    ResourceDescriptor,
    ServerInfo,
    ServerStatus,
    ToolDescriptor,
    ToolOutcome,
)

__all__ = [
    "ConnectionState",
    "MCPManager",
    "MCPSession",
    "ResourceDescriptor",
    "ServerInfo",
    "ServerStatus",
    "ToolDescriptor",
    "ToolOutcome",
]
