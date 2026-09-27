"""A real MCP server, for the studio's own tests.

Not a mock: it is a genuine ``MCPServer`` on stdio, so the client tests exercise
the SDK handshake, discovery, tool calls and resource reads for real. It is
deliberately trivial — the point is the transport, not the behaviour.
"""

from __future__ import annotations

import json
import sys

from mcp.server.mcpserver import MCPServer
from mcp_types import ContentBlock, ImageContent, TextContent

INSTRUCTIONS = "This is the test server for Blender AI Studio. Use echo to check a round trip."


def build() -> MCPServer:
    app = MCPServer(name="studio-test-mcp", instructions=INSTRUCTIONS)

    @app.tool()
    def echo(text: str) -> str:
        """Return the text, unchanged. For checking a round trip."""
        return text

    @app.tool()
    def add(a: float, b: float) -> dict:
        """Add two numbers."""
        return {"result": a + b}

    @app.tool()
    def fail(reason: str = "no reason given") -> str:
        """Always fail, to exercise the error path."""
        raise ValueError(reason)

    @app.tool()
    def refuse(code: str, why: str) -> str:
        """Fail with a structured, machine-readable error, as blender-mcp does."""
        from mcp.server.mcpserver.exceptions import ToolError

        raise ToolError(json.dumps({"success": False, "error": {"code": code, "message": why}}))

    @app.tool()
    def structured(value: int) -> dict:
        """Answer with a structured payload and no prose."""
        return {"structured": value}

    @app.tool()
    def slow(seconds: float = 30.0) -> str:
        """Sleep, for cancellation and timeout tests."""
        import time

        time.sleep(seconds)
        return "waited"

    @app.tool()
    def picture() -> list[ContentBlock]:
        """Answer with an image block, like a render preview does."""
        # 1x1 transparent PNG.
        import base64

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        return [
            TextContent(type="text", text=json.dumps({"rendered": True})),
            ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png"),
        ]

    @app.resource("test://greeting", name="Greeting", mime_type="text/plain")
    def greeting() -> str:
        """A plain text resource."""
        return "hello from the test server"

    @app.resource("test://numbers", name="Numbers", mime_type="application/json")
    def numbers() -> str:
        """A JSON resource."""
        return json.dumps({"numbers": [1, 2, 3]})

    return app


def main() -> int:
    build().run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
