"""A second MCP server, so "several servers" can be tested for real.

Different tool names on purpose: two servers defining the same tool is a clash the
manager must report, and testing that needs a genuinely different catalog rather
than a second copy of the first one.
"""

from __future__ import annotations

import json
import sys

from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = "This is a second test server, with filesystem-shaped tools."


def build() -> MCPServer:
    app = MCPServer(name="studio-test-mcp-two", instructions=INSTRUCTIONS)

    @app.tool()
    def read_note(name: str) -> str:
        """Read a note. The filesystem-shaped tool of a second server."""
        return f"note: {name}"

    @app.tool()
    def list_notes() -> list[str]:
        """Every note there is."""
        return ["one", "two"]

    @app.tool()
    def write_note(name: str, text: str) -> dict:
        """Write a note."""
        return {"written": name, "bytes": len(text)}

    @app.resource("test://notes", name="Notes", mime_type="application/json")
    def notes() -> str:
        """Every note, as JSON."""
        return json.dumps({"notes": ["one", "two"]})

    return app


def main() -> int:
    build().run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
