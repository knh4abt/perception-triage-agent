"""Expose the tools over MCP (Model Context Protocol), using the official SDK.

The agent no longer imports the tools; it starts this server as a subprocess and talks
to it over stdin/stdout. Any MCP client (Claude Desktop, an IDE, another agent) can use
the same tools without knowing they are Python.

Run standalone (waits for a client on stdio): python -m triage_agent.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from triage_agent import tools

mcp = FastMCP("perception-triage")

# FastMCP builds each tool's schema from the type hints and its description from the docstring.
for fn in tools.ALL_TOOLS:
    mcp.tool()(fn)


if __name__ == "__main__":
    # stdout is the protocol channel: nothing else may print to it.
    mcp.run(transport="stdio")
