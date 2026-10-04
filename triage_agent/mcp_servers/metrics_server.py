"""MCP server for detection metrics: size, confusions, false alarms, hard images.

Run standalone (waits for a client on stdio): python -m triage_agent.mcp_servers.metrics_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from triage_agent import tools

mcp = FastMCP("triage-metrics")

# FastMCP builds each tool's schema from the type hints and its description from the docstring.
for fn in tools.METRICS_TOOLS:
    mcp.tool()(fn)


if __name__ == "__main__":
    # stdout is the protocol channel: nothing else may print to it.
    mcp.run(transport="stdio")
