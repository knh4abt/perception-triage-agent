"""MCP server for image facts: brightness, object count, object size.

A separate server from metrics: it reads different data, and in a real system it could
sit next to the image store while the metrics server sits next to the evaluation results.

Run standalone (waits for a client on stdio): python -m triage_agent.mcp_servers.images_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from triage_agent import tools

mcp = FastMCP("triage-images")

for fn in tools.IMAGE_TOOLS:
    mcp.tool()(fn)


if __name__ == "__main__":
    mcp.run(transport="stdio")
