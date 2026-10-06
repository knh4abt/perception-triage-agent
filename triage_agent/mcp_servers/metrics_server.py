"""MCP server for detection metrics: size, confusions, false alarms, hard images.

Two ways to run it:
- local (default): the CLI starts it as a subprocess and talks over stdin/stdout.
- deployed: MCP_TRANSPORT=http serves the same tools over HTTP at /mcp, so agents
  on another machine can call them. This is how it runs in the cloud container.

    python -m triage_agent.mcp_servers.metrics_server
    MCP_TRANSPORT=http PORT=7860 python -m triage_agent.mcp_servers.metrics_server
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from triage_agent import tools

# 0.0.0.0 = accept connections from outside the container, not only from inside it.
mcp = FastMCP("triage-metrics", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))

# FastMCP builds each tool's schema from the type hints and its description from the docstring.
for fn in tools.METRICS_TOOLS:
    mcp.tool()(fn)


if __name__ == "__main__":
    if os.environ.get("MCP_TRANSPORT") == "http":
        mcp.run(transport="streamable-http")
    else:
        # stdout is the protocol channel: nothing else may print to it.
        mcp.run(transport="stdio")
