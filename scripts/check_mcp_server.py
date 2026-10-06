"""Call a running metrics MCP server over HTTP and print what it answers.

    python -m scripts.check_mcp_server --url http://localhost:7860/mcp
    python -m scripts.check_mcp_server --url https://<user>-<space>.hf.space/gradio_api/mcp/
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


async def check(url: str) -> None:
    client = MultiServerMCPClient({"metrics": {"transport": "streamable_http", "url": url}})
    tools = {t.name: t for t in await client.get_tools()}
    print("tools:", ", ".join(tools))
    result = await tools["size_breakdown"].ainvoke({})
    # MCP returns content blocks; the first one holds the result as text. The FastMCP
    # server writes JSON, the Gradio server writes a Python dict, so accept both.
    text = result[0]["text"] if isinstance(result, list) else result
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    print("largest failure:", data["largest_failure"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/mcp")
    asyncio.run(check(parser.parse_args().url))


if __name__ == "__main__":
    main()
