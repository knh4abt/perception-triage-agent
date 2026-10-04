"""Command-line entry point: python -m triage_agent.cli --images data/sample --out reports/

Add --approve to review the draft yourself before the report is written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Command

from triage_agent import tools
from triage_agent.config import load_config
from triage_agent.graph import build_graph
from triage_agent.llm import get_llm

MCP_SERVERS = {
    "metrics": "triage_agent.mcp_servers.metrics_server",
    "images": "triage_agent.mcp_servers.images_server",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find where an object detector fails.")
    parser.add_argument("--images", type=Path, default=Path("data/sample"),
                        help="folder with the input images")
    parser.add_argument("--out", type=Path, default=Path("reports"),
                        help="folder where report.md is written")
    parser.add_argument("--approve", action="store_true",
                        help="pause before writing the report so you can approve or comment")
    return parser.parse_args()


async def run(out_dir: Path, approve: bool) -> str:
    cfg = load_config()
    if not cfg["agent"]["use_mcp"]:
        local = [StructuredTool.from_function(f) for f in tools.METRICS_TOOLS + tools.IMAGE_TOOLS]
        return await run_graph(cfg, local, out_dir, approve)

    client = MultiServerMCPClient({name: {
        "transport": "stdio",
        # Same interpreter as the CLI, so each server sees the same .venv packages.
        "command": sys.executable,
        "args": ["-m", module],
        # The MCP SDK passes only a minimal environment to the subprocess by default;
        # forward ours so TRIAGE_IMAGES/TRIAGE_OUT and .env keys arrive.
        "env": dict(os.environ),
        "cwd": str(Path.cwd()),
    } for name, module in MCP_SERVERS.items()})
    # One open session per server for the whole run: one process each, not one per tool call.
    async with AsyncExitStack() as stack:
        agent_tools = []
        for name in MCP_SERVERS:
            session = await stack.enter_async_context(client.session(name))
            agent_tools += await load_mcp_tools(session)
        return await run_graph(cfg, agent_tools, out_dir, approve)


def ask_human(payload: dict) -> dict:
    print("\n--- draft summary ---\n" + payload["summary"])
    if payload["open_issues"]:
        print("open reviewer issues:", payload["open_issues"])
    if input("approve? [y/n] ").strip().lower().startswith("y"):
        return {"approved": True}
    return {"approved": False, "comment": input("comment for the editor: ").strip()}


async def run_graph(cfg: dict, agent_tools: list, out_dir: Path, approve: bool) -> str:
    graph = build_graph(get_llm(cfg), agent_tools, out_dir, cfg["agent"]["max_revisions"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.md").write_text(
        "```mermaid\n" + graph.get_graph().draw_mermaid() + "```\n", encoding="utf-8")

    # thread_id names this run in the checkpointer, so a paused run can be resumed.
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 60}
    pause = await stream(graph, {"human_review": approve, "trace": []}, config)
    while pause is not None:
        pause = await stream(graph, Command(resume=ask_human(pause)), config)

    final = graph.get_state(config).values
    (out_dir / "run_log.json").write_text(
        json.dumps({"trace": final["trace"], "reviews": final["review_log"]}, indent=1), encoding="utf-8")
    path = out_dir / "report.md"
    path.write_text(final["report"], encoding="utf-8")
    return str(path)


async def stream(graph, graph_input, config) -> dict | None:
    """Run until the end or the next pause; print each step as it finishes."""
    async for update in graph.astream(graph_input, config=config, stream_mode="updates"):
        for node, values in update.items():
            if node == "__interrupt__":
                return values[0].value
            for line in (values or {}).get("trace", []):
                print(f"  {line}", flush=True)
    return None


def main() -> None:
    args = parse_args()
    # Passed via the environment so the tools and the MCP server processes see them.
    os.environ["TRIAGE_IMAGES"] = str(args.images)
    os.environ["TRIAGE_OUT"] = str(args.out)
    print(f"report written to {asyncio.run(run(args.out, args.approve))}")


if __name__ == "__main__":
    main()
