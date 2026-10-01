"""Command-line entry point: python -m triage_agent.cli --images data/sample --out reports/"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from triage_agent import tools
from triage_agent.config import load_config
from triage_agent.graph import build_graph
from triage_agent.llm import get_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find where an object detector fails.")
    parser.add_argument("--images", type=Path, default=Path("data/sample"),
                        help="folder with the input images")
    parser.add_argument("--out", type=Path, default=Path("reports"),
                        help="folder where report.md is written")
    return parser.parse_args()


async def run(out_dir: Path) -> str:
    cfg = load_config()
    if not cfg["agent"]["use_mcp"]:
        return await run_graph(cfg, [StructuredTool.from_function(f) for f in tools.ALL_TOOLS], out_dir)

    client = MultiServerMCPClient({"triage": {
        "transport": "stdio",
        # Same interpreter as the CLI, so the server sees the same .venv packages.
        "command": sys.executable,
        "args": ["-m", "triage_agent.mcp_server"],
        # The MCP SDK passes only a minimal environment to the subprocess by default;
        # forward ours so TRIAGE_IMAGES/TRIAGE_OUT and .env keys arrive.
        "env": dict(os.environ),
        "cwd": str(Path.cwd()),
    }})
    # One session for the whole run: one server process instead of one per tool call.
    async with client.session("triage") as session:
        return await run_graph(cfg, await load_mcp_tools(session), out_dir)


async def run_graph(cfg: dict, agent_tools: list, out_dir: Path) -> str:
    graph = build_graph(get_llm(cfg), agent_tools, out_dir / "metrics.json",
                        cfg["agent"]["max_revisions"])
    # recursion_limit is a second safety net on top of max_revisions: no loop can spin forever.
    state = await graph.ainvoke(
        {"messages": [HumanMessage("Analyse the detector and write the report.")]},
        config={"recursion_limit": 40},
    )
    for msg in state["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"  tool call: {call['name']}")
    print(f"  reviewer sent the draft back {state.get('revisions', 0)} time(s); "
          f"unresolved issues: {len(state.get('issues', []))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text("# Where does YOLOv8n fail?\n\n" + state["report"].strip() + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    args = parse_args()
    # Passed via the environment so the tools (and later the MCP server process) see them.
    os.environ["TRIAGE_IMAGES"] = str(args.images)
    os.environ["TRIAGE_OUT"] = str(args.out)
    print(f"report written to {asyncio.run(run(args.out))}")


if __name__ == "__main__":
    main()
