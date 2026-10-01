"""The agents as an explicit LangGraph.

    START -> analyst <-> tools          (analyst calls tools until it has the numbers)
             analyst -> reviewer        (its last message is the report draft)
             reviewer -> analyst        (issues found, revisions left: rewrite)
             reviewer -> finish -> END  (approved, or out of revisions)

An explicit graph instead of a prebuilt agent, so every step and every loop limit is
visible in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from triage_agent.report import render_tables
from triage_agent.review import (
    REVIEWER_PROMPT,
    Review,
    check_coverage,
    check_numbers,
    review_facts,
)

ANALYST_PROMPT = """You analyse where an object detector (YOLOv8n) fails on COCO street images.

Steps:
1. Call run_detection, then compute_class_metrics, find_hard_examples and summarize_findings.
2. Then write your interpretation in Markdown with exactly these three headings:
   ## Summary (3 sentences: overall precision and recall, the weakest class, the main failure mode)
   ## Where it fails (the largest failure with its exact missed count, other size effects,
      class confusions, images with no detections)
   ## Recommendations (2-3 concrete next steps that follow from the failures)

Tables and the list of hard images are added automatically: do not write tables or list images.
Rules: use only numbers returned by the tools, copied exactly. Never estimate or round
differently. If a number is not in a tool result, do not write it."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    report: str
    revisions: int
    issues: list[str]


def build_graph(llm: BaseChatModel, tools: list[BaseTool], metrics_path: Path, max_revisions: int):
    llm_with_tools = llm.bind_tools(tools)
    reviewer_llm = llm.with_structured_output(Review)

    def analyst(state: State) -> dict:
        reply = llm_with_tools.invoke([SystemMessage(ANALYST_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def after_analyst(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else "reviewer"

    def reviewer(state: State) -> dict:
        draft = state["messages"][-1].content
        # Read metrics.json from disk, not from the analyst's messages: the reference
        # must not depend on what the analyst chose to look at.
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        issues = check_numbers(draft, metrics) + check_coverage(draft, metrics)
        verdict = reviewer_llm.invoke(REVIEWER_PROMPT.format(
            facts=json.dumps(review_facts(metrics), indent=1), report=draft))
        if not verdict.approved:
            issues += verdict.issues
        update = {"report": draft, "issues": issues}
        if issues and state.get("revisions", 0) < max_revisions:
            update["revisions"] = state.get("revisions", 0) + 1
            update["messages"] = [HumanMessage(
                "A reviewer checked your report against metrics.json and found these problems:\n"
                + "\n".join(f"- {i}" for i in issues)
                + "\nWrite the corrected text with the same three ## headings. "
                "Use only numbers from the tool results.")]
        return update

    def after_reviewer(state: State) -> str:
        # The reviewer only adds a message when it sends the draft back.
        return "analyst" if isinstance(state["messages"][-1], HumanMessage) else "finish"

    def finish(state: State) -> dict:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report = state["report"].strip() + "\n\n" + render_tables(metrics)
        if state.get("issues"):
            # Out of revisions: ship the report, but say openly what is still unverified.
            report += "\n\n## Reviewer notes (unresolved)\n" + "\n".join(f"- {i}" for i in state["issues"])
        return {"report": report}

    graph = StateGraph(State)
    graph.add_node("analyst", analyst)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("reviewer", reviewer)
    graph.add_node("finish", finish)
    graph.add_edge(START, "analyst")
    graph.add_conditional_edges("analyst", after_analyst, ["tools", "reviewer"])
    graph.add_edge("tools", "analyst")
    graph.add_conditional_edges("reviewer", after_reviewer, ["analyst", "finish"])
    graph.add_edge("finish", END)
    return graph.compile()
