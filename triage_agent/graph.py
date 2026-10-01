"""The agent as an explicit LangGraph.

    START -> analyst <-> tools        (analyst calls tools until it has the numbers)
             analyst -> finish -> END (its last message is the report draft)

An explicit graph instead of a prebuilt agent, so every step is visible and the
Reviewer can be added later as one more node and edge.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

ANALYST_PROMPT = """You analyse where an object detector (YOLOv8n) fails on COCO street images.

Steps:
1. Call run_detection, then compute_class_metrics, find_hard_examples and summarize_findings.
2. Then write the report in Markdown with exactly these sections:
   ## Summary (3 sentences: overall precision and recall, the weakest class, the main failure mode)
   ## Per-class results (a table: class, precision, recall, TP, FP, FN)
   ## Where it fails (object size, class confusions, images with no detections)
   ## Hardest images (file name and what went wrong)
   ## Recommendations (2-3 concrete next steps)

Rules: use only numbers returned by the tools, copied exactly. Never estimate or round
differently. If a number is not in a tool result, do not write it."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    report: str


def build_graph(llm: BaseChatModel, tools: list[BaseTool]):
    llm_with_tools = llm.bind_tools(tools)

    def analyst(state: State) -> dict:
        reply = llm_with_tools.invoke([SystemMessage(ANALYST_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def route(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else "finish"

    def finish(state: State) -> dict:
        return {"report": state["messages"][-1].content}

    graph = StateGraph(State)
    graph.add_node("analyst", analyst)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finish", finish)
    graph.add_edge(START, "analyst")
    graph.add_conditional_edges("analyst", route, ["tools", "finish"])
    graph.add_edge("tools", "analyst")
    graph.add_edge("finish", END)
    return graph.compile()
