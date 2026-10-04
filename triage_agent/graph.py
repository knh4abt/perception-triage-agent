"""The multi-agent workflow as an explicit LangGraph.

    START -> prepare                      code: detection, metrics, image stats
          -> specialist x3 (parallel)     size, confusion, hard_images (Send fan-out)
          -> editor                       summary + recommendations
          -> reviewer                     code checks + LLM check, per section
               failed sections -> back to ONLY the specialists that own them
               only editor issues -> editor
          -> human_approval               pauses for a person when enabled (interrupt)
          -> finish                       headings and tables added by code

Fan-out is fixed in code, not chosen by an LLM supervisor: every specialist must run for
a full report, so letting an 8B model route would only add a way to fail.
"""

from __future__ import annotations

import asyncio
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from triage_agent.agents import (
    EDITOR_PROMPT,
    SECTIONS,
    SPECIALISTS,
    build_specialist,
    parse_editor,
    specialist_request,
)
from triage_agent.report import render_report
from triage_agent.review import (
    REVIEWER_PROMPT,
    SectionReview,
    check_sections,
    grounded_issues,
    section_facts,
)


def merge_dicts(old: dict, new: dict) -> dict:
    # Parallel specialists each return {their_section: text}; merging keeps all of them,
    # and a rewrite replaces only the section it belongs to.
    return {**(old or {}), **new}


class State(TypedDict, total=False):
    overview: dict
    sections: Annotated[dict[str, str], merge_dicts]
    summary: str
    recommendations: list[str]
    issues: dict[str, list[str]]       # section -> open issues from the latest review
    revisions: int
    sent_back: bool                    # did the latest review send work back?
    human_review: bool                 # pause for a person before finishing
    human_comment: str
    report: str
    trace: Annotated[list[str], operator.add]   # what ran, in order, for the console and GUI
    review_log: Annotated[list[dict], operator.add]  # every review round: drafts and issues


class SpecialistTask(TypedDict):
    section: str
    feedback: list[str]


def fan_out(sections: list[str], issues: dict[str, list[str]]) -> list[Send]:
    return [Send("specialist", {"section": s, "feedback": issues.get(s, [])}) for s in sections]


def build_graph(llm: BaseChatModel, tools: list[BaseTool], out_dir: Path, max_revisions: int,
                prepare_fn=None):
    """prepare_fn runs detection + metrics; injectable so tests can skip YOLO."""
    if prepare_fn is None:
        from triage_agent.tools import prepare as prepare_fn

    by_name = {t.name: t for t in tools}
    # Least privilege: each specialist only sees the tools for its own question.
    specialists = {s.section: build_specialist(s, llm, [by_name[n] for n in s.tools])
                   for s in SPECIALISTS}
    reviewer_llm = llm.with_structured_output(SectionReview)

    def load_reference() -> tuple[dict, dict]:
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        stats = json.loads((out_dir / "image_stats.json").read_text(encoding="utf-8"))
        return metrics, stats

    def prepare(state: State) -> dict:
        return {"overview": prepare_fn(), "revisions": 0, "issues": {},
                "trace": ["prepare: detection, metrics, image stats (code)"]}

    def start_specialists(state: State) -> list[Send]:
        return fan_out([s.section for s in SPECIALISTS], {})

    async def specialist(task: SpecialistTask) -> dict:
        result = await specialists[task["section"]].ainvoke(
            {"messages": [specialist_request(task["feedback"])]},
            config={"recursion_limit": 12})
        calls = [c["name"] for m in result["messages"] for c in (getattr(m, "tool_calls", None) or [])]
        note = "revision" if task["feedback"] else "first draft"
        return {"sections": {task["section"]: result["messages"][-1].content.strip()},
                "trace": [f"{task['section']}_agent ({note}): tools {calls or 'none'}"]}

    async def editor(state: State) -> dict:
        sections = "\n\n".join(f"[{SECTIONS[k].heading}]\n{v}" for k, v in state["sections"].items())
        feedback = state.get("issues", {}).get("editor", [])
        if state.get("human_comment"):
            feedback = [*feedback, f"Comment from the human reviewer: {state['human_comment']}"]
        reply = await llm.ainvoke(EDITOR_PROMPT.format(
            overview=json.dumps(state["overview"], indent=1), sections=sections,
            feedback=("\nFix these problems from the review:\n" + "\n".join(f"- {f}" for f in feedback))
            if feedback else ""))
        summary, recommendations = parse_editor(reply.content)
        return {"summary": summary, "recommendations": recommendations,
                "trace": ["editor: summary + recommendations"]}

    async def reviewer(state: State) -> dict:
        # Reference read from disk, not from the agents' messages: it must not depend on
        # what the agents chose to look at.
        metrics, stats = load_reference()
        editor_text = state["summary"] + "\n" + "\n".join(state["recommendations"])
        issues = check_sections(state["sections"], editor_text, metrics, stats)

        # One small LLM call per section, each with only its own facts.
        texts = {**state["sections"], "editor": editor_text}
        facts = section_facts(metrics, stats)

        async def judge(name: str) -> list[str]:
            try:
                review = await reviewer_llm.ainvoke(REVIEWER_PROMPT.format(
                    facts=json.dumps(facts[name], indent=1), section=texts[name]))
                return grounded_issues(texts[name], review)
            except Exception:  # malformed structured output from a small model
                return []

        for name, found in zip(texts, await asyncio.gather(*(judge(n) for n in texts)), strict=True):
            if found:
                issues.setdefault(name, []).extend(found)

        sent_back = bool(issues) and state["revisions"] < max_revisions
        summary = ", ".join(f"{k} ({len(v)})" for k, v in issues.items()) or "all checks passed"
        record = {"round": state["revisions"], "sections": dict(state["sections"]),
                  "editor": editor_text, "issues": issues}
        return {"issues": issues, "sent_back": sent_back, "review_log": [record],
                "revisions": state["revisions"] + int(sent_back),
                "trace": [f"reviewer: {summary}" + (" -> sent back" if sent_back else "")]}

    def after_review(state: State):
        if not state["sent_back"]:
            return "human_approval"
        # Only the specialists whose sections failed rerun; the editor then runs again anyway.
        failed = [s for s in state["issues"] if s in SECTIONS]
        return fan_out(failed, state["issues"]) if failed else "editor"

    def human_approval(state: State) -> dict:
        if not state.get("human_review"):
            return {}
        answer = interrupt({"summary": state["summary"], "open_issues": state["issues"]})
        if answer.get("approved", True):
            return {"human_comment": "", "trace": ["human: approved"]}
        return {"human_comment": answer.get("comment", "Please revise."),
                "trace": [f"human: rejected ({answer.get('comment', '')})"]}

    def after_human(state: State) -> str:
        return "editor" if state.get("human_comment") else "finish"

    def finish(state: State) -> dict:
        metrics, _ = load_reference()
        report = render_report(state["summary"], state["sections"], state["recommendations"],
                               metrics, state.get("issues", {}))
        return {"report": report, "trace": ["finish: report assembled"]}

    graph = StateGraph(State)
    graph.add_node("prepare", prepare)
    graph.add_node("specialist", specialist)
    graph.add_node("editor", editor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("human_approval", human_approval)
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", start_specialists, ["specialist"])
    graph.add_edge("specialist", "editor")
    graph.add_edge("editor", "reviewer")
    graph.add_conditional_edges("reviewer", after_review, ["specialist", "editor", "human_approval"])
    graph.add_conditional_edges("human_approval", after_human, ["editor", "finish"])
    graph.add_edge("finish", END)
    # The checkpointer saves state after every step; interrupt() needs it to pause and resume.
    return graph.compile(checkpointer=MemorySaver())

