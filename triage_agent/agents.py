"""The specialist agents and the editor.

Each specialist answers one question with its own tools and writes one section of the
report. The editor writes the summary and recommendations from the three sections.
Code adds all headings and tables, so the LLMs only write plain paragraphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

RULES = """Rules: call your tools first. Use only numbers returned by the tools, copied exactly.
Never estimate, add up or round differently. Write 3 to 5 plain sentences: no headings,
no tables, no bullet lists."""


@dataclass(frozen=True)
class Specialist:
    section: str       # key in the report state
    heading: str       # heading code puts above the agent's text
    tools: tuple[str, ...]
    task: str


SPECIALISTS = (
    Specialist(
        section="size",
        heading="Where it fails: object size",
        tools=("size_breakdown",),
        task="Explain how object size affects detection. State the largest failure exactly as "
             "the tool's largest_failure sentence gives it, then compare small and large recall "
             "for the classes.",
    ),
    Specialist(
        section="confusion",
        heading="Where it fails: class confusions and false alarms",
        tools=("confusions", "false_alarms"),
        task="Explain which real classes were detected as a different class (keep the direction "
             "exactly as the tool states it) and where false alarms come from, including the "
             "class with the lowest precision.",
    ),
    Specialist(
        section="hard_images",
        heading="What the hard images have in common",
        tools=("hard_examples", "compare_hard_vs_all", "image_stats"),
        task="Explain what makes the hardest images hard. Call compare_hard_vs_all and compare "
             "the hardest images with all images on brightness, number of objects, object size "
             "and share of small objects. Say clearly which factors differ a lot and which do "
             "not. Then call hard_examples, pick the first file name and call image_stats for it; "
             "mention that image by its exact file name with its statistics.",
    ),
)
SECTIONS = {s.section: s for s in SPECIALISTS}


class SpecialistState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_specialist(spec: Specialist, llm: BaseChatModel, tools: list[BaseTool]):
    """A small agent <-> tools loop. Compiled once, reused for every revision round."""
    llm_with_tools = llm.bind_tools(tools)
    system = SystemMessage(f"You are one specialist in a team analysing where the object "
                           f"detector YOLOv8n fails on COCO street images.\n\n{spec.task}\n\n{RULES}")

    async def agent(state: SpecialistState) -> dict:
        return {"messages": [await llm_with_tools.ainvoke([system, *state["messages"]])]}

    def route(state: SpecialistState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(SpecialistState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


def specialist_request(feedback: list[str]) -> HumanMessage:
    text = "Write your section."
    if feedback:
        text += ("\nA reviewer checked your previous version against the data and found:\n"
                 + "\n".join(f"- {f}" for f in feedback)
                 + "\nWrite the corrected section.")
    return HumanMessage(text)


EDITOR_PROMPT = """You are the editor of a report on where the object detector YOLOv8n fails.
Three specialists wrote the sections below. Write the summary and 2 or 3 recommendations.
Use only numbers that appear in OVERVIEW or in the SECTIONS, copied exactly.

Answer in exactly this format and nothing else:
SUMMARY:
<3 sentences: overall precision and recall, the weakest class, the main failure mode>
RECOMMENDATIONS:
- <next step that follows from a finding>
- <next step that follows from a finding>

OVERVIEW:
{overview}

SECTIONS:
{sections}
{feedback}"""


def parse_editor(text: str) -> tuple[str, list[str]]:
    """Split the editor's answer at the two markers.

    Plain text with markers instead of JSON mode: forced JSON once made the model generate
    without end. If a marker is missing, the reviewer flags the empty part.
    """
    summary, _, recs = text.partition("RECOMMENDATIONS:")
    summary = summary.replace("SUMMARY:", "").strip()
    items = [line.strip().lstrip("-*0123456789. ").strip() for line in recs.splitlines()]
    return summary, [item for item in items if item][:3]
