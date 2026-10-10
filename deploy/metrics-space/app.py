"""The metrics tools as a web page for people and an MCP server for AI agents.

People see the results directly as tables (no buttons to press). Agents call the same
functions as MCP tools at /gradio_api/mcp/. The data is fixed (one local run), so the page
is rendered once at start-up.
"""

from __future__ import annotations

import gradio as gr

from triage_agent import tools


def size_page() -> str:
    data = tools.size_breakdown()
    rows = "\n".join(
        f"| {cls} | {s['small']['recall']} ({s['small']['missed']} of {s['small']['num_gt']} missed) "
        f"| {s['medium']['recall']} | {s['large']['recall']} |"
        for cls, s in data["recall_by_size"].items())
    return (f"**Biggest failure: {data['largest_failure']}.**\n\n"
            "Recall = share of real objects the detector found.\n\n"
            "| Class | Small | Medium | Large |\n|---|---:|---:|---:|\n" + rows +
            f"\n\nSize classes: {data['size_definition']}.")


def confusion_page() -> str:
    data = tools.confusions()
    found = "\n".join(f"- {c}" for c in data["confusions"]) or "- none"
    missed = "\n".join(f"| {c} | {n} |" for c, n in data["missed_completely"].items())
    return ("**Found, but called the wrong class:**\n\n" + found +
            "\n\n**Not found at all:**\n\n| Class | Missed objects |\n|---|---:|\n" + missed)


def false_alarm_page() -> str:
    data = tools.false_alarms()
    rows = "\n".join(f"| {c} | {data['false_positives_per_class'][c]} | {data['precision_per_class'][c]} |"
                     for c in data["precision_per_class"])
    return ("A false alarm is a box where there is no real object.\n\n"
            "| Class | False alarms | Precision |\n|---|---:|---:|\n" + rows +
            f"\n\n**Lowest precision: {data['lowest_precision_class']}.**")


def hard_images_page() -> str:
    data = tools.hard_examples(n=10)
    rows = "\n".join(f"| {h['file_name']} | {h['num_gt']} | {h['tp']} | {h['fn']} | {h['fp']} |"
                     for h in data["hard_images"])
    empty = ", ".join(data["images_without_detections"]) or "none"
    return ("The 10 photos with the most errors.\n\n"
            "| Photo | Objects | Found | Missed | False alarms |\n|---|---:|---:|---:|---:|\n" + rows +
            f"\n\n**Photos where nothing was detected:** {empty}")


AGENTS = """AI agents can use these results as tools through MCP (Model Context Protocol).

- **Address:** `/gradio_api/mcp/` on this site (streamable HTTP)
- **Tools:** `size_breakdown`, `confusions`, `false_alarms`, `hard_examples`

In the main project, three AI agents call these tools to write their report:
[perception-triage-agent](https://github.com/knh4abt/perception-triage-agent)"""

with gr.Blocks(title="Where does YOLOv8n fail?") as demo:
    gr.Markdown("# Where does YOLOv8n fail?\nResults of the object detector YOLOv8n on 200 COCO "
                "street photos (person, car, bus, truck).")
    with gr.Tab("Object size"):
        gr.Markdown(size_page())
    with gr.Tab("Mixed-up classes"):
        gr.Markdown(confusion_page())
    with gr.Tab("False alarms"):
        gr.Markdown(false_alarm_page())
    with gr.Tab("Hard photos"):
        gr.Markdown(hard_images_page())
    with gr.Tab("For AI agents"):
        gr.Markdown(AGENTS)

    # Hidden from the page, visible to agents: each function becomes one MCP tool.
    for fn in tools.METRICS_TOOLS:
        gr.api(fn, api_name=fn.__name__)


if __name__ == "__main__":
    demo.launch(mcp_server=True)
