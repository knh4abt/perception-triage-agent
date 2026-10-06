"""The metrics tools as a small web page and, at the same time, as an MCP server.

Runs on a free Hugging Face Space (Gradio SDK). Gradio turns each function below into a
tab you can click and, with mcp_server=True, into an MCP tool at /gradio_api/mcp/.
The functions are the same ones the local metrics MCP server exposes.
"""

from __future__ import annotations

import gradio as gr

from triage_agent import tools

ABOUT = """# Where does YOLOv8n fail?
Detection metrics for YOLOv8n on 200 COCO street images (person, car, bus, truck),
served to AI agents over MCP. Agents connect to `/gradio_api/mcp/`; people can click the tabs."""

tabs = [
    gr.Interface(tools.size_breakdown, inputs=None, outputs=gr.JSON(), api_name="size_breakdown",
                 description="Recall for small, medium and large objects."),
    gr.Interface(tools.confusions, inputs=None, outputs=gr.JSON(), api_name="confusions",
                 description="Which real classes were detected as a different class."),
    gr.Interface(tools.false_alarms, inputs=None, outputs=gr.JSON(), api_name="false_alarms",
                 description="Detections that match no labelled object."),
    gr.Interface(tools.hard_examples, inputs=gr.Number(value=5, precision=0, label="n"),
                 outputs=gr.JSON(), api_name="hard_examples",
                 description="The images with the most errors."),
]

with gr.Blocks(title="Perception triage metrics") as demo:
    gr.Markdown(ABOUT)
    gr.TabbedInterface(tabs, ["Object size", "Confusions", "False alarms", "Hard images"])


if __name__ == "__main__":
    demo.launch(mcp_server=True)
