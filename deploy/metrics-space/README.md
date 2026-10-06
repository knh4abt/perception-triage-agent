---
title: Perception Triage Metrics MCP
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
---

Detection metrics of [perception-triage-agent](https://github.com/knh4abt/perception-triage-agent)
(YOLOv8n on 200 COCO street images): object size, class confusions, false alarms, hard images.

- People: click the tabs.
- AI agents: connect an MCP client to `/gradio_api/mcp/` (streamable HTTP).

Read-only. Built from that repository with `python -m scripts.build_space`.
