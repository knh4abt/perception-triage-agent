"""report.md assembled by code: headings, tables and lists.

The agents write only plain paragraphs. Code places them under fixed headings and adds
every table straight from metrics.json: an 8B model invented per-class breakdowns for
hard images and changed heading formats when it wrote the whole report itself.
"""

from __future__ import annotations

from triage_agent.agents import SPECIALISTS


def render_report(summary: str, sections: dict[str, str], recommendations: list[str],
                  metrics: dict, open_issues: dict[str, list[str]]) -> str:
    parts = ["# Where does YOLOv8n fail?", "", "## Summary", "", summary.strip()]
    for spec in SPECIALISTS:
        parts += ["", f"## {spec.heading}", "", sections.get(spec.section, "(missing)").strip()]
    parts += ["", "## Recommendations", ""]
    parts += [f"{i}. {r.strip()}" for i, r in enumerate(recommendations, 1)]
    parts += ["", render_tables(metrics)]
    if open_issues:
        # Out of revisions: ship the report, but say openly what is still unverified.
        parts += ["", "## Reviewer notes (unresolved)", ""]
        parts += [f"- {section}: {issue}" for section, found in open_issues.items() for issue in found]
    return "\n".join(parts) + "\n"


def render_tables(metrics: dict) -> str:
    lines = ["## Per-class results", "",
             "| Class | Precision | Recall | TP | FP | FN |", "|---|---|---|---|---|---|"]
    for cls, r in metrics["per_class"].items():
        lines.append(f"| {cls} | {r['precision']} | {r['recall']} | {r['tp']} | {r['fp']} | {r['fn']} |")
    o = metrics["overall"]
    lines.append(f"| **all** | {o['precision']} | {o['recall']} | {o['tp']} | {o['fp']} | {o['fn']} |")

    lines += ["", "## Recall by object size", "",
              "| Class | Small | Medium | Large |", "|---|---|---|---|"]
    for cls, sizes in metrics["recall_by_size"].items():
        cells = [f"{s['recall']} ({s['missed']}/{s['num_gt']} missed)" for s in sizes.values()]
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    lines += ["", "Small < 32x32 px, large >= 96x96 px (COCO definition)."]

    lines += ["", "## Hardest images", "",
              "| Image | Objects | Found | Missed | False alarms | Missed classes |",
              "|---|---|---|---|---|---|"]
    for h in metrics["hard_images"]:
        lines.append(f"| {h['file_name']} | {h['num_gt']} | {h['tp']} | {h['fn']} | {h['fp']} | "
                     f"{', '.join(h['missed']) or '-'} |")
    if metrics["images_without_detections"]:
        lines += ["", "Images with no detections at all: "
                  + ", ".join(metrics["images_without_detections"])]
    return "\n".join(lines)
