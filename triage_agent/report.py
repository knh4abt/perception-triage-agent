"""The data parts of report.md, rendered by code straight from metrics.json.

The LLM writes only the interpretation (summary, where it fails, recommendations).
Tables and lists are copied by code: an 8B model invented per-class breakdowns for
hard images when it wrote them itself.
"""

from __future__ import annotations


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
