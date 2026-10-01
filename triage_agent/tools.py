"""The four tools the agent can call. Plain functions: no LLM, no framework.

Keeping them framework-free means pytest can call them directly and the MCP server
can expose the exact same functions. The docstrings are what the LLM reads to decide
which tool to call, so they describe when to use each tool.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from triage_agent.config import load_config
from triage_agent.detector import run_detection as _detect
from triage_agent.detector import save_predictions
from triage_agent.metrics import compute_metrics, largest_miss_group, load_ground_truth


def _paths(cfg: dict) -> tuple[Path, Path]:
    return Path(cfg["data"]["images"]), Path(cfg["output"]["dir"])


def _load_metrics() -> dict:
    cfg = load_config()
    _, out_dir = _paths(cfg)
    path = out_dir / "metrics.json"
    if not path.exists():
        compute_class_metrics()
    return json.loads(path.read_text(encoding="utf-8"))


def run_detection(rerun: bool = False) -> dict:
    """Run the YOLOv8 detector on all images. Call this first.

    Reuses existing predictions unless rerun is true. Returns how many boxes were
    found per class.
    """
    cfg = load_config()
    images_dir, out_dir = _paths(cfg)
    pred_path = out_dir / "predictions.json"
    if rerun or not pred_path.exists():
        save_predictions(_detect(images_dir, cfg), out_dir)
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    return {
        "num_images": len(list(images_dir.glob("*.jpg"))),
        "num_boxes": len(preds),
        "boxes_per_class": dict(Counter(p["category"] for p in preds)),
    }


def compute_class_metrics() -> dict:
    """Compare detections with ground truth. Returns precision and recall per class,
    recall by object size (small/medium/large), and which classes get confused.
    Also writes metrics.json, the reference the report is checked against.
    """
    cfg = load_config()
    images_dir, out_dir = _paths(cfg)
    file_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    classes = cfg["data"]["classes"]
    preds = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    gts = load_ground_truth(Path(cfg["data"]["annotations"]), set(file_names), classes)
    metrics = compute_metrics(preds, gts, file_names, classes,
                              cfg["eval"]["iou_threshold"], cfg["eval"]["num_hard_examples"])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return {k: metrics[k] for k in ("overall", "per_class", "recall_by_size", "confusion",
                                    "background_false_positives")}


def find_hard_examples(n: int = 5) -> dict:
    """Return the n images with the most errors (missed objects plus false alarms),
    and the images where the detector found nothing at all.
    """
    m = _load_metrics()
    return {"hard_images": m["hard_images"][:n],
            "images_without_detections": m["images_without_detections"]}


def summarize_findings() -> dict:
    """Return the key findings as precomputed facts: weakest classes, the gap between
    small and large objects, and the most common class confusions. Call this before
    writing the report and quote its numbers.
    """
    m = _load_metrics()
    per_class = m["per_class"]
    by_size = m["recall_by_size"]
    size, cls, missed = largest_miss_group(m)
    confusions = sorted(((gt, pred, n) for gt, row in m["confusion"].items()
                         for pred, n in row.items() if pred != "missed"), key=lambda t: -t[2])
    # Precomputing comparisons here keeps arithmetic out of the LLM, which an 8B model
    # gets wrong often enough to matter.
    return {
        "overall": m["overall"],
        "lowest_recall_class": min(per_class, key=lambda c: per_class[c]["recall"]),
        "lowest_precision_class": min(per_class, key=lambda c: per_class[c]["precision"]),
        "small_vs_large_recall": {c: {"small": s["small"]["recall"], "large": s["large"]["recall"],
                                      "num_small": s["small"]["num_gt"]} for c, s in by_size.items()},
        # Given as a ready sentence: without the missed count, the model once wrote the
        # total (360) as the number missed (227).
        "largest_failure": f"{size} {cls}: {missed} of {by_size[cls][size]['num_gt']} missed",
        "top_confusions": [f"{gt} detected as {pred}: {n}" for gt, pred, n in confusions[:3]],
        "num_images_without_detections": len(m["images_without_detections"]),
    }


ALL_TOOLS = [run_detection, compute_class_metrics, find_hard_examples, summarize_findings]
