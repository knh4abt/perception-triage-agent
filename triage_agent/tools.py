"""Tools for the specialist agents. Plain functions: no LLM, no framework.

Keeping them framework-free means pytest can call them directly and the MCP servers can
expose the exact same functions. Each agent only gets the tools for its own question,
so the docstrings describe one narrow job each.

prepare() is not an agent tool: the graph runs it as code before any agent starts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from triage_agent.config import load_config
from triage_agent.image_stats import build_image_stats, save_image_stats
from triage_agent.metrics import compute_metrics, largest_miss_group, load_ground_truth


def _paths(cfg: dict) -> tuple[Path, Path]:
    return Path(cfg["data"]["images"]), Path(cfg["output"]["dir"])


def _load(name: str) -> dict:
    _, out_dir = _paths(load_config())
    path = out_dir / name
    if not path.exists():
        prepare()
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(rerun_detection: bool = False) -> dict:
    """Detection, metrics and image statistics, written to disk. Returns an overview."""
    cfg = load_config()
    images_dir, out_dir = _paths(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    if rerun_detection or not pred_path.exists():
        # Imported here, not at the top: it pulls in ultralytics and torch (~2 GB), which the
        # deployed metrics server never needs because it only reads metrics.json.
        from triage_agent.detector import run_detection, save_predictions

        save_predictions(run_detection(images_dir, cfg), out_dir)
    preds = json.loads(pred_path.read_text(encoding="utf-8"))

    file_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    classes = cfg["data"]["classes"]
    gts = load_ground_truth(Path(cfg["data"]["annotations"]), set(file_names), classes)
    metrics = compute_metrics(preds, gts, file_names, classes,
                              cfg["eval"]["iou_threshold"], cfg["eval"]["num_hard_examples"])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    save_image_stats(build_image_stats(images_dir, gts, metrics), out_dir)

    per_class = metrics["per_class"]
    return {
        "num_images": len(file_names),
        "num_boxes": len(preds),
        "boxes_per_class": dict(Counter(p["category"] for p in preds)),
        "overall": metrics["overall"],
        "per_class": per_class,
        "lowest_recall_class": min(per_class, key=lambda c: per_class[c]["recall"]),
        "lowest_precision_class": min(per_class, key=lambda c: per_class[c]["precision"]),
    }


# --- metrics server tools ---------------------------------------------------------

def size_breakdown() -> dict:
    """Recall per class for small, medium and large objects, and the single largest
    failure as a ready sentence. Use it to explain how object size affects detection.
    """
    m = _load("metrics.json")
    size, cls, missed = largest_miss_group(m)
    total = m["recall_by_size"][cls][size]["num_gt"]
    return {
        "recall_by_size": m["recall_by_size"],
        # Given as a ready sentence: without the missed count, the model once wrote the
        # total (360) as the number missed (227).
        "largest_failure": f"{missed} of {total} {size} {cls} objects were missed",
        "size_definition": "small < 32x32 px, large >= 96x96 px (COCO)",
    }


def confusions() -> dict:
    """Which real classes were detected as a different class, with direction stated
    explicitly, and how many objects of each class were missed completely.
    """
    m = _load("metrics.json")
    pairs = sorted(((gt, pred, n) for gt, row in m["confusion"].items()
                    for pred, n in row.items() if pred != "missed"), key=lambda t: -t[2])
    return {
        "confusions": [f"a real {gt} was detected as {pred}: {n} times" for gt, pred, n in pairs],
        "missed_completely": {gt: row.get("missed", 0) for gt, row in m["confusion"].items()},
    }


def false_alarms() -> dict:
    """False alarms: detections that match no labelled object. Per class, plus precision
    and the class with the lowest precision.
    """
    m = _load("metrics.json")
    per_class = m["per_class"]
    return {
        "false_positives_per_class": {c: r["fp"] for c, r in per_class.items()},
        "background_false_positives": m["background_false_positives"],
        "precision_per_class": {c: r["precision"] for c, r in per_class.items()},
        "lowest_precision_class": min(per_class, key=lambda c: per_class[c]["precision"]),
    }


def hard_examples(n: int = 5) -> dict:
    """The n images with the most errors (missed objects plus false alarms), and the
    images where the detector found nothing at all.
    """
    m = _load("metrics.json")
    return {"hard_images": m["hard_images"][:n],
            "images_without_detections": m["images_without_detections"]}


# --- images server tools ----------------------------------------------------------

def compare_hard_vs_all() -> dict:
    """Brightness, number of objects and object size: the hardest images compared with
    all images. Use it to explain what makes the hard images hard.
    """
    return _load("image_stats.json")["comparison"]


def image_stats(file_name: str) -> dict:
    """Brightness (0-255), number of labelled objects, median object area in pixels and
    share of small objects for one image, for example '000000466416.jpg'.
    """
    stats = _load("image_stats.json")["per_image"]
    if file_name not in stats:
        return {"error": f"unknown image {file_name}"}
    return {"file_name": file_name, **stats[file_name]}


METRICS_TOOLS = [size_breakdown, confusions, false_alarms, hard_examples]
IMAGE_TOOLS = [compare_hard_vs_all, image_stats]
