"""Compare predictions.json with COCO ground truth and write metrics.json.

Run from the project root: python -m triage_agent.metrics
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from triage_agent.config import load_config

# COCO size buckets by object area in pixels: small < 32x32 <= medium < 96x96 <= large.
SIZE_LIMITS = [("small", 32**2), ("medium", 96**2), ("large", float("inf"))]


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two [x, y, w, h] boxes."""
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _intersection(a: list[float], b: list[float]) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def size_bucket(area: float) -> str:
    return next(name for name, limit in SIZE_LIMITS if area < limit)


def load_ground_truth(ann_path: Path, file_names: set[str], classes: list[str]) -> list[dict]:
    """GT boxes for our images and classes, in the same flat format as predictions."""
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]
                  if img["file_name"] in file_names}
    return [{"file_name": id_to_file[a["image_id"]], "category": names[a["category_id"]],
             "bbox": a["bbox"], "area": a["area"], "iscrowd": bool(a["iscrowd"])}
            for a in coco["annotations"]
            if a["image_id"] in id_to_file and names[a["category_id"]] in classes]


def match_image(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[list, list, list]:
    """Greedy matching per class, highest score first (the standard COCO/VOC procedure).

    Returns (true positives as (pred, gt) pairs, false-positive preds, missed gts).
    """
    normal = [g for g in gts if not g["iscrowd"]]
    crowd = [g for g in gts if g["iscrowd"]]
    used: set[int] = set()
    tps, fps = [], []
    for p in sorted(preds, key=lambda p: -p["score"]):
        best, best_iou = None, iou_thr
        for i, g in enumerate(normal):
            if i in used or g["category"] != p["category"]:
                continue
            o = iou(p["bbox"], g["bbox"])
            if o >= best_iou:
                best, best_iou = i, o
        if best is not None:
            used.add(best)
            tps.append((p, normal[best]))
        elif any(g["category"] == p["category"] and _inside_crowd(p, g, iou_thr) for g in crowd):
            # A box inside a labelled crowd is neither right nor wrong: COCO ignores it.
            continue
        else:
            fps.append(p)
    fns = [g for i, g in enumerate(normal) if i not in used]
    return tps, fps, fns


def _inside_crowd(pred: dict, crowd: dict, thr: float) -> bool:
    # Intersection over the prediction's own area: one person box inside a big crowd box
    # has a tiny IoU but lies fully inside it.
    area = pred["bbox"][2] * pred["bbox"][3]
    return area > 0 and _intersection(pred["bbox"], crowd["bbox"]) / area >= thr


def _ratio(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def compute_metrics(preds: list[dict], gts: list[dict], file_names: list[str],
                    classes: list[str], iou_thr: float, num_hard: int) -> dict:
    by_file_p, by_file_g = defaultdict(list), defaultdict(list)
    for p in preds:
        by_file_p[p["file_name"]].append(p)
    for g in gts:
        by_file_g[g["file_name"]].append(g)

    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    size_hits = {c: {s: [0, 0] for s, _ in SIZE_LIMITS} for c in classes}  # [found, total]
    confusion = {c: defaultdict(int) for c in classes}  # gt class -> what it became
    background_fp = defaultdict(int)  # preds that overlap no object at all
    per_image = []

    for f in file_names:
        tps, fps, fns = match_image(by_file_p[f], by_file_g[f], iou_thr)
        for p, g in tps:
            counts[p["category"]]["tp"] += 1
            size_hits[g["category"]][size_bucket(g["area"])][0] += 1
        for p in fps:
            counts[p["category"]]["fp"] += 1
        for g in fns:
            counts[g["category"]]["fn"] += 1

        # Explain each miss: was the object found but called a different class?
        unexplained_fps = list(fps)
        for g in fns:
            other = next((p for p in unexplained_fps
                          if p["category"] != g["category"] and iou(p["bbox"], g["bbox"]) >= iou_thr),
                         None)
            if other:
                confusion[g["category"]][other["category"]] += 1
                unexplained_fps.remove(other)
            else:
                confusion[g["category"]]["missed"] += 1
        for p in unexplained_fps:
            background_fp[p["category"]] += 1
        for g in by_file_g[f]:
            if not g["iscrowd"]:
                size_hits[g["category"]][size_bucket(g["area"])][1] += 1

        per_image.append({
            "file_name": f, "num_gt": sum(not g["iscrowd"] for g in by_file_g[f]),
            "tp": len(tps), "fp": len(fps), "fn": len(fns),
            "missed": sorted({g["category"] for g in fns}),
            "false_alarms": sorted({p["category"] for p in fps}),
        })

    per_class = {c: {**n, "precision": _ratio(n["tp"], n["tp"] + n["fp"]),
                     "recall": _ratio(n["tp"], n["tp"] + n["fn"])} for c, n in counts.items()}
    total = {k: sum(n[k] for n in counts.values()) for k in ("tp", "fp", "fn")}
    hard = sorted(per_image, key=lambda r: (-(r["fp"] + r["fn"]), r["file_name"]))[:num_hard]

    return {
        "settings": {"iou_threshold": iou_thr, "num_images": len(file_names), "classes": classes},
        "overall": {**total, "precision": _ratio(total["tp"], total["tp"] + total["fp"]),
                    "recall": _ratio(total["tp"], total["tp"] + total["fn"])},
        "per_class": per_class,
        "recall_by_size": {c: {s: {"recall": _ratio(hit, n), "num_gt": n}
                               for s, (hit, n) in sizes.items()} for c, sizes in size_hits.items()},
        "confusion": {c: dict(v) for c, v in confusion.items()},
        "background_false_positives": dict(background_fp),
        "images_without_detections": [r["file_name"] for r in per_image if r["tp"] + r["fp"] == 0],
        "hard_images": hard,
    }


def main() -> None:
    cfg = load_config()
    images_dir = Path(cfg["data"]["images"])
    out_dir = Path(cfg["output"]["dir"])
    file_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    classes = cfg["data"]["classes"]

    preds = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    gts = load_ground_truth(Path(cfg["data"]["annotations"]), set(file_names), classes)
    metrics = compute_metrics(preds, gts, file_names, classes,
                              cfg["eval"]["iou_threshold"], cfg["eval"]["num_hard_examples"])

    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    o = metrics["overall"]
    print(f"precision {o['precision']}  recall {o['recall']}  (tp {o['tp']}, fp {o['fp']}, fn {o['fn']})")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
