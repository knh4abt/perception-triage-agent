"""Per-image facts (brightness, object count, object size) and hard-vs-all comparison.

Written to image_stats.json next to metrics.json, so the Reviewer can check numbers the
hard-image agent quotes against a file on disk, the same way it checks metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageStat

from triage_agent.metrics import SIZE_LIMITS

DARK_THRESHOLD = 70  # mean grey value (0-255) below which a photo counts as dark, e.g. night


def brightness(path: Path) -> float:
    # A thumbnail is enough for a mean and keeps 200 images under a few seconds.
    with Image.open(path) as img:
        img = img.convert("L")
        img.thumbnail((128, 128))
        return round(ImageStat.Stat(img).mean[0], 1)


def per_image_stats(images_dir: Path, gts: list[dict]) -> dict[str, dict]:
    by_file: dict[str, list[dict]] = {}
    for g in gts:
        if not g["iscrowd"]:
            by_file.setdefault(g["file_name"], []).append(g)
    small_limit = SIZE_LIMITS[0][1]
    stats = {}
    for path in sorted(images_dir.glob("*.jpg")):
        areas = [g["area"] for g in by_file.get(path.name, [])]
        stats[path.name] = {
            "brightness": brightness(path),
            "num_objects": len(areas),
            "median_object_area": round(median(areas)) if areas else 0,
            "small_share": round(sum(a < small_limit for a in areas) / len(areas), 3) if areas else 0.0,
        }
    return stats


def _group_summary(rows: list[dict]) -> dict:
    return {
        "num_images": len(rows),
        "mean_brightness": round(mean(r["brightness"] for r in rows), 1),
        "dark_images": sum(r["brightness"] < DARK_THRESHOLD for r in rows),
        "mean_num_objects": round(mean(r["num_objects"] for r in rows), 1),
        "median_object_area": round(median(r["median_object_area"] for r in rows)),
        "mean_small_share": round(mean(r["small_share"] for r in rows), 3),
    }


def compare_groups(stats: dict[str, dict], hard_files: list[str]) -> dict:
    """Hard images vs all images, precomputed so the LLM never does the arithmetic."""
    return {
        "hard_images": _group_summary([stats[f] for f in hard_files if f in stats]),
        "all_images": _group_summary(list(stats.values())),
        "dark_threshold": DARK_THRESHOLD,
    }


def build_image_stats(images_dir: Path, gts: list[dict], metrics: dict) -> dict:
    stats = per_image_stats(images_dir, gts)
    hard = [h["file_name"] for h in metrics["hard_images"]]
    return {"per_image": stats, "comparison": compare_groups(stats, hard)}


def save_image_stats(result: dict, out_dir: Path) -> Path:
    path = out_dir / "image_stats.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path
