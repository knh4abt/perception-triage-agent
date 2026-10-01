"""Download COCO val2017 annotations and ~200 images that contain our target classes.

Run once from the project root: python -m scripts.download_data
Re-running is cheap: files that already exist are skipped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import requests

from triage_agent.config import load_config

# Official COCO hosts: plain http (port 80) is blocked on many corporate networks and the
# https S3 bucket is slow for the 250 MB annotation zip. A Hugging Face copy of the single
# file we need is ~25 MB and fast; check_annotations() verifies it matches the official stats.
ANNOTATIONS_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
IMAGES_BASE = "https://s3.amazonaws.com/images.cocodataset.org/val2017"
SEED = 0  # fixed so everyone who runs this gets the same 200 images


def download_annotations(dest: Path) -> None:
    if dest.exists():
        print(f"annotations already present: {dest}")
        return
    print("downloading instances_val2017.json (~25 MB)...")
    resp = requests.get(ANNOTATIONS_URL, timeout=300)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"saved {dest}")


def check_annotations(coco: dict) -> None:
    # Official COCO val2017 numbers; a mismatch means a corrupted or modified mirror.
    counts = (len(coco["images"]), len(coco["annotations"]), len(coco["categories"]))
    if counts != (5000, 36781, 80):
        raise ValueError(f"unexpected COCO val2017 content (images, boxes, classes): {counts}")


def pick_images(coco: dict, class_names: list[str], n: int) -> list[dict]:
    """Images with at least one non-crowd box of a target class, sampled reproducibly."""
    target_ids = {c["id"] for c in coco["categories"] if c["name"] in class_names}
    keep = {a["image_id"] for a in coco["annotations"]
            if a["category_id"] in target_ids and not a["iscrowd"]}
    candidates = sorted((img for img in coco["images"] if img["id"] in keep),
                        key=lambda img: img["id"])
    return random.Random(SEED).sample(candidates, n)


def fetch(url: str, attempts: int = 3) -> bytes:
    # Single connection timeouts happen on slow networks; retrying beats restarting the run.
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1}: {url}")
    raise AssertionError("unreachable")


def download_images(images: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images, 1):
        path = out_dir / img["file_name"]
        if path.exists():
            continue
        path.write_bytes(fetch(f"{IMAGES_BASE}/{img['file_name']}"))
        if i % 20 == 0:
            print(f"  {i}/{len(images)}")


def main() -> None:
    cfg = load_config()["data"]
    ann_path = Path(cfg["annotations"])
    download_annotations(ann_path)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    check_annotations(coco)
    images = pick_images(coco, cfg["classes"], cfg["num_images"])
    print(f"selected {len(images)} images containing {cfg['classes']}")
    download_images(images, Path(cfg["images"]))
    print(f"done: {cfg['images']}")


if __name__ == "__main__":
    main()
