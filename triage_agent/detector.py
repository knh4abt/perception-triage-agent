"""Run YOLOv8 on a folder of images and save the boxes as predictions.json.

Run from the project root: python -m triage_agent.detector
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from triage_agent.config import load_config


def run_detection(images_dir: Path, cfg: dict) -> list[dict]:
    """One dict per box: file_name, class name, bbox [x, y, w, h] in pixels, score."""
    model = YOLO(cfg["detector"]["weights"])
    wanted = set(cfg["data"]["classes"])
    # Map names to the model's own class ids instead of hardcoding COCO ids.
    class_ids = [i for i, name in model.names.items() if name in wanted]

    predictions = []
    # stream=True yields one result at a time, so memory stays flat for any folder size.
    results = model.predict(source=str(images_dir), conf=cfg["detector"]["conf_threshold"],
                            device=cfg["detector"]["device"], classes=class_ids,
                            stream=True, verbose=False)
    for r in results:
        file_name = Path(r.path).name
        for (x1, y1, x2, y2), cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                r.boxes.conf.tolist()):
            predictions.append({
                "file_name": file_name,
                "category": model.names[int(cls)],
                # Same [x, y, w, h] format as the COCO ground truth, so metrics.py compares like with like.
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(score, 3),
            })
    return predictions


def save_predictions(predictions: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.json"
    path.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    return path


def main() -> None:
    cfg = load_config()
    preds = run_detection(Path(cfg["data"]["images"]), cfg)
    path = save_predictions(preds, Path(cfg["output"]["dir"]))
    n_images = len({p["file_name"] for p in preds})
    print(f"{len(preds)} boxes on {n_images} images -> {path}")


if __name__ == "__main__":
    main()
