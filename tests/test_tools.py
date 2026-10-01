"""The agent's tools on a tiny fake dataset: no model download, no LLM call."""

import json

import pytest

from triage_agent import tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    images = tmp_path / "images"
    out = tmp_path / "out"
    images.mkdir()
    out.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        (images / name).write_bytes(b"")  # tools only list file names; content is never read

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 0},
            {"image_id": 2, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        ],
    }
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps(coco))
    preds = [{"file_name": "a.jpg", "category": "person", "bbox": [1, 1, 50, 50], "score": 0.9}]
    (out / "predictions.json").write_text(json.dumps(preds))

    cfg = {
        "data": {"annotations": str(ann), "images": str(images), "classes": ["person", "car"]},
        "detector": {"weights": "unused.pt", "conf_threshold": 0.25, "device": "cpu"},
        "eval": {"iou_threshold": 0.5, "num_hard_examples": 5},
        "output": {"dir": str(out)},
    }
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    return out


def test_run_detection_reuses_existing_predictions(workspace):
    # rerun=False must not load YOLO: the fake weights file does not exist.
    assert tools.run_detection() == {"num_images": 2, "num_boxes": 1, "boxes_per_class": {"person": 1}}


def test_compute_class_metrics_writes_metrics_json(workspace):
    result = tools.compute_class_metrics()
    assert result["per_class"]["person"]["recall"] == 1.0
    assert result["per_class"]["car"]["fn"] == 1
    assert (workspace / "metrics.json").exists()


def test_hard_examples_and_summary(workspace):
    hard = tools.find_hard_examples(n=1)
    assert hard["hard_images"][0]["file_name"] == "b.jpg"
    assert hard["images_without_detections"] == ["b.jpg"]

    summary = tools.summarize_findings()
    assert summary["lowest_recall_class"] == "car"
    assert summary["largest_failure"] == "small car: 1 of 1 missed"
