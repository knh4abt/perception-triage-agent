"""The agents' tools and image statistics on a tiny fake dataset: no model, no LLM call."""

import json

import pytest
from PIL import Image

from triage_agent import tools
from triage_agent.image_stats import brightness, compare_groups


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    images = tmp_path / "images"
    out = tmp_path / "out"
    images.mkdir()
    out.mkdir()
    Image.new("RGB", (64, 64), (200, 200, 200)).save(images / "a.jpg")  # bright
    Image.new("RGB", (64, 64), (20, 20, 20)).save(images / "b.jpg")     # dark

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


def test_prepare_reuses_predictions_and_writes_reference_files(workspace):
    # rerun_detection=False must not load YOLO: the fake weights file does not exist.
    overview = tools.prepare()
    assert overview["num_boxes"] == 1
    assert overview["lowest_recall_class"] == "car"
    assert (workspace / "metrics.json").exists() and (workspace / "image_stats.json").exists()


def test_metrics_tools(workspace):
    tools.prepare()
    assert tools.size_breakdown()["largest_failure"] == "1 of 1 small car objects were missed"
    assert tools.hard_examples(n=1)["hard_images"][0]["file_name"] == "b.jpg"
    assert tools.false_alarms()["lowest_precision_class"] == "car"


def test_image_tools(workspace):
    tools.prepare()
    assert tools.image_stats("b.jpg")["brightness"] < 70
    assert tools.image_stats("missing.jpg") == {"error": "unknown image missing.jpg"}
    comparison = tools.compare_hard_vs_all()
    assert comparison["all_images"]["num_images"] == 2


def test_brightness_of_plain_images(tmp_path):
    Image.new("RGB", (32, 32), (100, 100, 100)).save(tmp_path / "grey.png")
    assert brightness(tmp_path / "grey.png") == pytest.approx(100, abs=1)


def test_compare_groups_counts_dark_images():
    stats = {"a": {"brightness": 30.0, "num_objects": 10, "median_object_area": 100, "small_share": 1.0},
             "b": {"brightness": 150.0, "num_objects": 2, "median_object_area": 5000, "small_share": 0.0}}
    result = compare_groups(stats, ["a"])
    assert result["hard_images"]["dark_images"] == 1
    assert result["all_images"]["mean_num_objects"] == 6.0
