"""Matching and metrics on hand-made boxes, so every expected value can be checked by eye."""

import pytest

from triage_agent.metrics import compute_metrics, iou, largest_miss_group, match_image, size_bucket


def box(category, bbox, score=None, crowd=False, file_name="a.jpg"):
    d = {"file_name": file_name, "category": category, "bbox": bbox}
    if score is None:  # ground truth
        d.update(area=bbox[2] * bbox[3], iscrowd=crowd)
    else:
        d["score"] = score
    return d


def test_iou_identical_disjoint_and_half_overlap():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)


def test_size_bucket_uses_coco_limits():
    assert size_bucket(31 * 31) == "small"
    assert size_bucket(32 * 32) == "medium"
    assert size_bucket(96 * 96) == "large"


def test_match_true_positive_false_positive_and_miss():
    gts = [box("car", [0, 0, 10, 10]), box("car", [50, 50, 10, 10])]
    preds = [box("car", [1, 1, 10, 10], 0.9), box("car", [200, 200, 10, 10], 0.8)]
    tps, fps, fns = match_image(preds, gts, 0.5)
    assert len(tps) == 1 and len(fps) == 1 and len(fns) == 1
    assert fns[0]["bbox"] == [50, 50, 10, 10]


def test_wrong_class_is_not_a_match():
    tps, fps, fns = match_image([box("truck", [0, 0, 10, 10], 0.9)], [box("car", [0, 0, 10, 10])], 0.5)
    assert (len(tps), len(fps), len(fns)) == (0, 1, 1)


def test_higher_score_wins_the_ground_truth_box():
    gts = [box("person", [0, 0, 10, 10])]
    weak = box("person", [0, 0, 10, 10], 0.3)
    strong = box("person", [1, 0, 10, 10], 0.9)
    tps, fps, _ = match_image([weak, strong], gts, 0.5)
    assert tps[0][0] is strong and fps == [weak]


def test_prediction_inside_crowd_box_is_ignored():
    gts = [box("person", [0, 0, 100, 100], crowd=True)]
    tps, fps, fns = match_image([box("person", [10, 10, 10, 20], 0.9)], gts, 0.5)
    # A crowd box is never a miss, and a person found inside it is not a false alarm.
    assert (tps, fps, fns) == ([], [], [])


def test_compute_metrics_counts_confusion_and_size():
    gts = [box("car", [0, 0, 10, 10]), box("truck", [100, 100, 50, 50])]
    preds = [box("truck", [0, 0, 10, 10], 0.9), box("truck", [100, 100, 50, 50], 0.8)]
    m = compute_metrics(preds, gts, ["a.jpg"], ["car", "truck"], 0.5, num_hard=5)
    assert m["per_class"]["truck"] == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0}
    assert m["per_class"]["car"]["fn"] == 1
    assert m["confusion"]["car"] == {"truck": 1}  # the car was found, but called a truck
    assert m["recall_by_size"]["car"]["small"] == {"recall": 0.0, "num_gt": 1, "missed": 1}
    assert largest_miss_group(m) == ("small", "car", 1)


def test_image_without_predictions_counts_all_misses():
    gts = [box("person", [0, 0, 10, 10], file_name="empty.jpg")]
    m = compute_metrics([], gts, ["empty.jpg"], ["person"], 0.5, num_hard=5)
    assert m["images_without_detections"] == ["empty.jpg"]
    assert m["overall"]["recall"] == 0.0
