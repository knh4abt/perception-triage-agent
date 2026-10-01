"""The Reviewer's code checks. These are the guards that caught real LLM mistakes."""

import pytest

from triage_agent.report import render_tables
from triage_agent.review import check_coverage, check_numbers, review_facts


@pytest.fixture
def metrics():
    return {
        "overall": {"tp": 594, "fp": 129, "fn": 391, "precision": 0.822, "recall": 0.603},
        "per_class": {"person": {"tp": 533, "fp": 93, "fn": 290, "precision": 0.851, "recall": 0.648}},
        "recall_by_size": {"person": {
            "small": {"recall": 0.369, "num_gt": 360, "missed": 227},
            "medium": {"recall": 0.801, "num_gt": 286, "missed": 57},
            "large": {"recall": 0.966, "num_gt": 177, "missed": 6},
        }},
        "confusion": {"person": {"missed": 290}, "car": {"truck": 4, "missed": 66}},
        "images_without_detections": ["000000177213.jpg"],
        "hard_images": [{"file_name": "000000490936.jpg", "num_gt": 16, "tp": 4, "fp": 3, "fn": 12,
                         "missed": ["person"], "false_alarms": []}],
    }


def test_check_numbers_accepts_real_values_and_percentages(metrics):
    text = "Precision 0.822, recall 60.3%, 533 true positives in 000000490936.jpg."
    assert check_numbers(text, metrics) == []


def test_check_numbers_flags_invented_value(metrics):
    assert check_numbers("Recall is 0.61.", metrics) == ["The number 0.61 does not appear in metrics.json."]


def test_check_numbers_ignores_names_and_small_list_numbers(metrics):
    assert check_numbers("1. YOLOv8n on COCO val2017 misses objects under 32x32 px.", metrics) == []


def test_coverage_requires_largest_failure_with_exact_count(metrics):
    good = "## Where it fails\n227 of 360 small person objects were missed.\n## Recommendations\n"
    assert check_coverage(good, metrics) == []


def test_coverage_rejects_total_written_as_missed(metrics):
    # The real mistake from an earlier run: the total (360) written as the number missed.
    bad = "## Where it fails\n360 small persons were missed.\n"
    assert len(check_coverage(bad, metrics)) == 1


def test_coverage_rejects_missing_section(metrics):
    assert check_coverage("## Summary\nAll fine.", metrics) == ["The section 'Where it fails' is missing."]


def test_review_facts_state_confusion_direction(metrics):
    facts = review_facts(metrics)
    assert "car detected as truck: 4" in facts["class_confusions"]
    assert facts["largest_miss_groups"][0].startswith("small person: 227 of 360")


def test_rendered_tables_pass_the_number_check(metrics):
    # Code-rendered tables must never trip the Reviewer's own check.
    assert check_numbers(render_tables(metrics), metrics) == []
