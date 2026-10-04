"""The Reviewer's code checks. These are the guards that caught real LLM mistakes."""

import pytest

from triage_agent.report import render_report, render_tables
from triage_agent.review import (
    Claim,
    SectionReview,
    check_brightness_claims,
    check_confusion_claims,
    check_coverage,
    check_lowest_precision,
    check_numbers,
    check_raw_tool_calls,
    check_sections,
    grounded_issues,
    section_facts,
)


@pytest.fixture
def metrics():
    return {
        "overall": {"tp": 594, "fp": 129, "fn": 391, "precision": 0.822, "recall": 0.603},
        "per_class": {
            "person": {"tp": 533, "fp": 93, "fn": 290, "precision": 0.851, "recall": 0.648},
            "truck": {"tp": 10, "fp": 13, "fn": 20, "precision": 0.435, "recall": 0.333},
        },
        "recall_by_size": {"person": {
            "small": {"recall": 0.369, "num_gt": 360, "missed": 227},
            "medium": {"recall": 0.801, "num_gt": 286, "missed": 57},
            "large": {"recall": 0.966, "num_gt": 177, "missed": 6},
        }},
        "confusion": {"person": {"missed": 290}, "car": {"truck": 4, "missed": 66}},
        "background_false_positives": {"person": 93},
        "images_without_detections": ["000000177213.jpg"],
        "hard_images": [{"file_name": "000000490936.jpg", "num_gt": 16, "tp": 4, "fp": 3, "fn": 12,
                         "missed": ["person"], "false_alarms": []}],
    }


@pytest.fixture
def image_stats():
    group = {"num_images": 10, "mean_brightness": 128.0, "dark_images": 1, "mean_num_objects": 13.5,
             "median_object_area": 242, "mean_small_share": 0.836}
    return {"per_image": {}, "comparison": {"hard_images": group, "all_images": group, "dark_threshold": 70}}


def test_check_numbers_accepts_real_values_and_percentages(metrics):
    text = "Precision 0.822, recall 60.3%, 533 true positives in 000000490936.jpg."
    assert check_numbers(text, metrics) == []


def test_check_numbers_flags_invented_value(metrics):
    expected = ["The number 0.61 does not appear in the measured data."]
    assert check_numbers("Recall is 0.61.", metrics) == expected


def test_check_numbers_ignores_names_and_small_list_numbers(metrics):
    assert check_numbers("1. YOLOv8n on COCO val2017 misses objects under 32x32 px.", metrics) == []


def test_coverage_requires_largest_failure_with_exact_count(metrics):
    assert check_coverage("227 of 360 small person objects were missed.", metrics) == []


def test_coverage_rejects_total_written_as_missed(metrics):
    # The real mistake from an earlier run: the total (360) written as the number missed.
    assert len(check_coverage("360 small persons were missed.", metrics)) == 1


def test_coverage_rejects_empty_section(metrics):
    assert check_coverage("  ", metrics) == ["The size section is empty."]


def test_lowest_precision_must_name_the_right_class(metrics):
    # The real mistake: the confusion agent called person the lowest-precision class.
    assert check_lowest_precision("The class with the lowest precision is person.", metrics)
    assert check_lowest_precision("Truck has the lowest precision, 0.435.", metrics) == []


def test_issues_are_grouped_by_the_section_that_must_fix_them(metrics, image_stats):
    sections = {"size": "227 of 360 small person objects were missed.",
                "confusion": "A real car was detected as truck 4 times.",
                "hard_images": "Image 12345 has brightness 130."}  # invented, as in a real run
    issues = check_sections(sections, "Overall recall is 0.603.", metrics, image_stats)
    assert set(issues) == {"hard_images"}


def test_grounded_issues_drop_quotes_not_in_the_section():
    section = "A real truck was detected as bus 2 times. Small objects are missed most."
    review = SectionReview(issues=[
        Claim(sentence="A real truck was detected as bus 2 times.", fact="bus detected as truck: 2"),
        Claim(sentence="Trucks are never missed.", fact="truck missed: 20"),  # made up by the reviewer
    ])
    assert grounded_issues(section, review) == [
        '"A real truck was detected as bus 2 times." contradicts: bus detected as truck: 2']


def test_section_facts_state_confusion_direction(metrics, image_stats):
    facts = section_facts(metrics, image_stats)
    assert "a real car was detected as truck: 4 times" in facts["confusion"]["confusions"]
    assert facts["confusion"]["lowest_precision_class"] == "truck"
    assert facts["size"]["largest_failure"] == "227 of 360 small person objects were missed"


def test_rendered_tables_pass_the_number_check(metrics):
    # Code-rendered tables must never trip the Reviewer's own check.
    assert check_numbers(render_tables(metrics), metrics) == []


def test_report_lists_sections_in_order_and_unresolved_issues(metrics):
    report = render_report("Summary text.", {"size": "S.", "confusion": "C.", "hard_images": "H."},
                           ["Tile small objects."], metrics, {"size": ["still wrong"]})
    order = [report.index(h) for h in ("## Summary", "## Where it fails: object size",
                                       "## Where it fails: class confusions", "## What the hard images",
                                       "## Recommendations", "## Reviewer notes")]
    assert order == sorted(order)
    assert "- size: still wrong" in report


def test_reviewer_issue_dropped_when_its_fact_agrees_with_the_sentence():
    # Real run: the reviewer "contradicted" a correct sentence with the same fact.
    section = "A real car was detected as truck 4 times."
    review = SectionReview(issues=[Claim(sentence="A real car was detected as truck 4 times.",
                                         fact="a real car was detected as truck: 4 times")])
    assert grounded_issues(section, review) == []


def test_brightness_claim_must_match_measurement(image_stats):
    comparison = image_stats["comparison"]
    comparison["all_images"] = {**comparison["all_images"], "mean_brightness": 112.9}
    # Real run: hard images (128.0) were called darker than all images (112.9).
    assert check_brightness_claims("The hardest images are significantly darker.", comparison)
    assert check_brightness_claims("The hardest images are slightly brighter.", comparison) == []


def test_raw_tool_call_text_is_flagged():
    assert check_raw_tool_calls('Done. {"name": "hard_examples", "parameters": {"n": 1}}')
    assert check_raw_tool_calls("Image 000000466416.jpg has 12 objects.") == []


def test_confusion_claims_must_match_the_table(metrics):
    # Real run: 13 is the truck false-positive count, the truck->car confusion is 0 here.
    assert check_confusion_claims("The truck, which was detected as a car 13 times.", metrics)
    assert check_confusion_claims("A real car was detected as truck 4 times.", metrics) == []
    assert check_confusion_claims("Cars were detected as trucks 4 times.", metrics) == []
