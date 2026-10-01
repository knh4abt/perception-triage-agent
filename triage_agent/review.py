"""Checks the Reviewer runs on a report draft.

Two layers:
- check_numbers: deterministic. Every number in the report must exist in metrics.json.
- the LLM reviewer (prompt below): reasoning errors code cannot catch, such as a
  confusion stated in the wrong direction or the main failure left out.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from triage_agent.metrics import largest_miss_group

# Integers up to this size are skipped: list numbering, "3 sentences", "top 5".
# Small counts therefore go unchecked; the trade-off is no false alarms on prose.
SMALL_INT = 10

REVIEWER_PROMPT = """You review a report about where an object detector fails.
Compare the report with the FACTS, which come straight from metrics.json.

Flag only these problems:
- a statement that contradicts the FACTS (for example "A detected as B" when the FACTS say "B detected as A")
- the largest group of missed objects (first entry of largest_miss_groups) is not mentioned
- a required section is missing: Summary, Where it fails, Recommendations

Do not flag style, wording or length. If there is no such problem, approve.

FACTS:
{facts}

REPORT:
{report}"""


class Review(BaseModel):
    approved: bool = Field(description="true if the report has none of the listed problems")
    issues: list[str] = Field(default_factory=list, description="one short sentence per problem")


def _collect_numbers(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 3))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)


def check_numbers(report: str, metrics: dict) -> list[str]:
    """Numbers in the report that appear nowhere in metrics.json (also as percentages)."""
    allowed: set[float] = set()
    _collect_numbers(metrics, allowed)
    allowed |= {round(v * 100, 1) for v in allowed if v <= 1}  # 0.603 may be written as 60.3%
    # File names are made of digits; drop them before looking for numbers.
    text = re.sub(r"\d{12}\.jpg", "", report)
    issues = []
    # Skip digits glued to letters ("YOLOv8n", "val2017", "32x32"): names, not claims.
    for token in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)", text):
        value = float(token)
        if value.is_integer() and value <= SMALL_INT:
            continue
        if round(value, 3) not in allowed:
            issues.append(f"The number {token} does not appear in metrics.json.")
    return sorted(set(issues))


def check_coverage(report: str, metrics: dict) -> list[str]:
    """The biggest failure must be named in 'Where it fails', with its exact missed count.

    Added after two failures the LLM reviewer approved: a report that left out small
    persons (the largest group of misses), and one that wrote the total (360) as the
    number missed (227). A rule that code can check should not be left to an LLM.
    """
    size, cls, missed = largest_miss_group(metrics)
    match = re.search(r"## Where it fails(.*?)(?=\n## |\Z)", report, re.DOTALL | re.IGNORECASE)
    if not match:
        return ["The section 'Where it fails' is missing."]
    section = match.group(1).lower()
    names = {cls, "people"} if cls == "person" else {cls}
    if size in section and any(n in section for n in names) and str(missed) in section:
        return []
    total = metrics["recall_by_size"][cls][size]["num_gt"]
    return [(f"'Where it fails' must state the largest failure exactly: {missed} of {total} "
             f"{size} {cls} objects were missed. {total} is the total, not the number missed.")]


def review_facts(metrics: dict) -> dict:
    """The facts the LLM reviewer compares against, phrased so direction is explicit."""
    confusions = [f"{gt} detected as {pred}: {n}"
                  for gt, row in metrics["confusion"].items()
                  for pred, n in row.items() if pred != "missed"]
    misses = []
    for cls, sizes in metrics["recall_by_size"].items():
        for size, s in sizes.items():
            missed = s["missed"]
            misses.append((missed, f"{size} {cls}: {missed} of {s['num_gt']} missed (recall {s['recall']})"))
    misses.sort(key=lambda t: -t[0])
    return {
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "class_confusions": confusions,
        "largest_miss_groups": [text for _, text in misses[:3]],
        "images_without_detections": len(metrics["images_without_detections"]),
    }
