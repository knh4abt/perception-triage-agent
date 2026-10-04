"""Checks the Reviewer runs on each section of the draft.

Two layers:
- code checks: every number must exist in the reference files (metrics.json,
  image_stats.json), and the size section must state the largest failure exactly.
- the LLM reviewer (prompt below): one call per section, for reasoning errors code
  cannot catch, such as a confusion stated in the wrong direction. Each issue must quote
  a sentence that code then finds in the section.
Issues are grouped by section, so only the agent that wrote a section has to rewrite it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from triage_agent.metrics import largest_miss_group

# Integers up to this size are skipped: list numbering, "3 sentences", "top 5".
# Small counts therefore go unchecked; the trade-off is no false alarms on prose.
SMALL_INT = 10

REVIEWER_PROMPT = """You check one section of a report about where an object detector fails.
The FACTS come straight from the measured data. Most sections are correct.

Report a sentence only if a specific FACT shows that it is false, for example it says
"A detected as B" while the FACTS say "B detected as A", names the wrong class, or calls
two clearly different values "similar". A sentence that correctly restates a fact is NOT
an issue. Do not report style, wording, length or missing details.

For each issue, copy the false sentence exactly and the fact that contradicts it.
If nothing is false, return an empty list.

FACTS:
{facts}

SECTION:
{section}"""


class Claim(BaseModel):
    sentence: str = Field(description="the false sentence, copied exactly from the section")
    fact: str = Field(description="the fact from FACTS that shows the sentence is false")


class SectionReview(BaseModel):
    issues: list[Claim] = Field(default_factory=list)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()))


def _fact_agrees(sentence: str, fact: str) -> bool:
    # The reviewer sometimes "contradicts" a sentence with a fact that says the same thing
    # ("car detected as truck 4 times" vs "car was detected as truck: 4 times").
    s, f = _words(sentence), _words(fact)
    return bool(s) and len(s & f) / len(s) >= 0.8


def grounded_issues(section: str, review: SectionReview) -> list[str]:
    """Keep only LLM issues that quote a real sentence and a fact that actually disagrees.

    An 8B reviewer flagged nearly every sentence, correct ones included. Requiring an exact
    quote plus the contradicting fact lets code drop issues it made up (quote not in the
    section) and issues where the quoted fact agrees with the sentence.
    """
    text = _normalise(section)
    kept = []
    for c in review.issues:
        sentence, fact = c.sentence.strip(), c.fact.strip()
        if sentence and _normalise(sentence) in text and not _fact_agrees(sentence, fact):
            kept.append(f'"{sentence}" contradicts: {fact}')
    return sorted(set(kept))


def check_brightness_claims(text: str, comparison: dict) -> list[str]:
    """'Darker' or 'brighter' must match the measured mean brightness.

    Added after the hard-image agent and the editor both called the hardest images darker
    (mean brightness 128.0 vs 112.9 for all images: they are brighter).
    """
    hard = comparison["hard_images"]["mean_brightness"]
    overall = comparison["all_images"]["mean_brightness"]
    lower = text.lower()
    truth = "brighter" if hard > overall else "darker"
    wrong = "darker" if truth == "brighter" else "brighter"
    if wrong in lower:
        return [f"The hardest images are {truth}, not {wrong}: mean brightness {hard} vs {overall}."]
    return []


CLASS_WORDS = r"(person|people|car|truck|bus)e?s?"


def check_confusion_claims(text: str, metrics: dict) -> list[str]:
    """Every 'X detected as Y N times' must match the confusion table.

    Added after the confusion agent wrote "truck detected as a car 13 times": 13 exists in
    metrics.json (truck false positives), so the number check passed; the real count is 2.
    """
    issues = []
    pattern = rf"{CLASS_WORDS}\W+(?:\w+\W+){{0,4}}?detected as (?:an? )?{CLASS_WORDS}\W+(\d+) times?"
    for gt, pred, count in re.findall(pattern, text.lower()):
        gt, pred = ("person" if gt == "people" else gt), ("person" if pred == "people" else pred)
        real = metrics["confusion"].get(gt, {}).get(pred, 0)
        if int(count) != real:
            issues.append(f"A real {gt} was detected as {pred} {real} times, not {count}.")
    return issues


def check_raw_tool_calls(text: str) -> list[str]:
    # A small model sometimes writes a tool call as JSON text instead of calling the tool.
    if re.search(r'\{\s*"name"\s*:', text):
        return ["The section contains a tool call written as text. Call the tool, then write prose."]
    return []


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


def check_numbers(report: str, reference: dict) -> list[str]:
    """Numbers in the text that appear nowhere in the reference data (also as percentages)."""
    allowed: set[float] = set()
    _collect_numbers(reference, allowed)
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
            issues.append(f"The number {token} does not appear in the measured data.")
    return sorted(set(issues))


def check_coverage(size_section: str, metrics: dict) -> list[str]:
    """The size section must state the biggest failure with its exact missed count.

    Added after two failures the LLM reviewer approved: a report that left out small
    persons (the largest group of misses), and one that wrote the total (360) as the
    number missed (227). A rule that code can check should not be left to an LLM.
    """
    size, cls, missed = largest_miss_group(metrics)
    total = metrics["recall_by_size"][cls][size]["num_gt"]
    if not size_section.strip():
        return ["The size section is empty."]
    text = size_section.lower()
    names = {cls, "people"} if cls == "person" else {cls}
    if size in text and any(n in text for n in names) and str(missed) in text:
        return []
    return [(f"The size section must state the largest failure exactly: {missed} of {total} "
             f"{size} {cls} objects were missed. {total} is the total, not the number missed.")]


def check_lowest_precision(text: str, metrics: dict) -> list[str]:
    """A sentence about the lowest precision must name the right class.

    Added after the confusion agent called person the lowest-precision class (it is truck).
    """
    per_class = metrics["per_class"]
    worst = min(per_class, key=lambda c: per_class[c]["precision"])
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "lowest precision" in sentence.lower() and worst not in sentence.lower():
            return [(f"The class with the lowest precision is {worst} "
                     f"({per_class[worst]['precision']}), not what the section says.")]
    return []


def check_sections(sections: dict[str, str], editor_text: str, metrics: dict,
                   image_stats: dict) -> dict[str, list[str]]:
    """All code checks, grouped by the section (and so the agent) that has to fix them."""
    reference = {"metrics": metrics, "image_stats": image_stats}
    comparison = image_stats["comparison"]
    texts = {**sections, "editor": editor_text}
    issues = {name: check_numbers(text, reference) + check_raw_tool_calls(text)
              for name, text in texts.items()}
    issues["size"] = issues.get("size", []) + check_coverage(sections.get("size", ""), metrics)
    issues["confusion"] = (issues.get("confusion", [])
                           + check_lowest_precision(sections.get("confusion", ""), metrics)
                           + check_confusion_claims(sections.get("confusion", ""), metrics))
    issues["editor"] = issues.get("editor", []) + check_confusion_claims(editor_text, metrics)
    for name in ("hard_images", "editor"):
        issues[name] = issues.get(name, []) + check_brightness_claims(texts.get(name, ""), comparison)
    return {name: found for name, found in issues.items() if found}


def section_facts(metrics: dict, image_stats: dict) -> dict[str, dict]:
    """The facts each section is judged against, phrased so direction is explicit.

    Each reviewer call sees only its own section's facts: a short prompt is where a small
    model judges best.
    """
    per_class = metrics["per_class"]
    size, cls, missed = largest_miss_group(metrics)
    size_facts = {
        "recall_by_size": {c: {k: v["recall"] for k, v in sizes.items()}
                           for c, sizes in metrics["recall_by_size"].items()},
        "largest_failure": f"{missed} of {metrics['recall_by_size'][cls][size]['num_gt']} "
                           f"{size} {cls} objects were missed",
    }
    confusion_facts = {
        "confusions": [f"a real {gt} was detected as {pred}: {n} times"
                       for gt, row in metrics["confusion"].items()
                       for pred, n in row.items() if pred != "missed"],
        "precision_per_class": {c: r["precision"] for c, r in per_class.items()},
        "lowest_precision_class": min(per_class, key=lambda c: per_class[c]["precision"]),
        "false_positives_per_class": {c: r["fp"] for c, r in per_class.items()},
    }
    hard_facts = {"hard_vs_all_images": image_stats["comparison"]}
    editor_facts = {"overall": metrics["overall"],
                    "recall_per_class": {c: r["recall"] for c, r in per_class.items()},
                    **size_facts, **confusion_facts, **hard_facts}
    return {"size": size_facts, "confusion": confusion_facts, "hard_images": hard_facts,
            "editor": editor_facts}
