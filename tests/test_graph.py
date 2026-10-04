"""Graph wiring and routing, with a scripted fake LLM: no Ollama needed."""

import asyncio
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from triage_agent import tools
from triage_agent.agents import parse_editor
from triage_agent.graph import build_graph, fan_out, merge_dicts
from triage_agent.review import SectionReview


def test_merge_dicts_keeps_parallel_sections_and_replaces_rewrites():
    state = merge_dicts({}, {"size": "v1"})
    state = merge_dicts(state, {"confusion": "c1"})
    assert merge_dicts(state, {"size": "v2"}) == {"size": "v2", "confusion": "c1"}


def test_fan_out_sends_only_failed_sections_with_their_feedback():
    sends = fan_out(["size"], {"size": ["wrong count"], "editor": ["x"]})
    expected = [("specialist", {"section": "size", "feedback": ["wrong count"]})]
    assert [(s.node, s.arg) for s in sends] == expected


def test_parse_editor_splits_summary_and_recommendations():
    text = "SUMMARY:\nRecall is 0.5.\nRECOMMENDATIONS:\n- Tile images.\n2. Train on small objects."
    assert parse_editor(text) == ("Recall is 0.5.", ["Tile images.", "Train on small objects."])


class ScriptedLLM(GenericFakeChatModel):
    """Answers calls in order with fixed messages; the reviewer always finds nothing."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        class Structured:
            async def ainvoke(self, _prompt):
                return SectionReview()

        return Structured()


def test_full_graph_runs_and_assembles_report(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "overall": {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5},
        "per_class": {"car": {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5}},
        "recall_by_size": {"car": {"small": {"recall": 0.0, "num_gt": 1, "missed": 1},
                                   "medium": {"recall": 1.0, "num_gt": 1, "missed": 0},
                                   "large": {"recall": 0.0, "num_gt": 0, "missed": 0}}},
        "confusion": {"car": {"missed": 1}}, "background_false_positives": {},
        "images_without_detections": [], "hard_images": [],
    }))
    group = {"num_images": 1, "mean_brightness": 100.0, "dark_images": 0, "mean_num_objects": 2.0,
             "median_object_area": 500, "mean_small_share": 0.5}
    (tmp_path / "image_stats.json").write_text(json.dumps(
        {"per_image": {}, "comparison": {"hard_images": group, "all_images": group, "dark_threshold": 70}}))

    drafts = [AIMessage("1 of 1 small car objects were missed.")] * 3
    editor = AIMessage("SUMMARY:\nOverall recall is 0.5.\nRECOMMENDATIONS:\n- Tile images.")
    llm = ScriptedLLM(messages=iter([*drafts, editor]))
    agent_tools = [StructuredTool.from_function(f) for f in tools.METRICS_TOOLS + tools.IMAGE_TOOLS]
    graph = build_graph(llm, agent_tools, tmp_path, max_revisions=2, prepare_fn=lambda: {"overall": {}})

    config = {"configurable": {"thread_id": "t"}}
    state = asyncio.run(graph.ainvoke({"human_review": False, "trace": []}, config=config))
    assert "## Where it fails: object size" in state["report"]
    assert "Tile images." in state["report"]
    assert state["revisions"] == 0  # every check passed, nothing sent back
    assert sum(line.endswith("(first draft): tools none") for line in state["trace"]) == 3
