# Can Small AI Agents Explain Where a Detector Fails, Without Making Things Up?

![Python](https://img.shields.io/badge/python-3.11-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-1.x-purple) ![MCP](https://img.shields.io/badge/MCP-tools-black) ![License](https://img.shields.io/badge/license-MIT-green) [![ci](https://github.com/knh4abt/perception-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/knh4abt/perception-triage-agent/actions/workflows/ci.yml)

An object detector (YOLOv8n) looks at 200 street photos. A small team of AI agents writes
a report on where it fails. A checker compares every claim with the real numbers and sends
wrong parts back to be fixed.

Everything runs on a laptop, for free (Llama 3.1 8B through Ollama).

**Try the results online:** [triage-metrics.onrender.com](https://triage-metrics.onrender.com)
(a web page for people and an MCP server for AI agents; on the free plan it can take a
minute to wake up).

<p align="center">
  <img src="docs/agent-story.gif" width="600" alt="An agent calls the hard images darker, the checker says they are brighter, the agent fixes it" /><br/>
  <sub>A real catch: an agent wrote "darker", the measured data says "brighter".</sub>
</p>

---

## What we found

**1. The detector misses small objects.** It finds 97% of large persons but only 37% of
small ones: **227 of 360 small persons were missed**.

**2. The hardest photos are crowded, not dark.** They have almost 3 times more objects, and
the objects are much smaller.

| | 10 hardest photos | all 200 photos |
|---|---:|---:|
| objects per photo | 13.5 | 4.9 |
| typical object size (pixels) | 242 | 3728 |
| brightness (0-255) | 128 | 113 |

<p align="center">
  <img src="docs/detector-misses.gif" width="560" alt="YOLOv8n on a street scene: green boxes found, red boxes missed" /><br/>
  <sub>The hardest photo: 4 of 16 objects found. Green: found. Red: missed.</sub>
</p>

**3. The AI agents make convincing mistakes.** They copy numbers correctly but sometimes get
the meaning wrong, for example "360 missed" when 360 was the total. Every time that happened,
I added a check in code. Mistakes that are still left at the end are listed in the report,
not hidden.

---

## How it works

1. **Measure:** code runs the detector and compares it with the correct labels.
2. **Explain:** three AI agents each write one part: object size, mixed-up classes, hard photos.
3. **Summarise:** an editor agent writes the summary and advice.
4. **Check:** a checker compares every claim with the numbers. A wrong part goes back to the
   agent that wrote it (at most twice).
5. **Approve (optional):** you read the draft and approve it before it is saved.

Result: [`reports/report.md`](reports/report.md)

---

## Run it

You need Python 3.11 and [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
python -m scripts.download_data        # downloads the 200 photos
python -m triage_agent.cli             # writes reports/report.md (about 4 minutes)
```

Add `--approve` to read and approve the report yourself before it is saved.

To let the agents use the online tools instead of the local ones:

```bash
METRICS_MCP_URL=https://triage-metrics.onrender.com/gradio_api/mcp/ python -m triage_agent.cli
```

---

## Limits

- 200 photos is a small sample.
- Each run can produce a slightly different report.
- The checker only catches the kinds of mistakes it was built for.

<details>
<summary>Technical details (architecture, design decisions, tests)</summary>

<br/>

**Stack:** LangGraph (three specialist agents run in parallel, an editor, a reviewer, optional
human approval), tools served by two MCP servers (metrics and image statistics), Ollama for
the LLM, pytest (33 tests, no LLM needed), Docker, GitHub Actions.

**Design decisions**

| Decision | Instead of | Why |
|---|---|---|
| Three fixed specialists in parallel | An LLM "supervisor" that decides who runs | Every report needs all three; an 8B router only adds a way to fail |
| Each agent sees only its own tools | All tools for every agent | Shorter prompts for a small model, clear owner per section |
| A wrong section goes back to its owner only | Rewrite the whole report | Faster, and correct sections stay untouched |
| Code writes all tables and headings | The LLM writes everything | The model changed formats and invented numbers |
| The checker reads the data from disk | Trusting what the agents looked at | A check must not share the agents' blind spots |

**Rules the report must follow, enforced by code and tested**

| Rule | Code | Test |
|---|---|---|
| Every number exists in the measured data | `check_numbers` | `test_check_numbers_flags_invented_value` |
| The biggest failure is stated exactly | `check_coverage` | `test_coverage_rejects_total_written_as_missed` |
| "A detected as B N times" matches the data | `check_confusion_claims` | `test_confusion_claims_must_match_the_table` |
| "Darker/brighter" matches the brightness | `check_brightness_claims` | `test_brightness_claim_must_match_measurement` |
| A rewrite goes only to the section's owner | `fan_out` | `test_fan_out_sends_only_failed_sections_with_their_feedback` |
| The whole workflow runs end to end | `build_graph` | `test_full_graph_runs_and_assembles_report` |

**Engineering:** fixed seed and checked label counts for the data, a token limit on every LLM
call, pinned versions for the deployed container (`deploy/`, running on Render), CI on every push.

**Detector numbers:** overall precision 0.822, recall 0.603 (confidence 0.25, IoU 0.5,
985 labelled objects). Labels come from a Hugging Face copy of COCO, checked against the
official counts.

</details>

---

## References and license

COCO (Lin et al., ECCV 2014, CC BY 4.0) · Ultralytics YOLOv8 (AGPL-3.0) · LangGraph ·
Model Context Protocol · Meta Llama 3.1 via Ollama.

Code: MIT. The COCO photos are not included; the script downloads them.
