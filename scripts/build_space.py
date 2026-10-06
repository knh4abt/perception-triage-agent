"""Assemble the deployable metrics server into dist/metrics-space.

The same folder is used for the local Docker test and for the Hugging Face Space, so what
you test is exactly what you deploy.

Run from the project root: python -m scripts.build_space
"""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE = Path("deploy/metrics-space")
TARGET = Path("dist/metrics-space")


def main() -> None:
    metrics = Path("reports/metrics.json")
    if not metrics.exists():
        raise SystemExit("reports/metrics.json missing: run python -m triage_agent.cli first")

    if TARGET.exists():
        # Keep .git so a cloned Space repo in this folder survives a rebuild.
        for item in TARGET.iterdir():
            if item.name != ".git":
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    TARGET.mkdir(parents=True, exist_ok=True)

    for item in SOURCE.iterdir():
        shutil.copy2(item, TARGET / item.name)
    shutil.copytree("triage_agent", TARGET / "triage_agent",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree("configs", TARGET / "configs")
    # Same relative path as in the project, so the tools find it without any change.
    (TARGET / "reports").mkdir()
    shutil.copy2(metrics, TARGET / "reports" / "metrics.json")
    print(f"built {TARGET}: " + ", ".join(sorted(p.name for p in TARGET.iterdir() if p.name != ".git")))


if __name__ == "__main__":
    main()
