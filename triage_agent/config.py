"""Load the YAML config and the .env secrets."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    # Secrets go into os.environ, not into the returned dict, so the config
    # can be printed or logged without leaking keys.
    load_dotenv()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # The CLI sets these from --images/--out. Environment variables also reach the
    # MCP server, which runs as a separate process and cannot see the CLI's arguments.
    if os.environ.get("TRIAGE_IMAGES"):
        cfg["data"]["images"] = os.environ["TRIAGE_IMAGES"]
    if os.environ.get("TRIAGE_OUT"):
        cfg["output"]["dir"] = os.environ["TRIAGE_OUT"]
    return cfg
