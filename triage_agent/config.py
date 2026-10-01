"""Load the YAML config and the .env secrets."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    # Secrets go into os.environ, not into the returned dict, so the config
    # can be printed or logged without leaking keys.
    load_dotenv()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
