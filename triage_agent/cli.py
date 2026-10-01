"""Command-line entry point: python -m triage_agent.cli --images data/sample --out reports/"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find where an object detector fails.")
    parser.add_argument("--images", type=Path, default=Path("data/sample"),
                        help="folder with the input images")
    parser.add_argument("--out", type=Path, default=Path("reports"),
                        help="folder where report.md is written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"images: {args.images}")
    print(f"out:    {args.out}")


if __name__ == "__main__":
    main()
