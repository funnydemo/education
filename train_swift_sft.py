#!/usr/bin/env python3
"""Launch ModelScope SWIFT SFT training with a YAML config."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("swift_sft.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run `swift sft` with a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the SWIFT YAML training config.",
    )
    parser.add_argument(
        "swift_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments appended after `swift sft --config <config>`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.expanduser().resolve()

    if not config.is_file():
        raise SystemExit(f"Config file does not exist: {config}")
    if shutil.which("swift") is None:
        raise SystemExit(
            "Missing SWIFT CLI: install dependencies with `python3 -m pip install -r requirements.txt`."
        )

    command = ["swift", "sft", "--config", str(config), *args.swift_args]
    print("Running:", " ".join(command))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
