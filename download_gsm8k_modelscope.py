#!/usr/bin/env python3
"""Download modelscope/gsm8k to a local directory.

Default target:
    /Users/yuzhian/Downloads/modelscope_gsm8k

Install dependency first if needed:
    pip install "modelscope[datasets]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ID = "modelscope/gsm8k"
DEFAULT_SUBSET = "main"
DEFAULT_OUTPUT_DIR = Path("/Users/yuzhian/Downloads/modelscope_gsm8k")
DEFAULT_SPLITS = ("train", "test")


def to_plain_record(item: Any) -> dict[str, Any]:
    """Convert a dataset row to a JSON-serializable dict."""
    if isinstance(item, dict):
        return item

    if hasattr(item, "items"):
        return dict(item.items())

    raise TypeError(f"Unsupported dataset row type: {type(item)!r}")


def save_jsonl(dataset: Any, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for row in dataset:
            record = to_plain_record(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_split(
    dataset_id: str,
    subset_name: str,
    split: str,
    output_dir: Path,
    trust_remote_code: bool,
) -> int:
    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency for ModelScope datasets: {missing}\n"
            "Install project dependencies with:\n"
            "    python3 -m pip install -r requirements.txt"
        ) from exc

    print(f"Downloading {dataset_id} subset={subset_name!r} split={split!r} ...")

    dataset = MsDataset.load(
        dataset_id,
        subset_name=subset_name,
        split=split,
        cache_dir=str(output_dir / "cache"),
        trust_remote_code=trust_remote_code,
    )

    output_file = output_dir / f"{split}.jsonl"
    count = save_jsonl(dataset, output_file)
    print(f"Saved {count} rows to {output_file}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download modelscope/gsm8k and export splits as JSONL files."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--subset-name", default=DEFAULT_SUBSET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the ModelScope cache and exported JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to download, for example: train test",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable execution of the dataset loading script from ModelScope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in args.splits:
        total += download_split(
            dataset_id=args.dataset_id,
            subset_name=args.subset_name,
            split=split,
            output_dir=output_dir,
            trust_remote_code=args.trust_remote_code,
        )

    print(f"Done. Exported {total} rows under {output_dir}")


if __name__ == "__main__":
    main()
