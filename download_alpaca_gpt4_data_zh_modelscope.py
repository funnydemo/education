#!/usr/bin/env python3
"""Download the first 500 rows of AI-ModelScope/alpaca-gpt4-data-zh.

Default target:
    /Users/yuzhian/Downloads/modelscope_alpaca_gpt4_data_zh_500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ID = "AI-ModelScope/alpaca-gpt4-data-zh"
DEFAULT_OUTPUT_DIR = Path("/Users/yuzhian/Downloads/modelscope_alpaca_gpt4_data_zh_500")
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 500


def to_plain_record(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item

    if hasattr(item, "items"):
        return dict(item.items())

    raise TypeError(f"Unsupported dataset row type: {type(item)!r}")


def iter_limited(dataset: Iterable[Any], limit: int) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield to_plain_record(row)


def save_jsonl(dataset: Iterable[Any], output_file: Path, limit: int) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for record in iter_limited(dataset, limit):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def load_dataset(
    dataset_id: str,
    split: str,
    output_dir: Path,
    trust_remote_code: bool,
) -> Any:
    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency for ModelScope datasets: {missing}\n"
            "Install project dependencies with:\n"
            "    python3 -m pip install -r requirements.txt"
        ) from exc

    return MsDataset.load(
        dataset_id,
        split=split,
        cache_dir=str(output_dir / "cache"),
        trust_remote_code=trust_remote_code,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download AI-ModelScope/alpaca-gpt4-data-zh and export a limited "
            "number of rows as JSONL."
        )
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the ModelScope cache and exported JSONL file.",
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
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading {args.dataset_id} split={args.split!r}; "
        f"exporting first {args.limit} rows ..."
    )
    dataset = load_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        output_dir=output_dir,
        trust_remote_code=args.trust_remote_code,
    )

    output_file = output_dir / f"{args.split}_first_{args.limit}.jsonl"
    count = save_jsonl(dataset, output_file, args.limit)
    print(f"Saved {count} rows to {output_file}")


if __name__ == "__main__":
    main()
