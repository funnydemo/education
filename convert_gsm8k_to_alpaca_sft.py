#!/usr/bin/env python3
"""Convert GSM8K JSONL data to Alpaca-style SFT JSONL data.

Input row:
    {"question": "...", "answer": "..."}

Output row:
    {"instruction": "...", "input": null, "output": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/Users/yuzhian/Downloads/modelscope_gsm8k/train.jsonl")


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, record


def convert_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    question = record.get("question")
    answer = record.get("answer")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}: missing non-empty string field 'question'")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{source}: missing non-empty string field 'answer'")

    return {
        "instruction": question,
        "input": None,
        "output": answer,
    }


def convert_file(input_file: Path, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for line_number, record in read_jsonl(input_file):
            converted = convert_record(record, f"{input_file}:{line_number}")
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    return count


def default_output_for(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "alpaca_sft"
    return input_path.with_name(f"{input_path.stem}_alpaca_sft.jsonl")


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No .jsonl files found under {input_path}")
        return files
    raise SystemExit(f"Input path does not exist: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert modelscope/gsm8k JSONL to Alpaca-style SFT JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input GSM8K JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL file for a single input file, or output directory "
            "when --input is a directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output else default_output_for(input_path)
    )
    input_files = iter_input_files(input_path)

    total = 0
    if len(input_files) == 1 and input_path.is_file():
        count = convert_file(input_files[0], output_path)
        print(f"Converted {count} rows: {input_files[0]} -> {output_path}")
        total += count
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for input_file in input_files:
            output_file = output_path / f"{input_file.stem}_alpaca_sft.jsonl"
            count = convert_file(input_file, output_file)
            print(f"Converted {count} rows: {input_file} -> {output_file}")
            total += count

    print(f"Done. Converted {total} rows.")


if __name__ == "__main__":
    main()
