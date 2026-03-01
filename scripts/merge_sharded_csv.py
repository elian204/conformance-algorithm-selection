#!/usr/bin/env python3
"""Merge shard-level results.csv files into one canonical CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REQUIRED_COLUMNS = {"trace_hash", "method", "model_id"}


def merge_csv_files(
    input_paths: Sequence[Path],
    output_path: Path,
    sort_rows: bool = True,
    fail_on_duplicates: bool = True,
) -> Path:
    """Merge multiple shard CSVs after validating schema and uniqueness."""
    if not input_paths:
        raise ValueError("At least one input CSV is required")

    rows: List[dict] = []
    header: List[str] | None = None
    model_id: str | None = None
    seen_pairs: set[Tuple[str, str]] = set()

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        with input_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            current_header = list(reader.fieldnames or [])
            if not current_header:
                raise ValueError(f"Input CSV has no header: {input_path}")

            if header is None:
                header = current_header
                missing = REQUIRED_COLUMNS - set(header)
                if missing:
                    raise ValueError(f"Input CSV missing required columns {sorted(missing)}: {input_path}")
            elif current_header != header:
                raise ValueError(f"CSV headers do not match: {input_path}")

            for row in reader:
                row_model_id = row.get("model_id", "")
                if model_id is None:
                    model_id = row_model_id
                elif row_model_id != model_id:
                    raise ValueError(
                        f"All rows must share the same model_id; found {row_model_id!r} after {model_id!r}"
                    )

                key = (row.get("trace_hash", ""), row.get("method", ""))
                if fail_on_duplicates and key in seen_pairs:
                    raise ValueError(f"Duplicate (trace_hash, method) pair detected: {key}")
                seen_pairs.add(key)
                rows.append(row)

    if header is None:
        raise ValueError("No CSV header found across inputs")

    if sort_rows:
        rows.sort(key=lambda row: (row.get("trace_hash", ""), row.get("method", "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge shard results.csv files")
    p.add_argument("--inputs", nargs="+", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort merged rows by trace_hash, method (default: true)",
    )
    p.add_argument(
        "--fail-on-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject duplicate (trace_hash, method) pairs (default: true)",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    merge_csv_files(
        [Path(path) for path in args.inputs],
        Path(args.output),
        sort_rows=args.sort,
        fail_on_duplicates=args.fail_on_duplicates,
    )
    print(f"Merged CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
