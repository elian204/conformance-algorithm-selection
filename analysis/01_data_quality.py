#!/usr/bin/env python3
"""Basic data-quality checks for merged ML tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run basic data-quality checks")
    p.add_argument("--input", required=True, type=str, help="Path to merged ML table")
    p.add_argument("--out-dir", required=True, type=str)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    key_cols = [c for c in ["model_id", "trace_hash", "trace_id"] if c in df.columns]
    duplicate_rows = int(df.duplicated(subset=key_cols, keep=False).sum()) if key_cols else 0

    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
    }).sort_values("missing_rate", ascending=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "key_columns": key_cols,
        "duplicate_rows_on_keys": duplicate_rows,
    }

    if "symbolic_status" in df.columns:
        summary["symbolic_status_counts"] = {
            str(k): int(v) for k, v in df["symbolic_status"].value_counts(dropna=False).items()
        }
    if "symbolic_faster_flag" in df.columns:
        valid = df["symbolic_faster_flag"].dropna()
        summary["symbolic_faster_rate"] = float(valid.mean()) if len(valid) > 0 else None

    missing_path = out_dir / "data_quality_missingness.csv"
    summary_path = out_dir / "data_quality_summary.json"

    missing.to_csv(missing_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {missing_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
