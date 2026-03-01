#!/usr/bin/env python3
"""Validate symbolic baseline CSV contract."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd


REQUIRED_COLUMNS = ["model_id", "trace_hash", "symbolic_time_seconds", "symbolic_status"]
VALID_STATUS = {"ok", "timeout", "error", "no_solution", "max_expansions"}


def validate_symbolic_dataframe(
    df: pd.DataFrame,
    require_trace_id: bool = False,
) -> List[str]:
    errors: List[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    if require_trace_id and "trace_id" not in df.columns:
        errors.append("Column 'trace_id' is required but missing")

    for col in ["model_id", "trace_hash"] + (["trace_id"] if "trace_id" in df.columns else []):
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty_mask.any():
            errors.append(f"Column '{col}' contains empty values at rows {empty_mask[empty_mask].index.tolist()[:10]}")

    times = pd.to_numeric(df["symbolic_time_seconds"], errors="coerce")
    bad_time = times.isna() | ~times.apply(math.isfinite) | (times < 0)
    if bad_time.any():
        errors.append(
            "Column 'symbolic_time_seconds' must be finite numeric and >= 0 "
            f"(bad rows: {bad_time[bad_time].index.tolist()[:10]})"
        )

    statuses = df["symbolic_status"].astype(str).str.strip()
    bad_status = ~statuses.isin(VALID_STATUS)
    if bad_status.any():
        invalid_values = sorted(statuses[bad_status].unique().tolist())
        errors.append(
            "Column 'symbolic_status' contains invalid values "
            f"{invalid_values}. Allowed: {sorted(VALID_STATUS)}"
        )

    key_cols = ["model_id", "trace_hash"]
    if "trace_id" in df.columns:
        key_cols.append("trace_id")
    duplicates = df.duplicated(subset=key_cols, keep=False)
    if duplicates.any():
        errors.append(
            f"Duplicate keys found on {key_cols}; example rows: "
            f"{duplicates[duplicates].index.tolist()[:10]}"
        )

    return errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate symbolic baseline CSV")
    p.add_argument("--input", required=True, type=str, help="Path to symbolic CSV")
    p.add_argument("--require-trace-id", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    errors = validate_symbolic_dataframe(df, require_trace_id=args.require_trace_id)

    if errors:
        print("Validation FAILED")
        for e in errors:
            print(f" - {e}")
        return 1

    print("Validation PASSED")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique model_id: {df['model_id'].nunique()}")
    print(f"Join keys used: {[c for c in ['model_id', 'trace_hash', 'trace_id'] if c in df.columns]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
