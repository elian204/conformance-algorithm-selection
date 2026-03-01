#!/usr/bin/env python3
"""Starter runtime baseline analysis for merged experiments."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


NUMERIC_EXCLUDE = {
    "symbolic_faster_flag",
}


def _collect_astar_df(patterns: Optional[List[str]]) -> Optional[pd.DataFrame]:
    if not patterns:
        return None
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        return None

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["astar_source_file"] = path
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute baseline runtime summaries")
    p.add_argument("--merged-table", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--astar-results-glob", nargs="+", default=None,
                   help="Optional glob patterns for raw A* results.csv")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    merged_path = Path(args.merged_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(merged_path)

    summary = {
        "rows": int(len(merged)),
        "matched_symbolic_rows": int(merged["symbolic_time_seconds"].notna().sum()) if "symbolic_time_seconds" in merged.columns else 0,
    }

    if "best_astar_time_seconds" in merged.columns:
        best = pd.to_numeric(merged["best_astar_time_seconds"], errors="coerce")
        summary["best_astar_time_seconds_mean"] = float(best.mean())
        summary["best_astar_time_seconds_median"] = float(best.median())

    if "symbolic_time_seconds" in merged.columns:
        sym = pd.to_numeric(merged["symbolic_time_seconds"], errors="coerce")
        summary["symbolic_time_seconds_mean"] = float(sym.mean())
        summary["symbolic_time_seconds_median"] = float(sym.median())

    if "symbolic_faster_flag" in merged.columns:
        valid = merged["symbolic_faster_flag"].dropna()
        summary["symbolic_faster_rate"] = float(valid.mean()) if len(valid) > 0 else None

    # Correlations with best A* runtime
    corr_rows = []
    if "best_astar_time_seconds" in merged.columns:
        target = pd.to_numeric(merged["best_astar_time_seconds"], errors="coerce")
        numeric_cols = [
            c for c in merged.columns
            if pd.api.types.is_numeric_dtype(merged[c]) and c not in NUMERIC_EXCLUDE and c != "best_astar_time_seconds"
        ]
        for col in numeric_cols:
            x = pd.to_numeric(merged[col], errors="coerce")
            tmp = pd.DataFrame({"x": x, "y": target}).dropna()
            if len(tmp) < 2:
                continue
            corr = float(tmp["x"].corr(tmp["y"]))
            corr_rows.append({"feature": col, "corr_with_best_astar_time": corr, "abs_corr": abs(corr)})

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False) if corr_rows else pd.DataFrame(
        columns=["feature", "corr_with_best_astar_time", "abs_corr"]
    )

    method_df = _collect_astar_df(args.astar_results_glob)
    if method_df is not None and {"method", "time_seconds", "status"}.issubset(set(method_df.columns)):
        ok = method_df[method_df["status"] == "ok"].copy()
        runtime_by_method = (
            ok.groupby("method")["time_seconds"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )
    else:
        runtime_by_method = pd.DataFrame(columns=["method", "count", "mean", "median", "std", "min", "max"])

    summary_path = out_dir / "runtime_baseline_summary.json"
    corr_path = out_dir / "runtime_feature_correlations.csv"
    method_path = out_dir / "runtime_by_method.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    corr_df.to_csv(corr_path, index=False)
    runtime_by_method.to_csv(method_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {corr_path}")
    print(f"Wrote: {method_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
