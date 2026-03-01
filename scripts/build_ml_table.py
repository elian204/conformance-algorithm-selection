#!/usr/bin/env python3
"""Build canonical ML tables by merging A* and symbolic runtime outputs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scripts.validate_symbolic_results import validate_symbolic_dataframe
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from validate_symbolic_results import validate_symbolic_dataframe  # type: ignore


def _collect_astar_paths(patterns: List[str]) -> List[Path]:
    files = set()
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if p.is_file():
                files.add(p.resolve())
    return sorted(files)


def _load_astar_results(patterns: List[str]) -> Tuple[pd.DataFrame, List[Path]]:
    paths = _collect_astar_paths(patterns)
    if not paths:
        raise FileNotFoundError("No A* results files matched the provided glob patterns")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["astar_source_file"] = str(p)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    required = {"model_id", "trace_hash", "method", "time_seconds", "status"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"A* results missing required columns: {sorted(missing)}")
    return merged, paths


def _choose_join_columns(
    astar_df: pd.DataFrame,
    symbolic_df: pd.DataFrame,
    ignore_trace_id: bool,
) -> List[str]:
    cols = ["model_id", "trace_hash"]
    if not ignore_trace_id and "trace_id" in astar_df.columns and "trace_id" in symbolic_df.columns:
        cols.append("trace_id")
    return cols


def _aggregate_astar(astar_df: pd.DataFrame, join_cols: List[str]) -> pd.DataFrame:
    # Deterministic tie-break when multiple rows exist per join key:
    # keep the latest experiment_id (if present), otherwise first in stable order.
    sorted_df = astar_df.copy()
    sort_cols = []
    if "experiment_id" in sorted_df.columns:
        sort_cols.append("experiment_id")
    if "astar_source_file" in sorted_df.columns:
        sort_cols.append("astar_source_file")
    if sort_cols:
        sorted_df = sorted_df.sort_values(sort_cols)

    feature_cols = [
        c for c in [
            "experiment_id", "dataset_name", "log_path", "model_path", "model_name", "model_source",
            "model_id", "trace_id", "trace_hash", "trace_activities", "trace_length",
            "trace_unique_activities", "trace_repetition_ratio", "trace_unique_dfg_edges",
            "trace_self_loops", "trace_variant_frequency", "trace_impossible_activities",
            "sp_nodes", "sp_edges",
            "model_places", "model_transitions", "model_arcs", "model_silent_transitions",
            "model_visible_transitions", "model_place_in_degree_avg", "model_place_out_degree_avg",
            "model_place_in_degree_max", "model_place_out_degree_max", "model_transition_in_degree_avg",
            "model_transition_out_degree_avg", "model_transition_in_degree_max", "model_transition_out_degree_max",
        ] if c in sorted_df.columns
    ]

    base = sorted_df[feature_cols].drop_duplicates(subset=join_cols, keep="last")

    agg = (
        sorted_df.groupby(join_cols, dropna=False)
        .agg(
            n_methods_tested=("method", "nunique"),
            n_method_rows=("method", "size"),
            any_astar_ok=("status", lambda s: bool((s == "ok").any())),
        )
        .reset_index()
    )

    ok_df = sorted_df[sorted_df["status"] == "ok"].copy()
    if not ok_df.empty:
        idx = ok_df.groupby(join_cols, dropna=False)["time_seconds"].idxmin()
        best = ok_df.loc[idx, join_cols + ["method", "time_seconds"]].rename(
            columns={"method": "best_method", "time_seconds": "best_astar_time_seconds"}
        )
    else:
        best = pd.DataFrame(columns=join_cols + ["best_method", "best_astar_time_seconds"])

    out = base.merge(agg, on=join_cols, how="left")
    out = out.merge(best, on=join_cols, how="left")
    return out


def _build_merge_report(
    astar_df: pd.DataFrame,
    astar_agg_df: pd.DataFrame,
    symbolic_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    join_cols: List[str],
) -> Dict[str, object]:
    symbolic_matched = int(merged_df["symbolic_time_seconds"].notna().sum())
    astar_unmatched = int(len(merged_df) - symbolic_matched)

    astar_keys = astar_agg_df[join_cols].drop_duplicates()
    symbolic_keys = symbolic_df[join_cols].drop_duplicates()
    symbolic_unmatched = int(
        len(symbolic_keys.merge(astar_keys, on=join_cols, how="left", indicator=True).query("_merge == 'left_only'"))
    )

    per_model = []
    if "model_id" in merged_df.columns:
        grouped = merged_df.groupby("model_id", dropna=False)
        for model_id, g in grouped:
            matched = int(g["symbolic_time_seconds"].notna().sum())
            total = len(g)
            per_model.append({
                "model_id": model_id,
                "total_rows": total,
                "matched_rows": matched,
                "coverage": (matched / total) if total > 0 else 0.0,
            })

    return {
        "join_columns": join_cols,
        "astar_raw_rows": int(len(astar_df)),
        "astar_unique_rows": int(len(astar_agg_df)),
        "symbolic_rows": int(len(symbolic_df)),
        "merged_rows": int(len(merged_df)),
        "matched_symbolic_rows": symbolic_matched,
        "astar_rows_without_symbolic": astar_unmatched,
        "symbolic_rows_without_astar": symbolic_unmatched,
        "per_model_coverage": per_model,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build merged ML tables from A* and symbolic results")
    p.add_argument("--astar-results-glob", nargs="+", required=True,
                   help="One or more glob patterns for A* results.csv files")
    p.add_argument("--symbolic-csv", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--ignore-trace-id", action="store_true",
                   help="Always join on model_id + trace_hash")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    astar_df, astar_paths = _load_astar_results(args.astar_results_glob)

    symbolic_path = Path(args.symbolic_csv)
    if not symbolic_path.exists():
        raise FileNotFoundError(f"Symbolic CSV not found: {symbolic_path}")
    symbolic_df = pd.read_csv(symbolic_path)

    validation_errors = validate_symbolic_dataframe(symbolic_df)
    if validation_errors:
        raise ValueError("Symbolic CSV validation failed: " + " | ".join(validation_errors))

    join_cols = _choose_join_columns(astar_df, symbolic_df, ignore_trace_id=args.ignore_trace_id)

    astar_agg = _aggregate_astar(astar_df, join_cols)
    merged = astar_agg.merge(symbolic_df, on=join_cols, how="left", suffixes=("", "_symbolic"))

    merged["symbolic_vs_best_astar_ratio"] = np.where(
        (merged["best_astar_time_seconds"].notna()) & (merged["best_astar_time_seconds"] > 0) &
        merged["symbolic_time_seconds"].notna(),
        merged["symbolic_time_seconds"] / merged["best_astar_time_seconds"],
        np.nan,
    )
    merged["symbolic_faster_flag"] = np.where(
        merged["symbolic_time_seconds"].notna() & merged["best_astar_time_seconds"].notna(),
        merged["symbolic_time_seconds"] < merged["best_astar_time_seconds"],
        np.nan,
    )

    weighted = merged.copy()
    if "trace_variant_frequency" in weighted.columns:
        weighted["sample_weight"] = pd.to_numeric(weighted["trace_variant_frequency"], errors="coerce").fillna(1.0)
    else:
        weighted["sample_weight"] = 1.0

    report = _build_merge_report(
        astar_df=astar_df,
        astar_agg_df=astar_agg,
        symbolic_df=symbolic_df,
        merged_df=merged,
        join_cols=join_cols,
    )
    report["astar_sources"] = [str(p) for p in astar_paths]
    report["symbolic_source"] = str(symbolic_path.resolve())

    unweighted_path = out_dir / "ml_table_unweighted.csv"
    weighted_path = out_dir / "ml_table_weighted.csv"
    report_path = out_dir / "merge_report.json"

    merged.to_csv(unweighted_path, index=False)
    weighted.to_csv(weighted_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {unweighted_path}")
    print(f"Wrote: {weighted_path}")
    print(f"Wrote: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
