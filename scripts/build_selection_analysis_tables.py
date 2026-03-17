#!/usr/bin/env python3
"""Build analysis-ready selection tables from the canonical aggregate CSV.

The raw aggregate file remains untouched. This script produces derived tables:

- completed method rows enriched with parsed model metadata
- completed per-trace full feature table
- completed per-trace apriori feature table
- model metadata table
- summary JSON
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scripts.selection_semantics import (
        deduplicate_latest_rows,
        filter_consensus_optimal_rows,
    )
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from selection_semantics import deduplicate_latest_rows, filter_consensus_optimal_rows  # type: ignore


FIT_PREC_RE = re.compile(r"_f(?P<fitness>\d+\.\d+)_p(?P<precision>\d+\.\d+)_")
HM_RE = re.compile(r"(?P<label>HM|Heuristic)_d(?P<value>\d+\.\d+)")
IM_RE = re.compile(r"IMf?_n(?P<value>\d+\.\d+)")


ID_COLS = [
    "dataset_name",
    "model_id",
    "model_name",
    "model_path",
    "trace_id",
    "trace_hash",
]

TRACE_FEATURE_COLS = [
    "trace_length",
    "trace_unique_activities",
    "trace_repetition_ratio",
    "trace_unique_dfg_edges",
    "trace_self_loops",
    "trace_variant_frequency",
    "trace_impossible_activities",
]

MODEL_FEATURE_COLS = [
    "model_places",
    "model_transitions",
    "model_arcs",
    "model_silent_transitions",
    "model_visible_transitions",
    "model_place_in_degree_avg",
    "model_place_out_degree_avg",
    "model_place_in_degree_max",
    "model_place_out_degree_max",
    "model_transition_in_degree_avg",
    "model_transition_out_degree_avg",
    "model_transition_in_degree_max",
    "model_transition_out_degree_max",
]

POSTHOC_FEATURE_COLS = [
    "sp_nodes",
    "sp_edges",
    "optimal_cost",
    "deviation_cost",
]

MODEL_METADATA_COLS = [
    "model_fitness",
    "model_precision",
    "miner_family",
    "miner_parameter_name",
    "miner_parameter_value",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build analysis-ready selection tables")
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Canonical aggregate CSV produced by aggregate_astar_results.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for derived analysis tables",
    )
    return p


def parse_model_metadata(model_path: str, model_name: str) -> Dict[str, object]:
    basename = Path(model_path or model_name or "").name
    stem = Path(basename).stem

    fitness = np.nan
    precision = np.nan
    m = FIT_PREC_RE.search(stem)
    if m:
        fitness = float(m.group("fitness"))
        precision = float(m.group("precision"))

    miner_family = "unknown"
    param_name: Optional[str] = None
    param_value = np.nan

    hm = HM_RE.search(stem)
    im = IM_RE.search(stem)

    if "provided_baseline" in stem:
        miner_family = "provided_baseline"
    elif hm:
        miner_family = "heuristic_miner"
        param_name = "dependency_threshold"
        param_value = float(hm.group("value"))
    elif im:
        miner_family = "inductive_miner"
        param_name = "noise_threshold"
        param_value = float(im.group("value"))
    elif "Alpha" in stem:
        miner_family = "alpha"
    elif "ILP" in stem:
        miner_family = "ilp"
    elif "SM" in stem:
        miner_family = "split_miner"

    return {
        "model_fitness": fitness,
        "model_precision": precision,
        "miner_family": miner_family,
        "miner_parameter_name": param_name,
        "miner_parameter_value": param_value,
    }


def enrich_method_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["model_path", "model_name"]
    model_keys = df[key_cols].drop_duplicates().copy()
    parsed_rows = [
        parse_model_metadata(str(row["model_path"]), str(row["model_name"]))
        for _, row in model_keys.iterrows()
    ]
    parsed = pd.DataFrame(parsed_rows)
    model_keys = pd.concat([model_keys.reset_index(drop=True), parsed], axis=1)
    return df.merge(model_keys, on=key_cols, how="left")


def build_trace_table(enriched: pd.DataFrame, apriori_only: bool) -> pd.DataFrame:
    group_cols = ID_COLS

    base_feature_cols = (
        ["aggregate_run_name", "model_source", "trace_activities"]
        + TRACE_FEATURE_COLS
        + MODEL_FEATURE_COLS
        + MODEL_METADATA_COLS
    )
    if not apriori_only:
        base_feature_cols += POSTHOC_FEATURE_COLS

    existing_base_cols = [c for c in base_feature_cols if c in enriched.columns]

    base = enriched[group_cols + existing_base_cols].drop_duplicates(
        subset=group_cols,
        keep="last",
    )

    grouped = enriched.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_method_rows=("method", "size"),
        n_methods_tested=("method", "nunique"),
        n_methods_ok=("status", lambda s: int((s == "ok").sum())),
        n_methods_timeout=("status", lambda s: int((s == "timeout").sum())),
    ).reset_index()
    summary["any_timeout"] = summary["n_methods_timeout"] > 0

    valid_df = filter_consensus_optimal_rows(enriched)
    valid_df["time_seconds"] = pd.to_numeric(valid_df["time_seconds"], errors="coerce")
    valid_df = valid_df.dropna(subset=["time_seconds"])
    if valid_df.empty:
        return pd.DataFrame(columns=group_cols + existing_base_cols + [
            "n_method_rows",
            "n_methods_tested",
            "n_methods_ok",
            "n_methods_timeout",
            "any_timeout",
            "best_method",
            "best_time_seconds",
        ])
    else:
        valid_keys = valid_df[group_cols].drop_duplicates()
        best = (
            valid_df.sort_values(group_cols + ["time_seconds"], kind="stable")
            .drop_duplicates(subset=group_cols, keep="first")
            .loc[:, group_cols + ["method", "time_seconds"]]
            .rename(
                columns={"method": "best_method", "time_seconds": "best_time_seconds"}
            )
        )

    base = base.merge(valid_keys, on=group_cols, how="inner")
    summary = summary.merge(valid_keys, on=group_cols, how="inner")
    trace_table = base.merge(summary, on=group_cols, how="left")
    trace_table = trace_table.merge(best, on=group_cols, how="left")
    return trace_table.reset_index(drop=True)


def build_model_metadata_table(enriched: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["dataset_name", "model_id", "model_name", "model_path"] + MODEL_METADATA_COLS + MODEL_FEATURE_COLS if c in enriched.columns]
    model_df = (
        enriched[cols]
        .drop_duplicates(subset=["model_id"])
        .sort_values(["dataset_name", "model_name"], kind="stable")
        .reset_index(drop=True)
    )
    return model_df


def write_summary(
    out_path: Path,
    raw_df: pd.DataFrame,
    completed_df: pd.DataFrame,
    full_df: pd.DataFrame,
    apriori_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> None:
    summary = {
        "input_rows": int(len(raw_df)),
        "completed_rows": int(len(completed_df)),
        "datasets_completed": sorted(completed_df["dataset_name"].dropna().unique().tolist()),
        "n_datasets_completed": int(completed_df["dataset_name"].nunique()),
        "n_models_completed": int(model_df["model_id"].nunique()),
        "n_trace_instances_completed": int(len(full_df)),
        "n_apriori_rows": int(len(apriori_df)),
        "n_best_method_missing": int(full_df["best_method"].isna().sum()),
        "completed_timeout_rows": int((completed_df["status"] == "timeout").sum()),
    }
    out_path.write_text(json.dumps(summary, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    completed_df = raw_df[raw_df["aggregate_parent_complete"] == 1].copy()
    completed_df = deduplicate_latest_rows(
        completed_df,
        ["dataset_name", "model_id", "model_name", "model_path", "trace_id", "trace_hash", "method"],
    )
    enriched_df = enrich_method_rows(completed_df)

    full_df = build_trace_table(enriched_df, apriori_only=False)
    apriori_df = build_trace_table(enriched_df, apriori_only=True)
    model_df = build_model_metadata_table(enriched_df)

    method_rows_path = args.out_dir / "selection_method_rows_completed_enriched.csv"
    full_path = args.out_dir / "selection_trace_table_full.csv"
    apriori_path = args.out_dir / "selection_trace_table_apriori.csv"
    model_path = args.out_dir / "model_metadata.csv"
    summary_path = args.out_dir / "selection_analysis_summary.json"

    enriched_df.to_csv(method_rows_path, index=False)
    full_df.to_csv(full_path, index=False)
    apriori_df.to_csv(apriori_path, index=False)
    model_df.to_csv(model_path, index=False)
    write_summary(summary_path, raw_df, completed_df, full_df, apriori_df, model_df)

    print(f"Wrote: {method_rows_path}")
    print(f"Wrote: {full_path}")
    print(f"Wrote: {apriori_path}")
    print(f"Wrote: {model_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
