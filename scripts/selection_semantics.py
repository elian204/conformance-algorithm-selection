#!/usr/bin/env python3
"""Shared selection semantics for rerun deduplication and valid-method filtering."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


RUN_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def extract_timestamp_rank(series: pd.Series) -> pd.Series:
    """Extract comparable integer timestamps from run identifiers."""

    extracted = (
        series.fillna("")
        .astype(str)
        .str.extract(RUN_TIMESTAMP_RE, expand=False)
        .str.replace("_", "", regex=False)
    )
    return pd.to_numeric(extracted, errors="coerce").fillna(-1).astype(np.int64)


def deduplicate_latest_rows(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
) -> pd.DataFrame:
    """Keep the latest rerun for each logical row deterministically."""

    missing = [column for column in subset_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing deduplication columns: {missing}")

    work = df.copy()
    sort_columns = list(subset_cols)

    if "aggregate_parent_complete" in work.columns:
        work["_aggregate_parent_complete_rank"] = pd.to_numeric(
            work["aggregate_parent_complete"], errors="coerce"
        ).fillna(0)
        sort_columns.append("_aggregate_parent_complete_rank")

    if "aggregate_source_kind" in work.columns:
        source_kind_rank = {"partial_shard": 0, "merged": 1}
        work["_aggregate_source_kind_rank"] = (
            work["aggregate_source_kind"].map(source_kind_rank).fillna(-1)
        )
        sort_columns.append("_aggregate_source_kind_rank")

    if "experiment_id" in work.columns:
        work["_experiment_timestamp_rank"] = extract_timestamp_rank(work["experiment_id"])
    else:
        work["_experiment_timestamp_rank"] = -1

    if "aggregate_run_name" in work.columns:
        work["_aggregate_run_timestamp_rank"] = extract_timestamp_rank(
            work["aggregate_run_name"]
        )
    else:
        work["_aggregate_run_timestamp_rank"] = -1

    if "aggregate_source_file" in work.columns:
        work["_aggregate_source_timestamp_rank"] = extract_timestamp_rank(
            work["aggregate_source_file"]
        )
    else:
        work["_aggregate_source_timestamp_rank"] = -1

    work["_recency_rank"] = work["_experiment_timestamp_rank"]
    missing_experiment_rank = work["_recency_rank"] < 0
    work.loc[missing_experiment_rank, "_recency_rank"] = work.loc[
        missing_experiment_rank, "_aggregate_run_timestamp_rank"
    ]
    missing_run_rank = work["_recency_rank"] < 0
    work.loc[missing_run_rank, "_recency_rank"] = work.loc[
        missing_run_rank, "_aggregate_source_timestamp_rank"
    ]

    sort_columns.extend(
        [
            "_recency_rank",
            "_experiment_timestamp_rank",
            "_aggregate_run_timestamp_rank",
            "_aggregate_source_timestamp_rank",
        ]
    )
    sort_columns.extend(
        column
        for column in (
            "experiment_id",
            "aggregate_parent_run_id",
            "aggregate_source_file",
            "aggregate_run_name",
        )
        if column in work.columns
    )
    work = work.sort_values(sort_columns, kind="stable")
    work = work.drop_duplicates(subset=list(subset_cols), keep="last").copy()
    drop_columns = [
        column
        for column in (
            "_aggregate_parent_complete_rank",
            "_aggregate_source_kind_rank",
            "_experiment_timestamp_rank",
            "_aggregate_run_timestamp_rank",
            "_aggregate_source_timestamp_rank",
            "_recency_rank",
        )
        if column in work.columns
    ]
    return work.drop(columns=drop_columns)


def filter_consensus_optimal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only methods that match the trace consensus deviation cost."""

    required = {"status", "cost", "optimal_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns for consensus-optimal filtering: "
            f"{sorted(missing)}"
        )

    work = df.copy()
    work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
    work["optimal_cost"] = pd.to_numeric(work["optimal_cost"], errors="coerce")
    deviation_cost = work["cost"].round()
    consensus_cost = work["optimal_cost"].round()
    valid_mask = (
        (work["status"] == "ok")
        & work["cost"].notna()
        & work["optimal_cost"].notna()
        & np.isfinite(work["cost"])
        & np.isfinite(work["optimal_cost"])
        & deviation_cost.eq(consensus_cost)
    )
    return work[valid_mask].copy()


def safe_row_idxmin(frame: pd.DataFrame) -> pd.Series:
    """Return row-wise idxmin while preserving NA for all-null rows."""

    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    if frame.empty:
        return result

    valid_mask = frame.notna().any(axis=1)
    if valid_mask.any():
        result.loc[valid_mask] = frame.loc[valid_mask].idxmin(axis=1)
    return result
