#!/usr/bin/env python3
"""Compute Stage A selector baselines from the canonical aggregate CSV.

Outputs include:
- wide per-instance metric matrices for `expansions` and `time_seconds`
- VBS/SBS summaries
- pairwise dominance tables and dominance matrices
- per-instance best bidirectional vs. best forward ratio targets
- optional method-specific ratio targets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scripts.selection_semantics import (
        deduplicate_latest_rows,
        filter_consensus_optimal_rows,
        safe_row_idxmin,
    )
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from selection_semantics import (  # type: ignore
        deduplicate_latest_rows,
        filter_consensus_optimal_rows,
        safe_row_idxmin,
    )


METRICS = ("expansions", "time_seconds")
INSTANCE_KEY_BASE = ("model_id", "trace_hash")
OPTIONAL_INSTANCE_KEYS = ("dataset_name", "trace_id")
OUTPUT_MATRIX_TEMPLATE = "stage_a_metric_matrix_{metric}.csv"
OUTPUT_METHOD_TOTALS = "stage_a_method_totals.csv"
OUTPUT_PAIRWISE_TEMPLATE = "stage_a_pairwise_dominance_{metric}.csv"
OUTPUT_PAIRWISE_MATRIX_TEMPLATE = "stage_a_pairwise_dominance_matrix_{metric}.csv"
OUTPUT_TARGETS = "stage_a_ratio_targets.csv"
OUTPUT_SUMMARY = "stage_a_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Stage A VBS/SBS baselines and dominance matrices"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Canonical aggregate CSV produced by aggregate_astar_results.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for Stage A artifacts",
    )
    parser.add_argument(
        "--include-partial-parents",
        action="store_true",
        help="Include rows from incomplete parent runs when aggregate_parent_complete exists",
    )
    parser.add_argument(
        "--bidirectional-algorithms",
        nargs="+",
        default=["bidir_mm", "bidir_std", "dibbs"],
        help="Algorithms to treat as bidirectional for the family-level ratio target",
    )
    parser.add_argument(
        "--ratio-numerator-method",
        type=str,
        default=None,
        help="Optional method for a method-specific expansion/time ratio numerator",
    )
    parser.add_argument(
        "--ratio-denominator-method",
        type=str,
        default=None,
        help="Optional method for a method-specific expansion/time ratio denominator",
    )
    return parser


def infer_instance_keys(df: pd.DataFrame) -> List[str]:
    """Infer the per-instance grouping key from canonical columns."""

    missing = [column for column in INSTANCE_KEY_BASE if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required instance key columns: {missing}")

    keys = [column for column in OPTIONAL_INSTANCE_KEYS if column in df.columns]
    keys.extend(INSTANCE_KEY_BASE)
    return keys


def load_input(path: Path, include_partial_parents: bool) -> pd.DataFrame:
    """Load and optionally restrict the canonical aggregate CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if not include_partial_parents and "aggregate_parent_complete" in df.columns:
        complete_flag = pd.to_numeric(df["aggregate_parent_complete"], errors="coerce").fillna(0)
        df = df[complete_flag == 1].copy()
    return df


def deduplicate_latest_method_rows(
    df: pd.DataFrame,
    instance_keys: Sequence[str],
) -> pd.DataFrame:
    """Keep the latest rerun for each instance-method deterministically."""

    if "method" not in df.columns:
        raise ValueError("Input CSV is missing required column: method")
    return deduplicate_latest_rows(df, list(instance_keys) + ["method"])


def prepare_metric_rows(
    df: pd.DataFrame,
    instance_keys: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    """Keep one valid numeric row per instance-method pair for the metric."""

    required = set(instance_keys) | {"method", metric, "cost", "optimal_cost", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns for {metric}: {sorted(missing)}")

    work = filter_consensus_optimal_rows(df)
    work = work.loc[:, list(instance_keys) + ["method", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work[work[metric].notna()].copy()
    if work.empty:
        return work

    sort_cols = list(instance_keys) + ["method"]
    work = work.sort_values(sort_cols, kind="stable")
    work = work.drop_duplicates(subset=list(instance_keys) + ["method"], keep="last")
    return work


def build_metric_matrix(
    df: pd.DataFrame,
    instance_keys: Sequence[str],
    metric: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create a per-instance wide matrix for a single metric."""

    metric_rows = prepare_metric_rows(df, instance_keys, metric)
    if metric_rows.empty:
        return pd.DataFrame(columns=list(instance_keys)), []

    matrix = (
        metric_rows.pivot(index=list(instance_keys), columns="method", values=metric)
        .reset_index()
    )
    method_columns = sorted(
        [column for column in matrix.columns if column not in set(instance_keys)]
    )
    matrix = matrix.loc[:, list(instance_keys) + method_columns].copy()

    value_frame = matrix[method_columns]
    best_column = f"virtual_best_{metric}"
    best_method_column = f"virtual_best_{metric}_method"
    available_column = f"n_methods_available_{metric}"

    matrix[best_column] = value_frame.min(axis=1, skipna=True)
    matrix[best_method_column] = safe_row_idxmin(value_frame)
    matrix[available_column] = value_frame.notna().sum(axis=1)
    return matrix, method_columns


def summarize_metric_matrix(
    matrix: pd.DataFrame,
    method_columns: Sequence[str],
    metric: str,
) -> Dict[str, object]:
    """Summarize VBS/SBS statistics for a wide metric matrix."""

    if not method_columns:
        return {
            "metric": metric,
            "instances_with_any_valid_method": 0,
            "strict_sbs_comparable_instances": 0,
            "methods": [],
            "vbs_total": None,
            "vbs_mean": None,
            "strict_sbs_method": None,
            "strict_sbs_total": None,
        }

    best_column = f"virtual_best_{metric}"
    comparable_mask = matrix[list(method_columns)].notna().all(axis=1)
    comparable = matrix.loc[comparable_mask, list(method_columns)]
    strict_sbs_method: Optional[str] = None
    strict_sbs_total: Optional[float] = None
    if not comparable.empty:
        totals = comparable.sum(axis=0)
        strict_sbs_method = str(totals.idxmin())
        strict_sbs_total = float(totals.loc[strict_sbs_method])

    return {
        "metric": metric,
        "instances_with_any_valid_method": int(len(matrix)),
        "strict_sbs_comparable_instances": int(comparable_mask.sum()),
        "methods": list(method_columns),
        "vbs_total": float(matrix[best_column].sum()) if len(matrix) else None,
        "vbs_mean": float(matrix[best_column].mean()) if len(matrix) else None,
        "strict_sbs_method": strict_sbs_method,
        "strict_sbs_total": strict_sbs_total,
    }


def build_method_totals(
    matrix: pd.DataFrame,
    method_columns: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    """Summarize per-method coverage and aggregate metric values."""

    rows: List[Dict[str, object]] = []
    if not method_columns:
        return pd.DataFrame(
            columns=[
                "metric",
                "method",
                "coverage_instances",
                "coverage_rate",
                "total_metric",
                "mean_metric",
                "median_metric",
                "strict_sbs_eligible",
            ]
        )

    strict_sbs_eligible = matrix[list(method_columns)].notna().all(axis=1)
    total_instances = len(matrix)
    for method in method_columns:
        series = matrix[method].dropna()
        rows.append(
            {
                "metric": metric,
                "method": method,
                "coverage_instances": int(series.shape[0]),
                "coverage_rate": float(series.shape[0] / total_instances) if total_instances else 0.0,
                "total_metric": float(series.sum()) if not series.empty else np.nan,
                "mean_metric": float(series.mean()) if not series.empty else np.nan,
                "median_metric": float(series.median()) if not series.empty else np.nan,
                "strict_sbs_eligible": bool(strict_sbs_eligible.sum() > 0 and series.shape[0] == int(strict_sbs_eligible.sum())),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "mean_metric", "method"], kind="stable")


def build_pairwise_dominance(
    matrix: pd.DataFrame,
    method_columns: Sequence[str],
    metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build ordered pairwise dominance statistics and a decisive-win matrix."""

    rows: List[Dict[str, object]] = []
    if not method_columns:
        empty = pd.DataFrame(
            columns=[
                "metric",
                "left_method",
                "right_method",
                "comparable_instances",
                "wins",
                "losses",
                "ties",
                "win_rate",
                "decisive_win_rate",
                "win_loss_ratio",
            ]
        )
        return empty, empty

    for left_method in method_columns:
        for right_method in method_columns:
            left = matrix[left_method]
            right = matrix[right_method]
            comparable_mask = left.notna() & right.notna()
            comparable_instances = int(comparable_mask.sum())
            if comparable_instances == 0:
                wins = losses = ties = 0
                win_rate = decisive_win_rate = win_loss_ratio = np.nan
            else:
                left_values = left[comparable_mask]
                right_values = right[comparable_mask]
                wins = int((left_values < right_values).sum())
                losses = int((left_values > right_values).sum())
                ties = int((left_values == right_values).sum())
                win_rate = float(wins / comparable_instances)
                decisive_total = wins + losses
                decisive_win_rate = (
                    float(wins / decisive_total) if decisive_total > 0 else np.nan
                )
                if losses == 0:
                    win_loss_ratio = np.inf if wins > 0 else np.nan
                else:
                    win_loss_ratio = float(wins / losses)

            rows.append(
                {
                    "metric": metric,
                    "left_method": left_method,
                    "right_method": right_method,
                    "comparable_instances": comparable_instances,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate": win_rate,
                    "decisive_win_rate": decisive_win_rate,
                    "win_loss_ratio": win_loss_ratio,
                }
            )

    pairwise = pd.DataFrame(rows).sort_values(
        ["metric", "left_method", "right_method"], kind="stable"
    )
    dominance_matrix = pairwise.pivot(
        index="left_method",
        columns="right_method",
        values="decisive_win_rate",
    ).reset_index()
    return pairwise, dominance_matrix


def build_algorithm_family_table(
    df: pd.DataFrame,
    instance_keys: Sequence[str],
    bidirectional_algorithms: Iterable[str],
) -> pd.DataFrame:
    """Create per-instance best metrics by algorithm family and derived ratios."""

    required = set(instance_keys) | {"algorithm", "method", "status", "cost", "optimal_cost"} | set(METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns for family ratios: {sorted(missing)}")

    work = filter_consensus_optimal_rows(df)
    work = work.loc[:, list(instance_keys) + ["algorithm", "method", *METRICS]].copy()
    for metric in METRICS:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["algorithm"])

    if work.empty:
        return pd.DataFrame(columns=list(instance_keys))

    family_frames: List[pd.DataFrame] = []
    best_method_frames: List[pd.DataFrame] = []
    for metric in METRICS:
        metric_work = work.dropna(subset=[metric]).copy()
        if metric_work.empty:
            continue

        best_by_algorithm = (
            metric_work.groupby(list(instance_keys) + ["algorithm"], dropna=False)[metric]
            .min()
            .reset_index()
        )
        algorithm_wide = (
            best_by_algorithm.pivot(index=list(instance_keys), columns="algorithm", values=metric)
            .reset_index()
        )
        rename_map = {
            column: f"{column}_{metric}"
            for column in algorithm_wide.columns
            if column not in set(instance_keys)
        }
        algorithm_wide = algorithm_wide.rename(columns=rename_map)
        family_frames.append(algorithm_wide)

        best_method_by_algorithm = (
            metric_work.sort_values(metric, kind="stable")
            .drop_duplicates(subset=list(instance_keys) + ["algorithm"], keep="first")
            .loc[:, list(instance_keys) + ["algorithm", "method"]]
        )
        method_wide = (
            best_method_by_algorithm.pivot(
                index=list(instance_keys),
                columns="algorithm",
                values="method",
            )
            .reset_index()
        )
        method_rename_map = {
            column: f"{column}_{metric}_method"
            for column in method_wide.columns
            if column not in set(instance_keys)
        }
        method_wide = method_wide.rename(columns=method_rename_map)
        best_method_frames.append(method_wide)

    family_table: Optional[pd.DataFrame] = None
    for frame in family_frames + best_method_frames:
        family_table = frame if family_table is None else family_table.merge(
            frame, on=list(instance_keys), how="outer"
        )

    if family_table is None:
        return pd.DataFrame(columns=list(instance_keys))

    bidirectional_metric_columns = [
        f"{algorithm}_expansions"
        for algorithm in bidirectional_algorithms
        if f"{algorithm}_expansions" in family_table.columns
    ]
    if bidirectional_metric_columns:
        bidirectional_expansion_frame = family_table[bidirectional_metric_columns]
        family_table["best_bidirectional_expansions"] = bidirectional_expansion_frame.min(
            axis=1, skipna=True
        )
        family_table["best_bidirectional_expansions_algorithm"] = safe_row_idxmin(
            bidirectional_expansion_frame
        ).str.replace("_expansions", "", regex=False)
    else:
        family_table["best_bidirectional_expansions"] = np.nan
        family_table["best_bidirectional_expansions_algorithm"] = pd.Series(
            pd.NA, index=family_table.index, dtype="object"
        )

    bidirectional_time_columns = [
        f"{algorithm}_time_seconds"
        for algorithm in bidirectional_algorithms
        if f"{algorithm}_time_seconds" in family_table.columns
    ]
    if bidirectional_time_columns:
        bidirectional_time_frame = family_table[bidirectional_time_columns]
        family_table["best_bidirectional_time_seconds"] = bidirectional_time_frame.min(
            axis=1, skipna=True
        )
        family_table["best_bidirectional_time_seconds_algorithm"] = safe_row_idxmin(
            bidirectional_time_frame
        ).str.replace("_time_seconds", "", regex=False)
    else:
        family_table["best_bidirectional_time_seconds"] = np.nan
        family_table["best_bidirectional_time_seconds_algorithm"] = pd.Series(
            pd.NA, index=family_table.index, dtype="object"
        )

    family_table["best_bidirectional_vs_forward_expansion_ratio"] = safe_ratio_series(
        family_table.get("best_bidirectional_expansions"),
        family_table.get("forward_expansions"),
    )
    family_table["best_bidirectional_vs_forward_time_ratio"] = safe_ratio_series(
        family_table.get("best_bidirectional_time_seconds"),
        family_table.get("forward_time_seconds"),
    )
    family_table["dibbs_vs_forward_expansion_ratio"] = safe_ratio_series(
        family_table.get("dibbs_expansions"),
        family_table.get("forward_expansions"),
    )
    family_table["dibbs_vs_forward_time_ratio"] = safe_ratio_series(
        family_table.get("dibbs_time_seconds"),
        family_table.get("forward_time_seconds"),
    )
    return family_table.sort_values(list(instance_keys), kind="stable").reset_index(drop=True)


def safe_ratio_series(
    numerator: Optional[pd.Series],
    denominator: Optional[pd.Series],
) -> pd.Series:
    """Compute a ratio series while preserving zero-denominator edge cases."""

    if numerator is None or denominator is None:
        return pd.Series(dtype=float)

    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=num.index, dtype=float)

    finite_mask = num.notna() & den.notna()
    zero_zero_mask = finite_mask & (num == 0) & (den == 0)
    positive_over_zero_mask = finite_mask & den.eq(0) & num.ne(0)
    normal_mask = finite_mask & den.ne(0)

    result.loc[zero_zero_mask] = 1.0
    result.loc[positive_over_zero_mask] = np.inf
    result.loc[normal_mask] = num.loc[normal_mask] / den.loc[normal_mask]
    return result


def add_method_specific_ratios(
    target_table: pd.DataFrame,
    instance_keys: Sequence[str],
    metric_matrices: Dict[str, pd.DataFrame],
    numerator_method: Optional[str],
    denominator_method: Optional[str],
) -> pd.DataFrame:
    """Optionally add direct method-to-method ratio targets."""

    if not numerator_method or not denominator_method:
        return target_table

    out = target_table.copy()
    for metric, matrix in metric_matrices.items():
        if numerator_method not in matrix.columns or denominator_method not in matrix.columns:
            out[f"{numerator_method}_vs_{denominator_method}_{metric}_ratio"] = np.nan
            continue

        direct = matrix.loc[
            :,
            list(instance_keys) + [numerator_method, denominator_method],
        ].copy()
        direct[f"{numerator_method}_vs_{denominator_method}_{metric}_ratio"] = safe_ratio_series(
            direct[numerator_method],
            direct[denominator_method],
        )
        direct = direct.drop(columns=[numerator_method, denominator_method])
        out = out.merge(direct, on=list(instance_keys), how="left")
    return out


def build_summary(
    df: pd.DataFrame,
    instance_keys: Sequence[str],
    method_totals: pd.DataFrame,
    metric_summaries: Dict[str, Dict[str, object]],
    target_table: pd.DataFrame,
    bidirectional_algorithms: Sequence[str],
    numerator_method: Optional[str],
    denominator_method: Optional[str],
) -> Dict[str, object]:
    """Assemble a compact JSON summary of Stage A artifacts."""

    summary: Dict[str, object] = {
        "rows_after_filters": int(len(df)),
        "instance_keys": list(instance_keys),
        "methods": sorted(df["method"].dropna().astype(str).unique().tolist()) if "method" in df.columns else [],
        "algorithms": sorted(df["algorithm"].dropna().astype(str).unique().tolist()) if "algorithm" in df.columns else [],
        "bidirectional_algorithms": list(bidirectional_algorithms),
        "metric_summaries": metric_summaries,
        "target_rows": int(len(target_table)),
    }

    if not method_totals.empty:
        summary["method_coverage"] = method_totals.to_dict(orient="records")

    ratio_columns = [
        column for column in target_table.columns if column.endswith("_ratio")
    ]
    ratio_summary = []
    for column in ratio_columns:
        series = pd.to_numeric(target_table[column], errors="coerce")
        valid = series[np.isfinite(series)]
        ratio_summary.append(
            {
                "column": column,
                "non_null": int(series.notna().sum()),
                "finite_values": int(valid.shape[0]),
                "mean": float(valid.mean()) if not valid.empty else None,
                "median": float(valid.median()) if not valid.empty else None,
            }
        )
    summary["ratio_summaries"] = ratio_summary

    if numerator_method and denominator_method:
        summary["method_specific_ratio"] = {
            "numerator_method": numerator_method,
            "denominator_method": denominator_method,
        }

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_input(args.input_csv, include_partial_parents=args.include_partial_parents)
    rows_after_filters = int(len(raw_df))
    instance_keys = infer_instance_keys(raw_df)
    df = deduplicate_latest_method_rows(raw_df, instance_keys)

    metric_matrices: Dict[str, pd.DataFrame] = {}
    metric_summaries: Dict[str, Dict[str, object]] = {}
    method_total_frames: List[pd.DataFrame] = []
    for metric in METRICS:
        matrix, method_columns = build_metric_matrix(df, instance_keys, metric)
        metric_matrices[metric] = matrix
        metric_summaries[metric] = summarize_metric_matrix(matrix, method_columns, metric)
        method_total_frames.append(build_method_totals(matrix, method_columns, metric))

        pairwise, dominance_matrix = build_pairwise_dominance(matrix, method_columns, metric)
        matrix_path = args.out_dir / OUTPUT_MATRIX_TEMPLATE.format(metric=metric)
        pairwise_path = args.out_dir / OUTPUT_PAIRWISE_TEMPLATE.format(metric=metric)
        dominance_path = args.out_dir / OUTPUT_PAIRWISE_MATRIX_TEMPLATE.format(metric=metric)

        matrix.to_csv(matrix_path, index=False)
        pairwise.to_csv(pairwise_path, index=False)
        dominance_matrix.to_csv(dominance_path, index=False)

    method_totals = pd.concat(method_total_frames, ignore_index=True) if method_total_frames else pd.DataFrame()
    method_totals.to_csv(args.out_dir / OUTPUT_METHOD_TOTALS, index=False)

    target_table = build_algorithm_family_table(
        df,
        instance_keys=instance_keys,
        bidirectional_algorithms=args.bidirectional_algorithms,
    )
    target_table = add_method_specific_ratios(
        target_table,
        instance_keys=instance_keys,
        metric_matrices=metric_matrices,
        numerator_method=args.ratio_numerator_method,
        denominator_method=args.ratio_denominator_method,
    )
    target_table.to_csv(args.out_dir / OUTPUT_TARGETS, index=False)

    summary = build_summary(
        df=df,
        instance_keys=instance_keys,
        method_totals=method_totals,
        metric_summaries=metric_summaries,
        target_table=target_table,
        bidirectional_algorithms=args.bidirectional_algorithms,
        numerator_method=args.ratio_numerator_method,
        denominator_method=args.ratio_denominator_method,
    )
    summary["input_csv"] = str(args.input_csv.resolve())
    summary["include_partial_parents"] = bool(args.include_partial_parents)
    summary["rows_after_filters"] = rows_after_filters
    summary["rows_after_latest_rerun_dedup"] = int(len(df))

    summary_path = args.out_dir / OUTPUT_SUMMARY
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_dir / OUTPUT_METHOD_TOTALS}")
    print(f"Wrote: {args.out_dir / OUTPUT_TARGETS}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
