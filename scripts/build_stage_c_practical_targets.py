#!/usr/bin/env python3
"""Build practical heuristic-aware Stage C scenario tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


KEY_COLS = ["dataset_name", "model_id", "trace_id", "trace_hash"]
FAST_FEATURES = [
    "xor_splits",
    "xor_joins",
    "and_splits",
    "and_joins",
    "tau_ratio",
    "trace_length",
    "distinct_activities",
    "trace_entropy",
    "trace_max_repetitions",
    "log_only_prop",
    "model_only_prop",
    "alphabet_overlap",
    "branching_imbalance",
    "sp_places",
    "sp_transitions",
    "sp_branching_factor",
]
SCENARIOS: Dict[str, Dict[str, object]] = {
    "rq3a": {
        "baseline": "forward_me",
        "competitors": ["backward_me", "bidir_mm_me", "dibbs_me"],
        "caveated": {"dibbs_me", "bidir_mm_me"},
    },
    "rq3b": {
        "baseline": "forward_mmr",
        "competitors": ["backward_mmr", "bidir_mm_mmr", "dibbs_mmr"],
        "caveated": set(),
    },
    "rq3b_sens": {
        "baseline": "forward_mmr",
        "competitors": ["backward_mmr", "bidir_mm_mmr", "bidir_std_mmr", "dibbs_mmr"],
        "caveated": set(),
    },
    "rq3a_backward_only": {
        "baseline": "forward_me",
        "competitors": ["backward_me"],
        "caveated": set(),
    },
    "rq3a_oracle_filtered": {
        "baseline": "forward_me",
        "competitors": ["backward_me", "bidir_mm_me", "dibbs_me"],
        "caveated": {"dibbs_me", "bidir_mm_me"},
        "posthoc_filter_best_competitor_methods": {"dibbs_me", "bidir_mm_me"},
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Build Stage C practical scenario tables")
    parser.add_argument(
        "--metric-matrix-csv",
        type=Path,
        default=Path("tmp_smoke/stage_a_canonical_current/stage_a_metric_matrix_expansions.csv"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_full.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_smoke/stage_c_practical_targets"),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=sorted(SCENARIOS),
    )
    return parser


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Ensure a frame contains the expected columns."""

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def duplicate_key_count(df: pd.DataFrame) -> int:
    """Count duplicate 4-key rows."""

    return int(df.duplicated(KEY_COLS).sum())


def assert_no_duplicate_keys(df: pd.DataFrame, name: str) -> None:
    """Ensure a frame does not contain duplicate 4-key rows."""

    duplicates = duplicate_key_count(df)
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate {KEY_COLS} rows")


def _winner_breakdown(
    best_methods: pd.Series,
    use_nonforward: pd.Series,
    competitor_cols: List[str],
) -> Dict[str, int]:
    """Compute winner counts among strict non-forward wins."""

    counts = (
        best_methods[use_nonforward.astype(bool)]
        .value_counts(dropna=True)
        .reindex(competitor_cols, fill_value=0)
        .astype(int)
    )
    return {method: int(counts.loc[method]) for method in competitor_cols}


def build_scenario_target(
    metric_df: pd.DataFrame,
    scenario_name: str,
    baseline_col: str,
    competitor_cols: List[str],
    caveated_methods: set[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Construct a single practical scenario target frame."""

    cols = KEY_COLS + [baseline_col] + competitor_cols
    validate_columns(metric_df, cols)
    scenario_df = metric_df.loc[:, cols].copy()
    scenario_df = scenario_df.rename(columns={baseline_col: "forward_expansions"})
    scenario_df["forward_baseline_method"] = baseline_col
    baseline_present_mask = pd.to_numeric(
        scenario_df["forward_expansions"],
        errors="coerce",
    ).notna()
    scenario_df = scenario_df.loc[baseline_present_mask].copy()
    scenario_df["forward_expansions"] = pd.to_numeric(
        scenario_df["forward_expansions"],
        errors="coerce",
    )

    competitor_frame = scenario_df.loc[:, competitor_cols].apply(pd.to_numeric, errors="coerce")
    competitor_available = competitor_frame.notna().any(axis=1)
    scenario_df["competitor_available"] = competitor_available.astype(int)

    best_expansions = competitor_frame.min(axis=1, skipna=True)
    best_methods = pd.Series(pd.NA, index=scenario_df.index, dtype="object")
    if competitor_available.any():
        best_methods.loc[competitor_available] = competitor_frame.loc[
            competitor_available
        ].idxmin(axis=1)

    scenario_df["best_competitor_expansions"] = best_expansions.where(
        competitor_available,
        np.nan,
    )
    scenario_df["best_competitor_method"] = best_methods
    scenario_df["is_tie"] = (
        competitor_available
        & scenario_df["best_competitor_expansions"].eq(scenario_df["forward_expansions"])
    ).astype(int)
    scenario_df["use_nonforward"] = (
        competitor_available
        & scenario_df["best_competitor_expansions"].lt(scenario_df["forward_expansions"])
    ).astype(int)
    scenario_df["best_competitor_is_caveated"] = scenario_df[
        "best_competitor_method"
    ].isin(caveated_methods)
    scenario_df["oracle_expansions"] = np.where(
        scenario_df["competitor_available"].astype(bool),
        np.minimum(
            scenario_df["forward_expansions"],
            scenario_df["best_competitor_expansions"].fillna(np.inf),
        ),
        scenario_df["forward_expansions"],
    )
    scenario_df["oracle_method"] = np.where(
        scenario_df["use_nonforward"].astype(bool),
        scenario_df["best_competitor_method"],
        scenario_df["forward_baseline_method"],
    )

    assert_no_duplicate_keys(scenario_df, f"{scenario_name} target")

    eligible_rows = int(scenario_df["competitor_available"].sum())
    nonforward_wins = int(scenario_df["use_nonforward"].sum())
    ties = int(scenario_df["is_tie"].sum())
    forward_strict_wins = int(eligible_rows - nonforward_wins - ties)
    non_tie_rows = int(eligible_rows - ties)
    non_tie_nonforward_rate = (
        float(nonforward_wins / non_tie_rows) if non_tie_rows else float("nan")
    )

    summary = {
        "scenario_name": scenario_name,
        "forward_baseline_method": baseline_col,
        "competitor_pool": competitor_cols,
        "baseline_present_rows": int(len(scenario_df)),
        "eligible_rows": eligible_rows,
        "no_competitor_rows": int(len(scenario_df) - eligible_rows),
        "nonforward_wins": nonforward_wins,
        "nonforward_win_rate": float(nonforward_wins / eligible_rows)
        if eligible_rows
        else float("nan"),
        "ties": ties,
        "tie_rate": float(ties / eligible_rows) if eligible_rows else float("nan"),
        "forward_strict_wins": forward_strict_wins,
        "forward_strict_win_rate": float(forward_strict_wins / eligible_rows)
        if eligible_rows
        else float("nan"),
        "non_tie_rows": non_tie_rows,
        "non_tie_nonforward_rate": non_tie_nonforward_rate,
        "winner_breakdown": _winner_breakdown(
            scenario_df["best_competitor_method"],
            scenario_df["use_nonforward"],
            competitor_cols,
        ),
        "duplicate_key_counts": {
            "target_pre_join": 0,
        },
    }
    return scenario_df, summary


def join_features(
    scenario_df: pd.DataFrame,
    features_df: pd.DataFrame,
    scenario_name: str,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Join a scenario target frame to the fast-feature table."""

    assert_no_duplicate_keys(features_df, "fast feature table")
    join_check = scenario_df.merge(
        features_df,
        on=KEY_COLS,
        how="outer",
        indicator=True,
    )
    unmatched_target_rows = int((join_check["_merge"] == "left_only").sum())
    unmatched_feature_rows = int((join_check["_merge"] == "right_only").sum())

    joined = scenario_df.merge(features_df, on=KEY_COLS, how="left")
    assert_no_duplicate_keys(joined, f"{scenario_name} joined table")

    if unmatched_target_rows:
        raise ValueError(
            f"{scenario_name} has {unmatched_target_rows} target rows missing fast features"
        )

    return joined, {
        "joined_row_count": int(len(joined)),
        "unmatched_target_rows": unmatched_target_rows,
        "unmatched_feature_rows": unmatched_feature_rows,
        "duplicate_key_counts": {
            "feature_pre_join": 0,
            "joined_post_join": 0,
        },
    }


def main(argv: List[str] | None = None) -> int:
    """Build the practical scenario modeling tables."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_df = pd.read_csv(args.metric_matrix_csv)
    features_df = pd.read_csv(args.features_csv)
    validate_columns(metric_df, KEY_COLS)
    validate_columns(features_df, KEY_COLS + FAST_FEATURES)
    features_subset = features_df.loc[:, KEY_COLS + FAST_FEATURES].copy()

    scenario_index: Dict[str, str] = {}

    for scenario_name in args.scenarios:
        config = SCENARIOS[scenario_name]
        target_df, summary = build_scenario_target(
            metric_df=metric_df,
            scenario_name=scenario_name,
            baseline_col=str(config["baseline"]),
            competitor_cols=list(config["competitors"]),
            caveated_methods=set(config["caveated"]),
        )
        posthoc_methods = set(config.get("posthoc_filter_best_competitor_methods", set()))
        if posthoc_methods:
            filter_mask = ~target_df["best_competitor_method"].isin(posthoc_methods)
            target_df = target_df.loc[filter_mask].copy()
            eligible_rows = int(target_df["competitor_available"].sum())
            nonforward_wins = int(target_df["use_nonforward"].sum())
            ties = int(target_df["is_tie"].sum())
            forward_strict_wins = int(eligible_rows - nonforward_wins - ties)
            non_tie_rows = int(eligible_rows - ties)
            summary.update(
                {
                    "baseline_present_rows": int(len(target_df)),
                    "eligible_rows": eligible_rows,
                    "no_competitor_rows": int(len(target_df) - eligible_rows),
                    "nonforward_wins": nonforward_wins,
                    "nonforward_win_rate": float(nonforward_wins / eligible_rows)
                    if eligible_rows
                    else float("nan"),
                    "ties": ties,
                    "tie_rate": float(ties / eligible_rows) if eligible_rows else float("nan"),
                    "forward_strict_wins": forward_strict_wins,
                    "forward_strict_win_rate": float(forward_strict_wins / eligible_rows)
                    if eligible_rows
                    else float("nan"),
                    "non_tie_rows": non_tie_rows,
                    "non_tie_nonforward_rate": float(nonforward_wins / non_tie_rows)
                    if non_tie_rows
                    else float("nan"),
                    "winner_breakdown": _winner_breakdown(
                        target_df["best_competitor_method"],
                        target_df["use_nonforward"],
                        list(config["competitors"]),
                    ),
                    "posthoc_filter_best_competitor_methods": sorted(posthoc_methods),
                }
            )
            assert_no_duplicate_keys(target_df, f"{scenario_name} target post-filter")
        joined_df, join_stats = join_features(
            scenario_df=target_df,
            features_df=features_subset,
            scenario_name=scenario_name,
        )
        summary["key_join_integrity"] = {
            "joined_row_count": join_stats["joined_row_count"],
            "unmatched_target_rows": join_stats["unmatched_target_rows"],
            "unmatched_feature_rows": join_stats["unmatched_feature_rows"],
        }
        summary["duplicate_key_counts"].update(join_stats["duplicate_key_counts"])

        scenario_csv = args.out_dir / f"{scenario_name}_scenario.csv"
        scenario_json = args.out_dir / f"{scenario_name}_summary.json"
        joined_df.to_csv(scenario_csv, index=False)
        scenario_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        scenario_index[scenario_name] = str(scenario_csv.resolve())

        print(f"Wrote: {scenario_csv}")
        print(f"Wrote: {scenario_json}")

    (args.out_dir / "scenario_index.json").write_text(
        json.dumps(scenario_index, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {args.out_dir / 'scenario_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
