"""Tests for the Oracle Stage C pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_stage_c_oracle import main as oracle_main
from scripts.run_stage_c_oracle import prepare_merged_dataset


def _build_fast_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(15):
        rows.append(
            {
                "dataset_name": f"family_{idx}",
                "model_id": f"model_{idx}",
                "trace_id": f"trace_{idx}",
                "trace_hash": f"hash_{idx}",
                "dibbs_vs_forward_expansion_ratio": 0.8 + (idx * 0.1),
                "dibbs_zero_vs_forward_zero_expansions_ratio": 0.9 + (idx * 0.05),
                "uses_potentially_inconsistent_dibbs_target": 1 if idx == 0 else 0,
                "xor_splits": idx % 3,
                "xor_joins": (idx + 1) % 4,
                "and_splits": (idx + 2) % 5,
                "and_joins": (idx + 3) % 3,
                "tau_ratio": 0.05 * idx,
                "trace_length": 5 + idx,
                "distinct_activities": 2 + (idx % 4),
                "trace_entropy": 0.5 + (idx * 0.1),
                "trace_max_repetitions": 1 + (idx % 3),
                "log_only_prop": 0.01 * idx,
                "model_only_prop": 0.02 * idx,
                "alphabet_overlap": 0.9 - (idx * 0.03),
                "branching_imbalance": 1.0 + (idx * 0.05),
                "sp_places": 20 + idx,
                "sp_transitions": 25 + idx,
                "sp_branching_factor": (25 + idx) / (20 + idx),
            }
        )
    return rows


def _build_oracle_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(15):
        rows.append(
            {
                "dataset_name": f"family_{idx}",
                "model_id": f"model_{idx}",
                "trace_id": f"trace_{idx}",
                "trace_hash": f"hash_{idx}",
                "h_f": 1.0 + (idx * 0.2),
                "normalized_h_f": (1.0 + (idx * 0.2)) / (5 + idx),
            }
        )
    return rows


def test_prepare_merged_dataset_filters_inconsistent_dibbs_rows(tmp_path: Path):
    fast_csv = tmp_path / "fast.csv"
    oracle_csv = tmp_path / "oracle.csv"

    pd.DataFrame(_build_fast_rows()).to_csv(fast_csv, index=False)
    pd.DataFrame(_build_oracle_rows()).to_csv(oracle_csv, index=False)

    merged, feature_cols, _ = prepare_merged_dataset(
        features_csv=fast_csv,
        oracle_csv=oracle_csv,
        target_col="dibbs_vs_forward_expansion_ratio",
        group_col="dataset_name",
    )

    assert len(merged) == 14
    assert "h_f" in feature_cols
    assert "normalized_h_f" in feature_cols
    assert merged["uses_potentially_inconsistent_dibbs_target"].sum() == 0


def test_prepare_merged_dataset_drops_rows_with_missing_oracle_features(tmp_path: Path):
    fast_csv = tmp_path / "fast.csv"
    oracle_csv = tmp_path / "oracle.csv"

    fast_rows = _build_fast_rows()
    oracle_rows = _build_oracle_rows()
    oracle_rows[1]["h_f"] = None
    oracle_rows[1]["normalized_h_f"] = None

    pd.DataFrame(fast_rows).to_csv(fast_csv, index=False)
    pd.DataFrame(oracle_rows).to_csv(oracle_csv, index=False)

    merged, _, prep_stats = prepare_merged_dataset(
        features_csv=fast_csv,
        oracle_csv=oracle_csv,
        target_col="dibbs_vs_forward_expansion_ratio",
        group_col="dataset_name",
    )

    assert len(merged) == 13
    assert prep_stats["rows_missing_any_oracle"] == 1
    assert merged["h_f"].notna().all()
    assert merged["normalized_h_f"].notna().all()


def test_prepare_merged_dataset_keeps_zero_vs_zero_rows_without_dibbs_filter(
    tmp_path: Path,
):
    fast_csv = tmp_path / "fast.csv"
    oracle_csv = tmp_path / "oracle.csv"

    pd.DataFrame(_build_fast_rows()).to_csv(fast_csv, index=False)
    pd.DataFrame(_build_oracle_rows()).to_csv(oracle_csv, index=False)

    merged, _, prep_stats = prepare_merged_dataset(
        features_csv=fast_csv,
        oracle_csv=oracle_csv,
        target_col="dibbs_zero_vs_forward_zero_expansions_ratio",
        group_col="dataset_name",
    )

    assert len(merged) == 15
    assert prep_stats["consistency_policy"] == "none"


def test_stage_c_oracle_main_writes_outputs(tmp_path: Path):
    fast_csv = tmp_path / "fast.csv"
    oracle_csv = tmp_path / "oracle.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(_build_fast_rows()).to_csv(fast_csv, index=False)
    pd.DataFrame(_build_oracle_rows()).to_csv(oracle_csv, index=False)

    rc = oracle_main(
        [
            "--features-csv",
            str(fast_csv),
            "--oracle-csv",
            str(oracle_csv),
            "--out-dir",
            str(out_dir),
            "--target-transform",
            "log",
            "--skip-shap",
        ]
    )

    assert rc == 0
    assert (out_dir / "oracle_cv_metrics.csv").exists()
    assert (out_dir / "oracle_cv_permutation_importance.csv").exists()
    assert (out_dir / "oracle_summary.json").exists()

    summary = json.loads((out_dir / "oracle_summary.json").read_text())
    assert summary["group_col"] == "dataset_name"
    assert summary["rows_used"] == 14
    assert summary["consistency_policy"] == "dibbs_only"
    assert summary["target_transform"] == "log"
    assert summary["shap_available"] is False
    assert summary["oracle_feature_set_name"] == "reduced_oracle_initial_me"
