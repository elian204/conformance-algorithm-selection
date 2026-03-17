"""Tests for the practitioner Stage C baseline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_stage_c_practitioner import main as practitioner_main
from scripts.run_stage_c_practitioner import prepare_dataset


FEATURE_COLS = [
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


def test_prepare_dataset_uses_target_specific_consistency_policy():
    df = pd.DataFrame(
        [
            {
                "dataset_name": "fam_a",
                "dibbs_vs_forward_expansion_ratio": 0.8,
                "dibbs_zero_vs_forward_zero_expansions_ratio": 0.9,
                "target_dibbs_method": "dibbs_zero",
                "bidir_mm_expansions_method": "bidir_mm_me",
                "xor_splits": 1,
                "xor_joins": 1,
                "and_splits": 1,
                "and_joins": 1,
                "tau_ratio": 0.1,
                "trace_length": 5,
                "distinct_activities": 3,
                "trace_entropy": 1.0,
                "trace_max_repetitions": 2,
                "log_only_prop": 0.0,
                "model_only_prop": 0.1,
                "alphabet_overlap": 0.8,
                "branching_imbalance": 1.0,
                "sp_places": 10,
                "sp_transitions": 12,
                "sp_branching_factor": 1.2,
            },
            {
                "dataset_name": "fam_b",
                "dibbs_vs_forward_expansion_ratio": 1.2,
                "dibbs_zero_vs_forward_zero_expansions_ratio": 1.1,
                "target_dibbs_method": "dibbs_me",
                "bidir_mm_expansions_method": "bidir_mm_zero",
                "xor_splits": 2,
                "xor_joins": 2,
                "and_splits": 2,
                "and_joins": 2,
                "tau_ratio": 0.2,
                "trace_length": 6,
                "distinct_activities": 4,
                "trace_entropy": 1.1,
                "trace_max_repetitions": 3,
                "log_only_prop": 0.1,
                "model_only_prop": 0.2,
                "alphabet_overlap": 0.6,
                "branching_imbalance": 1.2,
                "sp_places": 11,
                "sp_transitions": 13,
                "sp_branching_factor": 1.18,
            },
        ]
    )

    dibbs_df = prepare_dataset(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col="dibbs_vs_forward_expansion_ratio",
        group_col="dataset_name",
        include_inconsistent=False,
        consistency_policy="auto",
    )
    assert len(dibbs_df) == 1
    assert dibbs_df.iloc[0]["dataset_name"] == "fam_a"
    assert dibbs_df.attrs["consistency_policy"] == "dibbs_only"

    zero_df = prepare_dataset(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col="dibbs_zero_vs_forward_zero_expansions_ratio",
        group_col="dataset_name",
        include_inconsistent=False,
        consistency_policy="auto",
    )
    assert len(zero_df) == 2
    assert zero_df.attrs["consistency_policy"] == "none"


def test_practitioner_main_writes_outputs(tmp_path: Path):
    features_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "out"

    rows = []
    for idx in range(15):
        rows.append(
            {
                "dataset_name": f"family_{idx}",
                "dibbs_zero_vs_forward_zero_expansions_ratio": 0.8 + (idx * 0.1),
                "target_dibbs_method": "dibbs_zero",
                "bidir_mm_expansions_method": "bidir_mm_zero",
                "uses_potentially_inconsistent_bidirectional_target": 0,
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

    pd.DataFrame(rows).to_csv(features_csv, index=False)

    rc = practitioner_main(
        [
            "--features-csv",
            str(features_csv),
            "--out-dir",
            str(out_dir),
            "--target-col",
            "dibbs_zero_vs_forward_zero_expansions_ratio",
            "--target-transform",
            "log",
        ]
    )

    assert rc == 0
    assert (out_dir / "practitioner_cv_metrics.csv").exists()
    assert (out_dir / "practitioner_tree_rules.txt").exists()
    assert (out_dir / "practitioner_summary.json").exists()

    summary = json.loads((out_dir / "practitioner_summary.json").read_text())
    assert summary["group_col"] == "dataset_name"
    assert summary["consistency_policy"] == "none"
    assert summary["rows_used"] == 15
    assert summary["target_transform"] == "log"
