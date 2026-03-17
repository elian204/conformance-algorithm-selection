"""Tests for heuristic-aware practical target building."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.build_stage_c_practical_targets import FAST_FEATURES
from scripts.build_stage_c_practical_targets import main as target_builder_main


def _feature_row(idx: int) -> dict[str, object]:
    return {
        "dataset_name": f"family_{idx}",
        "model_id": f"model_{idx}",
        "trace_id": f"trace_{idx}",
        "trace_hash": f"hash_{idx}",
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


def _metric_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset_name": "family_0",
            "model_id": "model_0",
            "trace_id": "trace_0",
            "trace_hash": "hash_0",
            "forward_me": 100.0,
            "backward_me": 90.0,
            "bidir_mm_me": 95.0,
            "dibbs_me": 120.0,
            "forward_mmr": 110.0,
            "backward_mmr": 111.0,
            "bidir_mm_mmr": 108.0,
            "bidir_std_mmr": 109.0,
            "dibbs_mmr": 107.0,
        },
        {
            "dataset_name": "family_1",
            "model_id": "model_1",
            "trace_id": "trace_1",
            "trace_hash": "hash_1",
            "forward_me": 100.0,
            "backward_me": 100.0,
            "bidir_mm_me": None,
            "dibbs_me": None,
            "forward_mmr": 90.0,
            "backward_mmr": None,
            "bidir_mm_mmr": None,
            "bidir_std_mmr": None,
            "dibbs_mmr": None,
        },
        {
            "dataset_name": "family_2",
            "model_id": "model_2",
            "trace_id": "trace_2",
            "trace_hash": "hash_2",
            "forward_me": 100.0,
            "backward_me": None,
            "bidir_mm_me": None,
            "dibbs_me": None,
            "forward_mmr": 80.0,
            "backward_mmr": 70.0,
            "bidir_mm_mmr": None,
            "bidir_std_mmr": 60.0,
            "dibbs_mmr": 75.0,
        },
    ]


def test_target_builder_writes_joined_scenario_tables(tmp_path: Path):
    metric_csv = tmp_path / "metrics.csv"
    features_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(_metric_rows()).to_csv(metric_csv, index=False)
    pd.DataFrame([_feature_row(idx) for idx in range(3)]).to_csv(features_csv, index=False)

    rc = target_builder_main(
        [
            "--metric-matrix-csv",
            str(metric_csv),
            "--features-csv",
            str(features_csv),
            "--out-dir",
            str(out_dir),
            "--scenarios",
            "rq3a",
            "rq3b_sens",
        ]
    )

    assert rc == 0
    rq3a_csv = out_dir / "rq3a_scenario.csv"
    rq3a_summary_json = out_dir / "rq3a_summary.json"
    rq3b_sens_csv = out_dir / "rq3b_sens_scenario.csv"
    assert rq3a_csv.exists()
    assert rq3a_summary_json.exists()
    assert rq3b_sens_csv.exists()

    rq3a_df = pd.read_csv(rq3a_csv)
    summary = json.loads(rq3a_summary_json.read_text())
    assert len(rq3a_df) == 3
    assert summary["baseline_present_rows"] == 3
    assert summary["eligible_rows"] == 2
    assert summary["no_competitor_rows"] == 1
    assert summary["nonforward_wins"] == 1
    assert summary["ties"] == 1
    assert summary["forward_strict_wins"] == 0
    assert summary["winner_breakdown"]["backward_me"] == 1
    assert summary["duplicate_key_counts"]["target_pre_join"] == 0
    assert summary["duplicate_key_counts"]["feature_pre_join"] == 0
    assert summary["duplicate_key_counts"]["joined_post_join"] == 0
    assert summary["key_join_integrity"]["joined_row_count"] == 3
    assert summary["key_join_integrity"]["unmatched_target_rows"] == 0
    assert "best_competitor_method" in rq3a_df.columns
    assert "best_competitor_is_caveated" in rq3a_df.columns
    for feature in FAST_FEATURES:
        assert feature in rq3a_df.columns

    rq3b_sens_df = pd.read_csv(rq3b_sens_csv)
    row = rq3b_sens_df.loc[rq3b_sens_df["dataset_name"] == "family_2"].iloc[0]
    assert row["best_competitor_method"] == "bidir_std_mmr"
    assert int(row["use_nonforward"]) == 1


def test_target_builder_rejects_duplicate_feature_keys(tmp_path: Path):
    metric_csv = tmp_path / "metrics.csv"
    features_csv = tmp_path / "features.csv"

    pd.DataFrame(_metric_rows()).to_csv(metric_csv, index=False)
    feature_rows = [_feature_row(0), _feature_row(0), _feature_row(1), _feature_row(2)]
    pd.DataFrame(feature_rows).to_csv(features_csv, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        target_builder_main(
            [
                "--metric-matrix-csv",
                str(metric_csv),
                "--features-csv",
                str(features_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--scenarios",
                "rq3a",
            ]
        )


def test_target_builder_writes_oracle_filtered_subset(tmp_path: Path):
    metric_csv = tmp_path / "metrics.csv"
    features_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "out"

    rows = _metric_rows() + [
        {
            "dataset_name": "family_3",
            "model_id": "model_3",
            "trace_id": "trace_3",
            "trace_hash": "hash_3",
            "forward_me": 100.0,
            "backward_me": 120.0,
            "bidir_mm_me": 90.0,
            "dibbs_me": 80.0,
            "forward_mmr": 100.0,
            "backward_mmr": 110.0,
            "bidir_mm_mmr": 105.0,
            "bidir_std_mmr": 104.0,
            "dibbs_mmr": 106.0,
        }
    ]
    feature_rows = [_feature_row(idx) for idx in range(4)]
    pd.DataFrame(rows).to_csv(metric_csv, index=False)
    pd.DataFrame(feature_rows).to_csv(features_csv, index=False)

    rc = target_builder_main(
        [
            "--metric-matrix-csv",
            str(metric_csv),
            "--features-csv",
            str(features_csv),
            "--out-dir",
            str(out_dir),
            "--scenarios",
            "rq3a_oracle_filtered",
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "rq3a_oracle_filtered_summary.json").read_text())
    scenario_df = pd.read_csv(out_dir / "rq3a_oracle_filtered_scenario.csv")
    assert summary["baseline_present_rows"] == 3
    assert summary["eligible_rows"] == 2
    assert summary["nonforward_wins"] == 1
    assert summary["winner_breakdown"]["backward_me"] == 1
    assert "family_3" not in set(scenario_df["dataset_name"])
