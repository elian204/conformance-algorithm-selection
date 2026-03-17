"""Tests for Stage A baseline analysis outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.analyze_stage_a_baselines import main as stage_a_main


def test_stage_a_baselines_outputs(tmp_path: Path):
    input_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "analysis"

    rows = [
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t1",
            "trace_id": "1",
            "method": "forward_zero",
            "algorithm": "forward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 10,
            "time_seconds": 1.0,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t1",
            "trace_id": "1",
            "method": "dibbs_zero",
            "algorithm": "dibbs",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 5,
            "time_seconds": 0.8,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t1",
            "trace_id": "1",
            "method": "bidir_std_zero",
            "algorithm": "bidir_std",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 7,
            "time_seconds": 0.9,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t1",
            "trace_id": "1",
            "method": "backward_zero",
            "algorithm": "backward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 12,
            "time_seconds": 1.1,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t2",
            "trace_id": "2",
            "method": "forward_zero",
            "algorithm": "forward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 20,
            "time_seconds": 2.0,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t2",
            "trace_id": "2",
            "method": "dibbs_zero",
            "algorithm": "dibbs",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 30,
            "time_seconds": 3.0,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t2",
            "trace_id": "2",
            "method": "bidir_std_zero",
            "algorithm": "bidir_std",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 15,
            "time_seconds": 1.5,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t2",
            "trace_id": "2",
            "method": "backward_zero",
            "algorithm": "backward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 18,
            "time_seconds": 2.2,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t3",
            "trace_id": "3",
            "method": "forward_zero",
            "algorithm": "forward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 40,
            "time_seconds": 4.0,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t3",
            "trace_id": "3",
            "method": "dibbs_zero",
            "algorithm": "dibbs",
            "status": "timeout",
            "optimal_cost": 1.0,
            "cost": None,
            "expansions": None,
            "time_seconds": None,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t3",
            "trace_id": "3",
            "method": "bidir_std_zero",
            "algorithm": "bidir_std",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 10,
            "time_seconds": 1.0,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-a",
            "trace_hash": "t3",
            "trace_id": "3",
            "method": "backward_zero",
            "algorithm": "backward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 45,
            "time_seconds": 4.5,
            "aggregate_parent_complete": 1,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-b",
            "trace_hash": "t4",
            "trace_id": "4",
            "method": "forward_zero",
            "algorithm": "forward",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 1,
            "time_seconds": 0.1,
            "aggregate_parent_complete": 0,
        },
        {
            "dataset_name": "demo",
            "model_id": "model-b",
            "trace_hash": "t4",
            "trace_id": "4",
            "method": "dibbs_zero",
            "algorithm": "dibbs",
            "status": "ok",
            "optimal_cost": 1.0,
            "cost": 1.0,
            "expansions": 1,
            "time_seconds": 0.1,
            "aggregate_parent_complete": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    rc = stage_a_main(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(out_dir),
            "--ratio-numerator-method",
            "dibbs_zero",
            "--ratio-denominator-method",
            "forward_zero",
        ]
    )

    assert rc == 0

    summary = json.loads((out_dir / "stage_a_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_after_filters"] == 12
    assert summary["metric_summaries"]["expansions"]["vbs_total"] == 30.0
    assert summary["metric_summaries"]["expansions"]["strict_sbs_method"] == "bidir_std_zero"
    assert summary["metric_summaries"]["expansions"]["strict_sbs_comparable_instances"] == 2

    pairwise = pd.read_csv(out_dir / "stage_a_pairwise_dominance_expansions.csv")
    dibbs_vs_forward = pairwise[
        (pairwise["left_method"] == "dibbs_zero")
        & (pairwise["right_method"] == "forward_zero")
    ].iloc[0]
    assert dibbs_vs_forward["comparable_instances"] == 2
    assert dibbs_vs_forward["wins"] == 1
    assert dibbs_vs_forward["losses"] == 1
    assert dibbs_vs_forward["decisive_win_rate"] == 0.5

    targets = pd.read_csv(out_dir / "stage_a_ratio_targets.csv")
    assert set(targets["trace_hash"]) == {"t1", "t2", "t3"}

    ratio_map = targets.set_index("trace_hash")[
        [
            "best_bidirectional_vs_forward_expansion_ratio",
            "dibbs_vs_forward_expansion_ratio",
            "dibbs_zero_vs_forward_zero_expansions_ratio",
        ]
    ]
    assert ratio_map.loc["t1", "best_bidirectional_vs_forward_expansion_ratio"] == 0.5
    assert ratio_map.loc["t2", "best_bidirectional_vs_forward_expansion_ratio"] == 0.75
    assert ratio_map.loc["t3", "best_bidirectional_vs_forward_expansion_ratio"] == 0.25
    assert ratio_map.loc["t1", "dibbs_vs_forward_expansion_ratio"] == 0.5
    assert ratio_map.loc["t2", "dibbs_vs_forward_expansion_ratio"] == 1.5
    assert pd.isna(ratio_map.loc["t3", "dibbs_vs_forward_expansion_ratio"])
    assert ratio_map.loc["t1", "dibbs_zero_vs_forward_zero_expansions_ratio"] == 0.5
    assert ratio_map.loc["t2", "dibbs_zero_vs_forward_zero_expansions_ratio"] == 1.5
    assert pd.isna(ratio_map.loc["t3", "dibbs_zero_vs_forward_zero_expansions_ratio"])


def test_stage_a_excludes_suboptimal_ok_rows(tmp_path: Path):
    input_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "forward_zero",
                "algorithm": "forward",
                "status": "ok",
                "optimal_cost": 2.0,
                "cost": 2.0,
                "expansions": 100,
                "time_seconds": 1.0,
                "aggregate_parent_complete": 1,
            },
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "dibbs_me",
                "algorithm": "dibbs",
                "status": "ok",
                "optimal_cost": 2.0,
                "cost": 3.0,
                "expansions": 1,
                "time_seconds": 0.01,
                "aggregate_parent_complete": 1,
            },
        ]
    ).to_csv(input_csv, index=False)

    rc = stage_a_main(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(out_dir),
            "--ratio-numerator-method",
            "dibbs_me",
            "--ratio-denominator-method",
            "forward_zero",
        ]
    )

    assert rc == 0

    summary = json.loads((out_dir / "stage_a_summary.json").read_text(encoding="utf-8"))
    assert summary["metric_summaries"]["expansions"]["vbs_total"] == 100.0
    assert summary["metric_summaries"]["expansions"]["strict_sbs_method"] == "forward_zero"

    targets = pd.read_csv(out_dir / "stage_a_ratio_targets.csv")
    assert pd.isna(targets.loc[0, "dibbs_vs_forward_expansion_ratio"])
    assert pd.isna(targets.loc[0, "dibbs_me_vs_forward_zero_expansions_ratio"])


def test_stage_a_uses_experiment_id_as_primary_rerun_recency_key(tmp_path: Path):
    input_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "forward_zero",
                "algorithm": "forward",
                "status": "ok",
                "optimal_cost": 1.0,
                "cost": 1.0,
                "expansions": 10,
                "time_seconds": 0.10,
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "zzz_old_named_run",
                "aggregate_parent_run_id": "parent_old",
                "experiment_id": "exp_20260301_120000",
            },
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "forward_zero",
                "algorithm": "forward",
                "status": "ok",
                "optimal_cost": 1.0,
                "cost": 1.0,
                "expansions": 999,
                "time_seconds": 9.99,
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "aaa_manual_run",
                "aggregate_parent_run_id": "parent_new",
                "experiment_id": "exp_20260302_120000",
            },
        ]
    ).to_csv(input_csv, index=False)

    rc = stage_a_main(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0

    matrix = pd.read_csv(out_dir / "stage_a_metric_matrix_expansions.csv")
    assert matrix.loc[0, "forward_zero"] == 999

    summary = json.loads((out_dir / "stage_a_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_after_filters"] == 2
    assert summary["rows_after_latest_rerun_dedup"] == 1


def test_stage_a_handles_all_na_bidirectional_family_rows(tmp_path: Path):
    input_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "forward_zero",
                "algorithm": "forward",
                "status": "ok",
                "optimal_cost": 1.0,
                "cost": 1.0,
                "expansions": 12,
                "time_seconds": 1.2,
                "aggregate_parent_complete": 1,
            },
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_hash": "t1",
                "trace_id": "1",
                "method": "dibbs_zero",
                "algorithm": "dibbs",
                "status": "timeout",
                "optimal_cost": 1.0,
                "cost": None,
                "expansions": None,
                "time_seconds": None,
                "aggregate_parent_complete": 1,
            },
        ]
    ).to_csv(input_csv, index=False)

    rc = stage_a_main(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0

    targets = pd.read_csv(out_dir / "stage_a_ratio_targets.csv")
    assert pd.isna(targets.loc[0, "best_bidirectional_expansions"])
    assert pd.isna(targets.loc[0, "best_bidirectional_expansions_algorithm"])
    assert pd.isna(targets.loc[0, "best_bidirectional_time_seconds"])
    assert pd.isna(targets.loc[0, "best_bidirectional_time_seconds_algorithm"])
