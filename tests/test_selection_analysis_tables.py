"""Tests for selection analysis table generation semantics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.build_selection_analysis_tables import main as build_selection_main


def test_selection_tables_use_latest_rerun_and_consensus_optimal_rows(tmp_path: Path):
    input_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "zzz_old_named_run",
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run_old/merged_results.csv",
                "aggregate_parent_run_id": "parent_old",
                "experiment_id": "exp_20260301_120000",
                "dataset_name": "demo",
                "model_id": "model-a",
                "model_name": "demo_model.pkl",
                "model_path": "/tmp/demo_model.pkl",
                "model_source": "file",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|Z",
                "trace_length": 2,
                "trace_unique_activities": 2,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 1,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 1,
                "method": "forward_zero",
                "status": "ok",
                "time_seconds": 0.10,
                "cost": 1.0,
                "optimal_cost": 1.0,
                "model_places": 3,
                "model_transitions": 2,
                "model_arcs": 4,
                "model_silent_transitions": 0,
                "model_visible_transitions": 2,
                "model_place_in_degree_avg": 1.0,
                "model_place_out_degree_avg": 1.0,
                "model_place_in_degree_max": 1,
                "model_place_out_degree_max": 1,
                "model_transition_in_degree_avg": 1.0,
                "model_transition_out_degree_avg": 1.0,
                "model_transition_in_degree_max": 1,
                "model_transition_out_degree_max": 1,
            },
            {
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "aaa_manual_run",
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run_new/merged_results.csv",
                "aggregate_parent_run_id": "parent_new",
                "experiment_id": "exp_20260302_120000",
                "dataset_name": "demo",
                "model_id": "model-a",
                "model_name": "demo_model.pkl",
                "model_path": "/tmp/demo_model.pkl",
                "model_source": "file",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|Z",
                "trace_length": 2,
                "trace_unique_activities": 2,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 1,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 1,
                "method": "forward_zero",
                "status": "ok",
                "time_seconds": 9.99,
                "cost": 1.0,
                "optimal_cost": 1.0,
                "model_places": 3,
                "model_transitions": 2,
                "model_arcs": 4,
                "model_silent_transitions": 0,
                "model_visible_transitions": 2,
                "model_place_in_degree_avg": 1.0,
                "model_place_out_degree_avg": 1.0,
                "model_place_in_degree_max": 1,
                "model_place_out_degree_max": 1,
                "model_transition_in_degree_avg": 1.0,
                "model_transition_out_degree_avg": 1.0,
                "model_transition_in_degree_max": 1,
                "model_transition_out_degree_max": 1,
            },
            {
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "aaa_manual_run",
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run_new/merged_results.csv",
                "aggregate_parent_run_id": "parent_new",
                "experiment_id": "exp_20260302_120000",
                "dataset_name": "demo",
                "model_id": "model-a",
                "model_name": "demo_model.pkl",
                "model_path": "/tmp/demo_model.pkl",
                "model_source": "file",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|Z",
                "trace_length": 2,
                "trace_unique_activities": 2,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 1,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 1,
                "method": "dibbs_zero",
                "status": "ok",
                "time_seconds": 0.05,
                "cost": 2.0,
                "optimal_cost": 1.0,
                "model_places": 3,
                "model_transitions": 2,
                "model_arcs": 4,
                "model_silent_transitions": 0,
                "model_visible_transitions": 2,
                "model_place_in_degree_avg": 1.0,
                "model_place_out_degree_avg": 1.0,
                "model_place_in_degree_max": 1,
                "model_place_out_degree_max": 1,
                "model_transition_in_degree_avg": 1.0,
                "model_transition_out_degree_avg": 1.0,
                "model_transition_in_degree_max": 1,
                "model_transition_out_degree_max": 1,
            },
            {
                "aggregate_parent_complete": 1,
                "aggregate_run_name": "aaa_manual_run",
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run_new/merged_results.csv",
                "aggregate_parent_run_id": "parent_new",
                "experiment_id": "exp_20260302_120000",
                "dataset_name": "demo",
                "model_id": "model-a",
                "model_name": "demo_model.pkl",
                "model_path": "/tmp/demo_model.pkl",
                "model_source": "file",
                "trace_id": "2",
                "trace_hash": "hash-2",
                "trace_activities": "Q",
                "trace_length": 1,
                "trace_unique_activities": 1,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 0,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 1,
                "method": "forward_zero",
                "status": "ok",
                "time_seconds": 0.03,
                "cost": 3.0,
                "optimal_cost": 1.0,
                "model_places": 3,
                "model_transitions": 2,
                "model_arcs": 4,
                "model_silent_transitions": 0,
                "model_visible_transitions": 2,
                "model_place_in_degree_avg": 1.0,
                "model_place_out_degree_avg": 1.0,
                "model_place_in_degree_max": 1,
                "model_place_out_degree_max": 1,
                "model_transition_in_degree_avg": 1.0,
                "model_transition_out_degree_avg": 1.0,
                "model_transition_in_degree_max": 1,
                "model_transition_out_degree_max": 1,
            },
        ]
    ).to_csv(input_csv, index=False)

    rc = build_selection_main(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    method_rows = pd.read_csv(out_dir / "selection_method_rows_completed_enriched.csv")
    assert len(method_rows) == 3
    forward_row = method_rows[method_rows["method"] == "forward_zero"].iloc[0]
    assert forward_row["time_seconds"] == 9.99

    apriori = pd.read_csv(out_dir / "selection_trace_table_apriori.csv")
    assert len(apriori) == 1
    assert apriori.loc[0, "best_method"] == "forward_zero"
    assert apriori.loc[0, "best_time_seconds"] == 9.99
    assert "hash-2" not in apriori["trace_hash"].tolist()
