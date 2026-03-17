"""Tests for Stage B feature extraction."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.petri_net import PetriNet, WorkflowNet, marking_from_places
from scripts.feature_engineering import main as feature_main


def _build_test_workflow() -> WorkflowNet:
    net = PetriNet("test")
    p0 = net.add_place("p0")
    p1 = net.add_place("p1")
    p2 = net.add_place("p2")
    p3 = net.add_place("p3")
    p4 = net.add_place("p4")
    p5 = net.add_place("p5")
    p6 = net.add_place("p6")

    t_a = net.add_transition("t_a", "A")
    t_b = net.add_transition("t_b", "B")
    t_tau = net.add_transition("t_tau", None)
    t_c = net.add_transition("t_c", "C")
    t_d = net.add_transition("t_d", "D")

    net.add_arc_place_to_transition(p0, t_a)
    net.add_arc_transition_to_place(t_a, p1)

    net.add_arc_place_to_transition(p0, t_b)
    net.add_arc_transition_to_place(t_b, p2)

    net.add_arc_place_to_transition(p1, t_tau)
    net.add_arc_transition_to_place(t_tau, p3)

    net.add_arc_place_to_transition(p2, t_c)
    net.add_arc_transition_to_place(t_c, p3)

    net.add_arc_place_to_transition(p3, t_d)
    net.add_arc_place_to_transition(p4, t_d)
    net.add_arc_transition_to_place(t_d, p5)
    net.add_arc_transition_to_place(t_d, p6)

    return WorkflowNet(
        net=net,
        initial_marking=marking_from_places(p0, p4),
        final_marking=marking_from_places(p5, p6),
    )


def test_feature_engineering_outputs_expected_features(tmp_path: Path, monkeypatch):
    driver_csv = tmp_path / "targets.csv"
    aggregate_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "out"
    model_path = tmp_path / "model_a.pkl"
    model_path.write_bytes(b"placeholder")

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "trace_id": "1",
                "model_id": "model-a",
                "trace_hash": "hash-1",
                "forward_expansions_method": "forward_zero",
                "dibbs_expansions_method": "dibbs_me",
                "bidir_mm_expansions_method": "bidir_mm_mmr",
                "dibbs_vs_forward_expansion_ratio": 0.5,
            }
        ]
    ).to_csv(driver_csv, index=False)

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "trace_id": "1",
                "model_id": "model-a",
                "trace_hash": "hash-1",
                "aggregate_parent_complete": 1,
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run/merged_results.csv",
                "aggregate_run_name": "demo_run_20260302_120000",
                "aggregate_parent_run_id": "parent-a",
                "experiment_id": "exp_20260302_120000",
                "model_path": str(model_path),
                "model_name": "model_a.pkl",
                "trace_activities": "A|Z|A",
                "method": "forward_zero",
            }
        ]
    ).to_csv(aggregate_csv, index=False)

    monkeypatch.setattr(
        "scripts.feature_engineering.load_model",
        lambda model_path: (_build_test_workflow(), None, None, None),
    )

    rc = feature_main(
        [
            "--driver-csv",
            str(driver_csv),
            "--aggregate-csv",
            str(aggregate_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0

    output = pd.read_csv(out_dir / "selection_features_full.csv")
    assert len(output) == 1
    row = output.iloc[0]
    assert row["_target_row_idx"] == 0
    assert row["xor_splits"] == 1
    assert row["xor_joins"] == 1
    assert row["and_splits"] == 1
    assert row["and_joins"] == 1
    assert math.isclose(row["tau_ratio"], 0.2)
    assert row["trace_length"] == 3
    assert row["distinct_activities"] == 2
    assert math.isclose(row["trace_entropy"], 0.9182958340544896)
    assert row["trace_max_repetitions"] == 2
    assert math.isclose(row["log_only_prop"], 1 / 3)
    assert math.isclose(row["model_only_prop"], 3 / 4)
    assert math.isclose(row["alphabet_overlap"], 1 / 5)
    assert math.isclose(row["branching_imbalance"], 2.0)
    assert row["sp_places"] == 11
    assert row["sp_transitions"] == 10
    assert math.isclose(row["sp_branching_factor"], 10 / 11)
    assert row["target_forward_method"] == "forward_zero"
    assert row["target_dibbs_method"] == "dibbs_me"
    assert row["target_mm_method"] == "bidir_mm_mmr"
    assert row["uses_potentially_inconsistent_dibbs_target"] == 1
    assert row["uses_potentially_inconsistent_mm_target"] == 0
    assert row["uses_potentially_inconsistent_bidirectional_target"] == 1

    failures = pd.read_csv(out_dir / "feature_extraction_failures.csv")
    assert failures.empty

    summary = json.loads((out_dir / "feature_extraction_summary.json").read_text())
    assert summary["rows_in_driver"] == 1
    assert summary["rows_in_output"] == 1
    assert summary["lp_features_included"] is False


def test_feature_engineering_resolves_missing_model_path_by_basename(
    tmp_path: Path,
    monkeypatch,
):
    driver_csv = tmp_path / "targets.csv"
    aggregate_csv = tmp_path / "aggregate.csv"
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fallback_model_path = data_dir / "shared_model.pkl"
    fallback_model_path.write_bytes(b"placeholder")

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "trace_id": "1",
                "model_id": "model-a",
                "trace_hash": "hash-1",
                "forward_expansions_method": "forward_zero",
                "dibbs_expansions_method": "dibbs_zero",
                "bidir_mm_expansions_method": "bidir_mm_zero",
                "dibbs_vs_forward_expansion_ratio": 0.8,
            }
        ]
    ).to_csv(driver_csv, index=False)

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "trace_id": "1",
                "model_id": "model-a",
                "trace_hash": "hash-1",
                "aggregate_parent_complete": 1,
                "aggregate_source_kind": "merged",
                "aggregate_source_file": "/tmp/run/merged_results.csv",
                "aggregate_run_name": "demo_run_20260302_120000",
                "aggregate_parent_run_id": "parent-a",
                "experiment_id": "exp_20260302_120000",
                "model_path": str(tmp_path / "missing" / "shared_model.pkl"),
                "model_name": "shared_model.pkl",
                "trace_activities": "A|B",
                "method": "forward_zero",
            }
        ]
    ).to_csv(aggregate_csv, index=False)

    monkeypatch.setattr("scripts.feature_engineering.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.feature_engineering.load_model",
        lambda model_path: (_build_test_workflow(), None, None, None),
    )

    rc = feature_main(
        [
            "--driver-csv",
            str(driver_csv),
            "--aggregate-csv",
            str(aggregate_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0

    output = pd.read_csv(out_dir / "selection_features_full.csv")
    assert len(output) == 1
    assert output.iloc[0]["model_path"] == str(fallback_model_path)

    failures = pd.read_csv(out_dir / "feature_extraction_failures.csv")
    assert failures.empty

    summary = json.loads((out_dir / "feature_extraction_summary.json").read_text())
    assert summary["missing_model_path_fallback_hits"] == 1
    assert summary["missing_model_path_fallback_misses"] == 0
