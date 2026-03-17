"""Tests for oracle LP feature extraction."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.petri_net import PetriNet, WorkflowNet, marking_from_places
from scripts.extract_oracle_features import main as oracle_main


def _build_test_workflow() -> WorkflowNet:
    net = PetriNet("oracle_test")
    p0 = net.add_place("p0")
    p1 = net.add_place("p1")
    p2 = net.add_place("p2")

    t_a = net.add_transition("t_a", "A")
    t_b = net.add_transition("t_b", "B")

    net.add_arc_place_to_transition(p0, t_a)
    net.add_arc_transition_to_place(t_a, p1)
    net.add_arc_place_to_transition(p1, t_b)
    net.add_arc_transition_to_place(t_b, p2)

    return WorkflowNet(
        net=net,
        initial_marking=marking_from_places(p0),
        final_marking=marking_from_places(p2),
    )


class _StubHeuristic:
    def __init__(self, value: float):
        self.value = value

    def estimate(self, marking) -> float:
        return self.value


def test_extract_oracle_features_serial_outputs_expected_columns(
    tmp_path: Path,
    monkeypatch,
):
    features_csv = tmp_path / "selection_features_full.csv"
    trace_source_csv = tmp_path / "selection_trace_table_apriori.csv"
    out_dir = tmp_path / "oracle_out"
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"placeholder")

    pd.DataFrame(
        [
            {
                "_target_row_idx": 0,
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "model_path": str(model_path),
                "trace_length": 2,
            }
        ]
    ).to_csv(features_csv, index=False)

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|B",
            }
        ]
    ).to_csv(trace_source_csv, index=False)

    monkeypatch.setattr(
        "scripts.extract_oracle_features.load_model",
        lambda model_path: (_build_test_workflow(), None, None, None),
    )
    monkeypatch.setattr(
        "scripts.extract_oracle_features.create_marking_equation_heuristic",
        lambda sp, direction, timeout_seconds=None: _StubHeuristic(
            5.0 if direction == "forward" else 3.0
        ),
    )

    rc = oracle_main(
        [
            "--features-csv",
            str(features_csv),
            "--trace-source-csv",
            str(trace_source_csv),
            "--out-dir",
            str(out_dir),
            "--jobs",
            "1",
            "--checkpoint-every",
            "1",
        ]
    )

    assert rc == 0

    output = pd.read_csv(out_dir / "selection_features_oracle.csv")
    assert len(output) == 1
    row = output.iloc[0]
    assert row["_target_row_idx"] == 0
    assert row["h_f"] == 5.0
    assert row["h_b"] == 3.0
    assert row["heuristic_asymmetry"] == 2.0
    assert row["normalized_h_f"] == 2.5
    assert row["model_path"] == str(model_path)

    failures = pd.read_csv(out_dir / "selection_features_oracle_failures.csv")
    assert failures.empty

    summary = json.loads(
        (out_dir / "selection_features_oracle_summary.json").read_text()
    )
    assert summary["rows_in_driver"] == 1
    assert summary["rows_in_output"] == 1
    assert summary["rows_with_failures"] == 0


def test_extract_oracle_features_deduplicates_driver_and_trace_source(
    tmp_path: Path,
    monkeypatch,
):
    features_csv = tmp_path / "selection_features_full.csv"
    trace_source_csv = tmp_path / "selection_trace_table_apriori.csv"
    out_dir = tmp_path / "oracle_out"
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"placeholder")

    pd.DataFrame(
        [
            {
                "_target_row_idx": 0,
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "model_path": str(model_path),
                "trace_length": 2,
                "extra": "old",
            },
            {
                "_target_row_idx": 1,
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "model_path": str(model_path),
                "trace_length": 2,
                "extra": "new",
            },
        ]
    ).to_csv(features_csv, index=False)

    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|B",
            },
            {
                "dataset_name": "demo",
                "model_id": "model-a",
                "trace_id": "1",
                "trace_hash": "hash-1",
                "trace_activities": "A|B",
            },
        ]
    ).to_csv(trace_source_csv, index=False)

    monkeypatch.setattr(
        "scripts.extract_oracle_features.load_model",
        lambda model_path: (_build_test_workflow(), None, None, None),
    )
    monkeypatch.setattr(
        "scripts.extract_oracle_features.create_marking_equation_heuristic",
        lambda sp, direction, timeout_seconds=None: _StubHeuristic(
            5.0 if direction == "forward" else 3.0
        ),
    )

    rc = oracle_main(
        [
            "--features-csv",
            str(features_csv),
            "--trace-source-csv",
            str(trace_source_csv),
            "--out-dir",
            str(out_dir),
            "--jobs",
            "1",
            "--checkpoint-every",
            "1",
        ]
    )

    assert rc == 0

    output = pd.read_csv(out_dir / "selection_features_oracle.csv")
    assert len(output) == 1
    assert output.loc[0, "_target_row_idx"] == 1

    summary = json.loads(
        (out_dir / "selection_features_oracle_summary.json").read_text()
    )
    assert summary["rows_in_driver"] == 1
    assert summary["rows_in_output"] == 1
    assert summary["driver_duplicate_rows_dropped"] == 1
    assert summary["trace_duplicate_rows_dropped"] == 1
