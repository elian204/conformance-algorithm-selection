"""Tests for the Stage C routing classifier pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_stage_c_classifier import main as classifier_main


def _build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(15):
        ratio = 0.8 if idx % 2 == 0 else 1.2
        rows.append(
            {
                "dataset_name": f"family_{idx}",
                "model_id": f"model_{idx}",
                "trace_id": f"trace_{idx}",
                "trace_hash": f"hash_{idx}",
                "dibbs_zero_vs_forward_zero_expansions_ratio": ratio + (idx * 0.01),
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


def test_stage_c_classifier_main_writes_outputs(tmp_path: Path):
    features_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_build_rows()).to_csv(features_csv, index=False)

    rc = classifier_main(
        [
            "--features-csv",
            str(features_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "classifier_tree_cv_metrics.csv").exists()
    assert (out_dir / "classifier_hgb_cv_metrics.csv").exists()
    assert (out_dir / "classifier_tree_feature_importances.csv").exists()
    assert (out_dir / "classifier_hgb_permutation_importance.csv").exists()
    assert (out_dir / "classifier_tree_rules.txt").exists()
    assert (out_dir / "classifier_summary.json").exists()

    summary = json.loads((out_dir / "classifier_summary.json").read_text())
    assert summary["group_col"] == "dataset_name"
    assert summary["rows_original"] == 15
    assert summary["rows_after_target_filter"] == 15
    assert summary["rows_after_feature_filter"] == 15
    assert summary["rows_used"] == 15
    assert summary["groups_used"] == 15
    assert summary["consistency_policy"] == "none"
    assert 0.0 <= summary["positive_class_rate"] <= 1.0
    assert summary["decision_tree"]["model_name"] == "DecisionTreeClassifier"
    assert (
        summary["hist_gradient_boosting"]["model_name"]
        == "HistGradientBoostingClassifier"
    )
