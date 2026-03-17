"""Tests for heuristic-aware practical classifier pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_stage_c_practical_classifier import FAST_FEATURES
from scripts.run_stage_c_practical_classifier import align_oof_predictions
from scripts.run_stage_c_practical_classifier import compute_regret
from scripts.run_stage_c_practical_classifier import main as practical_classifier_main


def _scenario_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(15):
        competitor_available = 0 if idx == 0 else 1
        use_nonforward = 1 if idx % 3 == 0 and competitor_available else 0
        is_tie = 1 if idx % 5 == 0 and competitor_available else 0
        best_comp = 90.0 if use_nonforward else 100.0
        backward_me = best_comp if competitor_available else None
        if idx == 4:
            backward_me = None
        row = {
            "dataset_name": f"family_{idx}",
            "model_id": f"model_{idx}",
            "trace_id": f"trace_{idx}",
            "trace_hash": f"hash_{idx}",
            "forward_baseline_method": "forward_me",
            "forward_expansions": 100.0,
            "best_competitor_method": "backward_me" if competitor_available else None,
            "best_competitor_expansions": best_comp if competitor_available else None,
            "competitor_available": competitor_available,
            "use_nonforward": use_nonforward,
            "is_tie": is_tie,
            "oracle_expansions": best_comp if use_nonforward else 100.0,
            "oracle_method": "backward_me" if use_nonforward else "forward_me",
            "best_competitor_is_caveated": False,
            "backward_me": backward_me,
            "bidir_mm_me": 95.0 if competitor_available else None,
            "dibbs_me": 97.0 if competitor_available else None,
        }
        for feature_idx, feature in enumerate(FAST_FEATURES):
            row[feature] = float(idx + feature_idx + 1)
        rows.append(row)
    return rows


def test_practical_classifier_main_writes_outputs(tmp_path: Path):
    scenario_csv = tmp_path / "rq3a.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_scenario_rows()).to_csv(scenario_csv, index=False)

    rc = practical_classifier_main(
        [
            "--scenario",
            "rq3a",
            "--scenario-csv",
            str(scenario_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "practical_tree_cv_metrics.csv").exists()
    assert (out_dir / "practical_hgb_cv_metrics.csv").exists()
    assert (out_dir / "practical_tree_feature_importances.csv").exists()
    assert (out_dir / "practical_hgb_permutation_importance.csv").exists()
    assert (out_dir / "practical_tree_rules.txt").exists()
    assert (out_dir / "practical_summary.json").exists()

    summary = json.loads((out_dir / "practical_summary.json").read_text())
    assert summary["scenario"] == "rq3a"
    assert summary["forward_baseline_method"] == "forward_me"
    assert summary["fixed_nonforward_action"] == "backward_me"
    assert summary["threshold_policy_main"] == 0.5
    assert summary["rows_baseline_present"] == 15
    assert summary["rows_eligible"] == 14
    assert summary["rows_no_competitor"] == 1
    assert summary["fixed_action_fallback_count_eligible"] >= 1
    assert "eligible_metrics" in summary["decision_tree"]
    assert "deployment_accounting_metrics" in summary["decision_tree"]
    assert "deployable_regret_eligible" in summary["decision_tree"]
    assert "best_competitor_switch_regret_eligible" in summary["decision_tree"]
    assert "posthoc_threshold_analysis" in summary["hist_gradient_boosting"]
    assert "always_forward_baseline_eligible" in summary
    assert "always_forward_baseline_baseline_present" in summary


def test_align_oof_predictions_restores_row_order_and_regret_invariant():
    eligible_df = pd.DataFrame(
        [
            {
                "dataset_name": "family_a",
                "model_id": "model_a",
                "trace_id": "trace_a",
                "trace_hash": "hash_a",
                "forward_expansions": 100.0,
                "best_competitor_expansions": 80.0,
                "use_nonforward": 1,
                "is_tie": 0,
                "competitor_available": 1,
                "backward_me": 90.0,
            },
            {
                "dataset_name": "family_b",
                "model_id": "model_b",
                "trace_id": "trace_b",
                "trace_hash": "hash_b",
                "forward_expansions": 100.0,
                "best_competitor_expansions": 120.0,
                "use_nonforward": 0,
                "is_tie": 0,
                "competitor_available": 1,
                "backward_me": 120.0,
            },
        ]
    )
    eligible_df["oracle_expansions"] = [80.0, 100.0]

    oof_df = pd.DataFrame(
        [
            {
                "row_id": 1,
                "fold": 1,
                "y_true": 0,
                "is_tie": 0,
                "probability_nonforward": 0.2,
                "prediction_main": 0,
                "prediction_always_forward": 0,
            },
            {
                "row_id": 0,
                "fold": 2,
                "y_true": 1,
                "is_tie": 0,
                "probability_nonforward": 0.8,
                "prediction_main": 1,
                "prediction_always_forward": 0,
            },
        ]
    )

    aligned = align_oof_predictions(eligible_df, oof_df)
    assert aligned["probability_nonforward"].tolist() == [0.8, 0.2]
    assert aligned["prediction_main"].tolist() == [1, 0]

    deployable = compute_regret(
        aligned,
        aligned["prediction_main"].to_numpy(),
        fixed_action_col="backward_me",
        use_best_competitor=False,
    )
    best_competitor_switch = compute_regret(
        aligned,
        aligned["prediction_main"].to_numpy(),
        fixed_action_col="backward_me",
        use_best_competitor=True,
    )
    assert best_competitor_switch["mean_excess_expansions"] <= deployable["mean_excess_expansions"]
