#!/usr/bin/env python3
"""Run heuristic-aware practical Stage C classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


RANDOM_STATE = 42
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
        "fixed_action": "backward_me",
    },
    "rq3b": {
        "baseline": "forward_mmr",
        "competitors": ["backward_mmr", "bidir_mm_mmr", "dibbs_mmr"],
        "fixed_action": "bidir_mm_mmr",
    },
    "rq3b_sens": {
        "baseline": "forward_mmr",
        "competitors": ["backward_mmr", "bidir_mm_mmr", "bidir_std_mmr", "dibbs_mmr"],
        "fixed_action": "bidir_mm_mmr",
    },
    "rq3a_backward_only": {
        "baseline": "forward_me",
        "competitors": ["backward_me"],
        "fixed_action": "backward_me",
    },
    "rq3a_oracle_filtered": {
        "baseline": "forward_me",
        "competitors": ["backward_me", "bidir_mm_me", "dibbs_me"],
        "fixed_action": "backward_me",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Run heuristic-aware practical classifiers")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group-col", default="dataset_name")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=3)
    return parser


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Ensure the expected columns are present."""

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _binary_metric_block(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute binary metrics with safe handling for single-class folds."""

    y_series = pd.Series(y_true)
    metrics = {
        "accuracy": float(accuracy_score(y_series, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_series, predictions)),
        "f1": float(f1_score(y_series, predictions, zero_division=0)),
        "roc_auc_valid": int(0),
        "pr_auc_valid": int(0),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if y_series.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_series, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_series, probabilities))
        metrics["roc_auc_valid"] = 1
        metrics["pr_auc_valid"] = 1
    return metrics


def _with_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Prefix all keys in a metrics dictionary."""

    return {f"{prefix}{key}": value for key, value in metrics.items()}


def fit_tree_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int,
) -> tuple[SimpleImputer, DecisionTreeClassifier]:
    """Fit the shallow tree classifier."""

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train_i, y_train)
    return imputer, model


def fit_hgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[SimpleImputer, HistGradientBoostingClassifier]:
    """Fit the HGB classifier."""

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_i, y_train)
    return imputer, model


def prepare_dataset(
    scenario_csv: Path,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load scenario data and split baseline-present vs eligible populations."""

    df = pd.read_csv(scenario_csv)
    validate_columns(
        df,
        KEY_COLS
        + FAST_FEATURES
        + [
            "forward_baseline_method",
            "forward_expansions",
            "best_competitor_method",
            "best_competitor_expansions",
            "use_nonforward",
            "is_tie",
            "competitor_available",
            group_col,
        ],
    )
    baseline_df = df.copy()
    baseline_df["forward_expansions"] = pd.to_numeric(
        baseline_df["forward_expansions"],
        errors="coerce",
    )
    baseline_df["best_competitor_expansions"] = pd.to_numeric(
        baseline_df["best_competitor_expansions"],
        errors="coerce",
    )
    baseline_df["competitor_available"] = (
        pd.to_numeric(baseline_df["competitor_available"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    baseline_df["use_nonforward"] = (
        pd.to_numeric(baseline_df["use_nonforward"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    baseline_df["is_tie"] = (
        pd.to_numeric(baseline_df["is_tie"], errors="coerce").fillna(0).astype(int)
    )
    baseline_df["oracle_expansions"] = np.where(
        baseline_df["competitor_available"].astype(bool),
        np.minimum(
            baseline_df["forward_expansions"],
            baseline_df["best_competitor_expansions"].fillna(np.inf),
        ),
        baseline_df["forward_expansions"],
    )
    baseline_df["row_id"] = np.arange(len(baseline_df))

    eligible_df = baseline_df.loc[baseline_df["competitor_available"].eq(1)].copy()
    eligible_df = eligible_df.dropna(subset=[group_col]).copy()
    eligible_df = eligible_df.dropna(subset=FAST_FEATURES, how="all").copy()
    eligible_df = eligible_df.reset_index(drop=True)
    return baseline_df, eligible_df


def run_model_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    is_tie: pd.Series,
    n_splits: int,
    max_depth: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, object, pd.DataFrame | None]:
    """Run grouped CV and collect OOF predictions plus fold metrics."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold, found {groups.nunique()}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, float]] = []
    oof_rows: List[pd.DataFrame] = []
    importance_rows: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        is_tie_test = is_tie.iloc[test_idx].astype(bool)

        if model_name == "decision_tree":
            imputer, model = fit_tree_classifier(X_train, y_train, max_depth=max_depth)
        elif model_name == "hgb":
            imputer, model = fit_hgb_classifier(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        X_test_i = imputer.transform(X_test)
        probabilities = model.predict_proba(X_test_i)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        always_forward_predictions = np.zeros(len(y_test), dtype=int)
        always_forward_scores = np.zeros(len(y_test), dtype=float)

        metrics = _binary_metric_block(y_test, predictions, probabilities)
        baseline_metrics = _binary_metric_block(
            y_test,
            always_forward_predictions,
            always_forward_scores,
        )
        non_tie_mask = ~is_tie_test
        if non_tie_mask.any():
            non_tie_metrics = _binary_metric_block(
                y_test.loc[non_tie_mask],
                predictions[non_tie_mask.to_numpy()],
                probabilities[non_tie_mask.to_numpy()],
            )
            baseline_non_tie_metrics = _binary_metric_block(
                y_test.loc[non_tie_mask],
                always_forward_predictions[non_tie_mask.to_numpy()],
                always_forward_scores[non_tie_mask.to_numpy()],
            )
        else:
            non_tie_metrics = _binary_metric_block(
                pd.Series([0, 1]),
                np.array([0, 1]),
                np.array([0.0, 1.0]),
            )
            baseline_non_tie_metrics = _binary_metric_block(
                pd.Series([0, 1]),
                np.array([0, 0]),
                np.array([0.0, 0.0]),
            )
            for key in non_tie_metrics:
                if key.endswith("_valid"):
                    non_tie_metrics[key] = 0
                    baseline_non_tie_metrics[key] = 0
                else:
                    non_tie_metrics[key] = float("nan")
                    baseline_non_tie_metrics[key] = float("nan")

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                "positive_rate_test": float(y_test.mean()),
                "positive_rate_pred": float(np.mean(predictions)),
                "non_tie_rows": int(non_tie_mask.sum()),
                **metrics,
                **_with_prefix(non_tie_metrics, "non_tie_"),
                **_with_prefix(baseline_metrics, "always_forward_"),
                **_with_prefix(baseline_non_tie_metrics, "always_forward_non_tie_"),
            }
        )

        oof_rows.append(
            pd.DataFrame(
                {
                    "row_id": X_test.index.to_numpy(),
                    "fold": fold,
                    "y_true": y_test.to_numpy(),
                    "is_tie": is_tie_test.to_numpy(),
                    "probability_nonforward": probabilities,
                    "prediction_main": predictions,
                    "prediction_always_forward": always_forward_predictions,
                }
            )
        )

        if model_name == "hgb":
            perm = permutation_importance(
                model,
                X_test_i,
                y_test,
                n_repeats=5,
                random_state=RANDOM_STATE,
                scoring="balanced_accuracy",
            )
            importance_rows.append(
                pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                        "fold": fold,
                    }
                )
            )

    if model_name == "decision_tree":
        final_imputer, final_model = fit_tree_classifier(X, y, max_depth=max_depth)
    else:
        final_imputer, final_model = fit_hgb_classifier(X, y)

    importance_df = (
        pd.concat(importance_rows, ignore_index=True) if importance_rows else None
    )
    return (
        pd.DataFrame(fold_rows),
        pd.concat(oof_rows, ignore_index=True),
        final_imputer,
        final_model,
        importance_df,
    )


def summarize_fold_metrics(fold_df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """Summarize means/stds for a fold metric block."""

    metrics = {}
    for name in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]:
        column = f"{prefix}{name}"
        metrics[f"{column}_mean"] = float(np.nanmean(fold_df[column]))
        metrics[f"{column}_std"] = float(np.nanstd(fold_df[column], ddof=0))
    metrics[f"{prefix}roc_auc_valid_folds"] = int(fold_df[f"{prefix}roc_auc_valid"].sum())
    metrics[f"{prefix}pr_auc_valid_folds"] = int(fold_df[f"{prefix}pr_auc_valid"].sum())
    return metrics


def threshold_scan(
    df: pd.DataFrame,
    fixed_action_col: str,
    thresholds: np.ndarray,
) -> Dict[str, object]:
    """Compute post-hoc threshold analysis on eligible OOF predictions."""

    best: Dict[str, object] | None = None
    for threshold in thresholds:
        predictions = (df["probability_nonforward"].to_numpy() >= threshold).astype(int)
        deployable = compute_regret(
            df,
            predictions,
            fixed_action_col,
            use_best_competitor=False,
        )
        metrics = _binary_metric_block(
            df["use_nonforward"],
            predictions,
            df["probability_nonforward"].to_numpy(),
        )
        candidate = {
            "threshold": float(threshold),
            "deployable_regret_mean_excess": deployable["mean_excess_expansions"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }
        if best is None:
            best = candidate
            continue
        if candidate["deployable_regret_mean_excess"] < best["deployable_regret_mean_excess"]:
            best = candidate
            continue
        if (
            candidate["deployable_regret_mean_excess"]
            == best["deployable_regret_mean_excess"]
            and candidate["balanced_accuracy"] > best["balanced_accuracy"]
        ):
            best = candidate
    return best or {}


def align_oof_predictions(
    eligible_df: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align OOF predictions back to eligible row order."""

    validate_columns(
        oof_df,
        [
            "row_id",
            "y_true",
            "is_tie",
            "probability_nonforward",
            "prediction_main",
            "prediction_always_forward",
        ],
    )
    if int(oof_df["row_id"].duplicated().sum()):
        raise ValueError("OOF prediction rows contain duplicate row_id values")

    aligned_oof = oof_df.sort_values("row_id", kind="stable").reset_index(drop=True)
    expected_row_ids = np.arange(len(eligible_df))
    if not np.array_equal(aligned_oof["row_id"].to_numpy(), expected_row_ids):
        raise ValueError("OOF prediction row_id values do not cover all eligible rows")

    eligible_aligned = eligible_df.reset_index(drop=True).copy()
    if not np.array_equal(
        aligned_oof["y_true"].astype(int).to_numpy(),
        eligible_aligned["use_nonforward"].astype(int).to_numpy(),
    ):
        raise ValueError("OOF y_true values do not align with eligible_df labels")
    if not np.array_equal(
        aligned_oof["is_tie"].astype(int).to_numpy(),
        eligible_aligned["is_tie"].astype(int).to_numpy(),
    ):
        raise ValueError("OOF is_tie values do not align with eligible_df rows")

    for column in [
        "fold",
        "probability_nonforward",
        "prediction_main",
        "prediction_always_forward",
    ]:
        eligible_aligned[column] = aligned_oof[column].to_numpy()
    return eligible_aligned


def compute_regret(
    df: pd.DataFrame,
    predictions: np.ndarray,
    fixed_action_col: str,
    use_best_competitor: bool,
) -> Dict[str, float]:
    """Compute regret metrics for a prediction vector."""

    forward = pd.to_numeric(df["forward_expansions"], errors="coerce").to_numpy()
    oracle = pd.to_numeric(df["oracle_expansions"], errors="coerce").to_numpy()
    competitor_available = (
        pd.to_numeric(df["competitor_available"], errors="coerce").fillna(0).astype(int).to_numpy()
    )
    selected = forward.copy()

    if use_best_competitor:
        competitor = pd.to_numeric(df["best_competitor_expansions"], errors="coerce").to_numpy()
        use_competitor = (predictions == 1) & (competitor_available == 1)
        selected[use_competitor] = competitor[use_competitor]
    else:
        fixed_action = pd.to_numeric(df[fixed_action_col], errors="coerce").to_numpy()
        use_fixed = (
            (predictions == 1)
            & (competitor_available == 1)
            & np.isfinite(fixed_action)
        )
        selected[use_fixed] = fixed_action[use_fixed]

    excess = selected - oracle
    normalized = excess / np.maximum(oracle, 1.0)
    return {
        "mean_excess_expansions": float(np.mean(excess)),
        "median_excess_expansions": float(np.median(excess)),
        "mean_normalized_regret_ratio": float(np.mean(normalized)),
    }


def build_population_predictions(
    baseline_df: pd.DataFrame,
    eligible_predictions_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Map eligible OOF predictions back onto the baseline-present population."""

    predictions = eligible_predictions_df.loc[
        :, KEY_COLS + ["prediction_main", "probability_nonforward"]
    ].copy()
    predictions["prediction_thresholded"] = (
        predictions["probability_nonforward"] >= threshold
    ).astype(int)
    baseline_with_predictions = baseline_df.merge(predictions, on=KEY_COLS, how="left")
    baseline_with_predictions["prediction_thresholded"] = np.where(
        baseline_with_predictions["competitor_available"].eq(0),
        0,
        pd.to_numeric(
            baseline_with_predictions["prediction_thresholded"],
            errors="coerce",
        ).fillna(0),
    ).astype(int)
    baseline_with_predictions["probability_nonforward"] = np.where(
        baseline_with_predictions["competitor_available"].eq(0),
        0.0,
        pd.to_numeric(
            baseline_with_predictions["probability_nonforward"],
            errors="coerce",
        ).fillna(0.0),
    )
    return baseline_with_predictions


def main(argv: List[str] | None = None) -> int:
    """Run the heuristic-aware practical classifier pipeline."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = SCENARIOS[args.scenario]
    fixed_action_col = str(config["fixed_action"])

    baseline_df, eligible_df = prepare_dataset(
        scenario_csv=args.scenario_csv,
        group_col=args.group_col,
    )
    validate_columns(eligible_df, [fixed_action_col] + list(config["competitors"]))
    X = eligible_df.loc[:, FAST_FEATURES]
    y = eligible_df["use_nonforward"].astype(int)
    groups = eligible_df[args.group_col].astype(str)
    is_tie = eligible_df["is_tie"].astype(int)

    tree_fold_df, tree_oof_df, _, tree_model, _ = run_model_cv(
        model_name="decision_tree",
        X=X,
        y=y,
        groups=groups,
        is_tie=is_tie,
        n_splits=args.n_splits,
        max_depth=args.max_depth,
    )
    hgb_fold_df, hgb_oof_df, _, hgb_model, hgb_importance_fold_df = run_model_cv(
        model_name="hgb",
        X=X,
        y=y,
        groups=groups,
        is_tie=is_tie,
        n_splits=args.n_splits,
        max_depth=args.max_depth,
    )

    tree_rules = export_text(tree_model, feature_names=FAST_FEATURES, decimals=6)
    tree_importance_df = (
        pd.DataFrame({"feature": FAST_FEATURES, "importance": tree_model.feature_importances_})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    hgb_importance_df = (
        hgb_importance_fold_df.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    tree_eligible_predictions = align_oof_predictions(eligible_df, tree_oof_df)
    hgb_eligible_predictions = align_oof_predictions(eligible_df, hgb_oof_df)
    tree_population = build_population_predictions(
        baseline_df=baseline_df,
        eligible_predictions_df=tree_eligible_predictions,
        threshold=0.5,
    )
    hgb_population = build_population_predictions(
        baseline_df=baseline_df,
        eligible_predictions_df=hgb_eligible_predictions,
        threshold=0.5,
    )

    # Main OOF regret summaries.
    tree_regret_eligible_deployable = compute_regret(
        tree_eligible_predictions,
        tree_eligible_predictions["prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    tree_regret_eligible_best_competitor_switch = compute_regret(
        tree_eligible_predictions,
        tree_eligible_predictions["prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )
    hgb_regret_eligible_deployable = compute_regret(
        hgb_eligible_predictions,
        hgb_eligible_predictions["prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    hgb_regret_eligible_best_competitor_switch = compute_regret(
        hgb_eligible_predictions,
        hgb_eligible_predictions["prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )

    non_tie_mask = eligible_df["is_tie"].eq(0).to_numpy()
    tree_regret_non_tie_deployable = compute_regret(
        tree_eligible_predictions.loc[non_tie_mask],
        tree_eligible_predictions.loc[non_tie_mask, "prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    tree_regret_non_tie_best_competitor_switch = compute_regret(
        tree_eligible_predictions.loc[non_tie_mask],
        tree_eligible_predictions.loc[non_tie_mask, "prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )
    hgb_regret_non_tie_deployable = compute_regret(
        hgb_eligible_predictions.loc[non_tie_mask],
        hgb_eligible_predictions.loc[non_tie_mask, "prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    hgb_regret_non_tie_best_competitor_switch = compute_regret(
        hgb_eligible_predictions.loc[non_tie_mask],
        hgb_eligible_predictions.loc[non_tie_mask, "prediction_main"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )

    tree_regret_deployment_deployable = compute_regret(
        tree_population,
        tree_population["prediction_thresholded"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    tree_regret_deployment_best_competitor_switch = compute_regret(
        tree_population,
        tree_population["prediction_thresholded"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )
    hgb_regret_deployment_deployable = compute_regret(
        hgb_population,
        hgb_population["prediction_thresholded"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=False,
    )
    hgb_regret_deployment_best_competitor_switch = compute_regret(
        hgb_population,
        hgb_population["prediction_thresholded"].astype(int).to_numpy(),
        fixed_action_col=fixed_action_col,
        use_best_competitor=True,
    )

    tree_threshold_posthoc = threshold_scan(
        tree_eligible_predictions,
        fixed_action_col=fixed_action_col,
        thresholds=np.linspace(0.0, 1.0, 101),
    )
    hgb_threshold_posthoc = threshold_scan(
        hgb_eligible_predictions,
        fixed_action_col=fixed_action_col,
        thresholds=np.linspace(0.0, 1.0, 101),
    )

    fixed_action_fallback_count = int(
        eligible_df[fixed_action_col].isna().sum()
    )
    fixed_action_fallback_positive_rows = int(
        (
            eligible_df[fixed_action_col].isna()
            & eligible_df["use_nonforward"].eq(1)
        ).sum()
    )

    # Deployment-accounting classification metrics.
    def _population_metric_block(pop_df: pd.DataFrame) -> Dict[str, float]:
        return _binary_metric_block(
            pop_df["use_nonforward"].astype(int),
            pop_df["prediction_thresholded"].astype(int).to_numpy(),
            pop_df["probability_nonforward"].to_numpy(),
        )

    def _always_forward_population_metric_block(pop_df: pd.DataFrame) -> Dict[str, float]:
        return _binary_metric_block(
            pop_df["use_nonforward"].astype(int),
            np.zeros(len(pop_df), dtype=int),
            np.zeros(len(pop_df), dtype=float),
        )

    tree_population_metrics = _population_metric_block(tree_population)
    hgb_population_metrics = _population_metric_block(hgb_population)
    population_always_forward_metrics = _always_forward_population_metric_block(baseline_df)

    (args.out_dir / "practical_tree_cv_metrics.csv").write_text(
        tree_fold_df.to_csv(index=False),
        encoding="utf-8",
    )
    (args.out_dir / "practical_hgb_cv_metrics.csv").write_text(
        hgb_fold_df.to_csv(index=False),
        encoding="utf-8",
    )
    tree_importance_df.to_csv(
        args.out_dir / "practical_tree_feature_importances.csv",
        index=False,
    )
    hgb_importance_df.to_csv(
        args.out_dir / "practical_hgb_permutation_importance.csv",
        index=False,
    )
    (args.out_dir / "practical_tree_rules.txt").write_text(tree_rules, encoding="utf-8")

    summary = {
        "scenario": args.scenario,
        "scenario_csv": str(args.scenario_csv.resolve()),
        "group_col": args.group_col,
        "n_splits": int(args.n_splits),
        "max_depth": int(args.max_depth),
        "forward_baseline_method": str(config["baseline"]),
        "competitor_pool": list(config["competitors"]),
        "fixed_nonforward_action": fixed_action_col,
        "threshold_policy_main": 0.5,
        "threshold_policy_posthoc": "exploratory_oof_regret_optimal",
        "rows_baseline_present": int(len(baseline_df)),
        "rows_eligible": int(len(eligible_df)),
        "rows_no_competitor": int((baseline_df["competitor_available"] == 0).sum()),
        "groups_used": int(groups.nunique()),
        "positive_class_rate_eligible": float(y.mean()),
        "positive_class_rate_baseline_present": float(baseline_df["use_nonforward"].mean()),
        "tie_rate_eligible": float(eligible_df["is_tie"].mean()),
        "fixed_action_fallback_count_eligible": fixed_action_fallback_count,
        "fixed_action_fallback_positive_rows_eligible": fixed_action_fallback_positive_rows,
        "best_competitor_switch_regret_note": (
            "If the classifier predicts non-forward, this variant routes to the row's "
            "best available competitor regardless of deployability; it is not a "
            "guaranteed upper bound on deployable regret."
        ),
        "always_forward_baseline_eligible": {
            **summarize_fold_metrics(
                tree_fold_df.loc[
                    :,
                    [
                        "always_forward_accuracy",
                        "always_forward_balanced_accuracy",
                        "always_forward_f1",
                        "always_forward_roc_auc",
                        "always_forward_pr_auc",
                        "always_forward_roc_auc_valid",
                        "always_forward_pr_auc_valid",
                    ]
                ].rename(
                    columns=lambda c: c.replace("always_forward_", "")
                )
            ),
        },
        "always_forward_baseline_baseline_present": population_always_forward_metrics,
        "decision_tree": {
            "model_name": "DecisionTreeClassifier",
            "eligible_metrics": summarize_fold_metrics(tree_fold_df),
            "eligible_non_tie_metrics": summarize_fold_metrics(tree_fold_df, prefix="non_tie_"),
            "deployment_accounting_metrics": tree_population_metrics,
            "deployable_regret_eligible": tree_regret_eligible_deployable,
            "deployable_regret_non_tie": tree_regret_non_tie_deployable,
            "deployable_regret_baseline_present": tree_regret_deployment_deployable,
            "best_competitor_switch_regret_eligible": tree_regret_eligible_best_competitor_switch,
            "best_competitor_switch_regret_non_tie": tree_regret_non_tie_best_competitor_switch,
            "best_competitor_switch_regret_baseline_present": tree_regret_deployment_best_competitor_switch,
            "posthoc_threshold_analysis": tree_threshold_posthoc,
            "top_feature_importances": tree_importance_df.head(10).to_dict(orient="records"),
        },
        "hist_gradient_boosting": {
            "model_name": "HistGradientBoostingClassifier",
            "eligible_metrics": summarize_fold_metrics(hgb_fold_df),
            "eligible_non_tie_metrics": summarize_fold_metrics(hgb_fold_df, prefix="non_tie_"),
            "deployment_accounting_metrics": hgb_population_metrics,
            "deployable_regret_eligible": hgb_regret_eligible_deployable,
            "deployable_regret_non_tie": hgb_regret_non_tie_deployable,
            "deployable_regret_baseline_present": hgb_regret_deployment_deployable,
            "best_competitor_switch_regret_eligible": hgb_regret_eligible_best_competitor_switch,
            "best_competitor_switch_regret_non_tie": hgb_regret_non_tie_best_competitor_switch,
            "best_competitor_switch_regret_baseline_present": hgb_regret_deployment_best_competitor_switch,
            "posthoc_threshold_analysis": hgb_threshold_posthoc,
            "top_cv_permutation_features": hgb_importance_df.head(10).to_dict(orient="records"),
        },
    }
    (args.out_dir / "practical_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Scenario: {args.scenario}")
    print(f"Rows baseline-present: {len(baseline_df)}")
    print(f"Rows eligible: {len(eligible_df)}")
    print(f"Groups used: {groups.nunique()}")
    print(
        "DecisionTreeClassifier eligible balanced_accuracy:",
        f"{tree_fold_df['balanced_accuracy'].mean():.6f}",
    )
    print(
        "HistGradientBoostingClassifier eligible balanced_accuracy:",
        f"{hgb_fold_df['balanced_accuracy'].mean():.6f}",
    )
    print(tree_rules)
    print(f"Wrote: {args.out_dir / 'practical_tree_cv_metrics.csv'}")
    print(f"Wrote: {args.out_dir / 'practical_hgb_cv_metrics.csv'}")
    print(f"Wrote: {args.out_dir / 'practical_tree_feature_importances.csv'}")
    print(f"Wrote: {args.out_dir / 'practical_hgb_permutation_importance.csv'}")
    print(f"Wrote: {args.out_dir / 'practical_tree_rules.txt'}")
    print(f"Wrote: {args.out_dir / 'practical_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
