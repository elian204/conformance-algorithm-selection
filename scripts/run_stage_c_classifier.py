#!/usr/bin/env python3
"""Run the Stage C practitioner routing classifiers."""

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


RANDOM_STATE = 42
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the Stage C routing classifiers",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_full.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_smoke/stage_c_classifier"),
    )
    parser.add_argument(
        "--target-col",
        default="dibbs_zero_vs_forward_zero_expansions_ratio",
    )
    parser.add_argument(
        "--group-col",
        default="dataset_name",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )
    return parser


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Ensure the expected columns are present."""

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_dataset(
    features_csv: Path,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Load and prepare the routing dataset."""

    df = pd.read_csv(features_csv)
    validate_columns(df, FAST_FEATURES + [target_col, group_col])
    original_rows = len(df)
    work = df.dropna(subset=[target_col, group_col]).copy()
    rows_after_target_filter = int(len(work))
    work = work.dropna(subset=FAST_FEATURES, how="all").copy()
    rows_after_feature_filter = int(len(work))
    ratios = pd.to_numeric(work[target_col], errors="coerce")
    work = work.loc[ratios.notna()].copy()
    work["use_dibbs"] = (ratios.loc[work.index] < 1.0).astype(int)
    work.attrs["rows_original"] = int(original_rows)
    work.attrs["rows_after_target_filter"] = rows_after_target_filter
    work.attrs["rows_after_feature_filter"] = rows_after_feature_filter
    work.attrs["consistency_policy"] = "none"
    return work.reset_index(drop=True)


def fit_tree_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int,
) -> tuple[SimpleImputer, DecisionTreeClassifier]:
    """Fit the shallow tree classifier."""

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_i, y_train)
    return imputer, model


def fit_hgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[SimpleImputer, HistGradientBoostingClassifier]:
    """Fit the HGB routing classifier."""

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


def _compute_binary_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute fold-level binary classification metrics."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if pd.Series(y_true).nunique() < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def run_tree_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    max_depth: int,
) -> tuple[pd.DataFrame, SimpleImputer, DecisionTreeClassifier]:
    """Run grouped CV for the shallow tree."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold, found {groups.nunique()}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        imputer, model = fit_tree_classifier(X_train, y_train, max_depth=max_depth)
        X_test_i = imputer.transform(X_test)
        predictions = model.predict(X_test_i)
        probabilities = model.predict_proba(X_test_i)[:, 1]
        metrics = _compute_binary_metrics(y_test, predictions, probabilities)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                **metrics,
                "positive_rate_test": float(y_test.mean()),
                "positive_rate_pred": float(np.mean(predictions)),
            }
        )

    final_imputer = SimpleImputer(strategy="median")
    X_full_i = final_imputer.fit_transform(X)
    final_model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_full_i, y)
    return pd.DataFrame(fold_rows), final_imputer, final_model


def run_hgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped CV for HGB and collect permutation importances."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold, found {groups.nunique()}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, float]] = []
    importance_rows: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        imputer, model = fit_hgb_classifier(X_train, y_train)
        X_test_i = imputer.transform(X_test)
        predictions = model.predict(X_test_i)
        probabilities = model.predict_proba(X_test_i)[:, 1]
        metrics = _compute_binary_metrics(y_test, predictions, probabilities)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                **metrics,
                "positive_rate_test": float(y_test.mean()),
                "positive_rate_pred": float(np.mean(predictions)),
            }
        )

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

    return pd.DataFrame(fold_rows), pd.concat(importance_rows, ignore_index=True)


def summarize_metrics(fold_df: pd.DataFrame) -> Dict[str, float]:
    """Build mean/std summary metrics."""

    return {
        "cv_accuracy_mean": float(fold_df["accuracy"].mean()),
        "cv_accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "cv_balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=0)),
        "cv_f1_mean": float(fold_df["f1"].mean()),
        "cv_f1_std": float(fold_df["f1"].std(ddof=0)),
        "cv_roc_auc_mean": float(fold_df["roc_auc"].mean()),
        "cv_roc_auc_std": float(fold_df["roc_auc"].std(ddof=0)),
    }


def main(argv: List[str] | None = None) -> int:
    """Run the routing classifiers and write artifacts."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_df = prepare_dataset(
        features_csv=args.features_csv,
        target_col=args.target_col,
        group_col=args.group_col,
    )
    X = model_df.loc[:, FAST_FEATURES]
    y = model_df["use_dibbs"].astype(int)
    groups = model_df[args.group_col].astype(str)

    tree_fold_df, _, tree_model = run_tree_cv(
        X=X,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
        max_depth=args.max_depth,
    )
    hgb_fold_df, hgb_importance_fold_df = run_hgb_cv(
        X=X,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
    )

    tree_rules = export_text(
        tree_model,
        feature_names=FAST_FEATURES,
        decimals=6,
    )
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

    (args.out_dir / "classifier_tree_cv_metrics.csv").write_text(
        tree_fold_df.to_csv(index=False),
        encoding="utf-8",
    )
    (args.out_dir / "classifier_hgb_cv_metrics.csv").write_text(
        hgb_fold_df.to_csv(index=False),
        encoding="utf-8",
    )
    tree_importance_df.to_csv(args.out_dir / "classifier_tree_feature_importances.csv", index=False)
    hgb_importance_df.to_csv(
        args.out_dir / "classifier_hgb_permutation_importance.csv",
        index=False,
    )
    (args.out_dir / "classifier_tree_rules.txt").write_text(tree_rules, encoding="utf-8")

    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "target_col": args.target_col,
        "binary_target": "use_dibbs",
        "group_col": args.group_col,
        "n_splits": int(args.n_splits),
        "max_depth": int(args.max_depth),
        "rows_original": int(model_df.attrs["rows_original"]),
        "rows_after_target_filter": int(model_df.attrs["rows_after_target_filter"]),
        "rows_after_feature_filter": int(model_df.attrs["rows_after_feature_filter"]),
        "rows_used": int(len(model_df)),
        "groups_used": int(groups.nunique()),
        "consistency_policy": str(model_df.attrs["consistency_policy"]),
        "positive_class_rate": float(y.mean()),
        "feature_cols": FAST_FEATURES,
        "decision_tree": {
            "model_name": "DecisionTreeClassifier",
            **summarize_metrics(tree_fold_df),
            "top_feature_importances": tree_importance_df.head(10).to_dict(orient="records"),
        },
        "hist_gradient_boosting": {
            "model_name": "HistGradientBoostingClassifier",
            **summarize_metrics(hgb_fold_df),
            "top_cv_permutation_features": hgb_importance_df.head(10).to_dict(orient="records"),
        },
    }
    (args.out_dir / "classifier_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Rows used: {len(model_df)}")
    print(f"Groups used: {groups.nunique()}")
    print(f"Positive class rate: {y.mean():.6f}")
    print("DecisionTreeClassifier:")
    print(f"  CV accuracy: {tree_fold_df['accuracy'].mean():.6f}")
    print(f"  CV balanced_accuracy: {tree_fold_df['balanced_accuracy'].mean():.6f}")
    print(f"  CV f1: {tree_fold_df['f1'].mean():.6f}")
    print(f"  CV roc_auc: {tree_fold_df['roc_auc'].mean():.6f}")
    print("HistGradientBoostingClassifier:")
    print(f"  CV accuracy: {hgb_fold_df['accuracy'].mean():.6f}")
    print(f"  CV balanced_accuracy: {hgb_fold_df['balanced_accuracy'].mean():.6f}")
    print(f"  CV f1: {hgb_fold_df['f1'].mean():.6f}")
    print(f"  CV roc_auc: {hgb_fold_df['roc_auc'].mean():.6f}")
    print()
    print(tree_rules)
    print(f"Wrote: {args.out_dir / 'classifier_tree_cv_metrics.csv'}")
    print(f"Wrote: {args.out_dir / 'classifier_hgb_cv_metrics.csv'}")
    print(f"Wrote: {args.out_dir / 'classifier_tree_feature_importances.csv'}")
    print(f"Wrote: {args.out_dir / 'classifier_hgb_permutation_importance.csv'}")
    print(f"Wrote: {args.out_dir / 'classifier_tree_rules.txt'}")
    print(f"Wrote: {args.out_dir / 'classifier_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
