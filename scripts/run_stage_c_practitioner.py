#!/usr/bin/env python3
"""Run the Stage C practitioner baseline with a shallow grouped regression tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor, export_text


RANDOM_STATE = 42
DEFAULT_FEATURES = [
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
        description="Run the Stage C practitioner baseline",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_full.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_smoke/stage_c_practitioner"),
    )
    parser.add_argument(
        "--target-col",
        default="dibbs_vs_forward_expansion_ratio",
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
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log", "log1p", "clip"],
        default="raw",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--include-inconsistent",
        action="store_true",
        help="Keep rows flagged as potentially inconsistent bidirectional targets.",
    )
    parser.add_argument(
        "--consistency-policy",
        choices=["auto", "none", "dibbs_only", "mm_only", "bidirectional_any"],
        default="auto",
        help="How to filter potentially inconsistent heuristic configurations.",
    )
    return parser


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Ensure the input frame contains the required columns."""

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_consistency_policy(target_col: str) -> str:
    """Infer the appropriate consistency policy for the requested target."""

    if target_col.startswith("dibbs_zero_vs_forward_zero"):
        return "none"
    if target_col.startswith("dibbs_vs_forward"):
        return "dibbs_only"
    if target_col.startswith("bidir_mm"):
        return "mm_only"
    if target_col.startswith("best_bidirectional"):
        return "bidirectional_any"
    return "none"


def build_consistency_mask(df: pd.DataFrame, policy: str) -> pd.Series:
    """Build a target-specific consistency mask."""

    if policy == "none":
        return pd.Series(True, index=df.index)

    if policy == "dibbs_only":
        if "uses_potentially_inconsistent_dibbs_target" in df.columns:
            source = df["uses_potentially_inconsistent_dibbs_target"]
        elif "target_dibbs_method" in df.columns:
            source = df["target_dibbs_method"].astype("string").str.endswith("_me")
        else:
            raise ValueError(
                "Missing DIBBS consistency columns: expected "
                "'uses_potentially_inconsistent_dibbs_target' or 'target_dibbs_method'"
            )
    elif policy == "mm_only":
        if "uses_potentially_inconsistent_mm_target" in df.columns:
            source = df["uses_potentially_inconsistent_mm_target"]
        elif "bidir_mm_expansions_method" in df.columns:
            source = df["bidir_mm_expansions_method"].astype("string").str.endswith("_me")
        elif "target_mm_method" in df.columns:
            source = df["target_mm_method"].astype("string").str.endswith("_me")
        else:
            raise ValueError(
                "Missing MM consistency columns: expected one of "
                "'uses_potentially_inconsistent_mm_target', "
                "'bidir_mm_expansions_method', or 'target_mm_method'"
            )
    elif policy == "bidirectional_any":
        if "uses_potentially_inconsistent_bidirectional_target" not in df.columns:
            raise ValueError(
                "Missing bidirectional consistency column: "
                "'uses_potentially_inconsistent_bidirectional_target'"
            )
        source = df["uses_potentially_inconsistent_bidirectional_target"]
    else:
        raise ValueError(f"Unsupported consistency policy: {policy}")

    return pd.to_numeric(source, errors="coerce").fillna(0).eq(0)


def build_target_transform(
    transform: str,
    clip_min: float,
    clip_max: float,
) -> tuple[
    Callable[[pd.Series], pd.Series],
    Callable[[np.ndarray], np.ndarray],
    Dict[str, float | str],
]:
    """Build forward and inverse transforms for the target."""

    if transform == "raw":
        return (
            lambda s: s.astype(float),
            lambda arr: np.asarray(arr, dtype=float),
            {"target_transform": "raw"},
        )

    if transform == "log":
        eps = 1e-12
        return (
            lambda s: np.log(np.clip(s.astype(float), eps, None)),
            lambda arr: np.exp(np.asarray(arr, dtype=float)),
            {"target_transform": "log", "log_epsilon": eps},
        )

    if transform == "log1p":
        return (
            lambda s: np.log1p(np.clip(s.astype(float), 0.0, None)),
            lambda arr: np.expm1(np.asarray(arr, dtype=float)),
            {"target_transform": "log1p"},
        )

    if transform == "clip":
        if clip_min <= 0 or clip_max <= 0 or clip_min >= clip_max:
            raise ValueError("clip_min and clip_max must satisfy 0 < clip_min < clip_max")
        return (
            lambda s: np.clip(s.astype(float), clip_min, clip_max),
            lambda arr: np.asarray(arr, dtype=float),
            {
                "target_transform": "clip",
                "clip_min": float(clip_min),
                "clip_max": float(clip_max),
            },
        )

    raise ValueError(f"Unsupported target transform: {transform}")


def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    include_inconsistent: bool,
    consistency_policy: str,
) -> pd.DataFrame:
    """Prepare the modeling frame with strict consistency filtering."""

    required_cols = feature_cols + [target_col, group_col]
    validate_columns(df, required_cols)

    work = df.copy()
    original_rows = len(work)
    work = work.dropna(subset=[target_col, group_col])
    rows_after_target_filter = int(len(work))

    resolved_policy = "none" if include_inconsistent else (
        infer_consistency_policy(target_col)
        if consistency_policy == "auto"
        else consistency_policy
    )
    rows_after_consistency_filter = rows_after_target_filter
    if resolved_policy != "none":
        consistent_mask = build_consistency_mask(work, resolved_policy)
        work = work[consistent_mask].copy()
        rows_after_consistency_filter = int(len(work))

    work = work.dropna(subset=feature_cols, how="all").copy()
    work.attrs["rows_original"] = original_rows
    work.attrs["rows_after_target_filter"] = rows_after_target_filter
    work.attrs["consistency_policy"] = resolved_policy
    work.attrs["rows_after_consistency_filter"] = rows_after_consistency_filter
    return work.reset_index(drop=True)


def run_grouped_cv(
    X: pd.DataFrame,
    y_model: pd.Series,
    y_raw: pd.Series,
    groups: pd.Series,
    n_splits: int,
    max_depth: int,
    inverse_transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[pd.DataFrame, SimpleImputer, DecisionTreeRegressor]:
    """Run grouped cross-validation and return fold metrics."""

    unique_groups = groups.nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold, found {unique_groups}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y_model, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_model.iloc[train_idx]
        y_test = y_model.iloc[test_idx]
        y_test_raw = y_raw.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_i, y_train)
        predictions_model = model.predict(X_test_i)
        predictions_raw = inverse_transform(predictions_model)

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                "mae_transformed": float(mean_absolute_error(y_test, predictions_model)),
                "r2_transformed": float(r2_score(y_test, predictions_model)),
                "mae_raw": float(mean_absolute_error(y_test_raw, predictions_raw)),
                "r2_raw": float(r2_score(y_test_raw, predictions_raw)),
                "y_test_mean_transformed": float(y_test.mean()),
                "prediction_mean_transformed": float(np.mean(predictions_model)),
                "y_test_mean_raw": float(y_test_raw.mean()),
                "prediction_mean_raw": float(np.mean(predictions_raw)),
            }
        )

    final_imputer = SimpleImputer(strategy="median")
    X_full_i = final_imputer.fit_transform(X)
    final_model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_full_i, y_model)

    return pd.DataFrame(fold_rows), final_imputer, final_model


def main(argv: List[str] | None = None) -> int:
    """Run the practitioner baseline and write outputs."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    model_df = prepare_dataset(
        df=df,
        feature_cols=DEFAULT_FEATURES,
        target_col=args.target_col,
        group_col=args.group_col,
        include_inconsistent=args.include_inconsistent,
        consistency_policy=args.consistency_policy,
    )

    X = model_df.loc[:, DEFAULT_FEATURES]
    y_raw = pd.to_numeric(model_df[args.target_col], errors="coerce")
    transform_target, inverse_transform, transform_meta = build_target_transform(
        transform=args.target_transform,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    y_model = pd.Series(transform_target(y_raw), index=y_raw.index, dtype=float)
    groups = model_df[args.group_col].astype(str)

    fold_df, _, final_model = run_grouped_cv(
        X=X,
        y_model=y_model,
        y_raw=y_raw,
        groups=groups,
        n_splits=args.n_splits,
        max_depth=args.max_depth,
        inverse_transform=inverse_transform,
    )

    tree_rules = export_text(
        final_model,
        feature_names=DEFAULT_FEATURES,
        decimals=6,
    )

    feature_importance_df = (
        pd.DataFrame(
            {
                "feature": DEFAULT_FEATURES,
                "importance": final_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "target_col": args.target_col,
        "group_col": args.group_col,
        "n_splits": int(args.n_splits),
        "max_depth": int(args.max_depth),
        "include_inconsistent": bool(args.include_inconsistent),
        **transform_meta,
        "consistency_policy": str(model_df.attrs["consistency_policy"]),
        "rows_original": int(model_df.attrs["rows_original"]),
        "rows_after_target_filter": int(model_df.attrs["rows_after_target_filter"]),
        "rows_after_consistency_filter": int(model_df.attrs["rows_after_consistency_filter"]),
        "rows_used": int(len(model_df)),
        "groups_used": int(groups.nunique()),
        "target_mean_raw": float(y_raw.mean()),
        "target_median_raw": float(y_raw.median()),
        "target_mean_transformed": float(y_model.mean()),
        "target_median_transformed": float(y_model.median()),
        "target_bidirectional_win_rate": float((y_raw < 1.0).mean()),
        "cv_mae_transformed_mean": float(fold_df["mae_transformed"].mean()),
        "cv_mae_transformed_std": float(fold_df["mae_transformed"].std(ddof=0)),
        "cv_r2_transformed_mean": float(fold_df["r2_transformed"].mean()),
        "cv_r2_transformed_std": float(fold_df["r2_transformed"].std(ddof=0)),
        "cv_mae_raw_mean": float(fold_df["mae_raw"].mean()),
        "cv_mae_raw_std": float(fold_df["mae_raw"].std(ddof=0)),
        "cv_r2_raw_mean": float(fold_df["r2_raw"].mean()),
        "cv_r2_raw_std": float(fold_df["r2_raw"].std(ddof=0)),
        "top_feature_importances": feature_importance_df.head(10).to_dict(orient="records"),
    }

    fold_path = args.out_dir / "practitioner_cv_metrics.csv"
    rules_path = args.out_dir / "practitioner_tree_rules.txt"
    summary_path = args.out_dir / "practitioner_summary.json"
    feature_importance_path = args.out_dir / "practitioner_feature_importances.csv"

    fold_df.to_csv(fold_path, index=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)
    rules_path.write_text(tree_rules, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rows used: {len(model_df)} / {model_df.attrs['rows_original']}")
    print(f"Groups used: {groups.nunique()}")
    print(f"Target transform: {args.target_transform}")
    print(f"CV mean MAE (transformed): {fold_df['mae_transformed'].mean():.6f}")
    print(f"CV mean R2 (transformed): {fold_df['r2_transformed'].mean():.6f}")
    print(f"CV mean MAE (raw): {fold_df['mae_raw'].mean():.6f}")
    print(f"CV mean R2 (raw): {fold_df['r2_raw'].mean():.6f}")
    print()
    print(tree_rules)
    print(f"Wrote: {fold_path}")
    print(f"Wrote: {feature_importance_path}")
    print(f"Wrote: {rules_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
