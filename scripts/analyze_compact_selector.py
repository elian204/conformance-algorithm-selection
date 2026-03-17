#!/usr/bin/env python3
"""Build a compact winner target and analyze apriori selection features.

Outputs:
- compact apriori trace table
- feature importance summaries for:
  - coarse apriori model (includes fitness/precision/miner metadata)
  - reduced apriori model (drops dominant coarse metadata)
- partial dependence data and plots for top reduced numeric features
- markdown and JSON summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


RANDOM_STATE = 42
PERMUTATION_SAMPLE_MAX = 12000
PD_SAMPLE_MAX = 15000
SHAP_BACKGROUND_MAX = 500
SHAP_SAMPLE_MAX = 2000
MAIN_CLASSES = [
    "dibbs_zero",
    "forward_zero",
    "bidir_mm_zero",
    "backward_zero",
    "bidir_std_zero",
]
ID_COLS = {
    "dataset_name",
    "model_id",
    "model_name",
    "model_path",
    "trace_id",
    "trace_hash",
    "aggregate_run_name",
    "model_source",
    "trace_activities",
}
TARGET_COLS = {
    "best_method",
    "best_method_compact",
    "best_time_seconds",
    "n_method_rows",
    "n_methods_tested",
    "n_methods_ok",
    "n_methods_timeout",
    "any_timeout",
}
COARSE_DROP_COLS = set()
REDUCED_DROP_COLS = {
    "model_fitness",
    "model_precision",
    "miner_family",
    "miner_parameter_name",
    "miner_parameter_value",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze compact best-method selector")
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def map_compact_method(method: object) -> object:
    if pd.isna(method):
        return np.nan
    method = str(method)
    return method if method in MAIN_CLASSES else "other"


def split_feature_types(df: pd.DataFrame, drop_cols: set[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if col in ID_COLS or col in TARGET_COLS or col in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def drop_all_missing_columns(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    dropped: List[str] = []
    keep_numeric: List[str] = []
    for col in numeric_cols:
        if df[col].notna().any():
            keep_numeric.append(col)
        else:
            dropped.append(col)
    keep_categorical: List[str] = []
    for col in categorical_cols:
        if df[col].notna().any():
            keep_categorical.append(col)
        else:
            dropped.append(col)
    keep_cols = keep_numeric + keep_categorical
    return df[keep_cols].copy(), keep_numeric, keep_categorical, dropped


def aggregate_importances(
    importances: np.ndarray,
    feature_names: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    aggregated: Dict[str, float] = {}
    for name, value in zip(feature_names, importances):
        if name.startswith("num__"):
            base = name[len("num__") :]
        elif name.startswith("cat__"):
            rest = name[len("cat__") :]
            base = rest
            for col in categorical_cols:
                prefix = f"{col}_"
                if rest.startswith(prefix):
                    base = col
                    break
        else:
            base = name
        aggregated[base] = aggregated.get(base, 0.0) + float(value)

    return (
        pd.DataFrame(
            [{"feature": k, "importance_mean": v} for k, v in aggregated.items()]
        )
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


def fit_and_score(
    df: pd.DataFrame,
    drop_cols: set[str],
    split_kind: str,
    groups: pd.Series | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object], Pipeline, pd.DataFrame, pd.Series, List[str]]:
    work = df[df["best_method_compact"].notna()].copy()
    numeric_cols, categorical_cols = split_feature_types(work, drop_cols)
    X = work[numeric_cols + categorical_cols]
    y = work["best_method_compact"]

    X, numeric_cols, categorical_cols, dropped_missing = drop_all_missing_columns(
        X, numeric_cols, categorical_cols
    )

    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
    else:
        groups = groups.loc[X.index]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    label_encoder = LabelEncoder().fit(y_train)
    proba = pipe.predict_proba(X_test)
    top3 = top_k_accuracy_score(
        label_encoder.transform(y_test),
        proba,
        k=min(3, proba.shape[1]),
        labels=np.arange(proba.shape[1]),
    )

    transformed_feature_names = pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
    if len(X_test) > PERMUTATION_SAMPLE_MAX:
        sample_idx = X_test.sample(PERMUTATION_SAMPLE_MAX, random_state=RANDOM_STATE).index
        X_perm = X_test.loc[sample_idx]
        y_perm = y_test.loc[sample_idx]
    else:
        X_perm = X_test
        y_perm = y_test

    importances = pipe.named_steps["classifier"].feature_importances_
    importance_df = aggregate_importances(
        importances,
        transformed_feature_names,
        categorical_cols,
    )

    summary = {
        "split_kind": split_kind,
        "rows_total": int(len(y)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "class_counts": y.value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "top3_accuracy": float(top3),
        "dropped_all_missing_features": dropped_missing,
        "top_features": importance_df.head(15).to_dict(orient="records"),
    }
    return importance_df, summary, pipe, X, y, numeric_cols


def partial_dependence_outputs(
    pipe: Pipeline,
    X: pd.DataFrame,
    numeric_cols: List[str],
    top_features: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    use_features = [f for f in top_features if f in numeric_cols][:6]
    if not use_features:
        return pd.DataFrame(columns=["feature", "target_class", "grid_value", "average_probability"])

    if len(X) > PD_SAMPLE_MAX:
        X = X.sample(PD_SAMPLE_MAX, random_state=RANDOM_STATE)

    target_classes = pipe.named_steps["classifier"].classes_.tolist()
    for feature in use_features:
        series = X[feature].dropna()
        if series.empty:
            continue
        lo = float(series.quantile(0.05))
        hi = float(series.quantile(0.95))
        if np.isclose(lo, hi):
            grid = np.array([lo])
        else:
            grid = np.linspace(lo, hi, 15)

        feature_rows: List[Dict[str, object]] = []
        for grid_value in grid:
            X_mod = X.copy()
            X_mod[feature] = grid_value
            proba = pipe.predict_proba(X_mod)
            avg_proba = proba.mean(axis=0)
            for class_idx, class_name in enumerate(target_classes):
                feature_rows.append(
                    {
                        "feature": feature,
                        "target_class": class_name,
                        "grid_value": float(grid_value),
                        "average_probability": float(avg_proba[class_idx]),
                    }
                )

        feature_df = pd.DataFrame(feature_rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        for class_name, sub in feature_df.groupby("target_class", sort=False):
            ax.plot(sub["grid_value"], sub["average_probability"], label=class_name)
        ax.set_title(f"Partial dependence: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Average predicted probability")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{feature}_partial_dependence.png", dpi=150)
        plt.close(fig)
        rows.extend(feature_rows)
    return pd.DataFrame(rows)


def write_markdown(
    out_path: Path,
    coarse_summary: Dict[str, object],
    reduced_summary: Dict[str, object],
    shap_top_features: List[Dict[str, object]],
) -> None:
    lines: List[str] = ["# Compact Selector Analysis", ""]
    lines.append("## Target")
    lines.append("- Classes: dibbs_zero, forward_zero, bidir_mm_zero, backward_zero, bidir_std_zero, other")
    lines.append("")
    lines.append("## Coarse Factors")
    lines.append("- `model_fitness` and `model_precision` should be read as dominant but coarse predictors.")
    lines.append("- They are useful for triage, but they mostly tell you which model regime you are in.")
    lines.append("")
    for label, summary in [("coarse_apriori", coarse_summary), ("reduced_apriori", reduced_summary)]:
        lines.append(f"## {label}")
        lines.append(f"- Accuracy: {summary['accuracy']:.4f}")
        lines.append(f"- Balanced accuracy: {summary['balanced_accuracy']:.4f}")
        lines.append(f"- Top-3 accuracy: {summary['top3_accuracy']:.4f}")
        lines.append("- Top features:")
        for rec in summary["top_features"][:10]:
            lines.append(f"  - {rec['feature']}: {rec['importance_mean']:.6f}")
        lines.append("")
    if shap_top_features:
        lines.append("## SHAP (reduced_apriori)")
        lines.append("- Top mean absolute SHAP features:")
        for rec in shap_top_features[:10]:
            lines.append(f"  - {rec['feature']}: {rec['mean_abs_shap']:.6f}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def compute_shap_outputs(
    pipe: Pipeline,
    X: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    if shap is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"]), []

    if len(X) > SHAP_SAMPLE_MAX:
        X_sample = X.sample(SHAP_SAMPLE_MAX, random_state=RANDOM_STATE)
    else:
        X_sample = X.copy()

    if len(X_sample) > SHAP_BACKGROUND_MAX:
        X_background = X_sample.sample(SHAP_BACKGROUND_MAX, random_state=RANDOM_STATE)
    else:
        X_background = X_sample

    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]
    X_background_t = pre.transform(X_background)
    X_sample_t = pre.transform(X_sample)
    feature_names = pre.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample_t)

    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
    mean_abs = np.abs(arr).mean(axis=(0, 1))

    shap_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    shap_df.to_csv(out_dir / "compact_selector_reduced_shap_transformed.csv", index=False)

    top_features = shap_df.head(20)["feature"].tolist()
    return shap_df, top_features


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    apriori_path = args.analysis_dir / "selection_trace_table_apriori.csv"
    df = pd.read_csv(apriori_path)
    df["best_method_compact"] = df["best_method"].map(map_compact_method)

    compact_table_path = args.out_dir / "selection_trace_table_apriori_compact.csv"
    df.to_csv(compact_table_path, index=False)

    grouped = df["model_id"]
    coarse_imp, coarse_summary, _, _, _, _ = fit_and_score(
        df, COARSE_DROP_COLS, "grouped_by_model_id", groups=grouped
    )
    reduced_imp, reduced_summary, reduced_pipe, reduced_X, _, reduced_numeric = fit_and_score(
        df, REDUCED_DROP_COLS, "grouped_by_model_id", groups=grouped
    )

    coarse_imp.to_csv(args.out_dir / "compact_selector_feature_importance_coarse.csv", index=False)
    reduced_imp.to_csv(args.out_dir / "compact_selector_feature_importance_reduced.csv", index=False)

    pd_dir = args.out_dir / "partial_dependence"
    pd_df = partial_dependence_outputs(
        reduced_pipe,
        reduced_X,
        reduced_numeric,
        reduced_imp["feature"].tolist(),
        pd_dir,
    )
    pd_df.to_csv(args.out_dir / "compact_selector_partial_dependence.csv", index=False)

    shap_df, _ = compute_shap_outputs(reduced_pipe, reduced_X, args.out_dir)

    summary = {
        "target_classes": MAIN_CLASSES + ["other"],
        "target_counts": df["best_method_compact"].value_counts(dropna=False).to_dict(),
        "coarse_apriori": coarse_summary,
        "reduced_apriori": reduced_summary,
        "partial_dependence_features": pd_df["feature"].drop_duplicates().tolist(),
        "shap_available": shap is not None,
        "shap_top_features": shap_df.head(15).to_dict(orient="records"),
        "notes": [
            "model_fitness and model_precision are dominant but coarse predictors",
            "reduced analysis removes model_fitness, model_precision, miner_family, and miner parameters",
            "SHAP is computed on the reduced apriori model using a bounded sample",
        ],
    }
    (args.out_dir / "compact_selector_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(
        args.out_dir / "compact_selector_report.md",
        coarse_summary,
        reduced_summary,
        shap_df.head(15).to_dict(orient="records"),
    )

    print(f"Wrote: {compact_table_path}")
    print(f"Wrote: {args.out_dir / 'compact_selector_feature_importance_coarse.csv'}")
    print(f"Wrote: {args.out_dir / 'compact_selector_feature_importance_reduced.csv'}")
    print(f"Wrote: {args.out_dir / 'compact_selector_partial_dependence.csv'}")
    if shap is not None:
        print(f"Wrote: {args.out_dir / 'compact_selector_reduced_shap_transformed.csv'}")
    print(f"Wrote: {args.out_dir / 'compact_selector_summary.json'}")
    print(f"Wrote: {args.out_dir / 'compact_selector_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
