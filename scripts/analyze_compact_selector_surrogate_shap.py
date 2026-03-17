#!/usr/bin/env python3
"""Run a compact-selector surrogate model with bounded SHAP analysis.

This is a lightweight follow-up to the compact selector analysis:
- uses only the top numeric apriori features
- grouped split by model_id
- computes compact-class performance
- emits SHAP summaries that are cheap enough to run reliably
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
TOP_FEATURES = 10
FIT_SAMPLE_MAX = 20000
SHAP_SAMPLE_MAX = 800


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact selector surrogate SHAP analysis")
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def pick_top_numeric_features(trace_df: pd.DataFrame, importance_df: pd.DataFrame, n: int) -> List[str]:
    numeric_cols = {c for c in trace_df.columns if pd.api.types.is_numeric_dtype(trace_df[c])}
    banned = {
        "best_time_seconds",
        "n_method_rows",
        "n_methods_tested",
        "n_methods_ok",
        "n_methods_timeout",
        "any_timeout",
    }
    selected: List[str] = []
    for feature in importance_df["feature"]:
        if feature in numeric_cols and feature not in banned:
            selected.append(feature)
        if len(selected) >= n:
            break
    return selected


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace_df = pd.read_csv(args.analysis_dir / "compact_selector_current" / "selection_trace_table_apriori_compact.csv")
    importance_df = pd.read_csv(args.analysis_dir / "compact_selector_current" / "compact_selector_feature_importance_reduced.csv")

    work = trace_df[trace_df["best_method_compact"].notna()].copy()
    feature_cols = pick_top_numeric_features(work, importance_df, TOP_FEATURES)

    if not feature_cols:
        raise SystemExit("No usable numeric features found for surrogate SHAP analysis")

    if len(work) > FIT_SAMPLE_MAX:
        work = work.sample(FIT_SAMPLE_MAX, random_state=RANDOM_STATE)

    X = work[feature_cols].copy()
    y = work["best_method_compact"].copy()
    groups = work["model_id"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train_i, y_train)

    y_pred = clf.predict(X_test_i)
    label_encoder = LabelEncoder().fit(y_train)
    proba = clf.predict_proba(X_test_i)
    top3 = top_k_accuracy_score(
        label_encoder.transform(y_test),
        proba,
        k=min(3, proba.shape[1]),
        labels=np.arange(proba.shape[1]),
    )

    # Global RF importances on the surrogate
    rf_importance = (
        pd.DataFrame({"feature": feature_cols, "importance_mean": clf.feature_importances_})
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    rf_importance.to_csv(args.out_dir / "surrogate_feature_importance.csv", index=False)

    # Bounded SHAP on a subset of test rows
    shap_n = min(SHAP_SAMPLE_MAX, X_test_i.shape[0])
    shap_idx = np.random.default_rng(RANDOM_STATE).choice(X_test_i.shape[0], shap_n, replace=False)
    X_shap = X_test_i[shap_idx]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_shap)

    if isinstance(sv, list):
        arr = np.stack(sv, axis=0)  # classes, samples, features
    else:
        arr = np.asarray(sv)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[0] == X_shap.shape[0]:
            arr = np.moveaxis(arr, 2, 0)

    global_mean_abs = np.abs(arr).mean(axis=(0, 1))
    shap_global = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": global_mean_abs})
        .sort_values("mean_abs_shap", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    shap_global.to_csv(args.out_dir / "surrogate_shap_global.csv", index=False)

    per_class_rows = []
    classes = clf.classes_.tolist()
    for class_idx, class_name in enumerate(classes):
        class_mean_abs = np.abs(arr[class_idx]).mean(axis=0)
        class_df = (
            pd.DataFrame({"feature": feature_cols, "mean_abs_shap": class_mean_abs})
            .sort_values("mean_abs_shap", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        for _, row in class_df.iterrows():
            per_class_rows.append(
                {
                    "target_class": class_name,
                    "feature": row["feature"],
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                }
            )
    pd.DataFrame(per_class_rows).to_csv(args.out_dir / "surrogate_shap_per_class.csv", index=False)

    plot_written = False

    summary = {
        "selected_features": feature_cols,
        "rows_total": int(len(work)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rows_used_for_shap": int(shap_n),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "top3_accuracy": float(top3),
        "classes": classes,
        "top_rf_features": rf_importance.head(10).to_dict(orient="records"),
        "top_shap_features": shap_global.head(10).to_dict(orient="records"),
        "plot_written": plot_written,
    }
    (args.out_dir / "surrogate_summary.json").write_text(json.dumps(summary, indent=2))

    report_lines = [
        "# Compact Selector Surrogate SHAP",
        "",
        "## Features",
        *[f"- {f}" for f in feature_cols],
        "",
        "## Performance",
        f"- Accuracy: {summary['accuracy']:.4f}",
        f"- Balanced accuracy: {summary['balanced_accuracy']:.4f}",
        f"- Top-3 accuracy: {summary['top3_accuracy']:.4f}",
        "",
        "## Top SHAP Features",
        *[f"- {rec['feature']}: {rec['mean_abs_shap']:.6f}" for rec in summary["top_shap_features"]],
    ]
    (args.out_dir / "surrogate_report.md").write_text("\n".join(report_lines) + "\n")

    print(f"Wrote: {args.out_dir / 'surrogate_feature_importance.csv'}")
    print(f"Wrote: {args.out_dir / 'surrogate_shap_global.csv'}")
    print(f"Wrote: {args.out_dir / 'surrogate_shap_per_class.csv'}")
    print(f"Wrote: {args.out_dir / 'surrogate_summary.json'}")
    print(f"Wrote: {args.out_dir / 'surrogate_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
