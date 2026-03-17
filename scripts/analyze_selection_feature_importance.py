#!/usr/bin/env python3
"""Analyze feature importance for algorithm selection.

Produces two analyses:
- apriori: only features available before alignment
- full: apriori + post-hoc explanatory features

The input tables are the outputs of build_selection_analysis_tables.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


RARE_CLASS_MIN = 100
RANDOM_STATE = 42

ID_LIKE_COLS = {
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
    "best_time_seconds",
    "n_method_rows",
    "n_methods_tested",
    "n_methods_ok",
    "n_methods_timeout",
    "any_timeout",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze feature importance for method selection")
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def group_rare_classes(y: pd.Series, min_count: int = RARE_CLASS_MIN) -> pd.Series:
    counts = y.value_counts()
    rare = set(counts[counts < min_count].index)
    return y.apply(lambda v: "other_rare" if v in rare else v)


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in ID_LIKE_COLS or col in TARGET_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def drop_all_missing_columns(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    dropped = []
    keep_numeric = []
    for col in numeric_cols:
        if df[col].notna().any():
            keep_numeric.append(col)
        else:
            dropped.append(col)
    keep_categorical = []
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

    imp_df = (
        pd.DataFrame(
            [{"feature": k, "importance_mean": v} for k, v in aggregated.items()]
        )
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return imp_df


def _fit_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    split_kind: str,
    groups: pd.Series | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X_work, numeric_cols, categorical_cols, dropped_cols = drop_all_missing_columns(X, numeric_cols, categorical_cols)

    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_work, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X_work, y, groups=groups))
        X_train, X_test = X_work.iloc[train_idx], X_work.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
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
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = aggregate_importances(
        perm.importances_mean,
        transformed_feature_names,
        categorical_cols,
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

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
        "dropped_all_missing_features": dropped_cols,
        "top_features": importance_df.head(15).to_dict(orient="records"),
        "classification_report": report,
    }
    return importance_df, summary


def run_analysis(df: pd.DataFrame, label: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    work = df.copy()
    work = work[work["best_method"].notna()].copy()
    work["best_method_grouped"] = group_rare_classes(work["best_method"])

    numeric_cols, categorical_cols = split_feature_types(work)
    feature_cols = numeric_cols + categorical_cols
    X = work[feature_cols]
    y = work["best_method_grouped"]

    random_importance, random_summary = _fit_and_score(
        X=X,
        y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        split_kind="random_row_split",
        groups=None,
    )
    random_importance["analysis"] = label
    random_importance["split_kind"] = "random_row_split"
    random_summary["analysis"] = label

    grouped_importance, grouped_summary = _fit_and_score(
        X=X,
        y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        split_kind="grouped_by_model_id",
        groups=work["model_id"],
    )
    grouped_importance["analysis"] = label
    grouped_importance["split_kind"] = "grouped_by_model_id"
    grouped_summary["analysis"] = label

    return (
        {
            "random_row_split": random_importance,
            "grouped_by_model_id": grouped_importance,
        },
        {
            "random_row_split": random_summary,
            "grouped_by_model_id": grouped_summary,
        },
    )


def write_markdown(
    out_path: Path,
    apriori_summary: Dict[str, object],
    full_summary: Dict[str, object],
) -> None:
    def lines_for(label: str, summary: Dict[str, object]) -> List[str]:
        lines = []
        lines.append(f"## {label.title()} Analysis")
        for split_name in ["random_row_split", "grouped_by_model_id"]:
            split = summary[split_name]
            lines.append(f"### {split_name}")
            lines.append(f"- Rows: {split['rows_total']}")
            lines.append(f"- Classes: {split['n_classes']}")
            lines.append(f"- Accuracy: {split['accuracy']:.4f}")
            lines.append(f"- Balanced accuracy: {split['balanced_accuracy']:.4f}")
            lines.append(f"- Top-3 accuracy: {split['top3_accuracy']:.4f}")
            lines.append("- Top features:")
            for rec in split["top_features"][:10]:
                lines.append(f"  - {rec['feature']}: {rec['importance_mean']:.6f}")
        return lines

    text = ["# Selection Feature Importance", ""]
    text.extend(lines_for("apriori", apriori_summary))
    text.append("")
    text.extend(lines_for("full", full_summary))
    out_path.write_text("\n".join(text) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    apriori_df = pd.read_csv(args.analysis_dir / "selection_trace_table_apriori.csv")
    full_df = pd.read_csv(args.analysis_dir / "selection_trace_table_full.csv")

    apriori_importances, apriori_summary = run_analysis(apriori_df, "apriori")
    full_importances, full_summary = run_analysis(full_df, "full")

    pd.concat(apriori_importances.values(), ignore_index=True).to_csv(
        args.out_dir / "feature_importance_apriori.csv", index=False
    )
    pd.concat(full_importances.values(), ignore_index=True).to_csv(
        args.out_dir / "feature_importance_full.csv", index=False
    )

    summary = {
        "apriori": apriori_summary,
        "full": full_summary,
    }
    (args.out_dir / "feature_importance_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(args.out_dir / "feature_importance_report.md", apriori_summary, full_summary)

    print(f"Wrote: {args.out_dir / 'feature_importance_apriori.csv'}")
    print(f"Wrote: {args.out_dir / 'feature_importance_full.csv'}")
    print(f"Wrote: {args.out_dir / 'feature_importance_summary.json'}")
    print(f"Wrote: {args.out_dir / 'feature_importance_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
