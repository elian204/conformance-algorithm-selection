"""Tests for symbolic validation and merged-table builder."""

import csv
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.build_ml_table import main as build_ml_main
from scripts.validate_symbolic_results import validate_symbolic_dataframe


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_validate_symbolic_rejects_duplicates():
    df = pd.DataFrame([
        {"model_id": "m1", "trace_hash": "h1", "symbolic_time_seconds": 1.0, "symbolic_status": "ok"},
        {"model_id": "m1", "trace_hash": "h1", "symbolic_time_seconds": 2.0, "symbolic_status": "ok"},
    ])
    errors = validate_symbolic_dataframe(df)
    assert any("Duplicate keys" in e for e in errors)


def test_build_ml_table_basic_merge(tmp_path: Path):
    astar = tmp_path / "results.csv"
    symbolic = tmp_path / "symbolic.csv"
    out_dir = tmp_path / "out"

    _write_csv(astar, [
        {
            "model_id": "m1",
            "trace_hash": "h1",
            "trace_id": "c1",
            "method": "forward_me",
            "time_seconds": 2.0,
            "status": "ok",
            "trace_variant_frequency": 3,
            "trace_length": 5,
        },
        {
            "model_id": "m1",
            "trace_hash": "h1",
            "trace_id": "c1",
            "method": "bidir_mm_me",
            "time_seconds": 1.0,
            "status": "ok",
            "trace_variant_frequency": 3,
            "trace_length": 5,
        },
    ])

    _write_csv(symbolic, [
        {
            "model_id": "m1",
            "trace_hash": "h1",
            "trace_id": "c1",
            "symbolic_time_seconds": 1.5,
            "symbolic_status": "ok",
        }
    ])

    rc = build_ml_main([
        "--astar-results-glob", str(astar),
        "--symbolic-csv", str(symbolic),
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    merged = pd.read_csv(out_dir / "ml_table_unweighted.csv")
    assert len(merged) == 1
    assert merged.loc[0, "best_method"] == "bidir_mm_me"
    assert merged.loc[0, "best_astar_time_seconds"] == 1.0
    assert merged.loc[0, "symbolic_time_seconds"] == 1.5
    assert merged.loc[0, "symbolic_faster_flag"] == 0

    weighted = pd.read_csv(out_dir / "ml_table_weighted.csv")
    assert weighted.loc[0, "sample_weight"] == 3
