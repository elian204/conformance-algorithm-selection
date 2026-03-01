"""Tests for shard CSV merging."""

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.merge_sharded_csv import merge_csv_files


HEADER = ["trace_hash", "method", "model_id", "value"]


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_merge_sharded_csv_success(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "merged.csv"

    _write_csv(a, [
        {"trace_hash": "bbb", "method": "m2", "model_id": "mid", "value": "2"},
    ])
    _write_csv(b, [
        {"trace_hash": "aaa", "method": "m1", "model_id": "mid", "value": "1"},
    ])

    merge_csv_files([a, b], out)

    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [(r["trace_hash"], r["method"]) for r in rows] == [("aaa", "m1"), ("bbb", "m2")]


def test_merge_sharded_csv_rejects_duplicate_pairs(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"

    dup_row = {"trace_hash": "aaa", "method": "m1", "model_id": "mid", "value": "1"}
    _write_csv(a, [dup_row])
    _write_csv(b, [dup_row])

    with pytest.raises(ValueError, match="Duplicate"):
        merge_csv_files([a, b], tmp_path / "merged.csv")


def test_merge_sharded_csv_rejects_mismatched_model_id(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"

    _write_csv(a, [{"trace_hash": "aaa", "method": "m1", "model_id": "mid1", "value": "1"}])
    _write_csv(b, [{"trace_hash": "bbb", "method": "m2", "model_id": "mid2", "value": "2"}])

    with pytest.raises(ValueError, match="model_id"):
        merge_csv_files([a, b], tmp_path / "merged.csv")


def test_merge_sharded_csv_rejects_header_mismatch(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"

    _write_csv(a, [{"trace_hash": "aaa", "method": "m1", "model_id": "mid", "value": "1"}])
    with b.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trace_hash", "method", "model_id", "different"])
        writer.writeheader()
        writer.writerow({"trace_hash": "bbb", "method": "m2", "model_id": "mid", "different": "2"})

    with pytest.raises(ValueError, match="headers do not match"):
        merge_csv_files([a, b], tmp_path / "merged.csv")
