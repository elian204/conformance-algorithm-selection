"""Tests for batch runner manifest/status behavior."""

import csv
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_batch_experiments import (
    _build_run_id,
    _interleave_shard_runs,
    main as batch_main,
)



def test_batch_dry_run_and_unmapped(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()

    (models_dir / "Sepsis_cases_model.pkl").write_text("x", encoding="utf-8")
    (models_dir / "unknown_model.pkl").write_text("x", encoding="utf-8")
    (logs_dir / "Sepsis_cases.xes").write_text("x", encoding="utf-8")

    rc = batch_main([
        "--models-dir", str(models_dir),
        "--logs-dir", str(logs_dir),
        "--output-dir", str(out_dir),
        "--dry-run",
    ])

    # one model is unmapped -> nonzero exit
    assert rc == 1

    manifest = pd.read_csv(out_dir / "batch_manifest.csv")
    status = pd.read_csv(out_dir / "batch_status.csv")

    assert len(manifest) == 2
    assert set(status["status"].tolist()) == {"dry_run", "unmapped"}



def test_batch_resume_skips_success(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()
    out_dir.mkdir()

    model = models_dir / "Sepsis_cases_model.pkl"
    log = logs_dir / "Sepsis_cases.xes"
    model.write_text("x", encoding="utf-8")
    log.write_text("x", encoding="utf-8")

    run_id = _build_run_id(
        model_path=model,
        log_path=log,
        algorithms=["all"],
        heuristics=["all"],
        max_traces=None,
        trace_shard_count=1,
        timeout=30.0,
        max_expansions=1_000_000,
    )

    status_csv = out_dir / "batch_status.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "run_id", "dataset_name", "model_path", "log_path", "status",
            "duration_seconds", "return_code", "results_dir", "mapped_by", "command", "message",
        ])
        writer.writeheader()
        writer.writerow({
            "timestamp": "2026-01-01T00:00:00",
            "run_id": run_id,
            "dataset_name": "Sepsis_cases",
            "model_path": str(model),
            "log_path": str(log),
            "status": "success",
            "duration_seconds": "1.0",
            "return_code": "0",
            "results_dir": "",
            "mapped_by": "prefix",
            "command": "",
            "message": "",
        })

    rc = batch_main([
        "--models-dir", str(models_dir),
        "--logs-dir", str(logs_dir),
        "--output-dir", str(out_dir),
        "--resume",
        "--dry-run",
    ])
    assert rc == 0

    status = pd.read_csv(status_csv)
    assert "skipped_resume" in set(status["status"].tolist())


def test_batch_shard_expansion_uses_child_run_ids(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()

    (models_dir / "Sepsis_cases_model.pkl").write_text("x", encoding="utf-8")
    (logs_dir / "Sepsis_cases.xes").write_text("x", encoding="utf-8")

    rc = batch_main([
        "--models-dir", str(models_dir),
        "--logs-dir", str(logs_dir),
        "--output-dir", str(out_dir),
        "--trace-shard-count", "3",
        "--dry-run",
    ])
    assert rc == 0

    manifest = pd.read_csv(out_dir / "batch_manifest.csv")
    status = pd.read_csv(out_dir / "batch_status.csv")

    assert len(manifest) == 3
    assert len(status) == 3
    assert set(status["status"].tolist()) == {"dry_run"}
    run_ids = manifest["run_id"].tolist()
    assert any(run_id.endswith("_s00") for run_id in run_ids)
    assert any(run_id.endswith("_s01") for run_id in run_ids)
    assert any(run_id.endswith("_s02") for run_id in run_ids)
    commands = status["command"].tolist()
    assert any("--trace-shard-index 0" in cmd for cmd in commands)
    assert any("--trace-shard-index 1" in cmd for cmd in commands)
    assert any("--trace-shard-index 2" in cmd for cmd in commands)


def test_batch_resume_skips_only_successful_shard_child_ids(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()
    out_dir.mkdir()

    model = models_dir / "Sepsis_cases_model.pkl"
    log = logs_dir / "Sepsis_cases.xes"
    model.write_text("x", encoding="utf-8")
    log.write_text("x", encoding="utf-8")

    parent_run_id = _build_run_id(
        model_path=model,
        log_path=log,
        algorithms=["all"],
        heuristics=["all"],
        max_traces=None,
        trace_shard_count=2,
        timeout=30.0,
        max_expansions=1_000_000,
    )

    status_csv = out_dir / "batch_status.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "run_id", "dataset_name", "model_path", "log_path", "status",
            "duration_seconds", "return_code", "results_dir", "mapped_by", "command", "message",
        ])
        writer.writeheader()
        writer.writerow({
            "timestamp": "2026-01-01T00:00:00",
            "run_id": f"{parent_run_id}_s00",
            "dataset_name": "Sepsis_cases",
            "model_path": str(model),
            "log_path": str(log),
            "status": "success",
            "duration_seconds": "1.0",
            "return_code": "0",
            "results_dir": "",
            "mapped_by": "prefix",
            "command": "",
            "message": "shard 0/2",
        })

    rc = batch_main([
        "--models-dir", str(models_dir),
        "--logs-dir", str(logs_dir),
        "--output-dir", str(out_dir),
        "--trace-shard-count", "2",
        "--resume",
        "--dry-run",
    ])
    assert rc == 0

    status = pd.read_csv(status_csv)
    skipped = status[status["status"] == "skipped_resume"]
    dry_runs = status[status["status"] == "dry_run"]
    assert len(skipped) == 1
    assert skipped.iloc[0]["run_id"].endswith("_s00")
    assert len(dry_runs) == 1
    assert dry_runs.iloc[0]["run_id"].endswith("_s01")


def test_batch_shard_workers_set_single_thread_env(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()

    (models_dir / "Sepsis_cases_model.pkl").write_text("x", encoding="utf-8")
    (logs_dir / "Sepsis_cases.xes").write_text("x", encoding="utf-8")

    captured_envs = []

    def fake_execute(cmd, repo_root, env=None):
        captured_envs.append(env or {})
        return {
            "return_code": 1,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "forced test failure",
            "results_dir": "",
        }

    with patch("scripts.run_batch_experiments._execute_run_command", side_effect=fake_execute):
        rc = batch_main([
            "--models-dir", str(models_dir),
            "--logs-dir", str(logs_dir),
            "--output-dir", str(out_dir),
            "--trace-shard-count", "2",
            "--jobs", "1",
            "--no-quality",
        ])

    assert rc == 1
    assert len(captured_envs) == 2
    for env in captured_envs:
        assert env["OMP_NUM_THREADS"] == "1"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["MKL_NUM_THREADS"] == "1"
        assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_interleave_shard_runs_round_robin_across_models(tmp_path: Path):
    model_a = tmp_path / "a.pkl"
    model_b = tmp_path / "b.pkl"
    log_a = tmp_path / "a.xes"
    log_b = tmp_path / "b.xes"
    for path in (model_a, model_b, log_a, log_b):
        path.write_text("x", encoding="utf-8")

    queue = []
    for parent_run_id, model_path, log_path in (
        ("parent_a", model_a, log_a),
        ("parent_b", model_b, log_b),
    ):
        for shard_index in range(2):
            spec = type("Spec", (), {
                "parent_run_id": parent_run_id,
                "trace_shard_index": shard_index,
            })()
            queue.append((spec, ["--model", str(model_path), "--log", str(log_path)], None))

    ordered = _interleave_shard_runs(queue)
    actual = [(item[0].parent_run_id, item[0].trace_shard_index) for item in ordered]
    assert actual == [
        ("parent_a", 0),
        ("parent_b", 0),
        ("parent_a", 1),
        ("parent_b", 1),
    ]


def test_batch_shards_execute_round_robin_across_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    models_dir.mkdir()
    logs_dir.mkdir()

    (models_dir / "A_model.pkl").write_text("x", encoding="utf-8")
    (models_dir / "B_model.pkl").write_text("x", encoding="utf-8")
    (logs_dir / "A.xes").write_text("x", encoding="utf-8")
    (logs_dir / "B.xes").write_text("x", encoding="utf-8")

    execution_order = []

    def fake_execute(cmd, repo_root, env=None):
        model_path = Path(cmd[cmd.index("--model") + 1]).name
        shard_index = int(cmd[cmd.index("--trace-shard-index") + 1])
        execution_order.append((model_path, shard_index))
        results_dir = out_dir / "fake_results" / f"{model_path}_s{shard_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results.csv").write_text(
            "model_id,trace_hash,method\n"
            f"{model_path},trace_{shard_index},forward_zero\n",
            encoding="utf-8",
        )
        return {
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "",
            "results_dir": str(results_dir),
        }

    with patch("scripts.run_batch_experiments._execute_run_command", side_effect=fake_execute):
        rc = batch_main([
            "--models-dir", str(models_dir),
            "--logs-dir", str(logs_dir),
            "--output-dir", str(out_dir),
            "--trace-shard-count", "2",
            "--jobs", "1",
            "--no-quality",
        ])

    assert rc == 0
    assert execution_order == [
        ("A_model.pkl", 0),
        ("B_model.pkl", 0),
        ("A_model.pkl", 1),
        ("B_model.pkl", 1),
    ]
