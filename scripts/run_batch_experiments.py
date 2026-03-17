#!/usr/bin/env python3
"""Batch runner for deterministic dataset experiments across many models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STATUS_FIELDS = [
    "timestamp",
    "run_id",
    "dataset_name",
    "model_path",
    "log_path",
    "status",
    "duration_seconds",
    "return_code",
    "results_dir",
    "mapped_by",
    "command",
    "message",
]

MANIFEST_FIELDS = [
    "run_id",
    "dataset_name",
    "model_path",
    "log_path",
    "mapped_by",
    "will_run",
    "skip_reason",
]


@dataclass
class RunSpec:
    run_id: str
    parent_run_id: str
    dataset_name: str
    model_path: Path
    log_path: Optional[Path]
    mapped_by: str
    will_run: bool
    trace_shard_count: int = 1
    trace_shard_index: int = 0
    skip_reason: str = ""


QueuedRun = Tuple[RunSpec, List[str], Optional[Dict[str, str]]]


def _load_mapping(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML mapping requires pyyaml. Use JSON or install pyyaml.") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Mapping file must be .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError("Mapping file must contain a JSON/YAML object")

    out = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("Mapping keys/values must be strings")
        out[k] = v
    return out


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_manifest(path: Path, specs: List[RunSpec]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for spec in specs:
            writer.writerow({
                "run_id": spec.run_id,
                "dataset_name": spec.dataset_name,
                "model_path": str(spec.model_path),
                "log_path": str(spec.log_path) if spec.log_path else "",
                "mapped_by": spec.mapped_by,
                "will_run": int(spec.will_run),
                "skip_reason": spec.skip_reason,
            })


def _load_success_results(status_csv: Path) -> Dict[str, str]:
    if not status_csv.exists():
        return {}
    done: Dict[str, str] = {}
    with status_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                done[row.get("run_id", "")] = row.get("results_dir", "")
    return done


def _parse_results_dir(output_text: str) -> str:
    matches = re.findall(r"Results:\s*(.+)", output_text)
    if not matches:
        return ""
    return matches[-1].strip()


def _execute_run_command(
    cmd: List[str],
    repo_root: Path,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Execute one dataset run command and collect process details."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {
        "return_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "results_dir": _parse_results_dir(combined_output),
    }


def _build_run_id(
    model_path: Path,
    log_path: Optional[Path],
    algorithms: List[str],
    heuristics: List[str],
    max_traces: Optional[int],
    trace_shard_count: int,
    timeout: float,
    max_expansions: int,
) -> str:
    payload = "|".join([
        str(model_path.resolve()),
        str(log_path.resolve()) if log_path else "",
        ",".join(algorithms),
        ",".join(heuristics),
        str(max_traces),
        str(trace_shard_count),
        str(timeout),
        str(max_expansions),
    ])
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _find_default_log(model_path: Path, logs_by_stem: Dict[str, Path]) -> Tuple[Optional[Path], str]:
    model_stem = model_path.stem

    candidates = sorted(logs_by_stem.keys(), key=len, reverse=True)
    for stem in candidates:
        if model_stem == stem or model_stem.startswith(stem + "_"):
            return logs_by_stem[stem], "prefix"
        if f"_{stem}_" in model_stem or model_stem.endswith("_" + stem):
            return logs_by_stem[stem], "contains"
    return None, "none"


def _resolve_log_for_model(
    model_path: Path,
    mapping: Dict[str, str],
    logs_by_stem: Dict[str, Path],
    logs_dir: Path,
) -> Tuple[Optional[Path], str]:
    candidates = [model_path.name, model_path.stem]
    for key in candidates:
        if key in mapping:
            mapped = Path(mapping[key])
            if not mapped.is_absolute():
                mapped = logs_dir / mapped
            if mapped.exists():
                return mapped, "mapping"
            return None, "mapping_missing"

    return _find_default_log(model_path, logs_by_stem)


def _interleave_shard_runs(run_queue: List[QueuedRun]) -> List[QueuedRun]:
    """Round-robin shard jobs across parent models while preserving per-model shard order."""
    if len(run_queue) <= 1:
        return run_queue

    grouped: Dict[str, List[QueuedRun]] = {}
    group_order: List[str] = []
    for queued_run in run_queue:
        spec = queued_run[0]
        parent_run_id = spec.parent_run_id
        if parent_run_id not in grouped:
            grouped[parent_run_id] = []
            group_order.append(parent_run_id)
        grouped[parent_run_id].append(queued_run)

    if len(group_order) <= 1:
        return run_queue

    interleaved: List[QueuedRun] = []
    while grouped:
        next_group_order: List[str] = []
        for parent_run_id in group_order:
            group_runs = grouped.get(parent_run_id)
            if not group_runs:
                continue
            interleaved.append(group_runs.pop(0))
            if group_runs:
                next_group_order.append(parent_run_id)
            else:
                grouped.pop(parent_run_id, None)
        group_order = next_group_order

    return interleaved


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run batch dataset experiments over many models")
    p.add_argument("--models-dir", required=True, type=str)
    p.add_argument("--logs-dir", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--mapping-file", type=str, default=None,
                   help="JSON/YAML file mapping model filename/stem -> log file path")

    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--trace-shard-count", type=int, default=1)
    p.add_argument("--algorithms", nargs="+", default=["all"])
    p.add_argument("--heuristics", nargs="+", default=["all"])
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-expansions", type=int, default=1_000_000)

    p.add_argument("--limit-models", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--sp-stats", action="store_true")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--jobs", type=int, default=1,
                   help="Maximum concurrent shard subprocesses (default: 1)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.trace_shard_count < 1:
        raise ValueError("--trace-shard-count must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"

    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_csv = output_dir / "batch_status.csv"
    manifest_csv = output_dir / "batch_manifest.csv"
    errors_log = output_dir / "batch_errors.log"
    if not status_csv.exists():
        with status_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            writer.writeheader()
    errors_log.touch(exist_ok=True)

    model_files = sorted(list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pnml")))
    if args.limit_models is not None:
        model_files = model_files[: args.limit_models]

    log_files = sorted(logs_dir.glob("*.xes"))
    logs_by_stem = {p.stem: p for p in log_files}

    mapping = _load_mapping(Path(args.mapping_file)) if args.mapping_file else {}

    specs: List[RunSpec] = []
    shard_groups: Dict[str, List[RunSpec]] = {}
    for model in model_files:
        log_path, mapped_by = _resolve_log_for_model(model, mapping, logs_by_stem, logs_dir)
        dataset_name = log_path.stem if log_path else ""
        parent_run_id = _build_run_id(
            model,
            log_path,
            args.algorithms,
            args.heuristics,
            args.max_traces,
            args.trace_shard_count,
            args.timeout,
            args.max_expansions,
        )

        if log_path is None:
            specs.append(RunSpec(
                run_id=parent_run_id,
                parent_run_id=parent_run_id,
                dataset_name=dataset_name,
                model_path=model,
                log_path=None,
                mapped_by=mapped_by,
                will_run=False,
                skip_reason="no_matching_log",
            ))
        else:
            shard_specs: List[RunSpec] = []
            for shard_index in range(args.trace_shard_count):
                child_run_id = (
                    parent_run_id
                    if args.trace_shard_count == 1
                    else f"{parent_run_id}_s{shard_index:02d}"
                )
                shard_specs.append(RunSpec(
                    run_id=child_run_id,
                    parent_run_id=parent_run_id,
                    dataset_name=dataset_name,
                    model_path=model,
                    log_path=log_path,
                    mapped_by=mapped_by,
                    will_run=True,
                    trace_shard_count=args.trace_shard_count,
                    trace_shard_index=shard_index,
                ))
            specs.extend(shard_specs)
            shard_groups[parent_run_id] = shard_specs

    _write_manifest(manifest_csv, specs)

    completed_success_rows = _load_success_results(status_csv) if args.resume else {}
    completed_success = set(completed_success_rows.keys())
    success_results = dict(completed_success_rows)
    failures = 0
    executed = 0
    run_queue: List[QueuedRun] = []

    for spec in specs:
        now = datetime.utcnow().isoformat()
        shard_message = (
            f"shard {spec.trace_shard_index}/{spec.trace_shard_count}"
            if spec.trace_shard_count > 1 else ""
        )

        if not spec.will_run:
            failures += 1
            row = {
                "timestamp": now,
                "run_id": spec.run_id,
                "dataset_name": spec.dataset_name,
                "model_path": str(spec.model_path),
                "log_path": "",
                "status": "unmapped",
                "duration_seconds": "0",
                "return_code": "",
                "results_dir": "",
                "mapped_by": spec.mapped_by,
                "command": "",
                "message": spec.skip_reason,
            }
            _append_csv_row(status_csv, STATUS_FIELDS, row)
            if args.fail_fast:
                break
            continue

        if spec.run_id in completed_success:
            row = {
                "timestamp": now,
                "run_id": spec.run_id,
                "dataset_name": spec.dataset_name,
                "model_path": str(spec.model_path),
                "log_path": str(spec.log_path),
                "status": "skipped_resume",
                "duration_seconds": "0",
                "return_code": "",
                "results_dir": "",
                "mapped_by": spec.mapped_by,
                "command": "",
                "message": (
                    f"{shard_message} | already successful in batch_status.csv"
                    if shard_message else "already successful in batch_status.csv"
                ),
            }
            _append_csv_row(status_csv, STATUS_FIELDS, row)
            continue

        if spec.trace_shard_count > 1:
            shard_output_root = output_dir / spec.parent_run_id / f"shard_{spec.trace_shard_index:02d}"
        else:
            shard_output_root = output_dir / spec.run_id

        cmd = [
            sys.executable,
            str(main_py),
            "--mode", "dataset",
            "--model", str(spec.model_path),
            "--log", str(spec.log_path),
            # Isolate each run under its own root to avoid dataset/timestamp collisions.
            "--output-dir", str(shard_output_root),
            "--timeout", str(args.timeout),
            "--max-expansions", str(args.max_expansions),
            "--algorithms",
            *args.algorithms,
            "--heuristics",
            *args.heuristics,
        ]

        if args.max_traces is not None:
            cmd.extend(["--max-traces", str(args.max_traces)])
        if spec.trace_shard_count > 1:
            cmd.extend([
                "--trace-shard-count", str(spec.trace_shard_count),
                "--trace-shard-index", str(spec.trace_shard_index),
            ])
        if args.sp_stats:
            cmd.append("--sp-stats")
        if args.no_quality:
            cmd.append("--no-quality")

        worker_env: Optional[Dict[str, str]] = None
        if spec.trace_shard_count > 1:
            worker_env = os.environ.copy()
            worker_env.update({
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            })

        if args.dry_run:
            row = {
                "timestamp": now,
                "run_id": spec.run_id,
                "dataset_name": spec.dataset_name,
                "model_path": str(spec.model_path),
                "log_path": str(spec.log_path),
                "status": "dry_run",
                "duration_seconds": "0",
                "return_code": "",
                "results_dir": "",
                "mapped_by": spec.mapped_by,
                "command": " ".join(cmd),
                "message": f"{shard_message} | planned only" if shard_message else "planned only",
            }
            _append_csv_row(status_csv, STATUS_FIELDS, row)
            continue

        run_queue.append((spec, cmd, worker_env))

    if args.trace_shard_count > 1:
        run_queue = _interleave_shard_runs(run_queue)

    def _record_run_result(
        spec: RunSpec,
        cmd: List[str],
        result: Dict[str, object],
    ) -> bool:
        nonlocal failures, executed
        executed += 1
        now = datetime.utcnow().isoformat()
        ok = int(result["return_code"]) == 0

        row = {
            "timestamp": now,
            "run_id": spec.run_id,
            "dataset_name": spec.dataset_name,
            "model_path": str(spec.model_path),
            "log_path": str(spec.log_path),
            "status": "success" if ok else "failed",
            "duration_seconds": f"{float(result['elapsed_seconds']):.6f}",
            "return_code": str(result["return_code"]),
            "results_dir": str(result["results_dir"]),
            "mapped_by": spec.mapped_by,
            "command": " ".join(cmd),
            "message": (
                f"shard {spec.trace_shard_index}/{spec.trace_shard_count}"
                if spec.trace_shard_count > 1 else ""
            ),
        }
        _append_csv_row(status_csv, STATUS_FIELDS, row)

        if not ok:
            failures += 1
            with errors_log.open("a", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write(f"{now} run_id={spec.run_id}\n")
                f.write("CMD: " + " ".join(cmd) + "\n")
                f.write("STDOUT:\n" + str(result["stdout"]) + "\n")
                f.write("STDERR:\n" + str(result["stderr"]) + "\n")
        else:
            success_results[spec.run_id] = str(result["results_dir"])
        return ok

    if not args.dry_run and run_queue:
        if args.jobs == 1:
            for spec, cmd, worker_env in run_queue:
                result = _execute_run_command(cmd, repo_root, env=worker_env)
                ok = _record_run_result(spec, cmd, result)
                if args.fail_fast and not ok:
                    break
        else:
            pending_iter = iter(run_queue)
            future_map = {}
            stop_submitting = False
            workers = min(args.jobs, len(run_queue))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in range(workers):
                    spec, cmd, worker_env = next(pending_iter)
                    future = executor.submit(_execute_run_command, cmd, repo_root, worker_env)
                    future_map[future] = (spec, cmd)

                while future_map:
                    done, _ = wait(list(future_map.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        spec, cmd = future_map.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover
                            result = {
                                "return_code": -1,
                                "elapsed_seconds": 0.0,
                                "stdout": "",
                                "stderr": f"batch worker exception: {exc}",
                                "results_dir": "",
                            }
                        ok = _record_run_result(spec, cmd, result)
                        if args.fail_fast and not ok:
                            stop_submitting = True

                    while (not stop_submitting) and len(future_map) < workers:
                        try:
                            spec, cmd, worker_env = next(pending_iter)
                        except StopIteration:
                            break
                        future = executor.submit(_execute_run_command, cmd, repo_root, worker_env)
                        future_map[future] = (spec, cmd)

    if not args.dry_run and args.trace_shard_count > 1 and shard_groups:
        try:
            from scripts.merge_sharded_csv import merge_csv_files
        except ImportError:
            from merge_sharded_csv import merge_csv_files

        for parent_run_id, shard_specs in shard_groups.items():
            child_ids = [spec.run_id for spec in shard_specs]
            if not all(child_id in success_results for child_id in child_ids):
                continue

            input_csvs: List[Path] = []
            merge_output = output_dir / parent_run_id / "merged_results.csv"

            try:
                for child_id in child_ids:
                    results_dir = success_results[child_id]
                    if not results_dir:
                        raise FileNotFoundError(f"Missing results_dir for shard {child_id}")
                    csv_path = Path(results_dir) / "results.csv"
                    input_csvs.append(csv_path)

                merge_output.parent.mkdir(parents=True, exist_ok=True)
                merge_csv_files(input_csvs, merge_output)
            except Exception as exc:
                failures += 1
                now = datetime.utcnow().isoformat()
                with errors_log.open("a", encoding="utf-8") as f:
                    f.write("=" * 100 + "\n")
                    f.write(f"{now} merge parent_run_id={parent_run_id}\n")
                    f.write("INPUTS:\n")
                    for path in input_csvs:
                        f.write(str(path) + "\n")
                    f.write(f"ERROR: {exc}\n")

    print(f"Manifest: {manifest_csv}")
    print(f"Status:   {status_csv}")
    if errors_log.exists():
        print(f"Errors:   {errors_log}")
    print(f"Specs: {len(specs)} | Queued: {len(run_queue)} | Executed: {executed} | Failures: {failures} | Jobs: {args.jobs}")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
