#!/usr/bin/env python3
"""Run an SP-stats pilot and report overhead/signal metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _parse_results_dir(output: str) -> Optional[Path]:
    matches = re.findall(r"Results:\s*(.+)", output)
    if not matches:
        return None
    return Path(matches[-1].strip())


def _run_experiment(
    repo_root: Path,
    model_path: Path,
    log_path: Path,
    output_dir: Path,
    max_traces: int,
    timeout: float,
    max_expansions: int,
    algorithms: List[str],
    heuristics: List[str],
    enable_sp_stats: bool,
    no_quality: bool,
) -> Tuple[bool, float, Optional[Path], str]:
    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--mode", "dataset",
        "--model", str(model_path),
        "--log", str(log_path),
        "--output-dir", str(output_dir),
        "--max-traces", str(max_traces),
        "--timeout", str(timeout),
        "--max-expansions", str(max_expansions),
        "--algorithms", *algorithms,
        "--heuristics", *heuristics,
    ]
    if enable_sp_stats:
        cmd.append("--sp-stats")
    if no_quality:
        cmd.append("--no-quality")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    results_dir = _parse_results_dir(output)
    ok = proc.returncode == 0
    return ok, elapsed, results_dir, output


def _extract_sp_metrics(results_dir: Optional[Path]) -> Dict[str, Optional[float]]:
    if results_dir is None:
        return {
            "missing_sp_nodes": None,
            "missing_sp_edges": None,
            "var_sp_nodes": None,
            "var_sp_edges": None,
            "corr_sp_nodes_time": None,
            "corr_sp_edges_time": None,
        }

    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return {
            "missing_sp_nodes": None,
            "missing_sp_edges": None,
            "var_sp_nodes": None,
            "var_sp_edges": None,
            "corr_sp_nodes_time": None,
            "corr_sp_edges_time": None,
        }

    df = pd.read_csv(csv_path)
    missing_sp_nodes = float(df["sp_nodes"].isna().mean()) if "sp_nodes" in df.columns else 1.0
    missing_sp_edges = float(df["sp_edges"].isna().mean()) if "sp_edges" in df.columns else 1.0

    sp_nodes_num = pd.to_numeric(df.get("sp_nodes"), errors="coerce") if "sp_nodes" in df.columns else pd.Series(dtype=float)
    sp_edges_num = pd.to_numeric(df.get("sp_edges"), errors="coerce") if "sp_edges" in df.columns else pd.Series(dtype=float)
    time_num = pd.to_numeric(df.get("time_seconds"), errors="coerce") if "time_seconds" in df.columns else pd.Series(dtype=float)

    var_sp_nodes = float(sp_nodes_num.dropna().var(ddof=0)) if len(sp_nodes_num.dropna()) > 1 else 0.0
    var_sp_edges = float(sp_edges_num.dropna().var(ddof=0)) if len(sp_edges_num.dropna()) > 1 else 0.0

    corr_sp_nodes_time = float(pd.concat([sp_nodes_num, time_num], axis=1).dropna().corr().iloc[0, 1]) \
        if len(pd.concat([sp_nodes_num, time_num], axis=1).dropna()) > 1 else 0.0
    corr_sp_edges_time = float(pd.concat([sp_edges_num, time_num], axis=1).dropna().corr().iloc[0, 1]) \
        if len(pd.concat([sp_edges_num, time_num], axis=1).dropna()) > 1 else 0.0

    return {
        "missing_sp_nodes": missing_sp_nodes,
        "missing_sp_edges": missing_sp_edges,
        "var_sp_nodes": var_sp_nodes,
        "var_sp_edges": var_sp_edges,
        "corr_sp_nodes_time": corr_sp_nodes_time,
        "corr_sp_edges_time": corr_sp_edges_time,
    }


def _select_models_for_dataset(models: List[Path], dataset: str, max_models: int) -> List[Path]:
    selected = []
    prefix = dataset + "_"
    exact = dataset + "_model"
    for m in models:
        s = m.stem
        if s == exact or s.startswith(prefix) or s.endswith("_" + dataset):
            selected.append(m)
    return sorted(selected)[:max_models]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SP-stats pilot and summarize overhead")
    p.add_argument("--models-dir", required=True, type=str)
    p.add_argument("--logs-dir", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument("--datasets", nargs="*", default=None,
                   help="Dataset stems (default: first 2 from logs-dir)")
    p.add_argument("--models-per-dataset", type=int, default=3)
    p.add_argument("--max-traces", type=int, default=100)

    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-expansions", type=int, default=1_000_000)
    p.add_argument("--algorithms", nargs="+", default=["all"])
    p.add_argument("--heuristics", nargs="+", default=["all"])
    p.add_argument("--no-quality", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pnml")))
    logs = sorted(logs_dir.glob("*.xes"))
    log_map = {p.stem: p for p in logs}

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(log_map.keys())[:2]

    run_rows = []
    any_failures = False

    for dataset in datasets:
        if dataset not in log_map:
            print(f"Skipping dataset without log: {dataset}")
            continue
        log_path = log_map[dataset]
        selected_models = _select_models_for_dataset(models, dataset, args.models_per_dataset)

        for model_path in selected_models:
            ok_no, t_no, dir_no, out_no = _run_experiment(
                repo_root=repo_root,
                model_path=model_path,
                log_path=log_path,
                output_dir=output_dir,
                max_traces=args.max_traces,
                timeout=args.timeout,
                max_expansions=args.max_expansions,
                algorithms=args.algorithms,
                heuristics=args.heuristics,
                enable_sp_stats=False,
                no_quality=args.no_quality,
            )

            ok_sp, t_sp, dir_sp, out_sp = _run_experiment(
                repo_root=repo_root,
                model_path=model_path,
                log_path=log_path,
                output_dir=output_dir,
                max_traces=args.max_traces,
                timeout=args.timeout,
                max_expansions=args.max_expansions,
                algorithms=args.algorithms,
                heuristics=args.heuristics,
                enable_sp_stats=True,
                no_quality=args.no_quality,
            )

            if not (ok_no and ok_sp):
                any_failures = True

            metrics_sp = _extract_sp_metrics(dir_sp)
            overhead_pct = ((t_sp - t_no) / t_no * 100.0) if t_no > 0 else None

            run_rows.append({
                "dataset": dataset,
                "model_path": str(model_path),
                "log_path": str(log_path),
                "ok_without_sp": int(ok_no),
                "ok_with_sp": int(ok_sp),
                "time_without_sp_seconds": t_no,
                "time_with_sp_seconds": t_sp,
                "overhead_pct": overhead_pct,
                "results_dir_without_sp": str(dir_no) if dir_no else "",
                "results_dir_with_sp": str(dir_sp) if dir_sp else "",
                **metrics_sp,
            })

    runs_csv = output_dir / "sp_stats_pilot_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        if run_rows:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)

    overhead_values = [r["overhead_pct"] for r in run_rows if r.get("overhead_pct") is not None]
    avg_overhead = mean(overhead_values) if overhead_values else None

    corr_nodes = [abs(r["corr_sp_nodes_time"]) for r in run_rows if r.get("corr_sp_nodes_time") is not None]
    corr_edges = [abs(r["corr_sp_edges_time"]) for r in run_rows if r.get("corr_sp_edges_time") is not None]
    mean_abs_corr_nodes = mean(corr_nodes) if corr_nodes else None
    mean_abs_corr_edges = mean(corr_edges) if corr_edges else None

    signal_strength = max(
        [v for v in [mean_abs_corr_nodes, mean_abs_corr_edges] if v is not None] or [0.0]
    )

    recommendation_enable = (
        avg_overhead is not None
        and avg_overhead <= 25.0
        and signal_strength >= 0.10
        and not any_failures
    )

    report = {
        "datasets": datasets,
        "models_per_dataset": args.models_per_dataset,
        "max_traces": args.max_traces,
        "num_run_pairs": len(run_rows),
        "avg_overhead_pct": avg_overhead,
        "mean_abs_corr_sp_nodes_time": mean_abs_corr_nodes,
        "mean_abs_corr_sp_edges_time": mean_abs_corr_edges,
        "signal_threshold": 0.10,
        "overhead_threshold_pct": 25.0,
        "recommend_enable_globally": recommendation_enable,
        "any_failures": any_failures,
    }

    report_json = output_dir / "sp_stats_pilot_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {runs_csv}")
    print(f"Wrote: {report_json}")
    print(f"Recommendation enable globally: {recommendation_enable}")

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
