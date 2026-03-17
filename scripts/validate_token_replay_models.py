#!/usr/bin/env python3
"""Validate discovered Petri net models with full-log token-replay metrics."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import pm4py


FIELDNAMES = [
    "dataset_name",
    "label",
    "model_path",
    "validated_fitness",
    "validated_precision",
    "fitness_eval_seconds",
    "precision_eval_seconds",
    "total_eval_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one dataset's candidate models using token replay."
    )
    parser.add_argument("--log", required=True, help="Path to the XES log.")
    parser.add_argument(
        "--models",
        required=True,
        nargs="+",
        help="One or more .pkl model files to validate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoint CSV/JSON files will be written.",
    )
    return parser.parse_args()


def write_checkpoints(out_dir: Path, rows: list[dict[str, str]]) -> None:
    csv_path = out_dir / "summary_full_token_replay.csv"
    json_path = out_dir / "summary_full_token_replay.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = log_path.stem
    print(f"[LOAD] {dataset_name} from {log_path}", flush=True)
    log = pm4py.read_xes(str(log_path))

    rows: list[dict[str, str]] = []
    for index, model_str in enumerate(args.models, start=1):
        model_path = Path(model_str)
        print(
            f"[{index}/{len(args.models)}] validating {model_path.name}",
            flush=True,
        )

        with model_path.open("rb") as handle:
            payload = pickle.load(handle)
        net = payload["net"]
        initial_marking = payload["im"]
        final_marking = payload["fm"]

        start = time.time()
        fitness = pm4py.fitness_token_based_replay(
            log, net, initial_marking, final_marking
        )["log_fitness"]
        fitness_seconds = time.time() - start

        start = time.time()
        precision = pm4py.precision_token_based_replay(
            log, net, initial_marking, final_marking
        )
        precision_seconds = time.time() - start

        rows.append(
            {
                "dataset_name": dataset_name,
                "label": model_path.stem,
                "model_path": str(model_path),
                "validated_fitness": f"{float(fitness):.12f}",
                "validated_precision": f"{float(precision):.12f}",
                "fitness_eval_seconds": f"{fitness_seconds:.6f}",
                "precision_eval_seconds": f"{precision_seconds:.6f}",
                "total_eval_seconds": f"{fitness_seconds + precision_seconds:.6f}",
            }
        )
        write_checkpoints(out_dir, rows)

        print(
            f"    => f={float(fitness):.4f}, p={float(precision):.4f}, "
            f"fit={fitness_seconds:.2f}s, prec={precision_seconds:.2f}s",
            flush=True,
        )

    print(f"[DONE] {dataset_name}", flush=True)


if __name__ == "__main__":
    main()
