#!/usr/bin/env python3
"""Convert ss-portfolio symbolic raw CSV into this project's symbolic contract."""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from scripts.validate_symbolic_results import validate_symbolic_dataframe
except ImportError:  # pragma: no cover
    from validate_symbolic_results import validate_symbolic_dataframe  # type: ignore


STATUS_PRIORITY = {
    "ok": 0,
    "timeout": 1,
    "no_solution": 2,
    "max_expansions": 3,
    "error": 4,
    "skipped": 5,
}


def _norm_status(value: object) -> str:
    s = str(value).strip().lower()
    mapping = {
        "ok": "ok",
        "success": "ok",
        "timeout": "timeout",
        "timed_out": "timeout",
        "error": "error",
        "failed": "error",
        "exception": "error",
        "no_solution": "no_solution",
        "nosolution": "no_solution",
        "max_expansions": "max_expansions",
        "maxexpansions": "max_expansions",
        "skipped": "skipped",
        "skip": "skipped",
    }
    if s in mapping:
        return mapping[s]
    if "timeout" in s:
        return "timeout"
    if "skip" in s:
        return "skipped"
    if "no" in s and "solution" in s:
        return "no_solution"
    return "error"


def _split_sequence(seq: object) -> List[str]:
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return []
    s = str(seq).strip()
    if not s:
        return []
    if "|" in s:
        parts = s.split("|")
    elif "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    return [p.strip() for p in parts if p.strip()]


def _trace_hash(tokens: List[str]) -> str:
    return hashlib.md5(",".join(tokens).encode("utf-8")).hexdigest()[:12]


def _norm_seq_key(tokens: List[str]) -> str:
    return "|".join(tokens)


def _collect_astar_paths(patterns: List[str]) -> List[Path]:
    files = set()
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if p.is_file():
                files.add(p.resolve())
    return sorted(files)


def _build_astar_reference(
    patterns: List[str],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], Tuple[str, str]]]:
    """
    Returns:
      - model_name -> model_id map (basename-based)
      - (model_id, normalized_sequence) -> (trace_hash, trace_id)
    """
    if not patterns:
        return {}, {}

    paths = _collect_astar_paths(patterns)
    if not paths:
        return {}, {}

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    model_map: Dict[str, str] = {}
    if {"model_path", "model_id"}.issubset(df.columns):
        for _, row in df[["model_path", "model_id"]].dropna().drop_duplicates().iterrows():
            name = Path(str(row["model_path"])).name
            mid = str(row["model_id"])
            if name in model_map and model_map[name] != mid:
                # Ambiguous mapping by basename; keep first deterministic mapping.
                continue
            model_map[name] = mid
    if {"model_name", "model_id"}.issubset(df.columns):
        for _, row in df[["model_name", "model_id"]].dropna().drop_duplicates().iterrows():
            name = str(row["model_name"])
            mid = str(row["model_id"])
            if name in model_map and model_map[name] != mid:
                continue
            model_map[name] = mid

    trace_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    needed = {"model_id", "trace_hash", "trace_activities"}
    if needed.issubset(df.columns):
        trace_id_col = "trace_id" if "trace_id" in df.columns else None
        rows = df[["model_id", "trace_hash", "trace_activities"] + ([trace_id_col] if trace_id_col else [])].drop_duplicates()
        for _, row in rows.iterrows():
            model_id = str(row["model_id"])
            tokens = _split_sequence(row["trace_activities"])
            if not tokens:
                continue
            key = (model_id, _norm_seq_key(tokens))
            trace_hash = str(row["trace_hash"])
            trace_id = str(row[trace_id_col]) if trace_id_col else trace_hash
            if key not in trace_map:
                trace_map[key] = (trace_hash, trace_id)

    return model_map, trace_map


def _resolve_model_path(raw_model_path: str, model_root: Optional[Path]) -> Optional[Path]:
    p = Path(raw_model_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        if model_root is not None:
            candidates.append(model_root / p)

    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _compute_model_id_from_path(resolved_model: Path) -> Optional[str]:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from experiments.benchmark_loader import load_model
        from experiments.runner import _compute_model_features, _compute_model_identity

        wf, *_ = load_model(str(resolved_model))
        features = _compute_model_features(wf)
        ident = _compute_model_identity(
            dataset_name=resolved_model.stem,
            model_path=str(resolved_model),
            model_features=features,
        )
        return str(ident["model_id"])
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert ss-portfolio symbolic raw CSV into symbolic contract")
    p.add_argument("--input-raw-csv", required=True, type=str)
    p.add_argument("--output-csv", required=True, type=str)
    p.add_argument("--astar-results-glob", nargs="*", default=[])
    p.add_argument("--model-root", type=str, default=None,
                   help="Optional root used to resolve relative model_path entries")
    p.add_argument("--python-only", action="store_true",
                   help="Keep only rows that look like Python symbolic mode (symbolic_sync_mode contains prebuild/py)")
    p.add_argument("--keep-skipped", action="store_true",
                   help="Keep symbolic_status=skipped rows (default: drop)")
    p.add_argument("--symbolic-ms-col", type=str, default="symbolic_ms")
    p.add_argument("--symbolic-status-col", type=str, default="symbolic_status")
    p.add_argument("--variant-seq-col", type=str, default="variant_seq")
    p.add_argument("--variant-id-col", type=str, default="variant_id")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input_raw_csv)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)

    if args.python_only and "symbolic_sync_mode" in raw.columns:
        mode = raw["symbolic_sync_mode"].astype(str).str.lower()
        raw = raw[mode.str.contains("prebuild|py|python", regex=True, na=False)].copy()

    model_map, trace_map = _build_astar_reference(args.astar_results_glob)
    model_root = Path(args.model_root) if args.model_root else None

    rows = []
    skipped_no_model = 0
    skipped_no_trace = 0

    for i, row in raw.iterrows():
        status = _norm_status(row.get(args.symbolic_status_col, ""))
        if status == "skipped" and not args.keep_skipped:
            continue

        ms = pd.to_numeric(row.get(args.symbolic_ms_col), errors="coerce")
        if pd.isna(ms) or not math.isfinite(float(ms)) or float(ms) < 0:
            continue
        symbolic_time_seconds = float(ms) / 1000.0

        raw_model = str(row.get("model_path", "")).strip()
        model_name = Path(raw_model).name if raw_model else ""

        model_id = model_map.get(model_name)
        if model_id is None and raw_model:
            resolved = _resolve_model_path(raw_model, model_root)
            if resolved is not None:
                model_id = _compute_model_id_from_path(resolved)

        if not model_id:
            skipped_no_model += 1
            continue

        tokens = _split_sequence(row.get(args.variant_seq_col))
        seq_key = _norm_seq_key(tokens)

        mapped = trace_map.get((model_id, seq_key)) if seq_key else None
        if mapped is not None:
            trace_hash, trace_id = mapped
        else:
            if not tokens:
                skipped_no_trace += 1
                continue
            trace_hash = _trace_hash(tokens)
            trace_id = str(row.get(args.variant_id_col, "")).strip() or trace_hash

        rows.append({
            "model_id": model_id,
            "trace_hash": trace_hash,
            "trace_id": trace_id,
            "symbolic_time_seconds": symbolic_time_seconds,
            "symbolic_status": status,
            "source_raw_row": int(i),
            "source_model_path": raw_model,
            "source_variant_id": str(row.get(args.variant_id_col, "")),
            "source_variant_seq": str(row.get(args.variant_seq_col, "")),
        })

    if not rows:
        print("ERROR: no rows produced after conversion/filtering")
        return 1

    out_df = pd.DataFrame(rows)

    out_df["_priority"] = out_df["symbolic_status"].map(STATUS_PRIORITY).fillna(99)
    out_df = out_df.sort_values(["_priority", "symbolic_time_seconds"]).drop_duplicates(
        subset=["model_id", "trace_hash", "trace_id"], keep="first"
    )
    out_df = out_df.drop(columns=["_priority"]).reset_index(drop=True)

    validation_errors = validate_symbolic_dataframe(out_df)
    if validation_errors:
        print("ERROR: converted CSV failed validation")
        for err in validation_errors:
            print(" -", err)
        return 1

    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Unique model_id: {out_df['model_id'].nunique()}")
    print(f"Dropped rows (no model_id): {skipped_no_model}")
    print(f"Dropped rows (no trace mapping/hash): {skipped_no_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
