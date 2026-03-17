#!/usr/bin/env python3
"""Copy the current canonical aggregate and analysis outputs into a static snapshot.

The snapshot is not auto-updated. Re-run this script when you want to refresh it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sync a static analysis snapshot from aggregated_current")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--snapshot-dir", type=Path, required=True)
    return p


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_clean_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_readme(snapshot_dir: Path) -> None:
    text = """# Analysis Snapshot

This folder is a static copy of the current canonical aggregate and selected analysis outputs.

It is not updated automatically.
To refresh it, rerun:

`python3 scripts/sync_analysis_snapshot.py --source-dir "tmp_smoke/aggregated_current" --snapshot-dir "tmp_smoke/analysis_snapshot_current"`

Key file:
- `merged/astar_results_canonical_snapshot.csv`

Source of truth remains under:
- `tmp_smoke/aggregated_current/`
"""
    (snapshot_dir / "README.md").write_text(text)


def main() -> int:
    args = build_parser().parse_args()
    source = args.source_dir
    snap = args.snapshot_dir
    ensure_clean_dir(snap)

    # Core canonical merged snapshot
    copy_file(
        source / "astar_results_canonical_current.csv",
        snap / "merged" / "astar_results_canonical_snapshot.csv",
    )
    copy_file(
        source / "astar_results_canonical_current_manifest.csv",
        snap / "merged" / "astar_results_canonical_snapshot_manifest.csv",
    )
    copy_file(
        source / "astar_results_canonical_current_summary.json",
        snap / "merged" / "astar_results_canonical_snapshot_summary.json",
    )

    # Core analysis-ready tables
    analysis = source / "analysis_ready"
    for name in [
        "selection_method_rows_completed_enriched.csv",
        "selection_trace_table_full.csv",
        "selection_trace_table_apriori.csv",
        "model_metadata.csv",
        "selection_analysis_summary.json",
    ]:
        copy_file(analysis / name, snap / "analysis_ready" / name)

    # Stable feature-importance outputs
    copy_tree(analysis / "feature_importance", snap / "analysis_ready" / "feature_importance")

    # Stable compact-selector outputs only, not all experimental versions
    compact_src = analysis / "compact_selector_current"
    compact_dst = snap / "analysis_ready" / "compact_selector"
    if compact_src.is_symlink():
        compact_src = compact_src.resolve()
    copy_tree(compact_src, compact_dst)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source),
        "snapshot_dir": str(snap),
        "note": "Static snapshot. Re-run sync_analysis_snapshot.py to refresh.",
    }
    (snap / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_readme(snap)

    print(f"Wrote snapshot: {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
