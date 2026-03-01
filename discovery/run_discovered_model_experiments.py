"""
Bidirectional A* Experiments with Discovered Models
====================================================

Runs conformance checking experiments using models discovered
with varying fitness/precision characteristics.

This module connects the model discovery pipeline with the
bidirectional A* search implementation for systematic comparison.

Usage:
    python run_discovered_model_experiments.py <log_path> <models_dir> [options]

Example:
    python run_discovered_model_experiments.py \\
        data/BPIC15_1.xes \\
        models/bpi2015 \\
        --max-traces 500 \\
        --methods Forward Backward Bidirectional \\
        --heuristics H0 ME

Output:
    - Detailed trace-level results CSV
    - Model-level aggregated results CSV
    - Method comparison summary
"""

import argparse
import json
import pickle
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import random

import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for bidirectional search experiments."""
    # Search methods to compare
    methods: List[str] = field(default_factory=lambda: [
        "Forward", "Backward", "Bidirectional", "MM"
    ])
    
    # Heuristics to test
    heuristics: List[str] = field(default_factory=lambda: ["H0", "ME"])
    
    # Trace limits
    max_traces: int = 1000
    timeout_per_trace: float = 60.0  # seconds
    
    # Model filters
    min_fitness: float = 0.0
    max_fitness: float = 1.0
    min_precision: float = 0.0
    max_precision: float = 1.0
    
    # Output settings
    save_trace_details: bool = True
    save_alignments: bool = False


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_discovered_models(
    models_dir: Path,
    dataset_name: str,
    config: ExperimentConfig
) -> List[Dict]:
    """
    Load discovered models that match filter criteria.
    
    Args:
        models_dir: Directory containing model pickle files
        dataset_name: Dataset name prefix to match
        config: Experiment configuration with filters
        
    Returns:
        List of model dictionaries with metadata
    """
    models = []
    
    # Find matching model files
    pattern = f"{dataset_name}_f*_p*_*_model.pkl"
    model_files = sorted(models_dir.glob(pattern))
    
    print(f"\nFound {len(model_files)} model files matching pattern")
    
    for model_path in model_files:
        try:
            # Parse filename for metadata
            filename = model_path.stem
            parts = filename.split("_")
            
            # Extract fitness and precision from filename
            fitness = None
            precision = None
            method = "Unknown"
            
            for part in parts:
                if part.startswith("f") and "." in part:
                    try:
                        fitness = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith("p") and "." in part:
                    try:
                        precision = float(part[1:])
                    except ValueError:
                        pass
            
            # Find method name (IMf, SM, ILP, HM, etc.)
            for method_name in ["IMf", "SM", "ILP", "HM", "Alpha"]:
                if method_name in filename:
                    method = method_name
                    break
            
            # Apply filters
            if fitness is not None:
                if fitness < config.min_fitness or fitness > config.max_fitness:
                    continue
            if precision is not None:
                if precision < config.min_precision or precision > config.max_precision:
                    continue
            
            # Load model
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            models.append({
                "path": model_path,
                "filename": filename,
                "net": model_data["net"],
                "im": model_data["im"],
                "fm": model_data["fm"],
                "fitness": fitness,
                "precision": precision,
                "method": method,
            })
            
        except Exception as e:
            print(f"  ⚠ Error loading {model_path.name}: {e}")
    
    print(f"Loaded {len(models)} models after filtering")
    return models


def load_event_log(log_path: Path) -> EventLog:
    """Load event log with caching."""
    log_path = Path(log_path)
    
    # Check for pickle cache
    pkl_path = log_path.with_suffix(log_path.suffix + ".pkl")
    
    if pkl_path.exists():
        print(f"Loading from cache: {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            log = pickle.load(f)
    else:
        print(f"Loading from XES: {log_path.name}")
        log = pm4py.read_xes(str(log_path))
        
        # Cache for next time
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(log, f)
        except Exception:
            pass
    
    # Ensure EventLog format
    if isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    
    print(f"  ✓ {len(log)} traces")
    return log


def sample_traces(log: EventLog, n: int, seed: int = 42) -> List[Tuple[int, Any]]:
    """Sample traces with their original indices."""
    if n >= len(log):
        return [(i, log[i]) for i in range(len(log))]
    
    rng = random.Random(seed)
    indices = list(range(len(log)))
    rng.shuffle(indices)
    indices = sorted(indices[:n])
    
    return [(i, log[i]) for i in indices]


def extract_trace_activities(trace) -> List[str]:
    """Extract activity labels from a trace."""
    activities = []
    for event in trace:
        if "concept:name" in event:
            activities.append(event["concept:name"])
        elif "Activity" in event:
            activities.append(event["Activity"])
    return activities


# ==============================================================================
# EXPERIMENT RUNNER (Placeholder for integration)
# ==============================================================================

def compute_alignment_stub(
    trace_activities: List[str],
    net,
    im,
    fm,
    method: str = "Forward",
    heuristic: str = "H0",
    timeout: float = 60.0
) -> Dict:
    """
    Placeholder for alignment computation.
    
    Replace this with actual bidirectional A* implementation:
    
    from solvers.bidirectional_astar import compute_alignment
    
    Args:
        trace_activities: List of activity labels
        net: Petri net
        im: Initial marking
        fm: Final marking
        method: Search method (Forward, Backward, Bidirectional, MM)
        heuristic: Heuristic function (H0, ME, REACH)
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with alignment results
    """
    # Stub implementation - replace with actual solver
    import time
    start = time.time()
    
    # TODO: Replace with actual implementation
    # Example integration:
    # from solvers.bidirectional_astar import BidirectionalAStar
    # solver = BidirectionalAStar(net, im, fm, method=method, heuristic=heuristic)
    # result = solver.compute_alignment(trace_activities, timeout=timeout)
    
    # Placeholder results
    elapsed = time.time() - start
    
    return {
        "cost": -1,  # Placeholder
        "states_explored": -1,
        "states_generated": -1,
        "time_seconds": elapsed,
        "status": "PLACEHOLDER",
        "alignment": None,
    }


def run_trace_experiment(
    trace_idx: int,
    trace_activities: List[str],
    model: Dict,
    config: ExperimentConfig
) -> List[Dict]:
    """
    Run experiments for a single trace across all method/heuristic combinations.
    """
    results = []
    
    for method in config.methods:
        for heuristic in config.heuristics:
            result = compute_alignment_stub(
                trace_activities,
                model["net"],
                model["im"],
                model["fm"],
                method=method,
                heuristic=heuristic,
                timeout=config.timeout_per_trace
            )
            
            results.append({
                "trace_idx": trace_idx,
                "trace_length": len(trace_activities),
                "model_filename": model["filename"],
                "model_fitness": model["fitness"],
                "model_precision": model["precision"],
                "model_method": model["method"],
                "search_method": method,
                "heuristic": heuristic,
                "cost": result["cost"],
                "states_explored": result["states_explored"],
                "states_generated": result["states_generated"],
                "time_seconds": result["time_seconds"],
                "status": result["status"],
            })
    
    return results


def run_model_experiments(
    log: EventLog,
    model: Dict,
    config: ExperimentConfig,
    trace_samples: List[Tuple[int, Any]]
) -> List[Dict]:
    """
    Run experiments for all sampled traces with a single model.
    """
    all_results = []
    
    print(f"\n  Model: {model['filename']}")
    print(f"    Fitness: {model['fitness']:.4f}, Precision: {model['precision']:.4f}")
    
    for trace_idx, trace in trace_samples:
        activities = extract_trace_activities(trace)
        
        if not activities:
            continue
        
        results = run_trace_experiment(
            trace_idx, activities, model, config
        )
        all_results.extend(results)
    
    print(f"    → {len(all_results)} experiments completed")
    return all_results


# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_results(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results by model and method."""
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Aggregate by model × method × heuristic
    agg = df.groupby([
        "model_filename", "model_fitness", "model_precision",
        "search_method", "heuristic"
    ]).agg({
        "cost": ["mean", "std"],
        "states_explored": ["mean", "std"],
        "time_seconds": ["mean", "sum"],
        "trace_idx": "count"
    }).reset_index()
    
    # Flatten column names
    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    
    return agg


def print_summary(results: List[Dict], models: List[Dict]):
    """Print experiment summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results collected!")
        return
    
    # Overall statistics
    print(f"\nTotal experiments: {len(df)}")
    print(f"Models tested: {df['model_filename'].nunique()}")
    print(f"Traces processed: {df['trace_idx'].nunique()}")
    
    # By search method
    print("\nResults by search method:")
    method_stats = df.groupby("search_method").agg({
        "states_explored": "mean",
        "time_seconds": "mean"
    })
    print(method_stats.to_string())
    
    # By model quality
    print("\nResults by model fitness band:")
    df["fitness_band"] = pd.cut(
        df["model_fitness"],
        bins=[0, 0.8, 0.9, 1.0],
        labels=["Low (<0.8)", "Medium (0.8-0.9)", "High (>0.9)"]
    )
    quality_stats = df.groupby("fitness_band").agg({
        "states_explored": "mean",
        "time_seconds": "mean"
    })
    print(quality_stats.to_string())


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run bidirectional A* experiments with discovered models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to XES event log"
    )
    
    parser.add_argument(
        "models_dir",
        type=str,
        help="Directory containing discovered model .pkl files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--max-traces",
        type=int,
        default=500,
        help="Maximum traces to process (default: 500)"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Forward", "Backward", "Bidirectional"],
        help="Search methods to test"
    )
    
    parser.add_argument(
        "--heuristics",
        nargs="+",
        default=["H0", "ME"],
        help="Heuristics to test"
    )
    
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.0,
        help="Minimum model fitness to include"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per trace in seconds"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    log_path = Path(args.log_path)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    
    if not log_path.exists():
        print(f"❌ Error: Log not found: {log_path}")
        return 1
    
    if not models_dir.exists():
        print(f"❌ Error: Models directory not found: {models_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = ExperimentConfig(
        methods=args.methods,
        heuristics=args.heuristics,
        max_traces=args.max_traces,
        timeout_per_trace=args.timeout,
        min_fitness=args.min_fitness,
    )
    
    dataset_name = log_path.stem
    
    # Header
    print("=" * 70)
    print("BIDIRECTIONAL A* EXPERIMENTS WITH DISCOVERED MODELS")
    print("=" * 70)
    print(f"Log:      {log_path}")
    print(f"Models:   {models_dir}")
    print(f"Methods:  {', '.join(config.methods)}")
    print(f"Heuristics: {', '.join(config.heuristics)}")
    print("=" * 70)
    
    # Load data
    log = load_event_log(log_path)
    models = load_discovered_models(models_dir, dataset_name, config)
    
    if not models:
        print("❌ No models found matching criteria!")
        return 1
    
    # Sample traces
    trace_samples = sample_traces(log, config.max_traces)
    print(f"\nSelected {len(trace_samples)} traces for experiments")
    
    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)
    
    all_results = []
    start_time = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]", end="")
        results = run_model_experiments(log, model, config, trace_samples)
        all_results.extend(results)
    
    elapsed = time.time() - start_time
    
    # Analyze and save
    print_summary(all_results, models)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / f"{dataset_name}_experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results: {results_path}")
    
    # Save aggregated results
    agg_df = analyze_results(all_results)
    agg_path = output_dir / f"{dataset_name}_aggregated_results.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"✓ Aggregated results: {agg_path}")
    
    print(f"\n{'=' * 70}")
    print(f"COMPLETE - Total time: {elapsed:.1f}s")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
