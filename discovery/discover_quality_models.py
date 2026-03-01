"""
Quality-Targeted Model Discovery for Conformance Checking Research
===================================================================

Discovers process models with specific fitness and precision targets.
Designed for bidirectional A* search research requiring models with
varying characteristics across the fitness-precision space.

Target Quality Metrics:
    - Fitness:   ≥ 0.80 (primary target)
    - Precision: ≥ 0.75 (secondary target, when achievable)

Discovery Algorithms:
    1. Inductive Miner (IMf) - Varying noise thresholds
    2. Split Miner - Parallelism and frequency thresholds
    3. ILP Miner - Global optimization approach
    4. Heuristic Miner - Dependency thresholds (backup)

Usage:
    python discover_quality_models.py <dataset_path> [options]
    
    # BPI Challenge 2015 example (all 5 municipalities)
    python discover_quality_models.py data/BPIC15_1.xes --output-dir models/bpi2015
    
    # With quality targets
    python discover_quality_models.py data/BPIC15_1.xes --min-fitness 0.85 --min-precision 0.75
    
    # Quick exploration mode (faster, fewer variants)
    python discover_quality_models.py data/BPIC15_1.xes --quick

Output:
    - {dataset}_f{fitness:.4f}_p{precision:.4f}_{method}_model.pkl
    - {dataset}_quality_models_summary.csv
    - {dataset}_quality_models_summary.json

Author: Bidirectional A* Conformance Checking Research
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any
import random

# Suppress PM4Py deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


# ==============================================================================
# QUALITY TARGETS
# ==============================================================================

@dataclass
class QualityTargets:
    """Quality targets for model discovery."""
    min_fitness: float = 0.80
    min_precision: float = 0.75
    ideal_fitness: float = 0.90
    ideal_precision: float = 0.80
    
    def meets_requirements(self, fitness: float, precision: float) -> bool:
        """Check if model meets minimum requirements."""
        return fitness >= self.min_fitness
    
    def meets_ideal(self, fitness: float, precision: float) -> bool:
        """Check if model meets ideal targets."""
        return fitness >= self.ideal_fitness and precision >= self.ideal_precision
    
    def quality_score(self, fitness: float, precision: float) -> float:
        """
        Compute quality score for ranking models.
        
        Prioritizes fitness (0.6 weight) over precision (0.4 weight),
        with bonus for meeting both targets.
        """
        base_score = fitness * 0.6 + precision * 0.4
        
        # Bonus for meeting requirements
        if self.meets_ideal(fitness, precision):
            base_score += 0.1
        elif self.meets_requirements(fitness, precision):
            base_score += 0.05
            
        return base_score


# ==============================================================================
# MODEL METADATA
# ==============================================================================

@dataclass
class DiscoveredModel:
    """Container for discovered model with full metadata."""
    dataset_name: str
    discovery_method: str
    parameters: Dict[str, Any]
    
    # Model structure
    places: int
    transitions: int
    arcs: int
    invisible_transitions: int
    
    # Quality metrics
    fitness: float
    precision: float
    
    # Timing
    discovery_time: float
    evaluation_time: float
    
    # File info
    filename: str
    
    # Quality assessment
    meets_min_fitness: bool = field(default=False)
    meets_min_precision: bool = field(default=False)
    quality_score: float = field(default=0.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['parameters'] = json.dumps(d['parameters'])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DiscoveredModel':
        """Create from dictionary."""
        d['parameters'] = json.loads(d['parameters'])
        return cls(**d)


# ==============================================================================
# EVENT LOG UTILITIES
# ==============================================================================

def load_event_log(log_path: Path, use_cache: bool = True) -> EventLog:
    """
    Load event log from XES with optional PKL caching.
    
    Args:
        log_path: Path to XES file
        use_cache: Whether to use/create pickle cache
        
    Returns:
        PM4Py EventLog object
    """
    log_path = Path(log_path)
    
    # Determine cache path
    if log_path.suffix.lower() == ".xes":
        pkl_path = log_path.with_suffix(log_path.suffix + ".pkl")
    else:
        pkl_path = log_path.with_suffix(".pkl")
    
    # Try cache first
    if use_cache and pkl_path.exists():
        print(f"  Loading from cache: {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            log = pickle.load(f)
        
        # Ensure EventLog format
        if isinstance(log, pd.DataFrame):
            log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
        
        print(f"  ✓ {len(log)} traces loaded")
        return log
    
    # Load from XES
    print(f"  Loading from XES: {log_path.name}")
    log = pm4py.read_xes(str(log_path))
    
    # Convert if needed
    if isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    
    print(f"  ✓ {len(log)} traces loaded")
    
    # Cache for future use
    if use_cache:
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(log, f)
            print(f"  ✓ Cached to {pkl_path.name}")
        except Exception as e:
            print(f"  ⚠ Cache failed: {e}")
    
    return log


def sample_log(log: EventLog, max_traces: int = 2000, seed: int = 42) -> EventLog:
    """Sample traces for faster evaluation."""
    if len(log) <= max_traces:
        return log
    
    rng = random.Random(seed)
    indices = list(range(len(log)))
    rng.shuffle(indices)
    indices = sorted(indices[:max_traces])
    
    sampled = EventLog()
    for i in indices:
        sampled.append(log[i])
    
    return sampled


def get_log_statistics(log: EventLog) -> Dict[str, Any]:
    """Compute event log statistics."""
    trace_lengths = [len(trace) for trace in log]
    activities = set()
    for trace in log:
        for event in trace:
            if "concept:name" in event:
                activities.add(event["concept:name"])
    
    return {
        "num_traces": len(log),
        "num_activities": len(activities),
        "min_trace_length": min(trace_lengths) if trace_lengths else 0,
        "max_trace_length": max(trace_lengths) if trace_lengths else 0,
        "avg_trace_length": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0,
    }


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def count_invisible_transitions(net) -> int:
    """Count invisible (tau) transitions in a Petri net."""
    return sum(1 for t in net.transitions if t.label is None)


def evaluate_model(log: EventLog, net, im, fm) -> Tuple[float, float]:
    """
    Evaluate fitness and precision using token-based replay.
    
    Returns:
        (fitness, precision) tuple
    """
    try:
        # Fitness via token-based replay
        fitness_result = pm4py.fitness_token_based_replay(log, net, im, fm)
        fitness = fitness_result["average_trace_fitness"]
        
        # Precision via token-based replay
        precision = pm4py.precision_token_based_replay(log, net, im, fm)
        
        return fitness, precision
        
    except Exception as e:
        print(f"    ⚠ Evaluation failed: {e}")
        return 0.0, 0.0


def get_model_structure(net, im, fm) -> Dict[str, int]:
    """Get model structural metrics."""
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
        "invisible_transitions": count_invisible_transitions(net),
    }


# ==============================================================================
# DISCOVERY ALGORITHMS
# ==============================================================================

def discover_inductive_miner(
    log: EventLog,
    noise_threshold: float = 0.0,
    eval_log: Optional[EventLog] = None
) -> Optional[Tuple]:
    """
    Discover model using Inductive Miner with frequency filtering (IMf).
    
    Args:
        log: Event log for discovery
        noise_threshold: Noise filtering threshold [0.0, 1.0]
                        Lower = more restrictive (higher precision)
                        Higher = more permissive (higher fitness)
        eval_log: Optional separate log for evaluation
        
    Returns:
        (net, im, fm, discovery_time, eval_time, fitness, precision, params)
    """
    params = {"noise_threshold": noise_threshold, "variant": "IMf"}
    
    # Discovery
    start_time = time.time()
    try:
        net, im, fm = pm4py.discover_petri_net_inductive(
            log,
            noise_threshold=noise_threshold
        )
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ IM discovery failed: {e}")
        return None
    
    # Evaluation
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = evaluate_model(eval_log, net, im, fm)
    eval_time = time.time() - eval_start
    
    return net, im, fm, discovery_time, eval_time, fitness, precision, params


def discover_split_miner(
    log: EventLog,
    parallelism_threshold: float = 0.1,
    frequency_threshold: float = 0.4,
    eval_log: Optional[EventLog] = None
) -> Optional[Tuple]:
    """
    Discover model using Split Miner.
    
    Split Miner excels at balancing fitness and precision for
    complex real-world processes with parallelism.
    
    Args:
        log: Event log for discovery
        parallelism_threshold: Threshold for detecting parallelism [0.0, 1.0]
        frequency_threshold: Threshold for filtering infrequent behavior [0.0, 1.0]
        eval_log: Optional separate log for evaluation
        
    Returns:
        (net, im, fm, discovery_time, eval_time, fitness, precision, params)
    """
    params = {
        "parallelism_threshold": parallelism_threshold,
        "frequency_threshold": frequency_threshold,
        "variant": "SplitMiner"
    }
    
    # Check if Split Miner is available
    try:
        from pm4py.algo.discovery.splitminer import algorithm as split_miner
    except ImportError:
        print("    ⚠ Split Miner not available in this PM4Py version")
        return None
    
    # Discovery
    start_time = time.time()
    try:
        # Split Miner produces BPMN, convert to Petri net
        bpmn_graph = split_miner.apply(
            log,
            parameters={
                "parallelism_threshold": parallelism_threshold,
                "frequency_threshold": frequency_threshold
            }
        )
        
        # Convert BPMN to Petri net
        from pm4py.objects.conversion.bpmn import converter as bpmn_converter
        net, im, fm = bpmn_converter.apply(bpmn_graph)
        
        discovery_time = time.time() - start_time
        
    except Exception as e:
        print(f"    ⚠ Split Miner discovery failed: {e}")
        return None
    
    # Evaluation
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = evaluate_model(eval_log, net, im, fm)
    eval_time = time.time() - eval_start
    
    return net, im, fm, discovery_time, eval_time, fitness, precision, params


def discover_ilp_miner(
    log: EventLog,
    alpha: float = 1.0,
    eval_log: Optional[EventLog] = None
) -> Optional[Tuple]:
    """
    Discover model using ILP Miner.
    
    ILP Miner uses integer linear programming to find globally
    optimal models. Good for structured processes.
    
    Args:
        log: Event log for discovery
        alpha: Weight parameter for the ILP formulation
        eval_log: Optional separate log for evaluation
        
    Returns:
        (net, im, fm, discovery_time, eval_time, fitness, precision, params)
    """
    params = {"alpha": alpha, "variant": "ILPMiner"}
    
    # Check if ILP Miner is available
    try:
        from pm4py.algo.discovery.ilp import algorithm as ilp_miner
    except ImportError:
        print("    ⚠ ILP Miner not available in this PM4Py version")
        return None
    
    # Discovery
    start_time = time.time()
    try:
        net, im, fm = ilp_miner.apply(log, parameters={"alpha": alpha})
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ ILP Miner discovery failed: {e}")
        return None
    
    # Evaluation
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = evaluate_model(eval_log, net, im, fm)
    eval_time = time.time() - eval_start
    
    return net, im, fm, discovery_time, eval_time, fitness, precision, params


def discover_heuristic_miner(
    log: EventLog,
    dependency_threshold: float = 0.8,
    and_threshold: float = 0.8,
    eval_log: Optional[EventLog] = None
) -> Optional[Tuple]:
    """
    Discover model using Heuristics Miner.
    
    Args:
        log: Event log for discovery
        dependency_threshold: Threshold for dependency relations [0.0, 1.0]
        and_threshold: Threshold for AND splits [0.0, 1.0]
        eval_log: Optional separate log for evaluation
        
    Returns:
        (net, im, fm, discovery_time, eval_time, fitness, precision, params)
    """
    params = {
        "dependency_threshold": dependency_threshold,
        "and_threshold": and_threshold,
        "variant": "HeuristicMiner"
    }
    
    # Discovery
    start_time = time.time()
    try:
        net, im, fm = pm4py.discover_petri_net_heuristics(
            log,
            dependency_threshold=dependency_threshold,
            and_threshold=and_threshold
        )
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ Heuristic Miner discovery failed: {e}")
        return None
    
    # Evaluation
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = evaluate_model(eval_log, net, im, fm)
    eval_time = time.time() - eval_start
    
    return net, im, fm, discovery_time, eval_time, fitness, precision, params


def discover_alpha_miner(
    log: EventLog,
    eval_log: Optional[EventLog] = None
) -> Optional[Tuple]:
    """
    Discover model using Alpha Miner.
    
    Alpha Miner typically produces high-precision models but
    may have lower fitness on noisy logs.
    
    Returns:
        (net, im, fm, discovery_time, eval_time, fitness, precision, params)
    """
    params = {"variant": "AlphaMiner"}
    
    # Discovery
    start_time = time.time()
    try:
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ Alpha Miner discovery failed: {e}")
        return None
    
    # Evaluation
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = evaluate_model(eval_log, net, im, fm)
    eval_time = time.time() - eval_start
    
    return net, im, fm, discovery_time, eval_time, fitness, precision, params


# ==============================================================================
# EXPLORATION STRATEGIES
# ==============================================================================

def explore_inductive_miner_space(
    log: EventLog,
    dataset_name: str,
    output_dir: Path,
    targets: QualityTargets,
    eval_log: Optional[EventLog] = None,
    quick_mode: bool = False
) -> List[DiscoveredModel]:
    """
    Systematically explore Inductive Miner parameter space.
    
    Strategy: Test noise thresholds from 0.0 (restrictive) to 0.8 (permissive)
    to find models meeting quality targets.
    """
    print("\n" + "─" * 70)
    print("📊 INDUCTIVE MINER (IMf) - Noise Threshold Exploration")
    print("─" * 70)
    
    # Define noise grid based on mode
    if quick_mode:
        noise_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    else:
        # Fine-grained grid for thorough exploration
        noise_grid = [
            0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18,
            0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80
        ]
    
    discovered = []
    seen_structures = set()  # Track unique models by structure
    
    for noise in noise_grid:
        result = discover_inductive_miner(log, noise, eval_log)
        
        if result is None:
            continue
        
        net, im, fm, disc_time, eval_time, fitness, precision, params = result
        structure = get_model_structure(net, im, fm)
        
        # Skip duplicate structures
        struct_key = (structure["places"], structure["transitions"], structure["arcs"])
        if struct_key in seen_structures:
            print(f"  noise={noise:.2f} → duplicate structure, skipping")
            continue
        seen_structures.add(struct_key)
        
        # Check quality
        meets_fitness = fitness >= targets.min_fitness
        meets_precision = precision >= targets.min_precision
        score = targets.quality_score(fitness, precision)
        
        status = "✓✓" if meets_fitness and meets_precision else ("✓" if meets_fitness else "✗")
        print(f"  noise={noise:.2f} → f={fitness:.4f} p={precision:.4f} "
              f"[{structure['places']}P, {structure['transitions']}T, "
              f"{structure['invisible_transitions']}τ] {status}")
        
        # Save model if meets minimum fitness requirement
        if meets_fitness:
            filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_IMf_n{noise:.2f}_model.pkl"
            filepath = output_dir / filename
            
            with open(filepath, "wb") as f:
                pickle.dump({"net": net, "im": im, "fm": fm}, f)
            
            model = DiscoveredModel(
                dataset_name=dataset_name,
                discovery_method="InductiveMiner_IMf",
                parameters=params,
                places=structure["places"],
                transitions=structure["transitions"],
                arcs=structure["arcs"],
                invisible_transitions=structure["invisible_transitions"],
                fitness=fitness,
                precision=precision,
                discovery_time=disc_time,
                evaluation_time=eval_time,
                filename=filename,
                meets_min_fitness=meets_fitness,
                meets_min_precision=meets_precision,
                quality_score=score
            )
            discovered.append(model)
    
    return discovered


def explore_split_miner_space(
    log: EventLog,
    dataset_name: str,
    output_dir: Path,
    targets: QualityTargets,
    eval_log: Optional[EventLog] = None,
    quick_mode: bool = False
) -> List[DiscoveredModel]:
    """
    Explore Split Miner parameter space.
    
    Split Miner is particularly good for processes with parallelism.
    """
    print("\n" + "─" * 70)
    print("📊 SPLIT MINER - Parallelism/Frequency Exploration")
    print("─" * 70)
    
    # Parameter grid
    if quick_mode:
        param_grid = [
            (0.1, 0.3), (0.1, 0.5), (0.2, 0.4)
        ]
    else:
        param_grid = [
            (0.05, 0.2), (0.05, 0.4), (0.05, 0.6),
            (0.1, 0.2), (0.1, 0.4), (0.1, 0.6),
            (0.2, 0.2), (0.2, 0.4), (0.2, 0.6),
            (0.3, 0.3), (0.3, 0.5),
        ]
    
    discovered = []
    seen_structures = set()
    
    for par_thresh, freq_thresh in param_grid:
        result = discover_split_miner(log, par_thresh, freq_thresh, eval_log)
        
        if result is None:
            continue
        
        net, im, fm, disc_time, eval_time, fitness, precision, params = result
        structure = get_model_structure(net, im, fm)
        
        # Skip duplicates
        struct_key = (structure["places"], structure["transitions"], structure["arcs"])
        if struct_key in seen_structures:
            continue
        seen_structures.add(struct_key)
        
        meets_fitness = fitness >= targets.min_fitness
        meets_precision = precision >= targets.min_precision
        score = targets.quality_score(fitness, precision)
        
        status = "✓✓" if meets_fitness and meets_precision else ("✓" if meets_fitness else "✗")
        print(f"  par={par_thresh:.2f} freq={freq_thresh:.2f} → "
              f"f={fitness:.4f} p={precision:.4f} {status}")
        
        if meets_fitness:
            filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_SM_p{par_thresh:.2f}_f{freq_thresh:.2f}_model.pkl"
            filepath = output_dir / filename
            
            with open(filepath, "wb") as f:
                pickle.dump({"net": net, "im": im, "fm": fm}, f)
            
            model = DiscoveredModel(
                dataset_name=dataset_name,
                discovery_method="SplitMiner",
                parameters=params,
                places=structure["places"],
                transitions=structure["transitions"],
                arcs=structure["arcs"],
                invisible_transitions=structure["invisible_transitions"],
                fitness=fitness,
                precision=precision,
                discovery_time=disc_time,
                evaluation_time=eval_time,
                filename=filename,
                meets_min_fitness=meets_fitness,
                meets_min_precision=meets_precision,
                quality_score=score
            )
            discovered.append(model)
    
    return discovered


def explore_ilp_miner_space(
    log: EventLog,
    dataset_name: str,
    output_dir: Path,
    targets: QualityTargets,
    eval_log: Optional[EventLog] = None,
    quick_mode: bool = False
) -> List[DiscoveredModel]:
    """
    Explore ILP Miner parameter space.
    """
    print("\n" + "─" * 70)
    print("📊 ILP MINER - Global Optimization")
    print("─" * 70)
    
    # ILP Miner has fewer parameters to tune
    alpha_values = [0.5, 1.0, 2.0] if quick_mode else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    discovered = []
    seen_structures = set()
    
    for alpha in alpha_values:
        result = discover_ilp_miner(log, alpha, eval_log)
        
        if result is None:
            continue
        
        net, im, fm, disc_time, eval_time, fitness, precision, params = result
        structure = get_model_structure(net, im, fm)
        
        struct_key = (structure["places"], structure["transitions"], structure["arcs"])
        if struct_key in seen_structures:
            continue
        seen_structures.add(struct_key)
        
        meets_fitness = fitness >= targets.min_fitness
        meets_precision = precision >= targets.min_precision
        score = targets.quality_score(fitness, precision)
        
        status = "✓✓" if meets_fitness and meets_precision else ("✓" if meets_fitness else "✗")
        print(f"  alpha={alpha:.2f} → f={fitness:.4f} p={precision:.4f} {status}")
        
        if meets_fitness:
            filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_ILP_a{alpha:.2f}_model.pkl"
            filepath = output_dir / filename
            
            with open(filepath, "wb") as f:
                pickle.dump({"net": net, "im": im, "fm": fm}, f)
            
            model = DiscoveredModel(
                dataset_name=dataset_name,
                discovery_method="ILPMiner",
                parameters=params,
                places=structure["places"],
                transitions=structure["transitions"],
                arcs=structure["arcs"],
                invisible_transitions=structure["invisible_transitions"],
                fitness=fitness,
                precision=precision,
                discovery_time=disc_time,
                evaluation_time=eval_time,
                filename=filename,
                meets_min_fitness=meets_fitness,
                meets_min_precision=meets_precision,
                quality_score=score
            )
            discovered.append(model)
    
    return discovered


def explore_heuristic_miner_space(
    log: EventLog,
    dataset_name: str,
    output_dir: Path,
    targets: QualityTargets,
    eval_log: Optional[EventLog] = None,
    quick_mode: bool = False
) -> List[DiscoveredModel]:
    """
    Explore Heuristics Miner parameter space.
    """
    print("\n" + "─" * 70)
    print("📊 HEURISTIC MINER - Dependency Threshold Exploration")
    print("─" * 70)
    
    if quick_mode:
        dep_grid = [0.5, 0.7, 0.9]
    else:
        dep_grid = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    discovered = []
    seen_structures = set()
    
    for dep_thresh in dep_grid:
        result = discover_heuristic_miner(log, dep_thresh, dep_thresh, eval_log)
        
        if result is None:
            continue
        
        net, im, fm, disc_time, eval_time, fitness, precision, params = result
        structure = get_model_structure(net, im, fm)
        
        struct_key = (structure["places"], structure["transitions"], structure["arcs"])
        if struct_key in seen_structures:
            continue
        seen_structures.add(struct_key)
        
        meets_fitness = fitness >= targets.min_fitness
        meets_precision = precision >= targets.min_precision
        score = targets.quality_score(fitness, precision)
        
        status = "✓✓" if meets_fitness and meets_precision else ("✓" if meets_fitness else "✗")
        print(f"  dep={dep_thresh:.2f} → f={fitness:.4f} p={precision:.4f} {status}")
        
        if meets_fitness:
            filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_HM_d{dep_thresh:.2f}_model.pkl"
            filepath = output_dir / filename
            
            with open(filepath, "wb") as f:
                pickle.dump({"net": net, "im": im, "fm": fm}, f)
            
            model = DiscoveredModel(
                dataset_name=dataset_name,
                discovery_method="HeuristicMiner",
                parameters=params,
                places=structure["places"],
                transitions=structure["transitions"],
                arcs=structure["arcs"],
                invisible_transitions=structure["invisible_transitions"],
                fitness=fitness,
                precision=precision,
                discovery_time=disc_time,
                evaluation_time=eval_time,
                filename=filename,
                meets_min_fitness=meets_fitness,
                meets_min_precision=meets_precision,
                quality_score=score
            )
            discovered.append(model)
    
    return discovered


# ==============================================================================
# MAIN EXPLORATION ORCHESTRATOR
# ==============================================================================

def explore_quality_model_space(
    log: EventLog,
    dataset_name: str,
    output_dir: Path,
    targets: QualityTargets,
    max_eval_traces: Optional[int] = None,
    quick_mode: bool = False,
    algorithms: Optional[List[str]] = None
) -> List[DiscoveredModel]:
    """
    Explore the fitness-precision space to discover quality models.
    
    Args:
        log: Event log
        dataset_name: Name for output files
        output_dir: Directory for saving models
        targets: Quality targets
        max_eval_traces: Maximum traces for evaluation (None = full log)
        quick_mode: Use coarse parameter grids for faster exploration
        algorithms: List of algorithms to use (default: all)
        
    Returns:
        List of discovered models meeting quality requirements
    """
    print("\n" + "=" * 70)
    print(f"EXPLORING MODEL SPACE: {dataset_name}")
    print("=" * 70)
    print(f"Targets: fitness ≥ {targets.min_fitness:.2f}, precision ≥ {targets.min_precision:.2f}")
    print(f"Mode: {'Quick' if quick_mode else 'Thorough'}")
    
    # Prepare evaluation log
    if max_eval_traces is not None:
        eval_log = sample_log(log, max_traces=max_eval_traces)
        print(f"Evaluation: sampled {len(eval_log)} traces (from {len(log)})")
    else:
        eval_log = log
        print(f"Evaluation: full log ({len(log)} traces)")
    
    # Default algorithms
    if algorithms is None:
        algorithms = ["IM", "SM", "ILP", "HM"]
    
    all_discovered = []
    
    # Inductive Miner
    if "IM" in algorithms:
        models = explore_inductive_miner_space(
            log, dataset_name, output_dir, targets, eval_log, quick_mode
        )
        all_discovered.extend(models)
        print(f"  → {len(models)} models meeting criteria")
    
    # Split Miner
    if "SM" in algorithms:
        models = explore_split_miner_space(
            log, dataset_name, output_dir, targets, eval_log, quick_mode
        )
        all_discovered.extend(models)
        print(f"  → {len(models)} models meeting criteria")
    
    # ILP Miner
    if "ILP" in algorithms:
        models = explore_ilp_miner_space(
            log, dataset_name, output_dir, targets, eval_log, quick_mode
        )
        all_discovered.extend(models)
        print(f"  → {len(models)} models meeting criteria")
    
    # Heuristic Miner
    if "HM" in algorithms:
        models = explore_heuristic_miner_space(
            log, dataset_name, output_dir, targets, eval_log, quick_mode
        )
        all_discovered.extend(models)
        print(f"  → {len(models)} models meeting criteria")
    
    return all_discovered


# ==============================================================================
# ANALYSIS AND REPORTING
# ==============================================================================

def analyze_discovered_models(models: List[DiscoveredModel], targets: QualityTargets):
    """Analyze and report on discovered models."""
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    
    if not models:
        print("⚠ No models meeting criteria were discovered!")
        return
    
    # Basic statistics
    print(f"\n📊 Total models discovered: {len(models)}")
    
    # By algorithm
    by_method = {}
    for m in models:
        method = m.discovery_method.split("_")[0]
        by_method[method] = by_method.get(method, 0) + 1
    
    print("\n   By algorithm:")
    for method, count in sorted(by_method.items()):
        print(f"      {method}: {count} models")
    
    # Quality distribution
    fitness_values = [m.fitness for m in models]
    precision_values = [m.precision for m in models]
    
    print(f"\n   Fitness range:   [{min(fitness_values):.4f}, {max(fitness_values):.4f}]")
    print(f"   Precision range: [{min(precision_values):.4f}, {max(precision_values):.4f}]")
    
    # Meeting both criteria
    meet_both = [m for m in models if m.meets_min_fitness and m.meets_min_precision]
    print(f"\n   Meeting both targets (f≥{targets.min_fitness}, p≥{targets.min_precision}): "
          f"{len(meet_both)}/{len(models)}")
    
    # Top models
    sorted_models = sorted(models, key=lambda m: m.quality_score, reverse=True)
    print("\n   Top 5 models by quality score:")
    for i, m in enumerate(sorted_models[:5], 1):
        print(f"      {i}. f={m.fitness:.4f} p={m.precision:.4f} "
              f"[{m.places}P, {m.transitions}T, {m.invisible_transitions}τ] "
              f"({m.discovery_method})")


def save_discovery_summary(
    models: List[DiscoveredModel],
    output_dir: Path,
    dataset_name: str,
    log_stats: Dict[str, Any]
):
    """Save discovery summary to CSV and JSON."""
    if not models:
        print("⚠ No models to save!")
        return
    
    # Create DataFrame
    df = pd.DataFrame([m.to_dict() for m in models])
    df = df.sort_values("quality_score", ascending=False)
    
    # Save CSV
    csv_path = output_dir / f"{dataset_name}_quality_models_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV summary: {csv_path}")
    
    # Save JSON with metadata
    json_data = {
        "dataset": dataset_name,
        "log_statistics": log_stats,
        "models": [m.to_dict() for m in models],
        "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    json_path = output_dir / f"{dataset_name}_quality_models_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ JSON summary: {json_path}")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Discover process models with targeted fitness and precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic discovery for BPI2015 municipality 1
    python discover_quality_models.py data/BPIC15_1.xes
    
    # Custom quality targets
    python discover_quality_models.py data/BPIC15_1.xes --min-fitness 0.85 --min-precision 0.80
    
    # Quick exploration (fewer parameter combinations)
    python discover_quality_models.py data/BPIC15_1.xes --quick
    
    # Specific algorithms only
    python discover_quality_models.py data/BPIC15_1.xes --algorithms IM SM
        """
    )
    
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to XES event log file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models (default: models)"
    )
    
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.80,
        help="Minimum fitness threshold (default: 0.80)"
    )
    
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.75,
        help="Minimum precision threshold (default: 0.75)"
    )
    
    parser.add_argument(
        "--max-eval-traces",
        type=int,
        default=None,
        help="Maximum traces for evaluation (default: full log)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with coarse parameter grids"
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["IM", "SM", "ILP", "HM", "Alpha"],
        default=None,
        help="Discovery algorithms to use (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        return 1
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup targets
    targets = QualityTargets(
        min_fitness=args.min_fitness,
        min_precision=args.min_precision
    )
    
    dataset_name = log_path.stem
    
    # Header
    print("=" * 70)
    print("QUALITY-TARGETED MODEL DISCOVERY")
    print("=" * 70)
    print(f"Dataset:   {log_path}")
    print(f"Output:    {output_dir}")
    print(f"Targets:   fitness ≥ {targets.min_fitness:.2f}, precision ≥ {targets.min_precision:.2f}")
    print("=" * 70)
    
    # Load log
    start_time = time.time()
    log = load_event_log(log_path)
    log_stats = get_log_statistics(log)
    
    print(f"\nLog statistics:")
    print(f"  Traces:     {log_stats['num_traces']}")
    print(f"  Activities: {log_stats['num_activities']}")
    print(f"  Avg length: {log_stats['avg_trace_length']:.1f} events")
    
    # Discover models
    models = explore_quality_model_space(
        log=log,
        dataset_name=dataset_name,
        output_dir=output_dir,
        targets=targets,
        max_eval_traces=args.max_eval_traces,
        quick_mode=args.quick,
        algorithms=args.algorithms
    )
    
    # Analysis
    analyze_discovered_models(models, targets)
    
    # Save summary
    save_discovery_summary(models, output_dir, dataset_name, log_stats)
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE - Total time: {total_time:.1f}s")
    print(f"Discovered {len(models)} quality models → {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
