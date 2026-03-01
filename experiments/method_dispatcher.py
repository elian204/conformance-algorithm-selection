"""
experiments/method_dispatcher.py
---------------------------------
Dispatches method configurations to the appropriate algorithm and heuristic.

This module provides a unified interface for running any of the 15 methods
without the runner needing to know the details of each algorithm.

Supported Algorithms:
  - forward: Standard forward A*
  - backward: Backward A* on reversed graph
  - bidir_std: Standard bidirectional A* (Front-to-End)
  - bidir_mm: MM algorithm (Holte et al. 2016)
  - dibbs: DIBBS (Sewell & Jacobson 2021)

Supported Heuristics:
  - zero: h(m) = 0 (Dijkstra's algorithm)
  - me: Marking equation LP relaxation
  - mmr: MMR/REACH heuristic

Cost comparison with τ-epsilon:
  All cost comparisons use round() to extract the integer deviation cost.
  This ensures correct handling of τ-epsilon costs.

Winner selection:
  The winner is the method with minimum CPU time (not expansions).
"""

from typing import Optional, Any, Tuple

from core.synchronous_product import SynchronousProduct

# Import algorithms
from algorithms.astar_forward import astar_forward
from algorithms.astar_backward import astar_backward
from algorithms.astar_bidirectional import (
    astar_bidirectional_standard,
    astar_bidirectional_mm,
    SearchResult,
)
from algorithms.astar_dibbs import astar_dibbs

# Import heuristics - handle different possible class names
from heuristics.zero import ZeroHeuristic

# Marking equation - use factory function for auto scipy fallback
try:
    from heuristics.marking_equation import create_marking_equation_heuristic
    _ME_AVAILABLE = True
except ImportError:
    _ME_AVAILABLE = False
    create_marking_equation_heuristic = None

# MMR/REACH - try different possible names
try:
    from heuristics.mmr import MMRHeuristic
except ImportError:
    try:
        from heuristics.reach import MMRHeuristic
    except ImportError:
        try:
            from heuristics.reach import REACHHeuristic as MMRHeuristic
        except ImportError:
            MMRHeuristic = None

from experiments.methods_config import MethodConfig


# =============================================================================
# Cost comparison utilities for τ-epsilon
# =============================================================================

def costs_equal(cost1: float, cost2: float) -> bool:
    """
    Check if two alignment costs represent the same deviation cost.

    With τ-epsilon costs, two alignments have equal deviation cost
    if round(cost1) == round(cost2).
    """
    return round(cost1) == round(cost2)


def deviation_cost(total_cost: float) -> int:
    """
    Extract integer deviation cost from total alignment cost.

    With τ-epsilon costs, deviation_cost = round(total_cost).
    """
    return round(total_cost)


def create_heuristic(
    heuristic_type: str,
    sp: SynchronousProduct,
    direction: str,  # "forward" or "backward"
    timeout_seconds: Optional[float] = None,
):
    """
    Factory function for creating heuristic instances.

    Parameters
    ----------
    heuristic_type : str
        One of "zero", "me", "mmr"
    sp : SynchronousProduct
        The synchronous product
    direction : str
        "forward" or "backward"

    Returns
    -------
    Heuristic instance with estimate(marking) -> float method
    """
    if heuristic_type == "zero":
        # Try different constructor signatures for ZeroHeuristic
        try:
            return ZeroHeuristic()
        except TypeError:
            try:
                return ZeroHeuristic(sp)
            except TypeError:
                return ZeroHeuristic(sp, direction=direction)

    elif heuristic_type == "me":
        if not _ME_AVAILABLE:
            raise ImportError("Marking equation heuristic not available. "
                              "Check heuristics/marking_equation.py")
        return create_marking_equation_heuristic(
            sp,
            direction=direction,
            timeout_seconds=timeout_seconds,
        )

    elif heuristic_type == "mmr":
        if MMRHeuristic is None:
            raise ImportError("MMRHeuristic not available. "
                              "Check heuristics/mmr.py or heuristics/reach.py")
        return MMRHeuristic(sp, direction=direction)

    else:
        raise ValueError(f"Unknown heuristic: {heuristic_type}")


def run_method(
    method: MethodConfig,
    sp: SynchronousProduct,
    max_expansions: int = 1_000_000,
    collect_stats: bool = True,
    timeout_seconds: Optional[float] = None,
) -> SearchResult:
    """
    Run a single method (algorithm + heuristic) on a synchronous product.

    Parameters
    ----------
    method : MethodConfig
        The method configuration specifying algorithm and heuristic.
    sp : SynchronousProduct
        The synchronous product of workflow net and trace.
    max_expansions : int
        Safety limit on number of expansions.
    collect_stats : bool
        Whether to collect detailed statistics.

    Returns
    -------
    SearchResult
        Contains optimal cost, alignment, expansions, timing, and stats.

    Examples
    --------
    >>> from experiments.methods_config import MethodConfig
    >>> method = MethodConfig("dibbs", "me")
    >>> result = run_method(method, sp)
    >>> print(f"C* = {result.optimal_cost}, expansions = {result.expansions}")
    """
    algorithm = method.algorithm
    heuristic = method.heuristic

    # =========================================================================
    # Forward A*
    # =========================================================================
    if algorithm == "forward":
        h = create_heuristic(heuristic, sp, "forward", timeout_seconds=timeout_seconds)
        return astar_forward(
            sp=sp,
            heuristic=h,
            max_expansions=max_expansions
        )

    # =========================================================================
    # Backward A*
    # =========================================================================
    elif algorithm == "backward":
        h = create_heuristic(heuristic, sp, "backward", timeout_seconds=timeout_seconds)
        return astar_backward(
            sp=sp,
            heuristic=h,
            max_expansions=max_expansions
        )

    # =========================================================================
    # Standard Bidirectional A*
    # =========================================================================
    elif algorithm == "bidir_std":
        h_f = create_heuristic(heuristic, sp, "forward", timeout_seconds=timeout_seconds)
        h_b = create_heuristic(heuristic, sp, "backward", timeout_seconds=timeout_seconds)
        return astar_bidirectional_standard(
            sp=sp,
            heuristic_f=h_f,
            heuristic_b=h_b,
            max_expansions=max_expansions,
            collect_stats=collect_stats
        )

    # =========================================================================
    # MM Bidirectional A* (Holte et al. 2016)
    # =========================================================================
    elif algorithm == "bidir_mm":
        h_f = create_heuristic(heuristic, sp, "forward", timeout_seconds=timeout_seconds)
        h_b = create_heuristic(heuristic, sp, "backward", timeout_seconds=timeout_seconds)
        return astar_bidirectional_mm(
            sp=sp,
            heuristic_f=h_f,
            heuristic_b=h_b,
            max_expansions=max_expansions,
            collect_stats=collect_stats
        )

    # =========================================================================
    # DIBBS: Dynamically Improved Bounds Bidirectional Search
    # (Sewell & Jacobson, 2021)
    # =========================================================================
    elif algorithm == "dibbs":
        h_f = create_heuristic(heuristic, sp, "forward", timeout_seconds=timeout_seconds)
        h_b = create_heuristic(heuristic, sp, "backward", timeout_seconds=timeout_seconds)
        return astar_dibbs(
            sp=sp,
            heuristic_f=h_f,
            heuristic_b=h_b,
            max_expansions=max_expansions,
            collect_stats=collect_stats
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_all_methods(
    sp: SynchronousProduct,
    methods: list = None,
    max_expansions: int = 1_000_000,
    collect_stats: bool = True
) -> dict:
    """
    Run all specified methods and return results as a dictionary.

    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product.
    methods : list of MethodConfig, optional
        Methods to run. Defaults to all 15 methods.
    max_expansions : int
        Safety limit per method.
    collect_stats : bool
        Whether to collect detailed statistics.

    Returns
    -------
    dict
        Mapping from method name to SearchResult.

    Example
    -------
    >>> results = run_all_methods(sp, methods=DIBBS_METHODS)
    >>> for name, result in results.items():
    ...     print(f"{name}: C*={result.optimal_cost}, exp={result.expansions}")
    """
    from experiments.methods_config import ALL_METHODS

    if methods is None:
        methods = ALL_METHODS

    results = {}
    for method in methods:
        try:
            result = run_method(method, sp, max_expansions, collect_stats)
            results[method.name] = result
        except Exception as e:
            # Log error but continue with other methods
            results[method.name] = SearchResult(
                optimal_cost=float('inf'),
                alignment=None,
                expansions=0,
                generations=0,
                time_seconds=0.0,
                algorithm=method.algorithm,
                heuristic=method.heuristic,
                stats=None
            )

    return results


def find_winner(results: dict) -> Tuple[Optional[str], float, float]:
    """
    Find the method with minimum CPU time among those finding optimal deviation cost.

    Parameters
    ----------
    results : dict
        Mapping from method name to SearchResult.

    Returns
    -------
    (winner_name, optimal_cost, min_time_seconds)
        - winner_name: Name of the winning method (or None if no solution)
        - optimal_cost: The optimal alignment cost (raw, may include ε)
        - min_time_seconds: The winner's CPU time in seconds

    Note
    ----
    With τ-epsilon costs, methods are considered to find the same optimum
    if round(cost1) == round(cost2). The winner is the method with
    minimum CPU time among those finding the optimal deviation cost.
    """
    # First find the minimum cost (for deviation cost comparison)
    finite_results = [(n, r) for n, r in results.items() if r.optimal_cost < float('inf')]

    if not finite_results:
        return (None, float('inf'), 0.0)

    # Find optimal deviation cost
    optimal_cost = min(r.optimal_cost for _, r in finite_results)
    optimal_deviation = round(optimal_cost)

    # Among methods finding optimal deviation cost, find minimum CPU time
    optimal_results = {
        name: r for name, r in finite_results
        if round(r.optimal_cost) == optimal_deviation
    }

    if not optimal_results:
        return (None, optimal_cost, 0.0)

    # Winner is the method with minimum CPU time
    winner_name = min(optimal_results.keys(), key=lambda n: optimal_results[n].time_seconds)
    winner = optimal_results[winner_name]

    return (winner_name, optimal_cost, winner.time_seconds)


def find_winner_by_expansions(results: dict) -> Tuple[Optional[str], float, int]:
    """
    Find the method with minimum expansions among those finding optimal deviation cost.

    This is an alternative winner criterion (legacy behavior).

    Parameters
    ----------
    results : dict
        Mapping from method name to SearchResult.

    Returns
    -------
    (winner_name, optimal_cost, min_expansions)
    """
    # First find the minimum cost (for deviation cost comparison)
    finite_results = [(n, r) for n, r in results.items() if r.optimal_cost < float('inf')]

    if not finite_results:
        return (None, float('inf'), 0)

    # Find optimal deviation cost
    optimal_cost = min(r.optimal_cost for _, r in finite_results)
    optimal_deviation = round(optimal_cost)

    # Among methods finding optimal deviation cost, find minimum expansions
    optimal_results = {
        name: r for name, r in finite_results
        if round(r.optimal_cost) == optimal_deviation
    }

    if not optimal_results:
        return (None, optimal_cost, 0)

    # Winner is the method with minimum expansions
    winner_name = min(optimal_results.keys(), key=lambda n: optimal_results[n].expansions)
    winner = optimal_results[winner_name]

    return (winner_name, optimal_cost, winner.expansions)
