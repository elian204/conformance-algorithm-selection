"""
algorithms/astar_backward.py
-----------------------------
Backward A* for optimal alignment computation.

Searches RG(SP)^rev from m_f toward m_i using admissible heuristic h_B.

Semantics:
- g_B(m) = best cost from m to m_f in the original graph
         = best cost from m_f to m in the reversed graph
- h_B(m) = lower bound on cost from m_i to m in the original graph
         = lower bound on cost from m to m_i in the reversed graph

OPTIMIZED: Lazy heuristic evaluation — h(m) is computed only when m is
popped for expansion, not when m is generated.
"""
from __future__ import annotations
import heapq
import time
from typing import Dict, List, Optional, Tuple

from core.petri_net import Marking
from core.synchronous_product import SynchronousProduct, SPTransition
from heuristics.base import Heuristic
from algorithms.astar_forward import SearchResult
from utils.alignment import Alignment, AlignmentMove


def astar_backward(
    sp: SynchronousProduct,
    heuristic: Heuristic,
    max_expansions: int = 10_000_000
) -> SearchResult:
    """
    Backward A* on RG(SP)^rev with LAZY heuristic evaluation.

    Parameters
    ----------
    sp           : SynchronousProduct
    heuristic    : admissible h_B(m) -> lower bound on cost from m_i to m
                   (i.e., lower bound on cost to reach m_i in reversed graph)
    max_expansions : safety cap

    Returns
    -------
    SearchResult with optimal cost, alignment, and search statistics.
    """
    t_start = time.perf_counter()

    m_i = sp.initial_marking
    m_f = sp.final_marking

    # Start backward from m_f; goal is m_i
    g: Dict[Marking, float] = {m_f: 0.0}
    
    # h_computed[m] = cached h value
    h_computed: Dict[Marking, float] = {}
    
    parent: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_f: (None, None)
    }

    counter = 0
    
    # Compute h for initial node (m_f in backward search)
    h_f = heuristic(m_f)
    h_computed[m_f] = h_f
    heuristic_calls = 1
    
    # (f, g, tie_id, marking, h_known)
    open_heap: List[Tuple[float, float, int, Marking, bool]] = [
        (h_f, 0.0, counter, m_f, True)
    ]

    closed: Dict[Marking, float] = {}

    expansions = 0
    generations = 1

    while open_heap:
        f_val, g_val, _, m, h_known = heapq.heappop(open_heap)

        if g_val > g.get(m, float('inf')) + 1e-12:
            continue

        if m in closed:
            if g_val >= closed[m]:
                continue

        # LAZY EVALUATION
        if not h_known:
            if m not in h_computed:
                h_computed[m] = heuristic(m)
                heuristic_calls += 1
            
            true_h = h_computed[m]
            true_f = g_val + true_h
            
            if true_f > f_val + 1e-9:
                counter += 1
                heapq.heappush(open_heap, (true_f, g_val, counter, m, True))
                continue

        closed[m] = g_val

        # Goal check (backward goal = initial marking)
        if m == m_i:
            alignment = _reconstruct_backward(parent, m_i, m_f)
            return SearchResult(
                optimal_cost=g_val,
                alignment=alignment,
                expansions=expansions,
                generations=generations,
                time_seconds=time.perf_counter() - t_start,
                algorithm="backward_astar",
                heuristic=heuristic.name,
                heuristic_calls=heuristic_calls,
            )

        expansions += 1
        if expansions > max_expansions:
            break

        # Predecessors in the original graph = successors in the reversed graph
        for sp_t, pred, cost in sp.predecessors(m):
            new_g = g_val + cost
            if new_g < g.get(pred, float('inf')):
                g[pred] = new_g
                parent[pred] = (m, sp_t)  # backward: pred -> m via sp_t
                counter += 1
                
                if pred in h_computed:
                    h_val = h_computed[pred]
                    heapq.heappush(open_heap, (new_g + h_val, new_g, counter, pred, True))
                else:
                    # LAZY: push with f = g
                    heapq.heappush(open_heap, (new_g, new_g, counter, pred, False))
                
                generations += 1

    return SearchResult(
        optimal_cost=float('inf'),
        alignment=None,
        expansions=expansions,
        generations=generations,
        time_seconds=time.perf_counter() - t_start,
        algorithm="backward_astar",
        heuristic=heuristic.name,
        heuristic_calls=heuristic_calls,
    )


def _reconstruct_backward(
    parent: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    start: Marking,  # m_i
    end: Marking,    # m_f
) -> Alignment:
    """
    Reconstruct alignment from backward search.
    The parent chain goes: m_i -> ... -> m_f (backward tree).
    Walk from m_i forward through the chain (which was built backward).
    """
    moves = []
    m = start
    while parent[m][0] is not None:
        next_m, sp_t = parent[m]
        moves.append(AlignmentMove(
            move_type=sp_t.move_type.name,
            model_label=sp_t.model_label,
            log_label=sp_t.log_label,
            cost=sp_t.cost
        ))
        m = next_m
    # Moves are in forward order (m_i to m_f) since we traversed backward tree
    return Alignment(moves=moves)
