"""
algorithms/astar_forward.py
----------------------------
Standard (forward) A* for optimal alignment computation.

Searches RG(SP) from m_i to m_f using an admissible heuristic h_F.

OPTIMIZED: Lazy heuristic evaluation — h(m) is computed only when m is
popped for expansion, not when m is generated. This dramatically reduces
the number of heuristic calls (especially important for expensive LP-based
heuristics like Marking Equation).

Approach:
- Push nodes with f = g (optimistic, assuming h=0)
- When popping, compute true h and re-insert if f changes
- Only expand when node is popped with its true f value
"""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from core.petri_net import Marking
from core.synchronous_product import SynchronousProduct, SPTransition
from heuristics.base import Heuristic
from utils.alignment import Alignment, AlignmentMove


@dataclass
class SearchResult:
    optimal_cost: float
    alignment: Optional[Alignment]
    expansions: int
    generations: int
    time_seconds: float
    algorithm: str
    heuristic: str
    heuristic_calls: int = 0  # Track actual h() invocations

    @property
    def solved(self) -> bool:
        return self.optimal_cost < float('inf')


def astar_forward(
    sp: SynchronousProduct,
    heuristic: Heuristic,
    max_expansions: int = 10_000_000
) -> SearchResult:
    """
    Forward A* on RG(SP) with LAZY heuristic evaluation.

    Parameters
    ----------
    sp           : SynchronousProduct
    heuristic    : admissible heuristic h_F(m) -> lower bound to m_f
    max_expansions : safety cap

    Returns
    -------
    SearchResult with optimal cost, alignment, and search statistics.
    """
    t_start = time.perf_counter()

    m_i = sp.initial_marking
    m_f = sp.final_marking

    # g[m] = best known cost from m_i to m
    g: Dict[Marking, float] = {m_i: 0.0}

    # h_computed[m] = cached h value (None if not yet computed)
    h_computed: Dict[Marking, float] = {}

    # parent[m] = (predecessor_marking, sp_transition_used)
    parent: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_i: (None, None)
    }

    # OPEN: min-heap of (f, g, tie_id, marking, h_known)
    # h_known: True if f = g + h(m), False if f = g (optimistic)
    counter = 0
    
    # For initial node, we DO compute h (needed for termination correctness)
    h_i = heuristic(m_i)
    h_computed[m_i] = h_i
    heuristic_calls = 1
    
    open_heap: List[Tuple[float, float, int, Marking, bool]] = [
        (h_i, 0.0, counter, m_i, True)
    ]

    closed: Dict[Marking, float] = {}  # closed[m] = g at time of expansion

    expansions = 0
    generations = 1  # m_i

    while open_heap:
        f_val, g_val, _, m, h_known = heapq.heappop(open_heap)

        # Skip stale heap entries that were superseded before expansion.
        if g_val > g.get(m, float('inf')) + 1e-12:
            continue

        # Skip if already expanded with same or better g
        if m in closed:
            if g_val >= closed[m]:
                continue

        # LAZY EVALUATION: If h not yet computed for this node, compute now
        if not h_known:
            if m not in h_computed:
                h_computed[m] = heuristic(m)
                heuristic_calls += 1
            
            true_h = h_computed[m]
            true_f = g_val + true_h
            
            # If true f is worse than what we popped with, re-insert with correct f
            if true_f > f_val + 1e-9:
                counter += 1
                heapq.heappush(open_heap, (true_f, g_val, counter, m, True))
                continue
        
        # Now we have the true f value - proceed with expansion
        closed[m] = g_val

        # Goal check
        if m == m_f:
            alignment = _reconstruct(parent, m_f, sp)
            return SearchResult(
                optimal_cost=g_val,
                alignment=alignment,
                expansions=expansions,
                generations=generations,
                time_seconds=time.perf_counter() - t_start,
                algorithm="forward_astar",
                heuristic=heuristic.name,
                heuristic_calls=heuristic_calls,
            )

        expansions += 1
        if expansions > max_expansions:
            break

        # Generate successors with LAZY h (push with f = g, h_known=False)
        for sp_t, succ, cost in sp.successors(m):
            new_g = g_val + cost
            if new_g < g.get(succ, float('inf')):
                g[succ] = new_g
                parent[succ] = (m, sp_t)
                counter += 1
                
                # Check if h already computed (from previous path)
                if succ in h_computed:
                    h_val = h_computed[succ]
                    heapq.heappush(open_heap, (new_g + h_val, new_g, counter, succ, True))
                else:
                    # LAZY: push with f = g (optimistic h=0)
                    heapq.heappush(open_heap, (new_g, new_g, counter, succ, False))
                
                generations += 1

    # Exhausted or hit cap — return best found (may be suboptimal)
    return SearchResult(
        optimal_cost=float('inf'),
        alignment=None,
        expansions=expansions,
        generations=generations,
        time_seconds=time.perf_counter() - t_start,
        algorithm="forward_astar",
        heuristic=heuristic.name,
        heuristic_calls=heuristic_calls,
    )


def _reconstruct(
    parent: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    goal: Marking,
    sp: SynchronousProduct
) -> Alignment:
    moves = []
    m = goal
    while parent[m][0] is not None:
        pred, sp_t = parent[m]
        moves.append(AlignmentMove(
            move_type=sp_t.move_type.name,
            model_label=sp_t.model_label,
            log_label=sp_t.log_label,
            cost=sp_t.cost
        ))
        m = pred
    moves.reverse()
    return Alignment(moves=moves)
