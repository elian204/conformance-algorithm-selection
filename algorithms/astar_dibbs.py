"""
algorithms/astar_dibbs.py
-------------------------
DIBBS: Dynamically Improved Bounds Bidirectional Search

Implementation of Sewell & Jacobson (2021), "Dynamically improved bounds 
bidirectional search", Artificial Intelligence, 291, 103405.

Key innovations over standard bidirectional A*:
1. Priority function: f̄_d(v) = 2g_d(v) + h_d(v) - h_{d̄}(v)
   - Uses BOTH heuristics in EACH direction
   - Dynamically improves bounds without additional computation
   
2. Termination condition: UB > (F̄_min^F + F̄_min^B) / 2
   - Terminates on or before the searches meet
   - No node is expanded in both directions
   
3. Best First Direction (BFD) rule: Expand from direction with smaller F̄_min
   - Balances exploration between directions
   - Exploits the easier direction dynamically

Theoretical guarantees (with consistent heuristics):
- Terminates on or before the first node would be expanded in both directions
- Never expands v if min{f̄*_F(v), f̄*_B(v)} > C*
- Can expand fewer nodes than either FA* or BA* alone

Notation (conformance checking):
  - m_i: initial marking
  - m_f: final marking  
  - g_F(m): cost from m_i to m (forward)
  - g_B(m): cost from m to m_f (backward, i.e., cost in original graph)
  - h_F(m): lower bound on cost from m to m_f
  - h_B(m): lower bound on cost from m_i to m
  - f̄_F(m) = 2g_F(m) + h_F(m) - h_B(m)  (forward priority)
  - f̄_B(m) = 2g_B(m) + h_B(m) - h_F(m)  (backward priority)
  - μ: upper bound on C* (best complete path found)
  - C*: optimal alignment cost

Reference:
  Sewell, E.C. & Jacobson, S.H. (2021). "Dynamically improved bounds 
  bidirectional search." Artificial Intelligence, 291, 103405.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set

from core.synchronous_product import SynchronousProduct, SPTransition
from core.petri_net import Marking

# Import Alignment and AlignmentMove
try:
    from utils.alignment import Alignment, AlignmentMove
except ImportError:
    from core.alignment import Alignment, AlignmentMove


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DIBBSStats:
    """
    Statistics for DIBBS search analysis.
    
    Attributes:
        expansions_fwd: Number of nodes expanded in forward direction
        expansions_bwd: Number of nodes expanded in backward direction
        generations_fwd: Number of nodes generated (pushed to open) forward
        generations_bwd: Number of nodes generated backward
        meeting_g_fwd: g-value at meeting point (forward direction)
        meeting_g_bwd: g-value at meeting point (backward direction)
        asymmetry_ratio: expansions_fwd / expansions_bwd (or inf)
        f_bar_min_f_at_term: F̄_min^F at termination
        f_bar_min_b_at_term: F̄_min^B at termination
        heuristic_calls_fwd: Number of h_F evaluations
        heuristic_calls_bwd: Number of h_B evaluations
    """
    expansions_fwd: int = 0
    expansions_bwd: int = 0
    generations_fwd: int = 0
    generations_bwd: int = 0
    wasted_expansions: int = 0  # For compatibility, always 0 in DIBBS
    meeting_g_fwd: float = float('inf')
    meeting_g_bwd: float = float('inf')
    asymmetry_ratio: float = 1.0
    f_bar_min_f_at_term: float = float('inf')
    f_bar_min_b_at_term: float = float('inf')
    heuristic_calls_fwd: int = 0
    heuristic_calls_bwd: int = 0
    tau_transitions_fwd: int = 0
    tau_transitions_bwd: int = 0
    g_level_expansions_fwd: Dict[int, int] = field(default_factory=dict)
    g_level_expansions_bwd: Dict[int, int] = field(default_factory=dict)
    
    def compute_asymmetry(self) -> None:
        """Compute asymmetry ratio after search completes."""
        if self.expansions_bwd > 0:
            self.asymmetry_ratio = self.expansions_fwd / self.expansions_bwd
        else:
            self.asymmetry_ratio = float('inf') if self.expansions_fwd > 0 else 1.0


@dataclass
class SearchResult:
    """
    Result of A* search for conformance checking.
    """
    optimal_cost: float
    alignment: Optional[Alignment]
    expansions: int
    generations: int
    time_seconds: float
    algorithm: str
    heuristic: str
    stats: Optional[DIBBSStats] = None

    @property
    def solved(self) -> bool:
        """Whether an optimal finite-cost solution was found."""
        return self.optimal_cost < float('inf')


# Type alias for heuristic functions
HeuristicFunc = Callable[[Marking], float]


# =============================================================================
# DIBBS Algorithm
# =============================================================================

def astar_dibbs(
    sp: SynchronousProduct,
    heuristic_f: HeuristicFunc,
    heuristic_b: HeuristicFunc,
    max_expansions: int = 1_000_000,
    collect_stats: bool = True,
    direction_rule: str = "bfd"  # "bfd" (Best First Direction) or "cardinality"
) -> SearchResult:
    """
    DIBBS: Dynamically Improved Bounds Bidirectional Search.
    
    Implements the algorithm from Sewell & Jacobson (2021).
    
    Priority function (Proposition 1 in paper):
        F̄_F(m) = 2·g_F(m) + h_F(m) - h_B(m)
        F̄_B(m) = 2·g_B(m) + h_B(m) - h_F(m)
    
    Termination condition (Theorem 11):
        UB ≤ (F̄_min^F + F̄_min^B) / 2
    
    Key property: Uses BOTH heuristics in BOTH directions to dynamically
    improve bounds, enabling expansion of fewer nodes than standard A*.
    
    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product of workflow net and trace.
    heuristic_f : Callable[[Marking], float]
        Forward heuristic h_F(m) -> estimated cost from m to m_f.
    heuristic_b : Callable[[Marking], float]
        Backward heuristic h_B(m) -> estimated cost from m_i to m.
    max_expansions : int
        Safety limit on total expansions (default 1M).
    collect_stats : bool
        If True, collect detailed statistics for analysis.
    direction_rule : str
        "bfd" - Best First Direction: expand from direction with smaller F̄_min
        "cardinality" - expand from direction with smaller open list
        
    Returns
    -------
    SearchResult
        Contains optimal cost, alignment, and search statistics.
        
    Notes
    -----
    DIBBS requires consistent heuristics for correctness. Both the marking
    equation and MMR/REACH heuristics used in conformance checking satisfy
    this requirement.
    
    Theoretical properties (Theorem 13):
    - If f̄*_F(v) > C*, then v is not expanded in forward direction
    - If f̄*_B(v) > C*, then v is not expanded in backward direction
    - DIBBS terminates on or before the searches meet (Theorem 12)
    """
    t_start = time.perf_counter()
    
    m_i = sp.initial_marking
    m_f = sp.final_marking
    
    # Statistics
    stats = DIBBSStats() if collect_stats else None
    
    # Cache heuristic values to avoid recomputation
    # h_F[m] = h_F(m), h_B[m] = h_B(m)
    h_F_cache: Dict[Marking, float] = {}
    h_B_cache: Dict[Marking, float] = {}
    
    def get_h_F(m: Marking) -> float:
        """Get cached h_F value or compute and cache."""
        if m not in h_F_cache:
            h_F_cache[m] = heuristic_f(m)
            if collect_stats:
                stats.heuristic_calls_fwd += 1
        return h_F_cache[m]
    
    def get_h_B(m: Marking) -> float:
        """Get cached h_B value or compute and cache."""
        if m not in h_B_cache:
            h_B_cache[m] = heuristic_b(m)
            if collect_stats:
                stats.heuristic_calls_bwd += 1
        return h_B_cache[m]
    
    # =========================================================================
    # Forward search state
    # =========================================================================
    # G_F[m] = best known g_F(m)
    G_F: Dict[Marking, float] = {m_i: 0.0}
    
    # F̄_F[m] = current f̄_F value (2*g_F + h_F - h_B)
    # Computed when node is generated/updated
    F_bar_F: Dict[Marking, float] = {}
    
    # Parent pointers for path reconstruction
    parent_F: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_i: (None, None)
    }
    
    # Closed set: markings that have been expanded
    closed_F: Set[Marking] = set()
    
    # Compute initial f̄_F(m_i) = 2*0 + h_F(m_i) - h_B(m_i) = h_F(m_i) - h_B(m_i)
    # But h_B(m_i) should be 0 (cost from m_i to m_i = 0)
    h_F_mi = get_h_F(m_i)
    h_B_mi = get_h_B(m_i)  # Should be ~0 for admissible h_B
    f_bar_mi = 2 * 0.0 + h_F_mi - h_B_mi
    F_bar_F[m_i] = f_bar_mi
    
    counter_F = 0
    # Heap entries: (f̄, g, counter, marking)
    open_F: List[Tuple[float, float, int, Marking]] = [
        (f_bar_mi, 0.0, counter_F, m_i)
    ]
    
    # =========================================================================
    # Backward search state
    # =========================================================================
    G_B: Dict[Marking, float] = {m_f: 0.0}
    F_bar_B: Dict[Marking, float] = {}
    
    parent_B: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_f: (None, None)
    }
    
    closed_B: Set[Marking] = set()
    
    # f̄_B(m_f) = 2*0 + h_B(m_f) - h_F(m_f)
    # h_F(m_f) should be 0 (cost from m_f to m_f = 0)
    h_B_mf = get_h_B(m_f)
    h_F_mf = get_h_F(m_f)  # Should be ~0
    f_bar_mf = 2 * 0.0 + h_B_mf - h_F_mf
    F_bar_B[m_f] = f_bar_mf
    
    counter_B = 0
    open_B: List[Tuple[float, float, int, Marking]] = [
        (f_bar_mf, 0.0, counter_B, m_f)
    ]
    
    # =========================================================================
    # Solution tracking
    # =========================================================================
    mu: float = float('inf')  # Upper bound on C*
    meeting_marking: Optional[Marking] = None
    
    expansions = 0
    generations = 2  # Initial markings
    
    if collect_stats:
        stats.generations_fwd = 1
        stats.generations_bwd = 1
    
    # =========================================================================
    # Main DIBBS loop
    # =========================================================================
    while open_F or open_B:
        
        if expansions >= max_expansions:
            break
        
        # Get F̄_min from both directions
        F_bar_min_F = open_F[0][0] if open_F else float('inf')
        F_bar_min_B = open_B[0][0] if open_B else float('inf')
        
        # ---------------------------------------------------------------------
        # DIBBS Termination condition (Theorem 11):
        #   UB ≤ (F̄_min^F + F̄_min^B) / 2
        # 
        # Equivalent to: UB * 2 ≤ F̄_min^F + F̄_min^B
        # ---------------------------------------------------------------------
        termination_bound = (F_bar_min_F + F_bar_min_B) / 2.0
        
        if mu <= termination_bound:
            if collect_stats:
                stats.f_bar_min_f_at_term = F_bar_min_F
                stats.f_bar_min_b_at_term = F_bar_min_B
            break
        
        # ---------------------------------------------------------------------
        # Direction selection
        # ---------------------------------------------------------------------
        if direction_rule == "bfd":
            # Best First Direction: expand from smaller F̄_min
            expand_forward = (F_bar_min_F <= F_bar_min_B) if open_F else False
            if not open_B:
                expand_forward = True
        else:
            # Cardinality rule: expand from smaller open list
            expand_forward = (len(open_F) <= len(open_B)) if open_F else False
            if not open_B:
                expand_forward = True
        
        if expand_forward and open_F:
            # -----------------------------------------------------------------
            # Expand forward
            # -----------------------------------------------------------------
            f_bar_m, g_m, _, m = heapq.heappop(open_F)
            
            # Skip if already closed
            if m in closed_F:
                continue
            
            # Skip if g-value is stale (a better path was found)
            if g_m > G_F.get(m, float('inf')):
                continue
            
            closed_F.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_fwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_fwd[g_int] = \
                    stats.g_level_expansions_fwd.get(g_int, 0) + 1
            
            # Check if we've reached the goal
            if m == m_f:
                candidate_cost = g_m
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    if collect_stats:
                        stats.meeting_g_fwd = g_m
                        stats.meeting_g_bwd = 0.0
            
            # Check for meeting with backward search
            if m in G_B:
                candidate_cost = G_F[m] + G_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    if collect_stats:
                        stats.meeting_g_fwd = G_F[m]
                        stats.meeting_g_bwd = G_B[m]
            
            # Expand successors
            for sp_t, m_next, cost in sp.successors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(sp_t, 'is_model_move') and sp_t.is_model_move:
                    if hasattr(sp_t, 'transition') and sp_t.transition is not None:
                        if hasattr(sp_t.transition, 'label') and sp_t.transition.label is None:
                            stats.tau_transitions_fwd += 1
                
                # Skip if not improving
                if g_new >= G_F.get(m_next, float('inf')):
                    continue
                
                G_F[m_next] = g_new
                parent_F[m_next] = (m, sp_t)
                
                # Compute f̄_F(m_next) = 2*g_F + h_F - h_B
                h_F_val = get_h_F(m_next)
                h_B_val = get_h_B(m_next)
                f_bar_new = 2 * g_new + h_F_val - h_B_val
                F_bar_F[m_next] = f_bar_new
                
                # Early pruning: skip if f̄ >= UB
                # (from Theorem 11: any path through this node costs >= UB)
                if f_bar_new >= 2 * mu:
                    continue
                
                counter_F += 1
                heapq.heappush(open_F, (f_bar_new, g_new, counter_F, m_next))
                generations += 1
                
                if collect_stats:
                    stats.generations_fwd += 1
                
                # Update UB if meeting with backward
                if m_next in G_B:
                    candidate_cost = g_new + G_B[m_next]
                    if candidate_cost < mu:
                        mu = candidate_cost
                        meeting_marking = m_next
                        if collect_stats:
                            stats.meeting_g_fwd = g_new
                            stats.meeting_g_bwd = G_B[m_next]
        
        elif open_B:
            # -----------------------------------------------------------------
            # Expand backward
            # -----------------------------------------------------------------
            f_bar_m, g_m, _, m = heapq.heappop(open_B)
            
            if m in closed_B:
                continue
            
            if g_m > G_B.get(m, float('inf')):
                continue
            
            closed_B.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_bwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_bwd[g_int] = \
                    stats.g_level_expansions_bwd.get(g_int, 0) + 1
            
            # Check if we've reached the backward goal (m_i)
            if m == m_i:
                candidate_cost = g_m
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    if collect_stats:
                        stats.meeting_g_fwd = 0.0
                        stats.meeting_g_bwd = g_m
            
            # Check for meeting with forward search
            if m in G_F:
                candidate_cost = G_F[m] + G_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    if collect_stats:
                        stats.meeting_g_fwd = G_F[m]
                        stats.meeting_g_bwd = G_B[m]
            
            # Expand predecessors
            for sp_t, m_prev, cost in sp.predecessors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(sp_t, 'is_model_move') and sp_t.is_model_move:
                    if hasattr(sp_t, 'transition') and sp_t.transition is not None:
                        if hasattr(sp_t.transition, 'label') and sp_t.transition.label is None:
                            stats.tau_transitions_bwd += 1
                
                if g_new >= G_B.get(m_prev, float('inf')):
                    continue
                
                G_B[m_prev] = g_new
                parent_B[m_prev] = (m, sp_t)
                
                # Compute f̄_B(m_prev) = 2*g_B + h_B - h_F
                h_B_val = get_h_B(m_prev)
                h_F_val = get_h_F(m_prev)
                f_bar_new = 2 * g_new + h_B_val - h_F_val
                F_bar_B[m_prev] = f_bar_new
                
                # Early pruning
                if f_bar_new >= 2 * mu:
                    continue
                
                counter_B += 1
                heapq.heappush(open_B, (f_bar_new, g_new, counter_B, m_prev))
                generations += 1
                
                if collect_stats:
                    stats.generations_bwd += 1
                
                # Update UB if meeting
                if m_prev in G_F:
                    candidate_cost = G_F[m_prev] + g_new
                    if candidate_cost < mu:
                        mu = candidate_cost
                        meeting_marking = m_prev
                        if collect_stats:
                            stats.meeting_g_fwd = G_F[m_prev]
                            stats.meeting_g_bwd = g_new
        
        else:
            break
    
    # =========================================================================
    # Reconstruct alignment
    # =========================================================================
    if collect_stats:
        stats.compute_asymmetry()
    
    if meeting_marking is not None and mu < float('inf'):
        alignment = _reconstruct_bidirectional(
            sp, parent_F, parent_B, meeting_marking
        )
        return SearchResult(
            optimal_cost=mu,
            alignment=alignment,
            expansions=expansions,
            generations=generations,
            time_seconds=time.perf_counter() - t_start,
            algorithm="dibbs",
            heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
            stats=stats
        )
    
    return SearchResult(
        optimal_cost=float('inf'),
        alignment=None,
        expansions=expansions,
        generations=generations,
        time_seconds=time.perf_counter() - t_start,
        algorithm="dibbs",
        heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
        stats=stats
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _reconstruct_bidirectional(
    sp: SynchronousProduct,
    parent_F: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    parent_B: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    meeting: Marking
) -> Alignment:
    """
    Reconstruct optimal alignment through the meeting point.
    
    The alignment is composed of:
      1. Forward path:  m_i -> ... -> meeting  (from parent_F)
      2. Backward path: meeting -> ... -> m_f  (from parent_B)
    """
    # --- Forward segment: walk back from meeting to m_i ---
    forward_moves = []
    m = meeting
    while parent_F.get(m, (None, None))[0] is not None:
        pred, sp_t = parent_F[m]
        forward_moves.append(AlignmentMove(
            move_type=sp_t.move_type.name if hasattr(sp_t.move_type, 'name') else str(sp_t.move_type),
            model_label=sp_t.model_label if hasattr(sp_t, 'model_label') else None,
            log_label=sp_t.log_label if hasattr(sp_t, 'log_label') else None,
            cost=sp_t.cost
        ))
        m = pred
    forward_moves.reverse()

    # --- Backward segment: walk from meeting toward m_f via parent_B ---
    backward_moves = []
    m = meeting
    while parent_B.get(m, (None, None))[0] is not None:
        next_m, sp_t = parent_B[m]
        backward_moves.append(AlignmentMove(
            move_type=sp_t.move_type.name if hasattr(sp_t.move_type, 'name') else str(sp_t.move_type),
            model_label=sp_t.model_label if hasattr(sp_t, 'model_label') else None,
            log_label=sp_t.log_label if hasattr(sp_t, 'log_label') else None,
            cost=sp_t.cost
        ))
        m = next_m

    all_moves = forward_moves + backward_moves
    return Alignment(moves=all_moves)
