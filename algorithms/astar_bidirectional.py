"""
algorithms/astar_bidirectional.py
----------------------------------
Bidirectional A* search variants for conformance checking.

Implements two algorithms:
  1. Standard Bidirectional A* (Front-to-End)
     - Priority: pr(m) = f(m) = g(m) + h(m)
     - Termination: max{f_min^F, f_min^B} >= μ
     
  2. MM Algorithm (Holte et al., 2016)
     - Priority: pr(m) = max{f(m), 2g(m)}
     - Termination: μ <= max{LB_F, LB_B, g_min^F + g_min^B}
     - Guarantees optimality with consistent heuristics
     - Reduces "overshooting" via the 2g penalty

Notation (standard process mining):
  - m_i: initial marking
  - m_f: final marking  
  - σ: trace (sequence of activities)
  - g(m): cost-so-far to reach marking m
  - h(m): heuristic estimate of remaining cost
  - f(m) = g(m) + h(m): estimated total cost through m
  - μ: upper bound on optimal cost (best complete path found)
  - C*: optimal alignment cost

References:
  - Holte et al. (2016). "Bidirectional Search That Is Guaranteed to 
    Meet in the Middle." AAAI 2016.
  - Adriansyah (2014). "Aligning Observed and Modeled Behavior." 
    PhD Thesis, TU Eindhoven.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Set

from core.synchronous_product import SynchronousProduct, SPTransition
from core.petri_net import Marking

# Import Alignment and AlignmentMove from the correct location
try:
    from utils.alignment import Alignment, AlignmentMove
except ImportError:
    try:
        from core.alignment import Alignment, AlignmentMove
    except ImportError:
        # Fallback definitions if imports fail
        @dataclass
        class AlignmentMove:
            move_type: str
            model_label: Optional[str]
            log_label: Optional[str]
            cost: float
        
        @dataclass
        class Alignment:
            moves: List[Any] = field(default_factory=list)
            
            @property
            def cost(self) -> float:
                return sum(m.cost for m in self.moves)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BidirectionalStats:
    """
    Statistics for bidirectional search analysis.
    
    Attributes:
        expansions_fwd: Number of nodes expanded in forward direction
        expansions_bwd: Number of nodes expanded in backward direction
        generations_fwd: Number of nodes generated (pushed to open) forward
        generations_bwd: Number of nodes generated backward
        wasted_expansions: Expansions after meeting but before termination
        meeting_g_fwd: g-value at meeting point (forward direction)
        meeting_g_bwd: g-value at meeting point (backward direction)
        asymmetry_ratio: expansions_fwd / expansions_bwd (or inf)
        g_level_expansions_fwd: Dict mapping g-level -> count (forward)
        g_level_expansions_bwd: Dict mapping g-level -> count (backward)
        tau_transitions_fwd: Count of τ-transitions fired (forward)
        tau_transitions_bwd: Count of τ-transitions fired (backward)
    """
    expansions_fwd: int = 0
    expansions_bwd: int = 0
    generations_fwd: int = 0
    generations_bwd: int = 0
    wasted_expansions: int = 0
    meeting_g_fwd: float = float('inf')
    meeting_g_bwd: float = float('inf')
    asymmetry_ratio: float = 1.0
    g_level_expansions_fwd: Dict[int, int] = field(default_factory=dict)
    g_level_expansions_bwd: Dict[int, int] = field(default_factory=dict)
    tau_transitions_fwd: int = 0
    tau_transitions_bwd: int = 0
    
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
    
    Attributes:
        optimal_cost: C* = optimal alignment cost (inf if no solution)
        alignment: Optimal alignment sequence (None if no solution)
        expansions: Total number of node expansions
        generations: Total number of nodes generated
        time_seconds: Wall-clock time for search
        algorithm: Algorithm identifier string
        heuristic: Heuristic identifier string
        stats: Detailed statistics (for bidirectional variants)
    """
    optimal_cost: float
    alignment: Optional[Alignment]
    expansions: int
    generations: int
    time_seconds: float
    algorithm: str
    heuristic: str
    stats: Optional[BidirectionalStats] = None

    @property
    def solved(self) -> bool:
        """Whether an optimal finite-cost solution was found."""
        return self.optimal_cost < float('inf')


# Type alias for heuristic functions
# h(marking) -> float (estimated cost to goal)
HeuristicFunc = Callable[[Marking], float]


# =============================================================================
# Standard Bidirectional A* (Front-to-End)
# =============================================================================

def astar_bidirectional_standard(
    sp: SynchronousProduct,
    heuristic_f: HeuristicFunc,
    heuristic_b: HeuristicFunc,
    max_expansions: int = 1_000_000,
    collect_stats: bool = True
) -> SearchResult:
    """
    Standard Bidirectional A* for conformance checking.
    
    Runs two simultaneous A* searches:
      - Forward:  m_i → m_f  using h_F (estimates cost to m_f)
      - Backward: m_f → m_i  using h_B (estimates cost to m_i)
    
    Priority function:
        pr(m) = f(m) = g(m) + h(m)
    
    Termination condition:
        max{f_min^F, f_min^B} >= μ
        
    where μ is the best complete path cost found so far.
    
    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product of workflow net and trace.
    heuristic_f : Callable[[Marking], float]
        Forward heuristic h_F(m) -> estimated cost from m to m_f.
    heuristic_b : Callable[[Marking], float]
        Backward heuristic h_B(m) -> estimated cost from m to m_i.
    max_expansions : int
        Safety limit on total expansions (default 1M).
    collect_stats : bool
        If True, collect detailed statistics for analysis.
        
    Returns
    -------
    SearchResult
        Contains optimal cost, alignment, and search statistics.
        
    Notes
    -----
    The algorithm alternates between directions, always expanding from
    the direction with the smaller f_min value. This ensures balanced
    exploration when heuristics are equally informative.
    
    Meeting detection: When a node m is expanded in one direction and
    already exists in the other direction's closed set, we have a
    candidate solution with cost g_F(m) + g_B(m).
    """
    t_start = time.perf_counter()
    
    m_i = sp.initial_marking
    m_f = sp.final_marking
    
    # Statistics collection
    stats = BidirectionalStats() if collect_stats else None
    meeting_found = False
    
    # =========================================================================
    # Forward search state
    # =========================================================================
    g_F: Dict[Marking, float] = {m_i: 0.0}
    parent_F: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_i: (None, None)
    }
    closed_F: Set[Marking] = set()
    
    h_mi = heuristic_f(m_i)
    counter_F = 0
    # Heap entries: (f, g, counter, marking)
    open_F: List[Tuple[float, float, int, Marking]] = [(h_mi, 0.0, counter_F, m_i)]
    
    # =========================================================================
    # Backward search state  
    # =========================================================================
    g_B: Dict[Marking, float] = {m_f: 0.0}
    parent_B: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_f: (None, None)
    }
    closed_B: Set[Marking] = set()
    
    h_mf = heuristic_b(m_f)
    counter_B = 0
    open_B: List[Tuple[float, float, int, Marking]] = [(h_mf, 0.0, counter_B, m_f)]
    
    # =========================================================================
    # Best solution tracking
    # =========================================================================
    mu: float = float('inf')  # Upper bound on C*
    meeting_marking: Optional[Marking] = None
    
    # Counters
    expansions = 0
    generations = 2  # Initial markings m_i and m_f
    
    if collect_stats:
        stats.generations_fwd = 1
        stats.generations_bwd = 1
    
    # =========================================================================
    # Main search loop
    # =========================================================================
    while open_F or open_B:
        
        if expansions >= max_expansions:
            break
        
        # Get lower bounds from both directions
        f_min_F = open_F[0][0] if open_F else float('inf')
        f_min_B = open_B[0][0] if open_B else float('inf')
        
        # ---------------------------------------------------------------------
        # Termination condition: max{f_min^F, f_min^B} >= μ
        # ---------------------------------------------------------------------
        if max(f_min_F, f_min_B) >= mu:
            break
        
        # Decide which direction to expand (lower f_min wins)
        expand_forward = (f_min_F <= f_min_B) if open_F else False
        if not open_B:
            expand_forward = True
        
        if expand_forward and open_F:
            # -----------------------------------------------------------------
            # Expand forward
            # -----------------------------------------------------------------
            _, g_m, _, m = heapq.heappop(open_F)
            
            # Skip if already closed with better g-value
            if m in closed_F:
                continue
            
            # Skip if g-value is stale
            if g_m > g_F.get(m, float('inf')):
                continue
            
            closed_F.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_fwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_fwd[g_int] = \
                    stats.g_level_expansions_fwd.get(g_int, 0) + 1
                if meeting_found:
                    stats.wasted_expansions += 1
            
            # Check for meeting with backward search
            if m in closed_B:
                candidate_cost = g_F[m] + g_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    meeting_found = True
                    if collect_stats:
                        stats.meeting_g_fwd = g_F[m]
                        stats.meeting_g_bwd = g_B[m]
            
            # Expand successors
            for t, m_next, cost in sp.successors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(t, 'is_model_move') and t.is_model_move:
                    if hasattr(t, 'transition') and t.transition is not None:
                        if hasattr(t.transition, 'label') and t.transition.label is None:
                            stats.tau_transitions_fwd += 1
                
                if g_new < g_F.get(m_next, float('inf')):
                    g_F[m_next] = g_new
                    parent_F[m_next] = (m, t)
                    h_val = heuristic_f(m_next)
                    f_new = g_new + h_val
                    
                    counter_F += 1
                    heapq.heappush(open_F, (f_new, g_new, counter_F, m_next))
                    generations += 1
                    
                    if collect_stats:
                        stats.generations_fwd += 1
                    
                    # Check if this creates a meeting
                    if m_next in g_B:
                        candidate_cost = g_new + g_B[m_next]
                        if candidate_cost < mu:
                            mu = candidate_cost
                            meeting_marking = m_next
                            meeting_found = True
                            if collect_stats:
                                stats.meeting_g_fwd = g_new
                                stats.meeting_g_bwd = g_B[m_next]
        
        elif open_B:
            # -----------------------------------------------------------------
            # Expand backward
            # -----------------------------------------------------------------
            _, g_m, _, m = heapq.heappop(open_B)
            
            if m in closed_B:
                continue
            
            if g_m > g_B.get(m, float('inf')):
                continue
            
            closed_B.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_bwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_bwd[g_int] = \
                    stats.g_level_expansions_bwd.get(g_int, 0) + 1
                if meeting_found:
                    stats.wasted_expansions += 1
            
            # Check for meeting with forward search
            if m in closed_F:
                candidate_cost = g_F[m] + g_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    meeting_found = True
                    if collect_stats:
                        stats.meeting_g_fwd = g_F[m]
                        stats.meeting_g_bwd = g_B[m]
            
            # Expand predecessors (backward direction)
            for t, m_prev, cost in sp.predecessors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(t, 'is_model_move') and t.is_model_move:
                    if hasattr(t, 'transition') and t.transition is not None:
                        if hasattr(t.transition, 'label') and t.transition.label is None:
                            stats.tau_transitions_bwd += 1
                
                if g_new < g_B.get(m_prev, float('inf')):
                    g_B[m_prev] = g_new
                    parent_B[m_prev] = (m, t)
                    h_val = heuristic_b(m_prev)
                    f_new = g_new + h_val
                    
                    counter_B += 1
                    heapq.heappush(open_B, (f_new, g_new, counter_B, m_prev))
                    generations += 1
                    
                    if collect_stats:
                        stats.generations_bwd += 1
                    
                    # Check if this creates a meeting
                    if m_prev in g_F:
                        candidate_cost = g_F[m_prev] + g_new
                        if candidate_cost < mu:
                            mu = candidate_cost
                            meeting_marking = m_prev
                            meeting_found = True
                            if collect_stats:
                                stats.meeting_g_fwd = g_F[m_prev]
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
            algorithm="bidirectional_standard",
            heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
            stats=stats
        )
    
    return SearchResult(
        optimal_cost=float('inf'),
        alignment=None,
        expansions=expansions,
        generations=generations,
        time_seconds=time.perf_counter() - t_start,
        algorithm="bidirectional_standard",
        heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
        stats=stats
    )


# =============================================================================
# MM Algorithm (Meet in the Middle)
# =============================================================================

def astar_bidirectional_mm(
    sp: SynchronousProduct,
    heuristic_f: HeuristicFunc,
    heuristic_b: HeuristicFunc,
    max_expansions: int = 1_000_000,
    collect_stats: bool = True
) -> SearchResult:
    """
    MM Algorithm: Bidirectional A* guaranteed to meet in the middle.
    
    Based on Holte et al. (2016), the MM algorithm modifies standard
    bidirectional A* to prevent either direction from "overshooting"
    the meeting point.
    
    Priority function (for expansion order):
        pr(m) = max{f(m), 2g(m)}
        
    The 2g(m) term penalizes deep exploration: once g(m) > C*/2, the
    priority exceeds C*, causing the algorithm to prefer expanding
    from the other direction.
    
    Termination condition:
        μ <= max{ LB_F, LB_B, g_min^F + g_min^B }
        
    where:
        - μ is the best complete path cost found
        - LB_F = min_{m ∈ OPEN_F} f_F(m)  (f-value lower bound, NOT pr)
        - LB_B = min_{m ∈ OPEN_B} f_B(m)  (f-value lower bound, NOT pr)
        - g_min^F = min_{m ∈ OPEN_F} g_F(m)
        - g_min^B = min_{m ∈ OPEN_B} g_B(m)
    
    IMPORTANT: The termination uses f-values (LB), not pr-values.
    The pr-values are only used for determining expansion order.
    
    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product of workflow net and trace.
    heuristic_f : Callable[[Marking], float]
        Forward heuristic h_F(m) -> estimated cost from m to m_f.
    heuristic_b : Callable[[Marking], float]
        Backward heuristic h_B(m) -> estimated cost from m to m_i.
    max_expansions : int
        Safety limit on total expansions (default 1M).
    collect_stats : bool
        If True, collect detailed statistics for analysis.
        
    Returns
    -------
    SearchResult
        Contains optimal cost, alignment, and search statistics.
        
    Notes
    -----
    MM is guaranteed to be optimal when both heuristics are consistent
    (i.e., h(m) <= c(m,m') + h(m') for all transitions m->m').
    
    For conformance checking:
        - h=0 is trivially consistent
        - Marking equation IS consistent (Adriansyah, 2014)
        - MMR (REACH) IS consistent (Casas-Ramos et al., 2024)
    
    The bound g_min^F + g_min^B <= C* relies on consistency: expanded
    nodes have optimal g-values, so the first node on any optimal path
    that remains in OPEN has g >= g_min.
    
    Theoretical guarantee (with consistent h):
        Both directions expand only nodes with g <= C*/2 before terminating.
    """
    t_start = time.perf_counter()
    
    m_i = sp.initial_marking
    m_f = sp.final_marking
    
    # Statistics
    stats = BidirectionalStats() if collect_stats else None
    meeting_found = False
    
    # =========================================================================
    # Forward search state
    # =========================================================================
    g_F: Dict[Marking, float] = {m_i: 0.0}
    parent_F: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_i: (None, None)
    }
    closed_F: Set[Marking] = set()
    
    h_mi = heuristic_f(m_i)
    f_mi = h_mi  # g = 0
    pr_mi = max(f_mi, 0.0)  # pr = max(f, 2g) = max(f, 0) = f
    
    counter_F = 0
    # Heap entries: (priority, g, f, counter, marking)
    # We store both g and f for the termination condition
    open_F: List[Tuple[float, float, float, int, Marking]] = [
        (pr_mi, 0.0, f_mi, counter_F, m_i)
    ]
    
    # =========================================================================
    # Backward search state
    # =========================================================================
    g_B: Dict[Marking, float] = {m_f: 0.0}
    parent_B: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]] = {
        m_f: (None, None)
    }
    closed_B: Set[Marking] = set()
    
    h_mf = heuristic_b(m_f)
    f_mf = h_mf
    pr_mf = max(f_mf, 0.0)
    
    counter_B = 0
    open_B: List[Tuple[float, float, float, int, Marking]] = [
        (pr_mf, 0.0, f_mf, counter_B, m_f)
    ]
    
    # =========================================================================
    # Solution tracking
    # =========================================================================
    mu: float = float('inf')
    meeting_marking: Optional[Marking] = None
    
    expansions = 0
    generations = 2
    
    if collect_stats:
        stats.generations_fwd = 1
        stats.generations_bwd = 1
    
    # =========================================================================
    # Main MM loop
    # =========================================================================
    while open_F or open_B:
        
        if expansions >= max_expansions:
            break
        
        # ---------------------------------------------------------------------
        # Compute bounds from open lists
        # 
        # MM uses TWO different quantities:
        #   1. pr_min = min priority (for expansion order)
        #   2. LB = min f-value (for termination condition)
        #   3. g_min = min g-value (for termination condition)
        #
        # Heap entries: (priority, g, f, counter, marking)
        # ---------------------------------------------------------------------
        if open_F:
            pr_min_F = open_F[0][0]                          # min priority
            LB_F = min(entry[2] for entry in open_F)         # min f-value
            g_min_F = min(entry[1] for entry in open_F)      # min g-value
        else:
            pr_min_F = float('inf')
            LB_F = float('inf')
            g_min_F = float('inf')
        
        if open_B:
            pr_min_B = open_B[0][0]                          # min priority
            LB_B = min(entry[2] for entry in open_B)         # min f-value
            g_min_B = min(entry[1] for entry in open_B)      # min g-value
        else:
            pr_min_B = float('inf')
            LB_B = float('inf')
            g_min_B = float('inf')
        
        # ---------------------------------------------------------------------
        # MM Termination condition (Equation 7 in paper):
        #
        #   μ <= max{ LB_F, LB_B, g_min^F + g_min^B }
        #
        # where LB_F, LB_B are f-value lower bounds (NOT pr-values).
        # This is the key difference from standard bidirectional A*.
        # ---------------------------------------------------------------------
        termination_bound = max(LB_F, LB_B, g_min_F + g_min_B)
        
        if mu <= termination_bound:
            break
        
        # Decide direction: expand from the one with smaller pr_min
        expand_forward = (pr_min_F <= pr_min_B) if open_F else False
        if not open_B:
            expand_forward = True
        
        if expand_forward and open_F:
            # -----------------------------------------------------------------
            # Expand forward (MM)
            # -----------------------------------------------------------------
            pr_m, g_m, f_m, _, m = heapq.heappop(open_F)
            
            if m in closed_F:
                continue
            
            if g_m > g_F.get(m, float('inf')):
                continue
            
            closed_F.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_fwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_fwd[g_int] = \
                    stats.g_level_expansions_fwd.get(g_int, 0) + 1
                if meeting_found:
                    stats.wasted_expansions += 1
            
            # Check for meeting
            if m in closed_B:
                candidate_cost = g_F[m] + g_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    meeting_found = True
                    if collect_stats:
                        stats.meeting_g_fwd = g_F[m]
                        stats.meeting_g_bwd = g_B[m]
            
            # Expand successors
            for t, m_next, cost in sp.successors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(t, 'is_model_move') and t.is_model_move:
                    if hasattr(t, 'transition') and t.transition is not None:
                        if hasattr(t.transition, 'label') and t.transition.label is None:
                            stats.tau_transitions_fwd += 1
                
                if g_new < g_F.get(m_next, float('inf')):
                    g_F[m_next] = g_new
                    parent_F[m_next] = (m, t)
                    
                    h_val = heuristic_f(m_next)
                    f_new = g_new + h_val
                    pr_new = max(f_new, 2 * g_new)  # MM priority
                    
                    counter_F += 1
                    heapq.heappush(open_F, (pr_new, g_new, f_new, counter_F, m_next))
                    generations += 1
                    
                    if collect_stats:
                        stats.generations_fwd += 1
                    
                    # Check meeting
                    if m_next in g_B:
                        candidate_cost = g_new + g_B[m_next]
                        if candidate_cost < mu:
                            mu = candidate_cost
                            meeting_marking = m_next
                            meeting_found = True
                            if collect_stats:
                                stats.meeting_g_fwd = g_new
                                stats.meeting_g_bwd = g_B[m_next]
        
        elif open_B:
            # -----------------------------------------------------------------
            # Expand backward (MM)
            # -----------------------------------------------------------------
            pr_m, g_m, f_m, _, m = heapq.heappop(open_B)
            
            if m in closed_B:
                continue
            
            if g_m > g_B.get(m, float('inf')):
                continue
            
            closed_B.add(m)
            expansions += 1
            
            if collect_stats:
                stats.expansions_bwd += 1
                g_int = int(g_m)
                stats.g_level_expansions_bwd[g_int] = \
                    stats.g_level_expansions_bwd.get(g_int, 0) + 1
                if meeting_found:
                    stats.wasted_expansions += 1
            
            # Check for meeting
            if m in closed_F:
                candidate_cost = g_F[m] + g_B[m]
                if candidate_cost < mu:
                    mu = candidate_cost
                    meeting_marking = m
                    meeting_found = True
                    if collect_stats:
                        stats.meeting_g_fwd = g_F[m]
                        stats.meeting_g_bwd = g_B[m]
            
            # Expand predecessors
            for t, m_prev, cost in sp.predecessors(m):
                g_new = g_m + cost
                
                if collect_stats and hasattr(t, 'is_model_move') and t.is_model_move:
                    if hasattr(t, 'transition') and t.transition is not None:
                        if hasattr(t.transition, 'label') and t.transition.label is None:
                            stats.tau_transitions_bwd += 1
                
                if g_new < g_B.get(m_prev, float('inf')):
                    g_B[m_prev] = g_new
                    parent_B[m_prev] = (m, t)
                    
                    h_val = heuristic_b(m_prev)
                    f_new = g_new + h_val
                    pr_new = max(f_new, 2 * g_new)  # MM priority
                    
                    counter_B += 1
                    heapq.heappush(open_B, (pr_new, g_new, f_new, counter_B, m_prev))
                    generations += 1
                    
                    if collect_stats:
                        stats.generations_bwd += 1
                    
                    # Check meeting
                    if m_prev in g_F:
                        candidate_cost = g_F[m_prev] + g_new
                        if candidate_cost < mu:
                            mu = candidate_cost
                            meeting_marking = m_prev
                            meeting_found = True
                            if collect_stats:
                                stats.meeting_g_fwd = g_F[m_prev]
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
            algorithm="bidirectional_mm",
            heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
            stats=stats
        )
    
    return SearchResult(
        optimal_cost=float('inf'),
        alignment=None,
        expansions=expansions,
        generations=generations,
        time_seconds=time.perf_counter() - t_start,
        algorithm="bidirectional_mm",
        heuristic=f"{getattr(heuristic_f, 'name', 'h_f')}+{getattr(heuristic_b, 'name', 'h_b')}",
        stats=stats
    )


# =============================================================================
# Unified Interface
# =============================================================================

def astar_bidirectional(
    sp: SynchronousProduct,
    heuristic_f: HeuristicFunc,
    heuristic_b: HeuristicFunc,
    variant: str = "standard",
    max_expansions: int = 1_000_000,
    collect_stats: bool = True
) -> SearchResult:
    """
    Unified interface for bidirectional A* variants.
    
    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product of workflow net and trace.
    heuristic_f : Callable[[Marking], float]
        Forward heuristic.
    heuristic_b : Callable[[Marking], float]
        Backward heuristic.
    variant : str
        Algorithm variant: "standard" or "mm"
    max_expansions : int
        Safety limit on expansions.
    collect_stats : bool
        Whether to collect detailed statistics.
        
    Returns
    -------
    SearchResult
        Search result with optimal alignment and statistics.
    """
    if variant == "mm":
        return astar_bidirectional_mm(
            sp, heuristic_f, heuristic_b, max_expansions, collect_stats
        )
    else:
        return astar_bidirectional_standard(
            sp, heuristic_f, heuristic_b, max_expansions, collect_stats
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _reconstruct_bidirectional(
    sp: SynchronousProduct,
    parent_F: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    parent_B: Dict[Marking, Tuple[Optional[Marking], Optional[SPTransition]]],
    meeting: Marking
):
    """
    Reconstruct optimal alignment through the meeting point.
    
    The alignment is composed of:
      1. Forward path:  m_i -> ... -> meeting  (from parent_F)
      2. Backward path: meeting -> ... -> m_f  (from parent_B)
      
    Parameters
    ----------
    sp : SynchronousProduct
        The synchronous product (for constructing Alignment object).
    parent_F : dict
        Forward parent pointers: marking -> (prev_marking, transition)
    parent_B : dict
        Backward parent pointers: marking -> (next_marking, transition)
    meeting : Marking
        The meeting point marking.
        
    Returns
    -------
    Alignment
        The optimal alignment sequence.
    """
    # --- Forward segment: walk back from meeting to m_i ---
    forward_moves = []
    m = meeting
    while parent_F.get(m, (None, None))[0] is not None:
        pred, sp_t = parent_F[m]
        # Convert SPTransition to AlignmentMove
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
        # The backward parent stored (next_marking_in_backward, transition)
        # The transition sp_t goes from m to next_m in the original graph
        backward_moves.append(AlignmentMove(
            move_type=sp_t.move_type.name if hasattr(sp_t.move_type, 'name') else str(sp_t.move_type),
            model_label=sp_t.model_label if hasattr(sp_t, 'model_label') else None,
            log_label=sp_t.log_label if hasattr(sp_t, 'log_label') else None,
            cost=sp_t.cost
        ))
        m = next_m
    # backward_moves already in correct forward order (meeting -> m_f)

    all_moves = forward_moves + backward_moves
    return Alignment(moves=all_moves)
