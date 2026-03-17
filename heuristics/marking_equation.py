"""
heuristics/marking_equation.py
--------------------------------
Marking-equation heuristic (admissible LP relaxation).

OPTIMIZED VERSION:
- Single shared Gurobi environment (created once at module load)
- Model created once per heuristic instance, reused for all LP calls
- RHS updated efficiently via constraint attributes
- Presolve disabled for speed on small LPs
- Per-instance caching (no cross-trace contamination)

For forward search, solves:
  min  Σ cost(t) x_t
  s.t. I_SP · x = m_f - m
       x ≥ 0

For backward search:
  s.t. I_SP · x = m - m_i

Reference: Van Dongen (2018) — "Efficiently Computing Alignments"
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from core.petri_net import Marking, Place
from core.synchronous_product import SynchronousProduct, SPTransition, MoveType
from heuristics.base import Heuristic

# ---------------------------------------------------------------------------
# Gurobi setup - single shared environment created ONCE at module load
# ---------------------------------------------------------------------------
_SHARED_ENV = None
GUROBI_AVAILABLE = False
SOLVER_INFO = "unknown"

try:
    import gurobipy as gp
    from gurobipy import GRB
    try:
        _SHARED_ENV = gp.Env(empty=True)
        _SHARED_ENV.setParam("OutputFlag", 0)
        _SHARED_ENV.setParam("LogToConsole", 0)
        _SHARED_ENV.start()
        # Quick validation
        _m = gp.Model(env=_SHARED_ENV)
        _m.dispose()
        GUROBI_AVAILABLE = True
        SOLVER_INFO = "Gurobi (licensed)"
    except gp.GurobiError as e:
        GUROBI_AVAILABLE = False
        SOLVER_INFO = f"scipy (Gurobi error: {e})"
except ImportError:
    GUROBI_AVAILABLE = False
    SOLVER_INFO = "scipy (gurobipy not installed)"

print(f"[marking_equation] LP solver: {SOLVER_INFO}")


# ---------------------------------------------------------------------------
# Gurobi-based heuristic (fast, requires license)
# ---------------------------------------------------------------------------

class MarkingEquationHeuristic(Heuristic):
    """
    Marking-equation LP heuristic using Gurobi.

    Optimizations:
    - Model structure built once in _setup()
    - Only RHS is updated per estimate() call
    - Uses primal simplex with no presolve for speed
    - Per-instance cache (fresh for each trace)
    """

    def __init__(
        self,
        sp: SynchronousProduct,
        direction: str = "forward",
        use_cache: bool = True,
        timeout_seconds: Optional[float] = None,
    ):
        self._cache: Dict[Marking, float] = {}  # Per-instance cache
        self._use_cache = use_cache
        self._timeout_seconds = timeout_seconds
        self._model = None
        self._x = None
        self._constrs = None
        super().__init__(sp, direction)

    def _setup(self):
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi required. Use MarkingEquationHeuristicScipy instead.")

        # Build index structures
        self._places: List[Place] = self.sp.places
        self._transitions: List[SPTransition] = self.sp.sp_transitions_list
        self._place_idx = {p: i for i, p in enumerate(self._places)}

        n_places = len(self._places)
        n_trans = len(self._transitions)

        # Build incidence matrix
        I = np.zeros((n_places, n_trans), dtype=np.float64)
        for j, sp_t in enumerate(self._transitions):
            sn_net = self.sp.sn.net
            tn_net = self.sp.tn.net
            if sp_t.sn_transition is not None:
                for p in sn_net.preset(sp_t.sn_transition):
                    if p in self._place_idx:
                        I[self._place_idx[p], j] -= 1.0
            if sp_t.tn_transition is not None:
                for p in tn_net.preset(sp_t.tn_transition):
                    if p in self._place_idx:
                        I[self._place_idx[p], j] -= 1.0
            if sp_t.sn_transition is not None:
                for p in sn_net.postset(sp_t.sn_transition):
                    if p in self._place_idx:
                        I[self._place_idx[p], j] += 1.0
            if sp_t.tn_transition is not None:
                for p in tn_net.postset(sp_t.tn_transition):
                    if p in self._place_idx:
                        I[self._place_idx[p], j] += 1.0

        self._I = I
        self._costs = np.array([t.cost for t in self._transitions], dtype=np.float64)

        # Pre-compute target vectors
        if self.direction == "forward":
            self._target = self._marking_to_vector(self.sp.final_marking)
        else:
            self._source = self._marking_to_vector(self.sp.initial_marking)

        # Build Gurobi model ONCE
        self._model = gp.Model(env=_SHARED_ENV)
        self._model.setParam("OutputFlag", 0)
        self._model.setParam("Threads", 1)
        self._model.setParam("Method", 0)  # Primal simplex
        self._model.setParam("Presolve", 0)  # Skip presolve for speed
        if self._timeout_seconds is not None:
            # Cap each LP solve independently; outer method watchdog remains the hard enforcer.
            lp_time_limit = max(0.001, min(float(self._timeout_seconds), 1.0))
            self._model.setParam("TimeLimit", lp_time_limit)

        # Add variables
        self._x = self._model.addVars(n_trans, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        # Objective: minimize cost
        self._model.setObjective(
            gp.quicksum(self._costs[j] * self._x[j] for j in range(n_trans)),
            GRB.MINIMIZE
        )

        # Add constraints (store references for RHS updates)
        self._constrs = []
        for i in range(n_places):
            c = self._model.addConstr(
                gp.quicksum(I[i, j] * self._x[j] for j in range(n_trans)) == 0.0,
                name=f"bal_{i}"
            )
            self._constrs.append(c)

        self._model.update()

    def _marking_to_vector(self, marking: Marking) -> np.ndarray:
        v = np.zeros(len(self._places), dtype=np.float64)
        for p, count in marking.items():
            if p in self._place_idx:
                v[self._place_idx[p]] += float(count)
        return v

    def estimate(self, marking: Marking) -> float:
        if self._use_cache and marking in self._cache:
            return self._cache[marking]

        val = self._solve_lp(marking)

        if self._use_cache:
            self._cache[marking] = val
        return val

    def _solve_lp(self, marking: Marking) -> float:
        m_vec = self._marking_to_vector(marking)

        if self.direction == "forward":
            rhs = self._target - m_vec
        else:
            rhs = m_vec - self._source

        try:
            # Update RHS values (fast operation)
            for i, c in enumerate(self._constrs):
                c.RHS = rhs[i]

            self._model.optimize()

            if self._model.Status == GRB.OPTIMAL:
                return max(0.0, self._model.ObjVal)
            return 0.0
        except Exception:
            return 0.0

    @property
    def name(self) -> str:
        return f"marking_eq_{self.direction}"


# ---------------------------------------------------------------------------
# Scipy-based heuristic (fallback, no license needed)
# ---------------------------------------------------------------------------

class MarkingEquationHeuristicScipy(Heuristic):
    """
    Marking-equation LP using scipy's HiGHS solver.
    Fast fallback when Gurobi is unavailable.
    
    Per-instance cache ensures no cross-trace contamination.
    """

    def __init__(
        self,
        sp: SynchronousProduct,
        direction: str = "forward",
        use_cache: bool = True,
        timeout_seconds: Optional[float] = None,
    ):
        self._cache: Dict[Marking, float] = {}  # Per-instance cache
        self._use_cache = use_cache
        self._timeout_seconds = timeout_seconds
        super().__init__(sp, direction)

    def _setup(self):
        from scipy.optimize import linprog
        self._linprog = linprog

        self._places: List[Place] = self.sp.places
        self._transitions: List[SPTransition] = self.sp.sp_transitions_list
        self._place_idx = {p: i for i, p in enumerate(self._places)}

        n_places = len(self._places)
        n_trans = len(self._transitions)

        # Build incidence matrix
        self._I = np.zeros((n_places, n_trans), dtype=np.float64)
        for j, sp_t in enumerate(self._transitions):
            sn_net = self.sp.sn.net
            tn_net = self.sp.tn.net
            if sp_t.sn_transition is not None:
                for p in sn_net.preset(sp_t.sn_transition):
                    if p in self._place_idx:
                        self._I[self._place_idx[p], j] -= 1.0
            if sp_t.tn_transition is not None:
                for p in tn_net.preset(sp_t.tn_transition):
                    if p in self._place_idx:
                        self._I[self._place_idx[p], j] -= 1.0
            if sp_t.sn_transition is not None:
                for p in sn_net.postset(sp_t.sn_transition):
                    if p in self._place_idx:
                        self._I[self._place_idx[p], j] += 1.0
            if sp_t.tn_transition is not None:
                for p in tn_net.postset(sp_t.tn_transition):
                    if p in self._place_idx:
                        self._I[self._place_idx[p], j] += 1.0

        self._costs = np.array([t.cost for t in self._transitions], dtype=np.float64)
        self._n_trans = n_trans

        # Pre-compute bounds (reused for all calls)
        self._bounds = [(0, None)] * n_trans

        # Pre-compute target vectors
        if self.direction == "forward":
            self._target = self._marking_to_vector(self.sp.final_marking)
        else:
            self._source = self._marking_to_vector(self.sp.initial_marking)

    def _marking_to_vector(self, marking: Marking) -> np.ndarray:
        v = np.zeros(len(self._places), dtype=np.float64)
        for p, count in marking.items():
            if p in self._place_idx:
                v[self._place_idx[p]] += float(count)
        return v

    def estimate(self, marking: Marking) -> float:
        if self._use_cache and marking in self._cache:
            return self._cache[marking]
        val = self._solve_lp(marking)
        if self._use_cache:
            self._cache[marking] = val
        return val

    def _solve_lp(self, marking: Marking) -> float:
        m_vec = self._marking_to_vector(marking)

        if self.direction == "forward":
            rhs = self._target - m_vec
        else:
            rhs = m_vec - self._source

        # Use dual simplex with reduced preprocessing for speed
        result = self._linprog(
            c=self._costs,
            A_eq=self._I,
            b_eq=rhs,
            bounds=self._bounds,
            method='highs-ds',  # Dual simplex - often faster for these LPs
            options={
                'presolve': False,
                # Cap each LP solve independently; outer method watchdog remains the hard enforcer.
                **({'time_limit': max(0.001, min(float(self._timeout_seconds), 1.0))}
                   if self._timeout_seconds is not None else {}),
            }
        )

        if result.success:
            return max(0.0, float(result.fun))
        return 0.0

    @property
    def name(self) -> str:
        return f"marking_eq_scipy_{self.direction}"


# ---------------------------------------------------------------------------
# Factory function: auto-select best available solver
# ---------------------------------------------------------------------------

def create_marking_equation_heuristic(sp: SynchronousProduct, 
                                       direction: str = "forward",
                                       use_cache: bool = True,
                                       timeout_seconds: Optional[float] = None) -> Heuristic:
    """
    Factory function that returns the best available ME heuristic.
    
    Returns Gurobi version if available, otherwise falls back to scipy.
    Each call returns a fresh instance with empty cache for fair comparison.
    """
    if GUROBI_AVAILABLE:
        return MarkingEquationHeuristic(
            sp,
            direction,
            use_cache,
            timeout_seconds=timeout_seconds,
        )
    else:
        return MarkingEquationHeuristicScipy(
            sp,
            direction,
            use_cache,
            timeout_seconds=timeout_seconds,
        )
