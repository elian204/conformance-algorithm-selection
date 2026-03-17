"""
core/synchronous_product.py
----------------------------
Builds the synchronous product SP of a workflow net SN and an event net TN.

  SP = (P, T, F, λ, m_i, m_f)
  P  = P_SN ⊎ P_TN
  T  = T_SM ⊎ T_MM ⊎ T_LM   (synchronous | model-only | log-only)

Move costs (parsimonious alignment cost function):
  c(t) = 0        if t ∈ T_SM                      [synchronous move]
  c(t) = ε        if t ∈ T_MM and λ(t) = (τ, ≫)   [silent model move]
  c(t) = 1        otherwise                        [deviation]

The small ε-cost for τ-moves ensures that among alignments with equal 
deviation cost, those with fewer silent moves are preferred. This:
  - Breaks ties in favor of simpler model executions
  - Eliminates zero-cost plateaus that cause search inefficiency
  - Preserves optimality (deviation cost = floor of total cost)

Reference: Van Dongen et al. on parsimonious alignments.

OPTIMIZED: Precomputes preset/postset for each SP transition at construction
time to avoid repeated dictionary lookups during search.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.petri_net import (
    PetriNet, WorkflowNet, Place, Transition, Marking, merge_markings
)

# -----------------------------------------------------------------------------
# Cost Configuration
# -----------------------------------------------------------------------------

# Small cost for τ-moves to prefer parsimonious alignments.
# Chosen such that: max_trace_length * max_tau_moves * TAU_EPSILON < 0.5
# This ensures safe recovery of integer deviation cost via round().
TAU_EPSILON = 1e-6


class MoveType(Enum):
    SYNCHRONOUS = auto()   # (a, a)
    MODEL_ONLY  = auto()   # (a, ≫) or (τ, ≫)
    LOG_ONLY    = auto()   # (≫, a)


@dataclass(frozen=True)
class SPTransition:
    """A transition in the synchronous product."""
    name: str
    move_type: MoveType
    model_label: Optional[str]   # None = τ or ≫
    log_label:   Optional[str]   # None = ≫
    # References to underlying net transitions (for incidence matrix)
    sn_transition: Optional[Transition]
    tn_transition: Optional[Transition]
    cost: float = 0.0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, SPTransition) and self.name == other.name

    def label_pair(self) -> Tuple[str, str]:
        ml = self.model_label if self.model_label is not None else "τ" if self.move_type == MoveType.MODEL_ONLY else "≫"
        ll = self.log_label   if self.log_label   is not None else "≫"
        return (ml, ll)

    @property
    def is_tau(self) -> bool:
        """True if this is a silent (τ) model move."""
        return self.move_type == MoveType.MODEL_ONLY and self.model_label is None

    @property
    def is_synchronous(self) -> bool:
        """True if this is a synchronous move."""
        return self.move_type == MoveType.SYNCHRONOUS

    @property
    def is_deviation(self) -> bool:
        """True if this is a deviation (log-only or non-τ model-only)."""
        return self.cost >= 0.5  # Works for both standard (1.0) and ε-cost


class SynchronousProduct:
    """
    Lazy/explicit synchronous product.

    The reachability graph is explored on-the-fly by the search algorithms;
    this class provides the transition relation given a marking.
    
    OPTIMIZED: Precomputes preset/postset for each SP transition at construction
    time to avoid repeated dictionary lookups during search.
    """

    def __init__(self, sn: WorkflowNet, tn: WorkflowNet):
        self.sn = sn
        self.tn = tn

        # Initial and final markings of SP
        self.initial_marking: Marking = merge_markings(sn.initial_marking, tn.initial_marking)
        self.final_marking:   Marking = merge_markings(sn.final_marking, tn.final_marking)

        # Pre-compute SP transitions (static set — same transitions are always available,
        # only enabledness depends on the current marking)
        self._sp_transitions: List[SPTransition] = []
        self._build_sp_transitions()

        # OPTIMIZATION: Precompute preset/postset for each transition
        self._precompute_transition_effects()

        # For heuristics: incidence matrix info
        self.places: List[Place] = (
            sorted(sn.places, key=lambda p: p.name) +
            sorted(tn.places, key=lambda p: p.name)
        )
        self.sp_transitions_list: List[SPTransition] = list(self._sp_transitions)

    def _precompute_transition_effects(self):
        """Precompute preset and postset for each SP transition (called once at init)."""
        self._cached_preset: Dict[SPTransition, FrozenSet[Place]] = {}
        self._cached_postset: Dict[SPTransition, FrozenSet[Place]] = {}
        
        sn_net = self.sn.net
        tn_net = self.tn.net
        
        for sp_t in self._sp_transitions:
            # Compute preset
            preset = set()
            if sp_t.sn_transition is not None:
                preset |= sn_net.preset(sp_t.sn_transition)
            if sp_t.tn_transition is not None:
                preset |= tn_net.preset(sp_t.tn_transition)
            self._cached_preset[sp_t] = frozenset(preset)
            
            # Compute postset
            postset = set()
            if sp_t.sn_transition is not None:
                postset |= sn_net.postset(sp_t.sn_transition)
            if sp_t.tn_transition is not None:
                postset |= tn_net.postset(sp_t.tn_transition)
            self._cached_postset[sp_t] = frozenset(postset)

    # ------------------------------------------------------------------
    # Build the static set of SP transitions
    # ------------------------------------------------------------------

    def _build_sp_transitions(self):
        sn_net = self.sn.net
        tn_net = self.tn.net

        # Group TN transitions by label for synchronous matching
        tn_by_label: Dict[str, List[Transition]] = {}
        for t_tn in tn_net.transitions:
            lbl = t_tn.label  # always non-None in event net
            tn_by_label.setdefault(lbl, []).append(t_tn)

        for t_sn in sn_net.transitions:
            sn_lbl = t_sn.label  # None = τ

            if sn_lbl is not None and sn_lbl in tn_by_label:
                # Synchronous moves: t_sn fires together with matching t_tn
                for t_tn in tn_by_label[sn_lbl]:
                    sp_t = SPTransition(
                        name=f"sync_{t_sn.name}_{t_tn.name}",
                        move_type=MoveType.SYNCHRONOUS,
                        model_label=sn_lbl,
                        log_label=sn_lbl,
                        sn_transition=t_sn,
                        tn_transition=t_tn,
                        cost=0.0
                    )
                    self._sp_transitions.append(sp_t)

            # Model-only move (τ always model-only; labeled also allowed as model-only)
            # τ-moves get small ε-cost to prefer parsimonious alignments
            cost = TAU_EPSILON if t_sn.is_silent else 1.0
            sp_t = SPTransition(
                name=f"model_{t_sn.name}",
                move_type=MoveType.MODEL_ONLY,
                model_label=t_sn.label,   # None = τ
                log_label=None,            # ≫
                sn_transition=t_sn,
                tn_transition=None,
                cost=cost
            )
            self._sp_transitions.append(sp_t)

        # Log-only moves: each TN transition fires alone
        for t_tn in tn_net.transitions:
            sp_t = SPTransition(
                name=f"log_{t_tn.name}",
                move_type=MoveType.LOG_ONLY,
                model_label=None,   # ≫
                log_label=t_tn.label,
                sn_transition=None,
                tn_transition=t_tn,
                cost=1.0
            )
            self._sp_transitions.append(sp_t)

    # ------------------------------------------------------------------
    # Firing semantics for the SP (OPTIMIZED with cached presets)
    # ------------------------------------------------------------------

    def is_enabled(self, marking: Marking, sp_t: SPTransition) -> bool:
        """Check if transition is enabled (uses cached preset)."""
        return all(marking.count(place) >= 1 for place in self._cached_preset[sp_t])

    def fire(self, marking: Marking, sp_t: SPTransition) -> Marking:
        """Fire transition and return new marking (uses cached pre/postsets)."""
        preset = self._cached_preset[sp_t]
        postset = self._cached_postset[sp_t]
        counts = marking.as_dict()

        for place in preset:
            available = counts.get(place, 0)
            if available <= 0:
                raise ValueError(f"Transition {sp_t.name} is not enabled at {marking}")
            if available == 1:
                counts.pop(place)
            else:
                counts[place] = available - 1

        for place in postset:
            counts[place] = counts.get(place, 0) + 1

        return Marking(counts.items())

    def successors(self, marking: Marking) -> List[Tuple[SPTransition, Marking, float]]:
        """Return list of (sp_transition, successor_marking, cost). OPTIMIZED."""
        result = []
        for sp_t in self._sp_transitions:
            preset = self._cached_preset[sp_t]
            if all(marking.count(place) >= 1 for place in preset):
                counts = marking.as_dict()
                for place in preset:
                    available = counts[place]
                    if available == 1:
                        counts.pop(place)
                    else:
                        counts[place] = available - 1

                postset = self._cached_postset[sp_t]
                for place in postset:
                    counts[place] = counts.get(place, 0) + 1

                succ = Marking(counts.items())
                result.append((sp_t, succ, sp_t.cost))
        return result

    def predecessors(self, marking: Marking) -> List[Tuple[SPTransition, Marking, float]]:
        """
        Return list of (sp_transition, predecessor_marking, cost) for the
        reversed graph RG(SP)^rev. OPTIMIZED.

        For each SP transition t with preset •t and postset t•, we need all
        predecessor markings `pred` such that:

            fire(pred, t) == marking

        Under multiset semantics with unit arc weights, the predecessor is
        unique whenever it exists. For each place p:

            marking(p) = pred(p) - 1[p in •t] + 1[p in t•]

        Therefore:

            pred(p) = marking(p) + 1[p in •t] - 1[p in t•]

        and a valid predecessor exists iff every resulting count is
        non-negative.
        """
        result = []
        for sp_t in self._sp_transitions:
            preset = self._cached_preset[sp_t]
            postset = self._cached_postset[sp_t]
            counts = marking.as_dict()
            valid = True

            for place in postset:
                available = counts.get(place, 0)
                if available <= 0:
                    valid = False
                    break
                if available == 1:
                    counts.pop(place)
                else:
                    counts[place] = available - 1

            if not valid:
                continue

            for place in preset:
                counts[place] = counts.get(place, 0) + 1

            pred = Marking(counts.items())
            if self.fire(pred, sp_t) == marking:
                result.append((sp_t, pred, sp_t.cost))
        return result
