"""
heuristics/reach.py
--------------------
REACH/MMR heuristic for conformance checking.

Based on: Casas-Ramos et al. (2024) "REACH: An Efficient Heuristic for
Conformance Checking", Expert Systems with Applications.

The MMR (Minimum Model Requirement) heuristic is LP-FREE and works by:
1. Structural analysis of the Petri net to find "forced" transitions
2. A place p with |p•| = 1 means its single output transition MUST fire
3. Forward propagation to count minimum required model moves

This is FASTER than the marking equation while still being admissible.

BUG FIX (2024): Changed forced_transitions from list to set to prevent
double-counting in AND-join structures.

OPTIMIZATION (2025): Added early termination when already at goal marking.
"""
from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, FrozenSet

from core.petri_net import Marking, Place, Transition
from core.synchronous_product import SynchronousProduct, SPTransition, MoveType
from heuristics.base import Heuristic


class REACHHeuristic(Heuristic):
    """
    MMR (Minimum Model Requirement) heuristic — LP-FREE admissible lower bound.

    Algorithm (from Casas-Ramos et al. 2024, Algorithm 3):
    -------------------------------------------------------
    For a marking m, compute the minimum number of model moves required
    to reach the final marking by identifying "forced" transitions:

    1. Project m onto the system net (SN) places
    2. For each marked place p where |p•| = 1 (single output transition):
       - That transition t MUST fire to empty p
       - If t is a labeled transition (not τ), it requires a sync or model move
    3. Propagate forward: firing t may mark new places with forced transitions
    4. Count minimum deviations needed

    This is admissible because it only counts unavoidable moves.
    """

    def __init__(self, sp: SynchronousProduct, direction: str = "forward",
                 use_cache: bool = True):
        self._cache: Dict[Marking, float] = {}
        self._use_cache = use_cache
        super().__init__(sp, direction)

    def _setup(self):
        """Precompute structural information about the net."""
        sn_net = self.sp.sn.net

        # Identify SN places
        self._sn_places: Set[Place] = set(self.sp.sn.places)
        self._sn_final: FrozenSet[Place] = frozenset(self.sp.sn.final_marking)
        self._sn_initial: FrozenSet[Place] = frozenset(self.sp.sn.initial_marking)

        # For each place, find output transitions: p•
        self._place_outputs: Dict[Place, Set[Transition]] = {}
        for p in sn_net.places:
            outputs = set()
            for t in sn_net.transitions:
                if p in sn_net.preset(t):
                    outputs.add(t)
            self._place_outputs[p] = outputs

        # Identify "forced" places: |p•| = 1
        self._forced_places: Set[Place] = {
            p for p, outs in self._place_outputs.items() if len(outs) == 1
        }

        # For transitions, get their preset and postset
        self._trans_preset: Dict[Transition, FrozenSet[Place]] = {
            t: frozenset(sn_net.preset(t)) for t in sn_net.transitions
        }
        self._trans_postset: Dict[Transition, FrozenSet[Place]] = {
            t: frozenset(sn_net.postset(t)) for t in sn_net.transitions
        }

        # Map labels to SN transitions that produce them
        self._label_to_trans: Dict[str, Set[Transition]] = {}
        for t in sn_net.transitions:
            if t.label is not None:
                self._label_to_trans.setdefault(t.label, set()).add(t)

        # Extract trace activities from trace net
        self._trace_activities: List[str] = self._extract_trace_activities()

        # OPTIMIZATION: Precompute mapping from TN place to trace position
        self._precompute_trace_positions()

        # Store reference to SN net for firing
        self._sn_net = sn_net

    def _extract_trace_activities(self) -> List[str]:
        """Extract ordered sequence of activities from the trace net."""
        tn_net = self.sp.tn.net
        acts = []
        current_places = set(self.sp.tn.initial_marking)
        visited_trans = set()

        while True:
            found = False
            for t in tn_net.transitions:
                if t in visited_trans:
                    continue
                if tn_net.preset(t).issubset(current_places):
                    acts.append(t.label)
                    current_places -= tn_net.preset(t)
                    current_places |= tn_net.postset(t)
                    visited_trans.add(t)
                    found = True
                    break
            if not found:
                break
        return acts

    def _precompute_trace_positions(self):
        """
        OPTIMIZATION: Precompute mapping from trace net place to trace position.
        Since trace net is a sequence, each place corresponds to a position.
        """
        tn_net = self.sp.tn.net
        self._tn_place_to_position: Dict[Place, int] = {}

        # Walk through trace net and record position for each place
        current_places = set(self.sp.tn.initial_marking)
        position = 0
        visited_trans = set()

        # Initial places = position 0 (before any activity)
        for p in current_places:
            self._tn_place_to_position[p] = position

        while True:
            found = False
            for t in tn_net.transitions:
                if t in visited_trans:
                    continue
                if tn_net.preset(t).issubset(current_places):
                    current_places -= tn_net.preset(t)
                    current_places |= tn_net.postset(t)
                    visited_trans.add(t)
                    position += 1
                    # Record position for new places
                    for p in tn_net.postset(t):
                        self._tn_place_to_position[p] = position
                    found = True
                    break
            if not found:
                break

        self._trace_length = len(self._trace_activities)

    def _project_to_sn(self, marking: Marking) -> FrozenSet[Place]:
        """Project SP marking onto SN places only."""
        return frozenset(p for p in marking if p in self._sn_places)

    def _get_remaining_trace(self, marking: Marking) -> List[str]:
        """
        Get trace activities still to be consumed from this marking.
        OPTIMIZED: Uses precomputed position mapping instead of walking the net.
        """
        # Find trace position from TN places in marking
        position = 0
        for p in marking:
            if p in self._tn_place_to_position:
                pos = self._tn_place_to_position[p]
                if pos > position:
                    position = pos

        # Return remaining activities from this position
        return self._trace_activities[position:]

    def _compute_mmr_forward(self, sn_marking: FrozenSet[Place],
                             remaining_trace: List[str]) -> int:
        """
        Compute MMR for forward search: minimum model moves from sn_marking to final.

        Algorithm:
        1. Find all transitions that MUST fire (from forced places)
        2. For each required transition:
           - If labeled and label NOT in remaining trace → model move needed
           - If labeled and label IN remaining trace → can sync (no extra cost)
        3. Propagate to find cascading requirements

        BUG FIX: Use SET for forced_transitions to avoid duplicates in AND-join
        scenarios where multiple places force the same transition.
        """
        # OPTIMIZATION: Early termination if already at final marking
        if sn_marking == self._sn_final:
            return 0

        required_model_moves = 0

        # Track which trace activities can still be used for sync
        available_trace = list(remaining_trace)

        # BFS to find all forced transitions
        current_marking = set(sn_marking)
        visited_markings = {frozenset(current_marking)}

        # Limit iterations to avoid infinite loops
        max_iterations = len(self._sn_places) * 10

        for _ in range(max_iterations):
            # Find forced transitions from current marking
            # BUG FIX: Use SET to avoid duplicates (critical for AND-joins where
            # multiple places may force the same transition)
            forced_transitions = set()

            for p in list(current_marking):
                if p in self._forced_places and p not in self._sn_final:
                    # This place has exactly one output - that transition must fire
                    forced_trans = next(iter(self._place_outputs[p]))

                    # Check if transition is enabled
                    if self._trans_preset[forced_trans].issubset(current_marking):
                        forced_transitions.add(forced_trans)

            if not forced_transitions:
                break

            # Process forced transitions (each transition exactly once)
            for t in forced_transitions:
                if t.label is not None:
                    # Labeled transition
                    if t.label in available_trace:
                        # Can synchronize with trace - no extra cost
                        available_trace.remove(t.label)
                    else:
                        # Must do model move
                        required_model_moves += 1

                # Fire the transition
                current_marking -= self._trans_preset[t]
                current_marking |= self._trans_postset[t]

            # Check if we've seen this marking before
            frozen_current = frozenset(current_marking)
            if frozen_current in visited_markings:
                break
            visited_markings.add(frozen_current)

        # Also count trace activities that cannot be matched by any model transition
        for act in available_trace:
            if act not in self._label_to_trans:
                # Activity not in model at all → log move required
                required_model_moves += 1

        return required_model_moves

    def _compute_mmr_backward(self, sn_marking: FrozenSet[Place],
                              consumed_trace: List[str]) -> int:
        """
        Compute MMR for backward search: minimum model moves from initial to sn_marking.

        For backward search, we estimate the cost of the PREFIX path.

        BUG FIX: Use SET for forced_transitions to avoid duplicates.
        """
        # OPTIMIZATION: Early termination if already at initial marking
        if sn_marking == self._sn_initial:
            return 0

        # For backward, we analyze what was needed to reach this marking
        # This is symmetric to forward but considers consumed activities

        required_model_moves = 0

        # Activities consumed to reach this point
        available_trace = list(consumed_trace)

        # Walk forward from initial to see required transitions
        current_marking = set(self._sn_initial)
        visited_markings = {frozenset(current_marking)}
        target = sn_marking

        max_iterations = len(self._sn_places) * 10

        for _ in range(max_iterations):
            if frozenset(current_marking) == target:
                break

            # BUG FIX: Use SET to avoid duplicates (critical for AND-joins)
            forced_transitions = set()

            for p in list(current_marking):
                if p in self._forced_places and p not in target:
                    forced_trans = next(iter(self._place_outputs[p]))
                    if self._trans_preset[forced_trans].issubset(current_marking):
                        forced_transitions.add(forced_trans)

            if not forced_transitions:
                break

            for t in forced_transitions:
                if t.label is not None:
                    if t.label in available_trace:
                        available_trace.remove(t.label)
                    else:
                        required_model_moves += 1

                current_marking -= self._trans_preset[t]
                current_marking |= self._trans_postset[t]

            frozen_current = frozenset(current_marking)
            if frozen_current in visited_markings:
                break
            visited_markings.add(frozen_current)

        return required_model_moves

    def estimate(self, marking: Marking) -> float:
        """Compute MMR heuristic value (LP-free, admissible)."""
        if self._use_cache and marking in self._cache:
            return self._cache[marking]

        sn_marking = self._project_to_sn(marking)

        if self.direction == "forward":
            remaining = self._get_remaining_trace(marking)
            val = float(self._compute_mmr_forward(sn_marking, remaining))
        else:
            # For backward: compute consumed trace
            remaining = self._get_remaining_trace(marking)
            all_acts = self._trace_activities
            consumed = all_acts[:len(all_acts) - len(remaining)]
            val = float(self._compute_mmr_backward(sn_marking, consumed))

        if self._use_cache:
            self._cache[marking] = val
        return val

    @property
    def name(self) -> str:
        return f"reach_{self.direction}"