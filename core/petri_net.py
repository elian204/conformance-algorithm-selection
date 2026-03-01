"""
core/petri_net.py
-----------------
Labeled Petri net and workflow net data structures.
Notation follows the paper:
  SN = (P_SN, T_SN, F_SN, λ_SN, m_i^SN, m_f^SN)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Place:
    name: str

    def __repr__(self):
        return f"Place({self.name})"


@dataclass(frozen=True)
class Transition:
    name: str
    label: Optional[str]  # None means silent (τ)

    @property
    def is_silent(self) -> bool:
        return self.label is None

    def __repr__(self):
        lbl = "τ" if self.is_silent else self.label
        return f"Transition({self.name}, {lbl})"


# A marking is a frozenset of (place, count) pairs (multi-set).
# For 1-safe nets a frozenset of places suffices.
Marking = FrozenSet[Place]


def marking_from_places(*places: Place) -> Marking:
    return frozenset(places)


# ---------------------------------------------------------------------------
# Petri Net
# ---------------------------------------------------------------------------

class PetriNet:
    """
    A labeled Petri net.
    Internally stores:
      - places: set of Place
      - transitions: set of Transition
      - preset:  t -> frozenset of Places (input places)
      - postset: t -> frozenset of Places (output places)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.places: Set[Place] = set()
        self.transitions: Set[Transition] = set()
        self._preset:  Dict[Transition, Set[Place]] = {}
        self._postset: Dict[Transition, Set[Place]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_place(self, name: str) -> Place:
        p = Place(name)
        self.places.add(p)
        return p

    def add_transition(self, name: str, label: Optional[str]) -> Transition:
        t = Transition(name, label)
        self.transitions.add(t)
        self._preset[t] = set()
        self._postset[t] = set()
        return t

    def add_arc_place_to_transition(self, place: Place, transition: Transition):
        self._preset[transition].add(place)

    def add_arc_transition_to_place(self, transition: Transition, place: Place):
        self._postset[transition].add(place)

    # ------------------------------------------------------------------
    # Firing semantics
    # ------------------------------------------------------------------

    def is_enabled(self, marking: Marking, transition: Transition) -> bool:
        return self._preset[transition].issubset(marking)

    def fire(self, marking: Marking, transition: Transition) -> Marking:
        """Return new marking after firing transition (assumes 1-safe / P/T net)."""
        new = set(marking)
        new -= self._preset[transition]
        new |= self._postset[transition]
        return frozenset(new)

    def enabled_transitions(self, marking: Marking) -> list:
        return [t for t in self.transitions if self.is_enabled(marking, t)]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def preset(self, t: Transition) -> FrozenSet[Place]:
        return frozenset(self._preset[t])

    def postset(self, t: Transition) -> FrozenSet[Place]:
        return frozenset(self._postset[t])

    def __repr__(self):
        return (f"PetriNet('{self.name}', "
                f"|P|={len(self.places)}, |T|={len(self.transitions)})")


# ---------------------------------------------------------------------------
# Workflow Net wrapper
# ---------------------------------------------------------------------------

@dataclass
class WorkflowNet:
    """
    A labeled workflow net:
      SN = (P, T, F, λ, m_i, m_f)
    """
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking

    @property
    def places(self):
        return self.net.places

    @property
    def transitions(self):
        return self.net.transitions
