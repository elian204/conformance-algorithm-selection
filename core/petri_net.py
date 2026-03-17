"""
core/petri_net.py
-----------------
Labeled Petri net and workflow net data structures.
Notation follows the paper:
  SN = (P_SN, T_SN, F_SN, λ_SN, m_i^SN, m_f^SN)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Set, Tuple


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


@dataclass(frozen=True, init=False)
class Marking:
    """
    Immutable multiset marking.

    Internally stores positive token counts keyed by place. Iteration and
    membership expose the support set for compatibility with existing code,
    while `items()` / `count()` expose multiplicities where needed.
    """
    _items: Tuple[Tuple[Place, int], ...]
    _counts: Dict[Place, int] = field(init=False, repr=False, compare=False, hash=False)

    def __init__(self, entries: Optional[Iterable[Tuple[Place, int]]] = None):
        counts: Dict[Place, int] = {}
        if entries is not None:
            for place, count in entries:
                count = int(count)
                if count <= 0:
                    continue
                counts[place] = counts.get(place, 0) + count

        items = tuple(sorted(counts.items(), key=lambda item: item[0].name))
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_counts", dict(items))

    def __iter__(self):
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    def __contains__(self, place: Place) -> bool:
        return self.count(place) > 0

    def __repr__(self) -> str:
        body = ", ".join(f"{place.name}:{count}" for place, count in self._items)
        return f"Marking({{{body}}})"

    def items(self) -> Tuple[Tuple[Place, int], ...]:
        return self._items

    def count(self, place: Place) -> int:
        return self._counts.get(place, 0)

    def get(self, place: Place, default: int = 0) -> int:
        return self._counts.get(place, default)

    def as_dict(self) -> Dict[Place, int]:
        return dict(self._counts)

    @property
    def total_tokens(self) -> int:
        return sum(self._counts.values())


def marking_from_places(*places: Place) -> Marking:
    counts: Dict[Place, int] = {}
    for place in places:
        counts[place] = counts.get(place, 0) + 1
    return Marking(counts.items())


def marking_from_dict(place_counts: Mapping[Place, int]) -> Marking:
    return Marking(place_counts.items())


def merge_markings(*markings: Marking) -> Marking:
    counts: Dict[Place, int] = {}
    for marking in markings:
        for place, count in marking.items():
            counts[place] = counts.get(place, 0) + count
    return Marking(counts.items())


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
        return all(marking.count(place) >= 1 for place in self._preset[transition])

    def fire(self, marking: Marking, transition: Transition) -> Marking:
        """Return new marking after firing transition with multiset semantics."""
        counts = marking.as_dict()
        for place in self._preset[transition]:
            available = counts.get(place, 0)
            if available <= 0:
                raise ValueError(f"Transition {transition.name} is not enabled at {marking}")
            if available == 1:
                counts.pop(place)
            else:
                counts[place] = available - 1

        for place in self._postset[transition]:
            counts[place] = counts.get(place, 0) + 1

        return Marking(counts.items())

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
