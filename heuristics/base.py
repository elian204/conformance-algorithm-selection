"""
heuristics/base.py
------------------
Abstract interface for heuristics used in A* conformance checking.
Both forward and backward heuristics must implement this interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.petri_net import Marking
    from core.synchronous_product import SynchronousProduct


class Heuristic(ABC):
    """
    Base class for admissible heuristics on the synchronous product.

    h(m) must be an admissible lower bound on the cost of reaching
    the goal marking from marking m.
    """

    def __init__(self, sp: "SynchronousProduct", direction: str = "forward"):
        """
        Parameters
        ----------
        sp        : SynchronousProduct
        direction : 'forward'  -> h estimates cost to m_f
                    'backward' -> h estimates cost to m_i (in reversed graph,
                                  i.e., from m_f toward m_i)
        """
        self.sp = sp
        self.direction = direction
        self._setup()

    def _setup(self):
        """Optional pre-computation (override in subclasses)."""
        pass

    @abstractmethod
    def estimate(self, marking: "Marking") -> float:
        """Return admissible lower bound h(m)."""
        ...

    def __call__(self, marking: "Marking") -> float:
        return self.estimate(marking)

    @property
    def name(self) -> str:
        return self.__class__.__name__
