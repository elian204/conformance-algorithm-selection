"""
heuristics/zero.py
------------------
The uninformed heuristic: h(m) = 0 for all m.
With this heuristic, A* reduces to Dijkstra's algorithm.
"""
from __future__ import annotations
from core.petri_net import Marking
from core.synchronous_product import SynchronousProduct
from heuristics.base import Heuristic


class ZeroHeuristic(Heuristic):
    """h ≡ 0  (Dijkstra / uninformed search)."""

    def estimate(self, marking: Marking) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "h0"
