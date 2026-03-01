"""
algorithms/__init__.py
----------------------
A* search algorithms for conformance checking.

Provides:
  - astar_forward: Standard forward A*
  - astar_backward: Backward A* on reversed graph
  - astar_bidirectional_standard: Bidirectional A* (Front-to-End)
  - astar_bidirectional_mm: Bidirectional A* (MM algorithm, Holte et al. 2016)
  - astar_dibbs: DIBBS (Dynamically Improved Bounds Bidirectional Search, Sewell & Jacobson 2021)
  - astar_bidirectional: Unified interface with variant parameter
"""

from algorithms.astar_forward import astar_forward
from algorithms.astar_backward import astar_backward
from algorithms.astar_bidirectional import (
    astar_bidirectional,
    astar_bidirectional_standard,
    astar_bidirectional_mm,
    SearchResult,
    BidirectionalStats,
)
from algorithms.astar_dibbs import astar_dibbs, DIBBSStats

__all__ = [
    "astar_forward",
    "astar_backward",
    "astar_bidirectional",
    "astar_bidirectional_standard",
    "astar_bidirectional_mm",
    "astar_dibbs",
    "SearchResult",
    "BidirectionalStats",
    "DIBBSStats",
]
