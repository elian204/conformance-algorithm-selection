"""
utils/alignment.py
------------------
Alignment representation and formatting utilities.

Cost interpretation (with τ-epsilon):
  - total_cost: sum of all move costs (float)
  - deviation_cost: number of non-synchronous, non-τ moves (int) = round(total_cost)
  - tau_count: number of τ-moves in the alignment
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlignmentMove:
    """A single move in an alignment."""
    move_type: str       # 'SYNCHRONOUS', 'MODEL_ONLY', 'LOG_ONLY'
    model_label: Optional[str]  # None = τ or ≫
    log_label:   Optional[str]  # None = ≫
    cost: float

    def log_str(self) -> str:
        return self.log_label if self.log_label is not None else "≫"

    def model_str(self) -> str:
        if self.model_label is None:
            return "τ" if self.move_type == "MODEL_ONLY" else "≫"
        return self.model_label

    @property
    def is_tau(self) -> bool:
        """True if this is a silent (τ) model move."""
        return self.move_type == "MODEL_ONLY" and self.model_label is None
    
    @property
    def is_synchronous(self) -> bool:
        """True if this is a synchronous move."""
        return self.move_type == "SYNCHRONOUS"
    
    @property
    def is_deviation(self) -> bool:
        """True if this is a deviation (log-only or non-τ model-only)."""
        return self.cost >= 0.5  # Works for both standard (1.0) and ε-cost

    @property
    def is_log_only(self) -> bool:
        """True if this is a log-only move (≫, a)."""
        return self.move_type == "LOG_ONLY"

    @property
    def is_model_only(self) -> bool:
        """True if this is a model-only move (a, ≫) or (τ, ≫)."""
        return self.move_type == "MODEL_ONLY"

    def __repr__(self):
        return f"({self.log_str()}/{self.model_str()}, cost={self.cost})"


@dataclass
class Alignment:
    """An alignment between a trace and a process model."""
    moves: List[AlignmentMove] = field(default_factory=list)

    @property
    def cost(self) -> float:
        """Total alignment cost (including ε for τ-moves)."""
        return sum(m.cost for m in self.moves)

    @property
    def deviation_cost(self) -> int:
        """
        Integer deviation cost (number of log/model moves excluding τ).

        With τ-epsilon costs, this equals round(total_cost).
        This is the standard alignment cost used in conformance metrics.
        """
        return round(self.cost)

    @property
    def tau_count(self) -> int:
        """Number of τ (silent) moves in the alignment."""
        return sum(1 for m in self.moves if m.is_tau)

    @property
    def sync_count(self) -> int:
        """Number of synchronous moves in the alignment."""
        return sum(1 for m in self.moves if m.is_synchronous)

    @property
    def log_move_count(self) -> int:
        """Number of log-only moves in the alignment."""
        return sum(1 for m in self.moves if m.is_log_only)

    @property
    def model_move_count(self) -> int:
        """Number of model-only moves (including τ) in the alignment."""
        return sum(1 for m in self.moves if m.is_model_only)

    @property
    def labeled_model_move_count(self) -> int:
        """Number of labeled (non-τ) model-only moves in the alignment."""
        return sum(1 for m in self.moves if m.is_model_only and not m.is_tau)

    @property
    def fitness(self) -> float:
        """Normalized fitness based on deviation cost."""
        # Use deviation_cost for proper fitness calculation
        max_cost = len(self.moves)  # Upper bound: all deviations
        return 1.0 - self.deviation_cost / max(1, max_cost)

    def __len__(self):
        return len(self.moves)

    def to_matrix_str(self) -> str:
        """Return alignment in the standard matrix notation used in the paper."""
        log_row   = " | ".join(m.log_str()   for m in self.moves)
        model_row = " | ".join(m.model_str() for m in self.moves)
        width = max(len(log_row), len(model_row))
        sep = "-" * (width + 4)
        return f"| {log_row.ljust(width)} |\n{sep}\n| {model_row.ljust(width)} |"

    def summary(self) -> str:
        """Return a concise summary of the alignment."""
        return (f"Alignment: deviation_cost={self.deviation_cost}, "
                f"τ_moves={self.tau_count}, sync_moves={self.sync_count}, "
                f"total_moves={len(self.moves)}")

    def __repr__(self):
        return f"Alignment(deviation_cost={self.deviation_cost}, τ={self.tau_count}, moves={len(self.moves)})"


def costs_equal(cost1: float, cost2: float) -> bool:
    """
    Check if two alignment costs represent the same deviation cost.

    With τ-epsilon costs, two alignments have equal deviation cost
    if round(cost1) == round(cost2).
    """
    return round(cost1) == round(cost2)


def deviation_cost(total_cost: float) -> int:
    """
    Extract integer deviation cost from total alignment cost.

    With τ-epsilon costs, deviation_cost = round(total_cost).
    """
    return round(total_cost)