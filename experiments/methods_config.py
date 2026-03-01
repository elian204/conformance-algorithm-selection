"""
experiments/methods_config.py
-----------------------------
Configuration for the 15 algorithm × heuristic combinations.

Method Matrix:
                    | H0 (zero) | ME (marking eq) | MMR (REACH) |
    ----------------|-----------|-----------------|-------------|
    Forward         |     ✓     |        ✓        |      ✓      |
    Backward        |     ✓     |        ✓        |      ✓      |
    Bidir Standard  |     ✓     |        ✓        |      ✓      |
    Bidir MM        |     ✓     |        ✓        |      ✓      |
    DIBBS           |     ✓     |        ✓        |      ✓      |
    
Total: 5 algorithms × 3 heuristics = 15 methods

References:
  - MM: Holte et al. (2016), "Bidirectional Search That Is Guaranteed to 
        Meet in the Middle", AAAI-16.
  - DIBBS: Sewell & Jacobson (2021), "Dynamically improved bounds 
           bidirectional search", Artificial Intelligence, 291, 103405.
"""

from dataclasses import dataclass
from typing import Literal, List

# Type definitions
Algorithm = Literal["forward", "backward", "bidir_std", "bidir_mm", "dibbs"]
Heuristic = Literal["zero", "me", "mmr"]


@dataclass
class MethodConfig:
    """Configuration for a single experimental method."""
    algorithm: Algorithm
    heuristic: Heuristic
    
    @property
    def name(self) -> str:
        return f"{self.algorithm}_{self.heuristic}"
    
    @property
    def is_bidirectional(self) -> bool:
        return self.algorithm in ("bidir_std", "bidir_mm", "dibbs")
    
    @property
    def is_mm(self) -> bool:
        return self.algorithm == "bidir_mm"
    
    @property
    def is_dibbs(self) -> bool:
        return self.algorithm == "dibbs"


# =============================================================================
# All 15 method combinations
# =============================================================================

ALL_METHODS: List[MethodConfig] = [
    # Forward variants
    MethodConfig("forward", "zero"),
    MethodConfig("forward", "me"),
    MethodConfig("forward", "mmr"),
    
    # Backward variants
    MethodConfig("backward", "zero"),
    MethodConfig("backward", "me"),
    MethodConfig("backward", "mmr"),
    
    # Standard Bidirectional variants
    MethodConfig("bidir_std", "zero"),
    MethodConfig("bidir_std", "me"),
    MethodConfig("bidir_std", "mmr"),
    
    # MM Bidirectional variants (Holte et al. 2016)
    MethodConfig("bidir_mm", "zero"),
    MethodConfig("bidir_mm", "me"),
    MethodConfig("bidir_mm", "mmr"),
    
    # DIBBS variants (Sewell & Jacobson, 2021)
    MethodConfig("dibbs", "zero"),
    MethodConfig("dibbs", "me"),
    MethodConfig("dibbs", "mmr"),
]


# =============================================================================
# Subsets for focused experiments
# =============================================================================

FORWARD_METHODS = [m for m in ALL_METHODS if m.algorithm == "forward"]
BACKWARD_METHODS = [m for m in ALL_METHODS if m.algorithm == "backward"]
BIDIR_STD_METHODS = [m for m in ALL_METHODS if m.algorithm == "bidir_std"]
BIDIR_MM_METHODS = [m for m in ALL_METHODS if m.algorithm == "bidir_mm"]
DIBBS_METHODS = [m for m in ALL_METHODS if m.algorithm == "dibbs"]

# Combined subsets
ALL_BIDIR_METHODS = BIDIR_STD_METHODS + BIDIR_MM_METHODS + DIBBS_METHODS
UNIDIRECTIONAL_METHODS = FORWARD_METHODS + BACKWARD_METHODS


# =============================================================================
# Utility functions
# =============================================================================

def get_methods(
    algorithms: List[Algorithm] = None,
    heuristics: List[Heuristic] = None
) -> List[MethodConfig]:
    """
    Filter methods by algorithm and/or heuristic.
    
    Parameters
    ----------
    algorithms : List[Algorithm], optional
        List of algorithms to include. None means all.
    heuristics : List[Heuristic], optional
        List of heuristics to include. None means all.
    
    Returns
    -------
    List[MethodConfig]
        Filtered list of method configurations.
    
    Examples
    --------
    >>> get_methods(algorithms=["forward", "backward"])
    [forward_zero, forward_me, forward_mmr, backward_zero, ...]
    
    >>> get_methods(heuristics=["me"])
    [forward_me, backward_me, bidir_std_me, bidir_mm_me, dibbs_me]
    
    >>> get_methods(algorithms=["dibbs"], heuristics=["me", "mmr"])
    [dibbs_me, dibbs_mmr]
    """
    methods = ALL_METHODS
    
    if algorithms is not None:
        methods = [m for m in methods if m.algorithm in algorithms]
    
    if heuristics is not None:
        methods = [m for m in methods if m.heuristic in heuristics]
    
    return methods


# Method name to config lookup
METHOD_LOOKUP = {m.name: m for m in ALL_METHODS}


def parse_method_name(name: str) -> MethodConfig:
    """
    Parse method name string to MethodConfig.
    
    Examples
    --------
    >>> parse_method_name("forward_me")
    MethodConfig(algorithm='forward', heuristic='me')
    
    >>> parse_method_name("dibbs_mmr")
    MethodConfig(algorithm='dibbs', heuristic='mmr')
    """
    if name in METHOD_LOOKUP:
        return METHOD_LOOKUP[name]
    
    raise ValueError(f"Unknown method: {name}. Valid methods: {list(METHOD_LOOKUP.keys())}")
