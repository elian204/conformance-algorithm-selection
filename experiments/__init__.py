"""
experiments/__init__.py
-----------------------
Experiment configuration and execution.
"""

from .methods_config import (
    ALL_METHODS,
    MethodConfig,
    get_methods,
    parse_method_name,
    FORWARD_METHODS,
    BACKWARD_METHODS,
    BIDIR_STD_METHODS,
    BIDIR_MM_METHODS,
    ALL_BIDIR_METHODS,
    UNIDIRECTIONAL_METHODS,
    Algorithm,
    Heuristic,
)

from .method_dispatcher import (
    run_method,
    run_all_methods,
    find_winner,
    create_heuristic,
)

__all__ = [
    # Method configuration
    "ALL_METHODS",
    "MethodConfig", 
    "get_methods",
    "parse_method_name",
    "FORWARD_METHODS",
    "BACKWARD_METHODS",
    "BIDIR_STD_METHODS",
    "BIDIR_MM_METHODS",
    "ALL_BIDIR_METHODS",
    "UNIDIRECTIONAL_METHODS",
    "Algorithm",
    "Heuristic",
    
    # Method dispatcher
    "run_method",
    "run_all_methods",
    "find_winner",
    "create_heuristic",
]
