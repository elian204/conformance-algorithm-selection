"""
Model Discovery Module for Bidirectional A* Conformance Checking
================================================================

This module provides quality-targeted process model discovery
for use in bidirectional A* search experiments.

Components:
    - discover_quality_models: Main discovery script
    - datasets_config: Dataset configuration and registry
    - run_discovered_model_experiments: Experiment runner

Usage:
    # Discover models for BPI Challenge 2015
    python -m discovery.discover_quality_models data/BPIC15_1.xes
    
    # Run experiments with discovered models
    python -m discovery.run_discovered_model_experiments \\
        data/BPIC15_1.xes models/bpi2015
"""

from .datasets_config import (
    DatasetConfig,
    BPI2015_DATASETS,
    BENCHMARK_DATASETS,
    ALL_DATASETS,
    get_dataset_config,
    list_bpi2015_datasets,
    list_all_datasets,
    find_dataset_file,
)

__all__ = [
    "DatasetConfig",
    "BPI2015_DATASETS",
    "BENCHMARK_DATASETS",
    "ALL_DATASETS",
    "get_dataset_config",
    "list_bpi2015_datasets",
    "list_all_datasets",
    "find_dataset_file",
]

__version__ = "1.0.0"
