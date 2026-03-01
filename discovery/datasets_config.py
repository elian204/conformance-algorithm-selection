"""
BPI Challenge 2015 Dataset Configuration
=========================================

The BPI Challenge 2015 dataset consists of event logs from 5 Dutch
municipalities regarding building permit applications.

Dataset Characteristics:
    - 5 separate event logs (one per municipality)
    - Complex processes with many activities
    - High variability between municipalities
    - Long-running cases (can span years)
    
For bidirectional A* research:
    - Suitable for testing backward search with complex process structures
    - High AND-join ratios in some variants
    - Variable trace lengths provide diverse test cases

Download:
    https://data.4tu.nl/articles/dataset/BPI_Challenge_2015/12712385
    
Files:
    - BPIC15_1.xes (Municipality 1)
    - BPIC15_2.xes (Municipality 2)
    - BPIC15_3.xes (Municipality 3)
    - BPIC15_4.xes (Municipality 4)
    - BPIC15_5.xes (Municipality 5)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# ==============================================================================
# DATASET DEFINITIONS
# ==============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    filename: str
    description: str
    
    # Expected characteristics (for validation)
    expected_traces_min: int = 0
    expected_traces_max: int = 100000
    expected_activities_min: int = 10
    expected_activities_max: int = 500
    
    # Discovery hints
    recommended_algorithms: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.recommended_algorithms is None:
            self.recommended_algorithms = ["IM", "SM", "HM"]


# BPI Challenge 2015 datasets
BPI2015_DATASETS = {
    "BPIC15_1": DatasetConfig(
        name="BPIC15_1",
        filename="BPIC15_1.xes",
        description="BPI Challenge 2015 - Municipality 1",
        expected_traces_min=1000,
        expected_traces_max=2000,
        expected_activities_min=200,
        expected_activities_max=400,
        recommended_algorithms=["IM", "SM"],
        notes="Largest activity set among BPI2015 variants"
    ),
    
    "BPIC15_2": DatasetConfig(
        name="BPIC15_2",
        filename="BPIC15_2.xes",
        description="BPI Challenge 2015 - Municipality 2",
        expected_traces_min=500,
        expected_traces_max=1500,
        expected_activities_min=200,
        expected_activities_max=350,
        recommended_algorithms=["IM", "SM"],
        notes="Moderate complexity"
    ),
    
    "BPIC15_3": DatasetConfig(
        name="BPIC15_3",
        filename="BPIC15_3.xes",
        description="BPI Challenge 2015 - Municipality 3",
        expected_traces_min=1000,
        expected_traces_max=2000,
        expected_activities_min=200,
        expected_activities_max=400,
        recommended_algorithms=["IM", "SM"],
        notes="Higher process variability"
    ),
    
    "BPIC15_4": DatasetConfig(
        name="BPIC15_4",
        filename="BPIC15_4.xes",
        description="BPI Challenge 2015 - Municipality 4",
        expected_traces_min=500,
        expected_traces_max=1500,
        expected_activities_min=200,
        expected_activities_max=350,
        recommended_algorithms=["IM", "SM"],
        notes="Smaller dataset, good for initial testing"
    ),
    
    "BPIC15_5": DatasetConfig(
        name="BPIC15_5",
        filename="BPIC15_5.xes",
        description="BPI Challenge 2015 - Municipality 5",
        expected_traces_min=500,
        expected_traces_max=1500,
        expected_activities_min=200,
        expected_activities_max=400,
        recommended_algorithms=["IM", "SM"],
        notes="Moderate complexity"
    ),
}


# Other commonly used process mining benchmark datasets
BENCHMARK_DATASETS = {
    "BPI2012": DatasetConfig(
        name="BPI2012",
        filename="BPI_Challenge_2012.xes",
        description="BPI Challenge 2012 - Loan Application Process",
        expected_traces_min=10000,
        expected_traces_max=15000,
        expected_activities_min=20,
        expected_activities_max=40,
        recommended_algorithms=["IM", "SM", "HM"],
        notes="Well-studied dataset with structured process"
    ),
    
    "BPI2017": DatasetConfig(
        name="BPI2017",
        filename="BPI_Challenge_2017.xes",
        description="BPI Challenge 2017 - Loan Application Process",
        expected_traces_min=30000,
        expected_traces_max=35000,
        expected_activities_min=20,
        expected_activities_max=30,
        recommended_algorithms=["IM", "SM"],
        notes="Larger version of similar process to BPI2012"
    ),
    
    "BPI2019": DatasetConfig(
        name="BPI2019",
        filename="BPI_Challenge_2019.xes",
        description="BPI Challenge 2019 - Purchase-to-Pay Process",
        expected_traces_min=200000,
        expected_traces_max=300000,
        expected_activities_min=30,
        expected_activities_max=50,
        recommended_algorithms=["IM", "SM"],
        notes="Very large dataset, consider sampling"
    ),
    
    "Sepsis": DatasetConfig(
        name="Sepsis",
        filename="Sepsis_cases.xes",
        description="Sepsis Cases - Hospital Process",
        expected_traces_min=1000,
        expected_traces_max=1500,
        expected_activities_min=10,
        expected_activities_max=20,
        recommended_algorithms=["IM", "SM", "HM", "ILP"],
        notes="Medical process with high variability"
    ),
    
    "BPI2013_incidents": DatasetConfig(
        name="BPI2013_incidents",
        filename="BPI_Challenge_2013_incidents.xes",
        description="BPI Challenge 2013 - Incident Management",
        expected_traces_min=7000,
        expected_traces_max=8000,
        expected_activities_min=10,
        expected_activities_max=15,
        recommended_algorithms=["IM", "SM", "HM"],
        notes="IT Service Management process"
    ),
}


# PLG2 synthetic datasets (if available)
SYNTHETIC_DATASETS = {
    "prAm6": DatasetConfig(
        name="prAm6",
        filename="prAm6.xes",
        description="PLG2 Synthetic - prAm6",
        expected_traces_min=500,
        expected_traces_max=5000,
        recommended_algorithms=["IM", "SM", "HM", "ILP"],
        notes="Synthetic process with known ground truth"
    ),
    
    "prDm6": DatasetConfig(
        name="prDm6",
        filename="prDm6.xes",
        description="PLG2 Synthetic - prDm6",
        expected_traces_min=500,
        expected_traces_max=5000,
        recommended_algorithms=["IM", "SM", "HM", "ILP"],
        notes="Synthetic process with known ground truth"
    ),
    
    "prEm6": DatasetConfig(
        name="prEm6",
        filename="prEm6.xes",
        description="PLG2 Synthetic - prEm6",
        expected_traces_min=500,
        expected_traces_max=5000,
        recommended_algorithms=["IM", "SM", "HM", "ILP"],
        notes="Synthetic process with known ground truth"
    ),
}


# Combined registry
ALL_DATASETS = {
    **BPI2015_DATASETS,
    **BENCHMARK_DATASETS,
    **SYNTHETIC_DATASETS,
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_dataset_config(name: str) -> Optional[DatasetConfig]:
    """Get configuration for a dataset by name."""
    return ALL_DATASETS.get(name)


def list_bpi2015_datasets() -> List[DatasetConfig]:
    """Get all BPI2015 dataset configurations."""
    return list(BPI2015_DATASETS.values())


def list_all_datasets() -> List[DatasetConfig]:
    """Get all registered dataset configurations."""
    return list(ALL_DATASETS.values())


def find_dataset_file(
    dataset_name: str,
    search_paths: List[Path] = None
) -> Optional[Path]:
    """
    Search for a dataset file in common locations.
    
    Args:
        dataset_name: Dataset name or filename
        search_paths: Additional paths to search
        
    Returns:
        Path to dataset file, or None if not found
    """
    # Default search paths
    if search_paths is None:
        search_paths = []
    
    default_paths = [
        Path("data"),
        Path("../data"),
        Path("../../data"),
        Path.home() / "data",
        Path.home() / "datasets",
        Path.home() / "process_mining" / "data",
    ]
    
    search_paths = search_paths + default_paths
    
    # Get filename
    config = get_dataset_config(dataset_name)
    if config:
        filename = config.filename
    else:
        # Assume it's already a filename
        filename = dataset_name if dataset_name.endswith(".xes") else f"{dataset_name}.xes"
    
    # Search
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        # Direct path
        full_path = base_path / filename
        if full_path.exists():
            return full_path
        
        # With .gz extension
        gz_path = base_path / f"{filename}.gz"
        if gz_path.exists():
            return gz_path
    
    return None


def validate_dataset(
    log_path: Path,
    config: DatasetConfig
) -> Dict[str, Any]:
    """
    Validate a loaded dataset against expected characteristics.
    
    Returns:
        Validation report dictionary
    """
    import pm4py
    
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {}
    }
    
    try:
        # Load log
        log = pm4py.read_xes(str(log_path))
        
        # Get statistics
        n_traces = len(log)
        activities = set()
        for trace in log:
            for event in trace:
                if "concept:name" in event:
                    activities.add(event["concept:name"])
        n_activities = len(activities)
        
        report["statistics"] = {
            "traces": n_traces,
            "activities": n_activities
        }
        
        # Validate traces
        if n_traces < config.expected_traces_min:
            report["warnings"].append(
                f"Fewer traces than expected: {n_traces} < {config.expected_traces_min}"
            )
        elif n_traces > config.expected_traces_max:
            report["warnings"].append(
                f"More traces than expected: {n_traces} > {config.expected_traces_max}"
            )
        
        # Validate activities
        if n_activities < config.expected_activities_min:
            report["warnings"].append(
                f"Fewer activities than expected: {n_activities} < {config.expected_activities_min}"
            )
        elif n_activities > config.expected_activities_max:
            report["warnings"].append(
                f"More activities than expected: {n_activities} > {config.expected_activities_max}"
            )
        
    except Exception as e:
        report["valid"] = False
        report["errors"].append(str(e))
    
    return report


# ==============================================================================
# CLI HELPER
# ==============================================================================

def print_dataset_info():
    """Print information about all registered datasets."""
    print("=" * 70)
    print("REGISTERED DATASETS")
    print("=" * 70)
    
    print("\n📁 BPI Challenge 2015 (5 municipalities):")
    for name, config in BPI2015_DATASETS.items():
        print(f"   {name}: {config.description}")
        print(f"      File: {config.filename}")
        print(f"      Algorithms: {', '.join(config.recommended_algorithms)}")
    
    print("\n📁 Other Benchmark Datasets:")
    for name, config in BENCHMARK_DATASETS.items():
        print(f"   {name}: {config.description}")
        print(f"      File: {config.filename}")
    
    print("\n📁 Synthetic Datasets:")
    for name, config in SYNTHETIC_DATASETS.items():
        print(f"   {name}: {config.description}")
        print(f"      File: {config.filename}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_dataset_info()
