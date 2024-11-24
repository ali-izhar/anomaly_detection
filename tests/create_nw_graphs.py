import sys
import os
import numpy as np
from typing import List, Dict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph import GraphGenerator

# Configuration parameters for NW graphs
NW_CONFIG = {
    "n": 30,  # Number of nodes
    "neighbors": {
        "initial": 4,  # Initial number of nearest neighbors
        "change1": 6,  # First change
        "change2": 3,  # Second change
        "change3": 5,  # Third change
    },
    "rewiring_prob": {
        "initial": 0.1,  # Initial rewiring probability
        "change1": 0.3,  # First change
        "change2": 0.05,  # Second change
        "change3": 0.2,  # Third change
    },
    "sequence_length": {
        "before_change": 50,  # Graphs before first change
        "after_change1": 50,  # Graphs after first change
        "after_change2": 50,  # Graphs after second change
        "after_change3": 50,  # Graphs after third change
    },
}


def generate_nw_graphs(config: Dict = NW_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Newman-Watts small-world graph sequence with multiple parameter changes.

    Args:
        config: Configuration dictionary with NW parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    # Generate first segment
    graphs1 = generator.newman_watts(
        n=config["n"],
        k1=config["neighbors"]["initial"],
        k2=config["neighbors"]["change1"],
        p1=config["rewiring_prob"]["initial"],
        p2=config["rewiring_prob"]["change1"],
        set1=config["sequence_length"]["before_change"],
        set2=config["sequence_length"]["after_change1"],
    )

    # Generate second segment
    graphs2 = generator.newman_watts(
        n=config["n"],
        k1=config["neighbors"]["change1"],
        k2=config["neighbors"]["change2"],
        p1=config["rewiring_prob"]["change1"],
        p2=config["rewiring_prob"]["change2"],
        set1=config["sequence_length"]["after_change1"],
        set2=config["sequence_length"]["after_change2"],
    )[
        config["sequence_length"]["after_change1"] :
    ]  # Skip first set

    # Generate third segment
    graphs3 = generator.newman_watts(
        n=config["n"],
        k1=config["neighbors"]["change2"],
        k2=config["neighbors"]["change3"],
        p1=config["rewiring_prob"]["change2"],
        p2=config["rewiring_prob"]["change3"],
        set1=config["sequence_length"]["after_change2"],
        set2=config["sequence_length"]["after_change3"],
    )[
        config["sequence_length"]["after_change2"] :
    ]  # Skip first set

    # Combine all graphs
    all_graphs = graphs1 + graphs2 + graphs3

    # Calculate change points
    change_points = [
        config["sequence_length"]["before_change"],
        config["sequence_length"]["before_change"]
        + config["sequence_length"]["after_change1"],
        config["sequence_length"]["before_change"]
        + config["sequence_length"]["after_change1"]
        + config["sequence_length"]["after_change2"],
    ]

    return {
        "graphs": all_graphs,
        "params": config,
        "change_points": change_points,
    }


if __name__ == "__main__":
    # Generate graphs
    result = generate_nw_graphs()
    print(f"Generated {len(result['graphs'])} NW graphs")
    print(f"Change points at t={result['change_points']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
