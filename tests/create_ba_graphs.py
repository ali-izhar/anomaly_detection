import sys
import os
import numpy as np
from typing import List, Dict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph import GraphGenerator

# Configuration parameters for BA graphs
BA_CONFIG = {
    "n": 50,  # Number of nodes
    "edges": {
        "initial": 3,  # Initial m1 parameter (edges per new node)
        "change1": 7,  # First change
        "change2": 4,  # Second change
        "change3": 6,  # Third change
    },
    "sequence_length": {
        "before_change": 50,  # Graphs before first change
        "after_change1": 50,  # Graphs after first change
        "after_change2": 50,  # Graphs after second change
        "after_change3": 50,  # Graphs after third change
    },
}


def generate_ba_graphs(config: Dict = BA_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Barabási-Albert graph sequence with multiple parameter changes.

    Args:
        config: Configuration dictionary with BA parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    # Generate first segment
    graphs1 = generator.barabasi_albert(
        n=config["n"],
        m1=config["edges"]["initial"],
        m2=config["edges"]["change1"],
        set1=config["sequence_length"]["before_change"],
        set2=config["sequence_length"]["after_change1"],
    )

    # Generate second segment
    graphs2 = generator.barabasi_albert(
        n=config["n"],
        m1=config["edges"]["change1"],
        m2=config["edges"]["change2"],
        set1=config["sequence_length"]["after_change1"],
        set2=config["sequence_length"]["after_change2"],
    )[
        config["sequence_length"]["after_change1"] :
    ]  # Skip first set

    # Generate third segment
    graphs3 = generator.barabasi_albert(
        n=config["n"],
        m1=config["edges"]["change2"],
        m2=config["edges"]["change3"],
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
    result = generate_ba_graphs()
    print(f"Generated {len(result['graphs'])} BA graphs")
    print(f"Change points at t={result['change_points']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
