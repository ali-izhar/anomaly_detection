import sys
import os
import numpy as np
from typing import List, Dict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph import GraphGenerator

# Configuration parameters for ER graphs
ER_CONFIG = {
    "n": 50,  # Number of nodes
    "probabilities": {
        "initial": 0.4,  # Initial edge probability
        "change1": 0.7,  # First change
        "change2": 0.3,  # Second change
        "change3": 0.5,  # Third change
    },
    "sequence_length": {
        "before_change": 50,  # Graphs before first change
        "after_change1": 50,  # Graphs after first change
        "after_change2": 50,  # Graphs after second change
        "after_change3": 50,  # Graphs after third change
    },
}


def generate_er_graphs(config: Dict = ER_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Erdős-Rényi graph sequence with multiple parameter changes.

    Args:
        config: Configuration dictionary with ER parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    # Generate first segment
    graphs1 = generator.erdos_renyi(
        n=config["n"],
        p1=config["probabilities"]["initial"],
        p2=config["probabilities"]["change1"],
        set1=config["sequence_length"]["before_change"],
        set2=config["sequence_length"]["after_change1"],
    )

    # Generate second segment
    graphs2 = generator.erdos_renyi(
        n=config["n"],
        p1=config["probabilities"]["change1"],
        p2=config["probabilities"]["change2"],
        set1=config["sequence_length"]["after_change1"],
        set2=config["sequence_length"]["after_change2"],
    )[
        config["sequence_length"]["after_change1"] :
    ]  # Skip first set

    # Generate third segment
    graphs3 = generator.erdos_renyi(
        n=config["n"],
        p1=config["probabilities"]["change2"],
        p2=config["probabilities"]["change3"],
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
    result = generate_er_graphs()
    print(f"Generated {len(result['graphs'])} ER graphs")
    print(f"Change points at t={result['change_points']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
