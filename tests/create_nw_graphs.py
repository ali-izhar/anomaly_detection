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
        "changed": 6,  # Changed number of nearest neighbors
    },
    "rewiring_prob": {
        "initial": 0.1,  # Initial rewiring probability
        "changed": 0.3,  # Changed rewiring probability
    },
    "sequence_length": {
        "before_change": 100,  # Number of graphs before change
        "after_change": 100,  # Number of graphs after change
    },
}


def generate_nw_graphs(config: Dict = NW_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Newman-Watts small-world graph sequence with parameter changes.

    Args:
        config: Configuration dictionary with NW parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    graphs = generator.newman_watts(
        n=config["n"],
        k1=config["neighbors"]["initial"],
        k2=config["neighbors"]["changed"],
        p1=config["rewiring_prob"]["initial"],
        p2=config["rewiring_prob"]["changed"],
        set1=config["sequence_length"]["before_change"],
        set2=config["sequence_length"]["after_change"],
    )

    return {
        "graphs": graphs,
        "params": config,
        "change_point": config["sequence_length"]["before_change"],
    }


if __name__ == "__main__":
    # Generate graphs
    result = generate_nw_graphs()
    print(f"Generated {len(result['graphs'])} NW graphs")
    print(f"Change point at t={result['change_point']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
