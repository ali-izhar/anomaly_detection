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
        "changed": 7,  # Changed m2 parameter
    },
    "sequence_length": {
        "before_change": 100,  # Number of graphs before change
        "after_change": 100,  # Number of graphs after change
    },
}


def generate_ba_graphs(config: Dict = BA_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Barabási-Albert graph sequence with parameter change.

    Args:
        config: Configuration dictionary with BA parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    graphs = generator.barabasi_albert(
        n=config["n"],
        m1=config["edges"]["initial"],
        m2=config["edges"]["changed"],
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
    result = generate_ba_graphs()
    print(f"Generated {len(result['graphs'])} BA graphs")
    print(f"Change point at t={result['change_point']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
