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
        "changed": 0.7,  # Changed edge probability
    },
    "sequence_length": {
        "before_change": 100,  # Number of graphs before change
        "after_change": 100,  # Number of graphs after change
    },
}


def generate_er_graphs(config: Dict = ER_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Erdős-Rényi graph sequence with parameter change.

    Args:
        config: Configuration dictionary with ER parameters

    Returns:
        Dictionary containing graph sequences and parameters
    """
    generator = GraphGenerator()

    graphs = generator.erdos_renyi(
        n=config["n"],
        p1=config["probabilities"]["initial"],
        p2=config["probabilities"]["changed"],
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
    result = generate_er_graphs()
    print(f"Generated {len(result['graphs'])} ER graphs")
    print(f"Change point at t={result['change_point']}")
    print(f"Graph shape: {result['graphs'][0].shape}")
