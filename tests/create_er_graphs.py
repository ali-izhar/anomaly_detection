# tests/create_er_graphs.py

"""
Erdős-Rényi (ER) Graph Sequence Generator

This module generates sequences of ER graphs with multiple parameter changes for testing
change point detection. It creates a sequence of graphs where the edge probability
changes at specific points, simulating network evolution with structural changes.

Usage:
    1. Update the ER_CONFIG dictionary below with desired parameters
    2. Run this file directly: python tests/create_er_graphs.py
"""

import sys
import os
import numpy as np
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.graph import GraphGenerator

# -----------------------------------------------------------------------------#
#                              USER CONFIGURATION                              #
# -----------------------------------------------------------------------------#

ER_CONFIG = {
    "n": 50,  # Number of nodes in each graph
    "probabilities": {
        "initial": 0.4,  # Initial edge probability
        "change1": 0.7,  # First change
        "change2": 0.3,  # Second change
        "change3": 0.5,  # Third change
    },
    "sequence_length": {
        "before_change": 50,  # Number of graphs before first change
        "after_change1": 50,  # Number of graphs after first change
        "after_change2": 50,  # Number of graphs after second change
        "after_change3": 50,  # Number of graphs after third change
    },
}

# -----------------------------------------------------------------------------#
#                           IMPLEMENTATION DETAILS                             #
# -----------------------------------------------------------------------------#


def _generate_graph_segment(
    generator: GraphGenerator,
    n: int,
    p1: float,
    p2: float,
    set1: int,
    set2: int,
    skip_first: bool = False,
) -> List[np.ndarray]:
    """Generate a segment of ER graphs with parameter change."""
    graphs = generator.erdos_renyi(n=n, p1=p1, p2=p2, set1=set1, set2=set2)
    if skip_first:
        return graphs[set1:]
    return graphs


def _calculate_change_points(config: Dict) -> List[int]:
    """Calculate the indices where parameter changes occur."""
    seq_len = config["sequence_length"]
    return [
        seq_len["before_change"],
        seq_len["before_change"] + seq_len["after_change1"],
        seq_len["before_change"] + seq_len["after_change1"] + seq_len["after_change2"],
    ]


def generate_er_graphs(config: Dict = ER_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Erdős-Rényi graph sequence with multiple parameter changes."""
    generator = GraphGenerator()

    graphs1 = _generate_graph_segment(
        generator,
        config["n"],
        config["probabilities"]["initial"],
        config["probabilities"]["change1"],
        config["sequence_length"]["before_change"],
        config["sequence_length"]["after_change1"],
    )

    graphs2 = _generate_graph_segment(
        generator,
        config["n"],
        config["probabilities"]["change1"],
        config["probabilities"]["change2"],
        config["sequence_length"]["after_change1"],
        config["sequence_length"]["after_change2"],
        skip_first=True,
    )

    graphs3 = _generate_graph_segment(
        generator,
        config["n"],
        config["probabilities"]["change2"],
        config["probabilities"]["change3"],
        config["sequence_length"]["after_change2"],
        config["sequence_length"]["after_change3"],
        skip_first=True,
    )

    all_graphs = graphs1 + graphs2 + graphs3
    change_points = _calculate_change_points(config)

    return {
        "graphs": all_graphs,
        "params": config,
        "change_points": change_points,
    }


# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#


def main():
    """Main entry point for graph generation."""
    print("\nGenerating Erdős-Rényi Graph Sequence")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Nodes per graph: {ER_CONFIG['n']}")
    print(f"  - Probability parameters: {ER_CONFIG['probabilities']}")
    print(f"  - Sequence lengths: {ER_CONFIG['sequence_length']}")

    result = generate_er_graphs(ER_CONFIG)

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Graph shape: {result['graphs'][0].shape}")


if __name__ == "__main__":
    main()
