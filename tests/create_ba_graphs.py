# tests/create_ba_graphs.py

"""
Barabási-Albert (BA) Graph Sequence Generator

This module generates sequences of BA graphs with multiple parameter changes for testing
change point detection. It creates a sequence of graphs where the number of 
edges per new node (m parameter) changes at specific points, simulating network evolution
with structural changes.

Usage:
    1. Update the BA_CONFIG dictionary below with desired parameters
    2. Run this file directly: python tests/create_ba_graphs.py
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

BA_CONFIG = {
    "n": 50,  # Number of nodes in each graph
    "edges": {
        "initial": 3,  # Initial m1 parameter (edges per new node)
        "change1": 7,  # First change in edge count
        "change2": 4,  # Second change in edge count
        "change3": 6,  # Third change in edge count
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
    m1: int,
    m2: int,
    set1: int,
    set2: int,
    skip_first: bool = False,
) -> List[np.ndarray]:
    """Generate a segment of BA graphs with parameter change."""
    graphs = generator.barabasi_albert(n=n, m1=m1, m2=m2, set1=set1, set2=set2)
    if skip_first:
        return graphs[set1:]
    return graphs


def generate_ba_graphs() -> Dict:
    """Generate BA graph sequence with multiple parameter changes."""
    config = BA_CONFIG
    n = config["n"]
    edges = config["edges"]
    seq_len = config["sequence_length"]

    # Define actual change points where graph parameters change
    change_points = [
        config["sequence_length"]["before_change"],
        config["sequence_length"]["before_change"]
        + config["sequence_length"]["after_change1"],
        config["sequence_length"]["before_change"]
        + config["sequence_length"]["after_change1"]
        + config["sequence_length"]["after_change2"],
    ]

    # Generate graphs with parameter changes at these points
    generator = GraphGenerator()
    graphs1 = _generate_graph_segment(
        generator,
        n,
        edges["initial"],
        edges["change1"],
        config["sequence_length"]["before_change"],
        config["sequence_length"]["after_change1"],
    )

    graphs2 = _generate_graph_segment(
        generator,
        n,
        edges["change1"],
        edges["change2"],
        config["sequence_length"]["after_change1"],
        config["sequence_length"]["after_change2"],
        skip_first=True,
    )

    graphs3 = _generate_graph_segment(
        generator,
        n,
        edges["change2"],
        edges["change3"],
        config["sequence_length"]["after_change2"],
        config["sequence_length"]["after_change3"],
        skip_first=True,
    )

    all_graphs = graphs1 + graphs2 + graphs3

    return {
        "graphs": all_graphs,
        "change_points": change_points,
        "params": {
            "edges": [
                edges["initial"],
                edges["change1"],
                edges["change2"],
                edges["change3"],
            ]
        },
    }


# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#


def main():
    """Main entry point for graph generation."""
    print("\nGenerating Barabási-Albert Graph Sequence")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Nodes per graph: {BA_CONFIG['n']}")
    print(f"  - Edge parameters: {BA_CONFIG['edges']}")
    print(f"  - Sequence lengths: {BA_CONFIG['sequence_length']}")

    result = generate_ba_graphs()

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Graph shape: {result['graphs'][0].shape}")


if __name__ == "__main__":
    main()
