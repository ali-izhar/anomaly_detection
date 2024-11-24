# tests/create_nw_graphs.py

"""
Newman-Watts (NW) Small-World Graph Sequence Generator

This module generates sequences of NW small-world graphs with multiple parameter changes 
for testing change point detection. It creates a sequence of graphs where both the number 
of nearest neighbors and rewiring probability change at specific points, simulating 
network evolution with structural changes.

Usage:
    1. Update the NW_CONFIG dictionary below with desired parameters
    2. Run this file directly: python tests/create_nw_graphs.py
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

NW_CONFIG = {
    "n": 30,  # Number of nodes in each graph
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
    k1: int,
    k2: int,
    p1: float,
    p2: float,
    set1: int,
    set2: int,
    skip_first: bool = False,
) -> List[np.ndarray]:
    """Generate a segment of NW graphs with parameter changes."""
    graphs = generator.newman_watts(
        n=n, k1=k1, k2=k2, p1=p1, p2=p2, set1=set1, set2=set2
    )
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


def generate_nw_graphs(config: Dict = NW_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Newman-Watts small-world graph sequence with multiple parameter changes."""
    generator = GraphGenerator()

    graphs1 = _generate_graph_segment(
        generator,
        config["n"],
        config["neighbors"]["initial"],
        config["neighbors"]["change1"],
        config["rewiring_prob"]["initial"],
        config["rewiring_prob"]["change1"],
        config["sequence_length"]["before_change"],
        config["sequence_length"]["after_change1"],
    )

    graphs2 = _generate_graph_segment(
        generator,
        config["n"],
        config["neighbors"]["change1"],
        config["neighbors"]["change2"],
        config["rewiring_prob"]["change1"],
        config["rewiring_prob"]["change2"],
        config["sequence_length"]["after_change1"],
        config["sequence_length"]["after_change2"],
        skip_first=True,
    )

    graphs3 = _generate_graph_segment(
        generator,
        config["n"],
        config["neighbors"]["change2"],
        config["neighbors"]["change3"],
        config["rewiring_prob"]["change2"],
        config["rewiring_prob"]["change3"],
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
    print("\nGenerating Newman-Watts Small-World Graph Sequence")
    print("------------------------------------------------")
    print(f"Configuration:")
    print(f"  - Nodes per graph: {NW_CONFIG['n']}")
    print(f"  - Neighbor parameters: {NW_CONFIG['neighbors']}")
    print(f"  - Rewiring probabilities: {NW_CONFIG['rewiring_prob']}")
    print(f"  - Sequence lengths: {NW_CONFIG['sequence_length']}")

    result = generate_nw_graphs(NW_CONFIG)

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Graph shape: {result['graphs'][0].shape}")


if __name__ == "__main__":
    main()
