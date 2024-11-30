# src/models/data.py

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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.graph import GraphGenerator
from src.graph.features import extract_centralities, compute_embeddings, adjacency_to_graph

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


def _calculate_change_points(config: Dict) -> List[int]:
    """Calculate the indices where parameter changes occur."""
    seq_len = config["sequence_length"]
    return [
        seq_len["before_change"],
        seq_len["before_change"] + seq_len["after_change1"],
        seq_len["before_change"] + seq_len["after_change1"] + seq_len["after_change2"],
    ]


def generate_ba_graphs(config: Dict = BA_CONFIG) -> Dict[str, List[np.ndarray]]:
    """Generate Barabási-Albert graph sequence with multiple parameter changes."""
    generator = GraphGenerator()
    graphs1 = _generate_graph_segment(
        generator,
        config["n"],
        config["edges"]["initial"],
        config["edges"]["change1"],
        config["sequence_length"]["before_change"],
        config["sequence_length"]["after_change1"],
    )

    graphs2 = _generate_graph_segment(
        generator,
        config["n"],
        config["edges"]["change1"],
        config["edges"]["change2"],
        config["sequence_length"]["after_change1"],
        config["sequence_length"]["after_change2"],
        skip_first=True,
    )

    graphs3 = _generate_graph_segment(
        generator,
        config["n"],
        config["edges"]["change2"],
        config["edges"]["change3"],
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


def extract_features(graphs: List[np.ndarray]) -> Dict[str, List[float]]:
    """Extract features from a list of graphs."""
    results = []
    for adj_matrix in graphs:
        G = adjacency_to_graph(adj_matrix)
        degrees = [d for n, d in G.degree()]
        centralities = extract_centralities([adj_matrix])
        svd_emb = compute_embeddings([adj_matrix], method="svd")
        lsvd_emb = compute_embeddings([adj_matrix], method="lsvd")
        results.append({
            "degree": degrees,
            "centralities": centralities,
            "svd": svd_emb,
            "lsvd": lsvd_emb,
        })
    return results

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

    result = generate_ba_graphs(BA_CONFIG)

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Graph shape: {result['graphs'][0].shape}")

    print("First graph:")
    print(result["graphs"][0])

    features = extract_features(result["graphs"])
    print(f"Number of features: {len(features)}")

    print(f"First feature:")
    print(features[0])

    features = {
        'degree': len(features[0]['degree']),  # Direct list
        'centralities': {
            'degree': len(features[0]['centralities']['degree'][0]),
            'betweenness': len(features[0]['centralities']['betweenness'][0]),
            'eigenvector': len(features[0]['centralities']['eigenvector'][0]),
            'closeness': len(features[0]['centralities']['closeness'][0])
        },
        'svd': len(features[0]['svd'][0]),
        'lsvd': len(features[0]['lsvd'][0])
    }

    print("Feature counts:")
    print(f"Degree: {features['degree']} values")
    print("\nCentrality measures:")
    for measure, count in features['centralities'].items():
        print(f"- {measure}: {count} values")
    print(f"\nSVD: {features['svd']} values")
    print(f"LSVD: {features['lsvd']} values")


if __name__ == "__main__":
    main()
