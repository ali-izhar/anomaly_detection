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
        "initial": 3,  # Initial m parameter (edges per new node)
        "change_min": 2,  # Minimum change in edge count
        "change_max": 8,  # Maximum change in edge count
    },
    "sequence_length": 200,  # Total number of graphs
    "min_segment_length": 30,  # Minimum length between changes
    "noise_std": 0.1,  # Standard deviation for noise
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
    skip_first: bool = False
) -> List[np.ndarray]:
    """Generate a segment of BA graphs with parameter change."""
    graphs = generator.barabasi_albert(n=n, m1=m1, m2=m2, set1=set1, set2=set2)
    if skip_first:
        return graphs[set1:]
    return graphs

def generate_ba_graphs() -> Dict:
    """Generate BA graph sequence with multiple random parameter changes."""
    config = BA_CONFIG
    n = config["n"]
    
    # Generate three segments with random change points
    min_len = config["min_segment_length"]
    total_len = config["sequence_length"]
    
    # Generate random but well-separated change points
    cp1 = np.random.randint(min_len, total_len // 3)
    cp2 = np.random.randint(cp1 + min_len, 2 * total_len // 3)
    cp3 = np.random.randint(cp2 + min_len, total_len - min_len)
    change_points = [cp1, cp2, cp3]
    
    # Generate random edge parameters within bounds
    m_values = [config["edges"]["initial"]]  # Start with initial value
    for _ in range(3):  # Generate 3 more values for changes
        m = np.random.randint(
            config["edges"]["change_min"],
            config["edges"]["change_max"] + 1
        )
        m_values.append(m)
    
    # Generate graph segments using the GraphGenerator interface
    generator = GraphGenerator()
    all_graphs = []
    
    # First segment: initial -> first change
    graphs = _generate_graph_segment(
        generator,
        n=n,
        m1=m_values[0],
        m2=m_values[1],
        set1=cp1,
        set2=cp2 - cp1
    )
    all_graphs.extend(graphs)
    
    # Second segment: first change -> second change
    graphs = _generate_graph_segment(
        generator,
        n=n,
        m1=m_values[1],
        m2=m_values[2],
        set1=cp2 - cp1,
        set2=cp3 - cp2,
        skip_first=True
    )
    all_graphs.extend(graphs)
    
    # Third segment: second change -> end
    graphs = _generate_graph_segment(
        generator,
        n=n,
        m1=m_values[2],
        m2=m_values[3],
        set1=cp3 - cp2,
        set2=total_len - cp3,
        skip_first=True
    )
    all_graphs.extend(graphs)
    
    # Add noise to all graphs
    noisy_graphs = []
    for adj in all_graphs:
        noise = np.random.normal(0, config["noise_std"], adj.shape)
        noisy = adj.copy()
        noisy += noise
        noisy = (noisy + noisy.T) / 2  # Ensure symmetry
        noisy = np.clip(noisy, 0, 1)  # Clip values to [0, 1]
        noisy_graphs.append(noisy)
    
    return {
        "graphs": noisy_graphs,
        "change_points": change_points,
        "params": {
            "edges": m_values,
            "noise_std": config["noise_std"]
        }
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
    print(f"  - Initial edges: {BA_CONFIG['edges']['initial']}")
    print(f"  - Edge range: [{BA_CONFIG['edges']['change_min']}, {BA_CONFIG['edges']['change_max']}]")
    print(f"  - Sequence length: {BA_CONFIG['sequence_length']}")
    print(f"  - Minimum segment length: {BA_CONFIG['min_segment_length']}")
    print(f"  - Noise std: {BA_CONFIG['noise_std']}")
    print(f"  - Number of changes: {len(BA_CONFIG['change_points'])}")

    result = generate_ba_graphs()

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Edge parameters: {result['params']['edges']}")
    print(f"  - Graph shape: {result['graphs'][0].shape}")

if __name__ == "__main__":
    main()
