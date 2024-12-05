# tests/create_ba_graphs.py

"""
Barabási-Albert (BA) Graph Sequence Generator

This module generates sequences of BA graphs with a random number of parameter changes.
Each sequence has between 3-7 change points at random locations, with random edge parameters.
"""

import numpy as np
from typing import List, Dict, Tuple
from src.graph import GraphGenerator

BA_CONFIG = {
    "n": 50,  # Number of nodes in each graph
    "edges": {
        "initial": 3,  # Initial m parameter
        "min": 2,     # Minimum edges after change
        "max": 8,     # Maximum edges after change
    },
    "sequence_length": 200,  # Total sequence length
    "min_segment": 20,      # Minimum length between changes
    "change_points": {
        "min": 3,    # Minimum number of change points
        "max": 7     # Maximum number of change points
    }
}

def _generate_random_change_points(config: Dict) -> Tuple[List[int], List[int]]:
    """Generate random change points and corresponding edge parameters.
    
    Returns:
        Tuple[List[int], List[int]]: (change_points, edge_parameters)
    """
    seq_len = config["sequence_length"]
    min_seg = config["min_segment"]
    
    # Randomly choose number of change points
    num_changes = np.random.randint(
        config["change_points"]["min"],
        config["change_points"]["max"] + 1
    )
    
    # Generate random change points with minimum separation
    points = []
    available_positions = list(range(min_seg, seq_len - min_seg))
    
    for _ in range(num_changes):
        if len(available_positions) < min_seg:
            break
            
        # Choose a random position
        point = np.random.choice(available_positions)
        points.append(point)
        
        # Remove nearby positions to ensure minimum separation
        mask = np.abs(np.array(available_positions) - point) >= min_seg
        available_positions = [p for i, p in enumerate(available_positions) if mask[i]]
    
    points = sorted(points)
    
    # Generate random edge parameters
    edge_params = [config["edges"]["initial"]]
    for _ in range(len(points)):
        m = np.random.randint(config["edges"]["min"], config["edges"]["max"] + 1)
        edge_params.append(m)
    
    return points, edge_params

def _generate_graph_segment(
    generator: GraphGenerator,
    n: int,
    m1: int,
    m2: int,
    length: int
) -> List[np.ndarray]:
    """Generate a segment of BA graphs."""
    return generator.barabasi_albert(n=n, m1=m1, m2=m2, set1=length, set2=0)

def generate_ba_graphs() -> Dict:
    """Generate BA graph sequence with random number of parameter changes."""
    config = BA_CONFIG
    n = config["n"]
    
    # Generate random change points and parameters
    change_points, edge_params = _generate_random_change_points(config)
    
    # Generate graph segments
    generator = GraphGenerator()
    all_graphs = []
    
    # First segment
    if len(change_points) > 0:
        segment_length = change_points[0]
        graphs = _generate_graph_segment(
            generator, n, edge_params[0], edge_params[0], segment_length
        )
        all_graphs.extend(graphs)
        
        # Middle segments
        for i in range(len(change_points) - 1):
            segment_length = change_points[i + 1] - change_points[i]
            graphs = _generate_graph_segment(
                generator, n, edge_params[i + 1], edge_params[i + 1], segment_length
            )
            all_graphs.extend(graphs)
        
        # Last segment
        segment_length = config["sequence_length"] - change_points[-1]
        graphs = _generate_graph_segment(
            generator, n, edge_params[-1], edge_params[-1], segment_length
        )
        all_graphs.extend(graphs)
    
    return {
        "graphs": all_graphs,
        "change_points": change_points,
        "params": {
            "edges": edge_params,
            "num_changes": len(change_points)
        }
    }

def main():
    """Main entry point for graph generation."""
    print("\nGenerating Barabási-Albert Graph Sequence")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Nodes per graph: {BA_CONFIG['n']}")
    print(f"  - Initial edges: {BA_CONFIG['edges']['initial']}")
    print(f"  - Edge range: [{BA_CONFIG['edges']['min']}, {BA_CONFIG['edges']['max']}]")
    print(f"  - Sequence length: {BA_CONFIG['sequence_length']}")
    print(f"  - Minimum segment length: {BA_CONFIG['min_segment']}")
    print(f"  - Change points range: [{BA_CONFIG['change_points']['min']}, {BA_CONFIG['change_points']['max']}]")

    result = generate_ba_graphs()

    print("\nResults:")
    print(f"  - Generated {len(result['graphs'])} graphs")
    print(f"  - Number of change points: {result['params']['num_changes']}")
    print(f"  - Change points at t={result['change_points']}")
    print(f"  - Edge parameters: {result['params']['edges']}")

if __name__ == "__main__":
    main()
