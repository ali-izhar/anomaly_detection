# synthetic_data/create_graph_sequences.py

"""
Graph Sequence Generator

This module generates sequences of graphs (BA, ER, or NW) with a random number of parameter changes.
Each sequence has between 1-3 change points at random locations, with random parameters.
"""

import sys
import numpy as np
import yaml
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph import GraphGenerator, SyntheticDataGenerator
from synthetic_data.visualize_graphs import GraphVisualizer


class GraphType(Enum):
    BA = "barabasi_albert"
    ER = "erdos_renyi"
    NW = "newman_watts"


@dataclass
class GraphConfig:
    """Configuration for graph generation"""

    graph_type: GraphType
    nodes: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    params: Dict

    @classmethod
    def from_yaml(cls, graph_type: GraphType, config_path: str = None) -> "GraphConfig":
        """Create configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs/graph_config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        common = config["common"]
        graph_params = config[graph_type.name.lower()]

        return cls(
            graph_type=graph_type,
            nodes=common["nodes"],
            seq_len=common["seq_len"],
            min_segment=common["min_segment"],
            min_changes=common["min_changes"],
            max_changes=common["max_changes"],
            params=graph_params,
        )


def _generate_random_change_points(
    config: GraphConfig,
) -> Tuple[List[int], List[Dict], int, int]:
    """Generate random change points and corresponding parameters."""
    # Generate random sequence length and number of nodes
    seq_len = config.seq_len
    n = config.nodes
    min_seg = config.min_segment

    # Generate change points
    num_changes = np.random.randint(config.min_changes, config.max_changes + 1)

    # Keep trying until we get enough valid change points
    while True:
        points = []
        available_positions = list(range(min_seg, seq_len - min_seg))

        for _ in range(num_changes):
            if len(available_positions) < min_seg:
                break
            point = np.random.choice(available_positions)
            points.append(point)
            # Remove positions that are too close to the chosen point
            mask = np.abs(np.array(available_positions) - point) >= min_seg
            available_positions = [
                p for i, p in enumerate(available_positions) if mask[i]
            ]

        points = sorted(points)
        # Only accept if we have at least minimum required changes
        if len(points) >= config.min_changes:
            break

    # Generate parameters based on graph type
    params = []
    if config.graph_type == GraphType.BA:
        # Use the specified initial edges from config
        m_initial = config.params["edges"]
        params.append({"m1": m_initial, "m2": m_initial})

        # For each change point, ensure significant parameter change
        prev_m = m_initial
        for _ in range(len(points)):
            while True:
                m = np.random.randint(
                    config.params["min_edges"], config.params["max_edges"] + 1
                )
                # Ensure significant change (at least 30% difference)
                if abs(m - prev_m) / prev_m >= 0.3:
                    break
            params.append({"m1": m, "m2": m})
            prev_m = m

    elif config.graph_type == GraphType.ER:
        # Use the specified initial probability from config
        p_initial = config.params["p"]
        params.append({"p1": p_initial, "p2": p_initial})

        # For each change point, ensure significant parameter change
        prev_p = p_initial
        for _ in range(len(points)):
            while True:
                p = np.random.uniform(config.params["min_p"], config.params["max_p"])
                # Ensure significant change (at least 30% difference)
                if abs(p - prev_p) / prev_p >= 0.3:
                    break
            params.append({"p1": p, "p2": p})
            prev_p = p

    elif config.graph_type == GraphType.NW:
        # Use the specified initial k and p from config
        k_initial = config.params["k"]
        p_initial = config.params["p"]
        params.append(
            {"k1": k_initial, "k2": k_initial, "p1": p_initial, "p2": p_initial}
        )

        # For each change point, ensure significant parameter change
        prev_k, prev_p = k_initial, p_initial
        for _ in range(len(points)):
            while True:
                k = np.random.randint(
                    config.params["min_k"], config.params["max_k"] + 1
                )
                p = np.random.uniform(config.params["min_p"], config.params["max_p"])
                # Ensure significant change in either k or p
                if (abs(k - prev_k) / prev_k >= 0.3) or (
                    abs(p - prev_p) / prev_p >= 0.3
                ):
                    break
            params.append({"k1": k, "k2": k, "p1": p, "p2": p})
            prev_k, prev_p = k, p

    return points, params, seq_len, n


def generate_graph_sequence(config: GraphConfig) -> Dict:
    """Generate graph sequence with random number of parameter changes."""
    # Generate random change points, parameters, sequence length and number of nodes
    change_points, params, seq_length, n = _generate_random_change_points(config)

    # Initialize generators
    graph_generator = GraphGenerator()
    syn_generator = SyntheticDataGenerator(graph_generator)

    all_graphs = []
    generator_method = getattr(syn_generator._generator, config.graph_type.value)

    # Generate graph segments
    for i in range(len(change_points)):
        start = change_points[i - 1] if i > 0 else 0
        end = change_points[i]
        length = end - start

        # Create generation config for this segment
        gen_config = {"n": n, "set1": length, "set2": 0, **params[i]}

        # Generate segment
        segment = generator_method(**gen_config)
        all_graphs.extend(segment)

    # Generate final segment
    final_length = seq_length - (change_points[-1] if change_points else 0)
    final_config = {"n": n, "set1": final_length, "set2": 0, **params[-1]}
    final_segment = generator_method(**final_config)
    all_graphs.extend(final_segment)

    return {
        "graphs": all_graphs,
        "change_points": change_points,
        "params": params,
        "graph_type": config.graph_type.value,
        "num_changes": len(change_points),
        "n": n,
        "sequence_length": seq_length,
    }


def main(visualize: bool = False):
    """Main entry point for graph generation."""
    graph_types = [GraphType.BA, GraphType.ER, GraphType.NW]

    for graph_type in graph_types:
        config = GraphConfig.from_yaml(
            graph_type, config_path="configs/graph_config.yaml"
        )
        print(f"\nGenerating {graph_type.value} Graph Sequence")
        print("-" * 50)
        print(f"Configuration:")
        print(f"  - Graph type: {config.graph_type.value}")
        print(f"  - Nodes per graph: {config.nodes}")
        print(f"  - Sequence length: {config.seq_len}")
        print(f"  - Minimum segment length: {config.min_segment}")
        print(f"  - Change points range: [{config.min_changes}, {config.max_changes}]")
        print(f"  - Parameters: {config.params}")

        result = generate_graph_sequence(config)

        print("\nResults:")
        print(
            f"  - Generated {len(result['graphs'])} graphs with {result['n']} nodes each"
        )
        print(f"  - Sequence length: {result['sequence_length']}")
        print(f"  - Number of change points: {result['num_changes']}")
        print(f"  - Change points at t={result['change_points']}")
        print(f"  - Parameters at each segment: {result['params']}")

        if visualize:
            visualizer = GraphVisualizer(
                graphs=result["graphs"],
                change_points=result["change_points"],
                graph_type=result["graph_type"],
            )
            visualizer.create_dashboard()


if __name__ == "__main__":
    main(visualize=True)
