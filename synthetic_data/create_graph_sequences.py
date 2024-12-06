# synthetic_data/create_graph_sequences.py

"""
Graph Sequence Generator

This module generates sequences of graphs (BA, ER, or NW) with a random number of parameter changes.
Each sequence has between 3-7 change points at random locations, with random parameters.
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
from tests.visualize_graphs import GraphVisualizer


class GraphType(Enum):
    BA = "barabasi_albert"
    ER = "erdos_renyi"
    NW = "newman_watts"


@dataclass
class GraphConfig:
    """Configuration for graph generation"""

    graph_type: GraphType
    n: int
    sequence_length: int
    min_segment: int
    min_changes: int
    max_changes: int
    params: Dict

    @classmethod
    def from_yaml(cls, graph_type: GraphType, config_path: str = None) -> "GraphConfig":
        """Create configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / "graph_config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get common parameters (used for all graph types)
        common = config["common"]

        # Get graph-specific parameters
        graph_params = config[graph_type.name.lower()]

        return cls(
            graph_type=graph_type,
            n=common["n"],
            sequence_length=common["sequence_length"],
            min_segment=common["min_segment"],
            min_changes=common["min_changes"],
            max_changes=common["max_changes"],
            params=graph_params,
        )


def _generate_random_change_points(config: GraphConfig) -> Tuple[List[int], List[Dict]]:
    """Generate random change points and corresponding parameters."""
    seq_len = config.sequence_length
    min_seg = config.min_segment

    # Generate change points
    num_changes = np.random.randint(config.min_changes, config.max_changes + 1)
    points = []
    available_positions = list(range(min_seg, seq_len - min_seg))

    for _ in range(num_changes):
        if len(available_positions) < min_seg:
            break
        point = np.random.choice(available_positions)
        points.append(point)
        mask = np.abs(np.array(available_positions) - point) >= min_seg
        available_positions = [p for i, p in enumerate(available_positions) if mask[i]]

    points = sorted(points)

    # Generate parameters based on graph type
    params = []
    if config.graph_type == GraphType.BA:
        params.append(
            {"m1": config.params["initial_edges"], "m2": config.params["initial_edges"]}
        )
        for _ in range(len(points)):
            m = np.random.randint(
                config.params["min_edges"], config.params["max_edges"] + 1
            )
            params.append({"m1": m, "m2": m})

    elif config.graph_type == GraphType.ER:
        params.append(
            {"p1": config.params["initial_p"], "p2": config.params["initial_p"]}
        )
        for _ in range(len(points)):
            p = np.random.uniform(config.params["min_p"], config.params["max_p"])
            params.append({"p1": p, "p2": p})

    elif config.graph_type == GraphType.NW:
        params.append(
            {
                "k1": config.params["initial_k"],
                "k2": config.params["initial_k"],
                "p1": config.params["initial_p"],
                "p2": config.params["initial_p"],
            }
        )
        for _ in range(len(points)):
            k = np.random.randint(config.params["min_k"], config.params["max_k"] + 1)
            p = np.random.uniform(config.params["min_p"], config.params["max_p"])
            params.append({"k1": k, "k2": k, "p1": p, "p2": p})

    return points, params


def generate_graph_sequence(config: GraphConfig) -> Dict:
    """Generate graph sequence with random number of parameter changes."""
    # Generate random change points and parameters
    change_points, params = _generate_random_change_points(config)

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
        gen_config = {"n": config.n, "set1": length, "set2": 0, **params[i]}

        # Generate segment
        segment = generator_method(**gen_config)
        all_graphs.extend(segment)

    # Generate final segment
    final_length = config.sequence_length - (change_points[-1] if change_points else 0)
    final_config = {"n": config.n, "set1": final_length, "set2": 0, **params[-1]}
    final_segment = generator_method(**final_config)
    all_graphs.extend(final_segment)

    return {
        "graphs": all_graphs,
        "change_points": change_points,
        "params": params,
        "graph_type": config.graph_type.value,
        "num_changes": len(change_points),
    }


def main(visualize: bool = False):
    """Main entry point for graph generation."""
    graph_types = [GraphType.BA, GraphType.ER, GraphType.NW]

    for graph_type in graph_types:
        config = GraphConfig.from_yaml(graph_type, config_path="graph_config.yaml")
        print(f"\nGenerating {graph_type.value} Graph Sequence")
        print("-" * 50)
        print(f"Configuration:")
        print(f"  - Graph type: {config.graph_type.value}")
        print(f"  - Nodes per graph: {config.n}")
        print(f"  - Sequence length: {config.sequence_length}")
        print(f"  - Minimum segment length: {config.min_segment}")
        print(f"  - Change points range: [{config.min_changes}, {config.max_changes}]")
        print(f"  - Parameters: {config.params}")

        result = generate_graph_sequence(config)

        print("\nResults:")
        print(f"  - Generated {len(result['graphs'])} graphs")
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
