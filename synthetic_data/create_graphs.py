"""
Graph Sequence Generator
This module generates sequences of graphs (BA, ER, or NW) with config/graph_config.yaml params.
"""

import sys
import numpy as np
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph import GraphGenerator, SyntheticDataGenerator


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

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        
        if self.min_changes > self.max_changes:
            raise ValueError("min_changes cannot be greater than max_changes")
        
        # Check if sequence length can accommodate minimum segments
        min_total_length = self.min_segment * (self.max_changes + 1)
        if self.seq_len < min_total_length:
            raise ValueError(
                f"Sequence length ({self.seq_len}) too short to accommodate "
                f"{self.max_changes} changes with minimum segment length {self.min_segment}"
            )
        
        # Validate graph-specific parameters
        if self.graph_type == GraphType.BA:
            self._validate_ba_params()
        elif self.graph_type == GraphType.ER:
            self._validate_er_params()
        elif self.graph_type == GraphType.NW:
            self._validate_nw_params()

    def _validate_ba_params(self):
        required = {"edges", "min_edges", "max_edges"}
        if not required.issubset(self.params.keys()):
            raise ValueError(f"BA graph requires parameters: {required}")
        if self.params["min_edges"] > self.params["max_edges"]:
            raise ValueError("min_edges cannot be greater than max_edges")
        if self.params["edges"] < 1:
            raise ValueError("Initial edges must be positive")

    def _validate_er_params(self):
        required = {"p", "min_p", "max_p"}
        if not required.issubset(self.params.keys()):
            raise ValueError(f"ER graph requires parameters: {required}")
        if not 0 <= self.params["p"] <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        if self.params["min_p"] > self.params["max_p"]:
            raise ValueError("min_p cannot be greater than max_p")

    def _validate_nw_params(self):
        required = {"k", "p", "min_k", "max_k", "min_p", "max_p"}
        if not required.issubset(self.params.keys()):
            raise ValueError(f"NW graph requires parameters: {required}")
        if self.params["min_k"] > self.params["max_k"]:
            raise ValueError("min_k cannot be greater than max_k")
        if not 0 <= self.params["p"] <= 1:
            raise ValueError("Probability p must be between 0 and 1")

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
) -> Tuple[List[int], List[Dict[str, Union[int, float]]], int, int]:
    """Generate random change points and corresponding parameters."""
    seq_len = config.seq_len
    n = config.nodes
    min_seg = config.min_segment
    
    MAX_ATTEMPTS = 100  # Prevent infinite loops
    num_changes = np.random.randint(config.min_changes, config.max_changes + 1)
    
    for _ in range(MAX_ATTEMPTS):
        points = []
        available_positions = list(range(min_seg, seq_len - min_seg))
        
        if len(available_positions) < num_changes:
            raise ValueError(
                f"Cannot generate {num_changes} changes with current constraints. "
                f"Available positions: {len(available_positions)}"
            )

        for _ in range(num_changes):
            if len(available_positions) < min_seg:
                break
            point = np.random.choice(available_positions)
            points.append(point)
            # Remove positions that are too close to the chosen point
            mask = np.abs(np.array(available_positions) - point) >= min_seg
            available_positions = [p for i, p in enumerate(available_positions) if mask[i]]

        points = sorted(points)
        if len(points) >= config.min_changes:
            break
    else:
        raise RuntimeError(
            f"Failed to generate valid change points after {MAX_ATTEMPTS} attempts"
        )

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
    try:
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

            try:
                # Generate segment
                segment = generator_method(**gen_config)
                all_graphs.extend(segment)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate graph segment {i} with parameters {gen_config}"
                ) from e

        # Generate final segment
        final_length = seq_length - (change_points[-1] if change_points else 0)
        final_config = {"n": n, "set1": final_length, "set2": 0, **params[-1]}
        
        try:
            final_segment = generator_method(**final_config)
            all_graphs.extend(final_segment)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate final segment with parameters {final_config}"
            ) from e

        return {
            "graphs": all_graphs,
            "change_points": change_points,
            "params": params,
            "graph_type": config.graph_type.value,
            "num_changes": len(change_points),
            "n": n,
            "sequence_length": seq_length,
        }

    except Exception as e:
        raise RuntimeError("Failed to generate graph sequence") from e


def main():
    """Main entry point for graph generation."""
    try:
        graph_types = [GraphType.BA, GraphType.ER, GraphType.NW]

        for graph_type in graph_types:
            try:
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

            except Exception as e:
                print(f"Error generating {graph_type.value} sequence: {str(e)}")
                continue

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
