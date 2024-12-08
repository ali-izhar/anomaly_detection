# synthetic_data/create_dataset.py

"""
Dataset Generator for Graph Anomaly Detection

This module creates synthetic graph datasets with labeled anomalies based on 
martingale computations and different graph types (BA, ER, NW).
"""

import logging
import os
import sys
import h5py
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import cupy as cp

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graph_sequences import GraphConfig, GraphType, generate_graph_sequence
from src.changepoint import ChangePointDetector

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig(GraphConfig):
    """Configuration for dataset generation, extends GraphConfig."""

    num_sequences: int = 20
    graph_types: List[str] = None
    split_ratio: Dict[str, float] = None
    output_dir: str = "dataset"
    graph_params: Dict[str, Dict] = None

    @classmethod
    def from_yaml(
        cls, config_path: str = None, graph_config_path: str = None
    ) -> "DatasetConfig":
        if config_path is None:
            config_path = Path(__file__).parent / "configs/dataset_config.yaml"
        if graph_config_path is None:
            graph_config_path = Path(__file__).parent / "configs/graph_config.yaml"

        with open(config_path, "r") as f:
            dataset_config = yaml.safe_load(f)["dataset"]

        with open(graph_config_path, "r") as f:
            graph_config_data = yaml.safe_load(f)

        graph_params = {
            "ba": graph_config_data["ba"],
            "er": graph_config_data["er"],
            "nw": graph_config_data["nw"],
        }

        graph_config = super().from_yaml(
            GraphType[dataset_config["graph_types"][0]], graph_config_path
        )

        return cls(
            graph_type=graph_config.graph_type,
            nodes=graph_config_data["common"]["nodes"],
            min_seq_length=graph_config_data["common"]["min_seq_length"],
            max_seq_length=graph_config_data["common"]["max_seq_length"],
            min_segment=graph_config_data["common"]["min_segment"],
            min_changes=graph_config_data["common"]["min_changes"],
            max_changes=graph_config_data["common"]["max_changes"],
            params=graph_config.params,
            num_sequences=dataset_config["num_sequences"],
            graph_types=dataset_config["graph_types"],
            split_ratio=dataset_config["split_ratio"],
            output_dir=dataset_config["output_dir"],
            graph_params=graph_params,
        )


class DatasetGenerator:
    """Generator for graph sequences dataset."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def _compute_features(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Extract and aggregate global features for each graph.
        Returns a (seq_len, 6) array where columns are:
        [mean_degree, mean_betweenness, mean_eigenvector, mean_closeness, mean_svd, mean_lsvd]

        Each of these features is initially (seq_len, n_nodes), and we take the mean over
        the node dimension to get a (seq_len,) vector for each feature.
        """
        validate_graph_shapes(graphs)

        detector = ChangePointDetector()
        detector.initialize(graphs)
        raw_features = detector.extract_features()
        # raw_features keys: degree, betweenness, closeness, eigenvector, svd, lsvd
        # Each is (seq_len, n_nodes)

        seq_len = len(graphs)
        n_nodes = graphs[0].shape[0]

        feature_order = [
            "degree",
            "betweenness",
            "eigenvector",
            "closeness",
            "svd",
            "lsvd",
        ]
        global_features = np.zeros((seq_len, len(feature_order)), dtype=np.float32)

        for i, name in enumerate(feature_order):
            if name not in raw_features:
                raise ValueError(f"Missing required feature: {name}")
            feat = np.array(raw_features[name], dtype=np.float32)  # (seq_len, n_nodes)
            if feat.shape != (seq_len, n_nodes):
                raise ValueError(
                    f"{name} feature has shape {feat.shape}, expected {(seq_len, n_nodes)}"
                )
            # Average over nodes to get (seq_len,)
            mean_feat = feat.mean(axis=1)
            global_features[:, i] = mean_feat

        return global_features  # (seq_len, 6)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize the (num_sequences, max_seq_len, 6) feature array independently."""
        # features shape: (N, T, 6)
        feat_cp = cp.asarray(features)
        mean = cp.mean(feat_cp, axis=(0, 1), keepdims=True)
        std = cp.std(feat_cp, axis=(0, 1), keepdims=True) + 1e-8
        normalized = (feat_cp - mean) / std
        return cp.asnumpy(normalized)

    def _pad_data(self, data: Dict) -> Dict:
        """Pad and normalize the data.

        - data["features"] is a list of arrays (seq_len, 6)
        - data["graphs"] is a list of arrays (seq_len, n_nodes, n_nodes)
        We pad them to (num_sequences, max_seq_len, ...) and normalize features.
        """
        max_seq_len = max(data["sequence_lengths"])
        num_sequences = len(data["graphs"])
        n_nodes = data["graphs"][0][0].shape[0]

        # Pad features
        feat_cp = cp.zeros((num_sequences, max_seq_len, 6), dtype=cp.float32)
        for i, feat_seq in enumerate(data["features"]):
            seq_cp = cp.asarray(feat_seq)  # (seq_len, 6)
            feat_cp[i, : seq_cp.shape[0], :] = seq_cp
        padded_features = cp.asnumpy(feat_cp)

        # Pad graphs
        graphs_cp = cp.zeros(
            (num_sequences, max_seq_len, n_nodes, n_nodes), dtype=cp.float32
        )
        for i, graphs in enumerate(data["graphs"]):
            g_cp = cp.asarray(np.stack(graphs, axis=0), dtype=cp.float32)
            graphs_cp[i, : g_cp.shape[0]] = g_cp
        padded_graphs = cp.asnumpy(graphs_cp)

        # Normalize features
        normalized_features = self._normalize_features(padded_features)

        sequence_lengths = np.array(data["sequence_lengths"])
        graph_types = data["graph_types"]

        # Validate shapes
        if padded_graphs.shape != (num_sequences, max_seq_len, n_nodes, n_nodes):
            raise ValueError("Invalid padded graphs shape")

        if normalized_features.shape != (num_sequences, max_seq_len, 6):
            raise ValueError("Invalid padded features shape")

        return {
            "features": {"all": normalized_features},
            "graphs": padded_graphs,
            "sequence_lengths": sequence_lengths,
            "graph_types": graph_types,
            "change_points": data["change_points"],
        }

    def _save_to_hdf5(self, data_split: Dict):
        os.makedirs(self.config.output_dir, exist_ok=True)

        for split_name, data in data_split.items():
            split_dir = os.path.join(self.config.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            with h5py.File(os.path.join(split_dir, "data.h5"), "w") as hf:
                num_sequences = len(data["graphs"])
                max_seq_len = data["graphs"].shape[1]
                n_nodes = data["graphs"].shape[2]

                hf.create_dataset(
                    "lengths", data=data["sequence_lengths"], compression="gzip"
                )

                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset("graph_types", data=data["graph_types"], dtype=dt)

                adj_group = hf.create_group("adjacency")
                for seq_idx, graphs in enumerate(data["graphs"]):
                    adj_group.create_dataset(
                        f"sequence_{seq_idx}", data=graphs, compression="gzip"
                    )

                feat_group = hf.create_group("features")
                feat_group.create_dataset(
                    "all", data=data["features"]["all"], compression="gzip"
                )

                cp_group = hf.create_group("change_points")
                for seq_idx, cp in enumerate(data["change_points"]):
                    cp_group.create_dataset(
                        f"sequence_{seq_idx}", data=cp, compression="gzip"
                    )

    def _split_dataset(
        self,
        sequences: Dict[str, np.ndarray],
        graphs: np.ndarray,
        sequence_lengths: np.ndarray,
        change_points: List,
        graph_types: List[str],
    ) -> Dict:
        num_sequences = len(sequence_lengths)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)

        train_end = int(self.config.split_ratio["train"] * num_sequences)
        val_end = train_end + int(self.config.split_ratio["val"] * num_sequences)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def subset_features(features, idx):
            return {k: v[idx] for k, v in features.items()}

        return {
            "train": {
                "features": subset_features(sequences, train_idx),
                "graphs": graphs[train_idx],
                "sequence_lengths": sequence_lengths[train_idx],
                "change_points": [change_points[i] for i in train_idx],
                "graph_types": [graph_types[i] for i in train_idx],
            },
            "val": {
                "features": subset_features(sequences, val_idx),
                "graphs": graphs[val_idx],
                "sequence_lengths": sequence_lengths[val_idx],
                "change_points": [change_points[i] for i in val_idx],
                "graph_types": [graph_types[i] for i in val_idx],
            },
            "test": {
                "features": subset_features(sequences, test_idx),
                "graphs": graphs[test_idx],
                "sequence_lengths": sequence_lengths[test_idx],
                "change_points": [change_points[i] for i in test_idx],
                "graph_types": [graph_types[i] for i in test_idx],
            },
        }

    def _generate_single_sequence(self, graph_config: GraphConfig) -> Dict:
        """Generate a single graph sequence and its features."""
        result = generate_graph_sequence(graph_config)
        features = self._compute_features(result["graphs"])
        return {
            "graphs": result["graphs"],
            "features": features,  # (seq_len, 6)
            "length": len(result["graphs"]),
            "graph_type": result["graph_type"],
            "change_points": result["change_points"],
        }

    def generate_sequences(self) -> Dict:
        data = {
            "features": [],
            "graphs": [],
            "sequence_lengths": [],
            "graph_types": [],
            "change_points": [],
        }

        tasks = []
        for graph_type_str in self.config.graph_types:
            graph_type = GraphType[graph_type_str.upper()]
            for _ in range(self.config.num_sequences):
                graph_config = GraphConfig(
                    graph_type=graph_type,
                    nodes=self.config.nodes,
                    min_seq_length=self.config.min_seq_length,
                    max_seq_length=self.config.max_seq_length,
                    min_segment=self.config.min_segment,
                    min_changes=self.config.min_changes,
                    max_changes=self.config.max_changes,
                    params=self.config.graph_params[graph_type_str.lower()],
                )
                tasks.append(graph_config)

        # Use multiprocessing to generate sequences
        with Pool(processes=min(cpu_count(), len(tasks))) as p:
            results = list(
                tqdm(
                    p.imap(self._generate_single_sequence, tasks),
                    total=len(tasks),
                    desc="Generating Sequences",
                    position=0,
                )
            )

        for res in results:
            data["features"].append(res["features"])  # (seq_len, 6)
            data["graphs"].append(res["graphs"])  # list of (n_nodes, n_nodes)
            data["sequence_lengths"].append(res["length"])
            data["graph_types"].append(res["graph_type"])
            data["change_points"].append(res["change_points"])

        # Pad and validate
        return self._pad_data(data)


def create_dataset(config: Optional[DatasetConfig] = None) -> Dict:
    if config is None:
        config = DatasetConfig.from_yaml()

    generator = DatasetGenerator(config)
    data = generator.generate_sequences()
    split_data = generator._split_dataset(
        sequences=data["features"],
        graphs=data["graphs"],
        sequence_lengths=data["sequence_lengths"],
        change_points=data["change_points"],
        graph_types=data["graph_types"],
    )

    generator._save_to_hdf5(split_data)
    return split_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph sequences dataset"
    )
    parser.add_argument(
        "--dataset-config",
        "-dc",
        type=str,
        help="Path to dataset config YAML file",
        default="configs/dataset_config.yaml",
    )
    parser.add_argument(
        "--graph-config",
        "-gc",
        type=str,
        help="Path to graph config YAML file",
        default="configs/graph_config.yaml",
    )
    parser.add_argument("--output", "-o", type=str, help="Override output directory")
    parser.add_argument("--graph-types", "-gt", nargs="+", help="Override graph types")
    parser.add_argument(
        "--sequences", "-s", type=int, help="Override number of sequences"
    )

    args = parser.parse_args()

    config = DatasetConfig.from_yaml(
        config_path=args.dataset_config, graph_config_path=args.graph_config
    )

    if args.output:
        config.output_dir = args.output
    if args.graph_types:
        config.graph_types = args.graph_types
    if args.sequences:
        config.num_sequences = args.sequences

    print("\nGenerating Graph Sequences Dataset")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Graph types: {config.graph_types}")
    print(f"  - Sequences per type: {config.num_sequences}")
    print(
        f"  - Sequence length range: [{config.min_seq_length}, {config.max_seq_length}]"
    )
    print(f"  - Nodes per graph: {config.nodes}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"\nGraph Parameters:")
    for graph_type in config.graph_types:
        print(f"  - {graph_type}: {config.graph_params[graph_type.lower()]}")
    print(f"\nDataset Parameters:")
    print(f"  - Split ratio: {config.split_ratio}")

    data = create_dataset(config)

    for split_name, split_data in data.items():
        print(f"\n{split_name.capitalize()} set:")
        print(f"  - Number of sequences: {len(split_data['graphs'])}")
        print(f"  - Feature types: {list(split_data['features'].keys())}")
        for feat_name, feat_array in split_data["features"].items():
            print(f"  - {feat_name} shape: {feat_array.shape}")
        print(f"  - Graph sequence lengths: {split_data['sequence_lengths']}")


def validate_graph_shapes(graphs: List[np.ndarray]) -> None:
    if not graphs:
        raise ValueError("Empty graph sequence")

    n_nodes = graphs[0].shape[0]
    if graphs[0].shape != (n_nodes, n_nodes):
        raise ValueError(
            f"First graph has invalid shape: {graphs[0].shape}, expected square matrix"
        )

    for i, g in enumerate(graphs):
        if not isinstance(g, np.ndarray):
            raise ValueError(f"Graph {i} is not a numpy array")
        if g.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"Graph {i} has inconsistent shape: {g.shape}, expected {(n_nodes, n_nodes)}"
            )


if __name__ == "__main__":
    main()
