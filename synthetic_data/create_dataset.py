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
            min_n=graph_config_data["common"]["min_n"],
            max_n=graph_config_data["common"]["max_n"],
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

    def _compute_features(self, graphs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract node-level features for each graph."""
        validate_graph_shapes(graphs)

        detector = ChangePointDetector()
        detector.initialize(graphs)
        raw_features = detector.extract_features()

        processed_features = {}

        # Centrality features
        for name in ["degree", "betweenness", "closeness", "eigenvector"]:
            if name in raw_features:
                processed_features[name] = np.array(
                    raw_features[name], dtype=np.float32
                )

        # Embedding features
        embedding_dim = 16  # fixed embedding dimension
        for name in ["svd", "lsvd"]:
            if name in raw_features:
                feat = np.array(raw_features[name], dtype=np.float32)
                if len(feat.shape) == 2:
                    seq_len, n_nodes = feat.shape
                    feat = feat.reshape(seq_len, n_nodes, 1)
                    feat = np.tile(feat, (1, 1, embedding_dim))
                processed_features[name] = feat

        validate_feature_shapes(processed_features, len(graphs), graphs[0].shape[0])
        return processed_features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        # Convert to GPU
        feat_cp = cp.asarray(features)
        mean = cp.mean(feat_cp, axis=(0, 1), keepdims=True)
        std = cp.std(feat_cp, axis=(0, 1), keepdims=True) + 1e-8
        normalized = (feat_cp - mean) / std
        return cp.asnumpy(normalized)

    def _split_dataset(
        self,
        sequences: Dict[str, np.ndarray],
        graphs: np.ndarray,
        sequence_lengths: np.ndarray,
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
            },
            "val": {
                "features": subset_features(sequences, val_idx),
                "graphs": graphs[val_idx],
                "sequence_lengths": sequence_lengths[val_idx],
            },
            "test": {
                "features": subset_features(sequences, test_idx),
                "graphs": graphs[test_idx],
                "sequence_lengths": sequence_lengths[test_idx],
            },
        }

    def _generate_single_sequence(self, graph_config: GraphConfig) -> Dict:
        """Generate a single graph sequence and its features."""
        result = generate_graph_sequence(graph_config)
        features = self._compute_features(result["graphs"])
        return {
            "graphs": result["graphs"],
            "features": features,
            "length": len(result["graphs"]),
            "graph_type": graph_config.graph_type.value,
        }

    def generate_sequences(self) -> Dict:
        data = {
            "features": {
                "degree": [],
                "betweenness": [],
                "closeness": [],
                "eigenvector": [],
                "svd": [],
                "lsvd": [],
            },
            "graphs": [],
            "sequence_lengths": [],
            "graph_types": [],
        }

        # Prepare tasks for parallel generation
        tasks = []
        for graph_type_str in self.config.graph_types:
            graph_type = GraphType[graph_type_str.upper()]
            for _ in range(self.config.num_sequences):
                graph_config = GraphConfig(
                    graph_type=graph_type,
                    min_n=self.config.min_n,
                    max_n=self.config.min_n,
                    min_seq_length=self.config.min_seq_length,
                    max_seq_length=self.config.max_seq_length,
                    min_segment=self.config.min_segment,
                    min_changes=self.config.min_changes,
                    max_changes=self.config.max_changes,
                    params=self.config.graph_params[graph_type_str.lower()],
                )
                tasks.append(graph_config)

        # Use multiprocessing to generate sequences
        # Note: generate_graph_sequence and feature extraction are CPU-based
        with Pool(processes=min(cpu_count(), len(tasks))) as p:
            results = list(
                tqdm(
                    p.imap(self._generate_single_sequence, tasks),
                    total=len(tasks),
                    desc="Generating Sequences",
                    position=0,
                )
            )

        # Aggregate results
        for res in results:
            for feat_name in res["features"]:
                data["features"][feat_name].append(res["features"][feat_name])
            data["graphs"].append(res["graphs"])
            data["sequence_lengths"].append(res["length"])
            data["graph_types"].append(res["graph_type"])

        # Pad and validate
        return self._pad_data(data)

    def _pad_data(self, data: Dict) -> Dict:
        max_seq_len = max(data["sequence_lengths"])
        num_sequences = len(data["graphs"])
        n_nodes = data["graphs"][0][0].shape[0]

        # Move data to GPU for large array ops
        # Padded features
        padded_features = {}
        for feat_name in ["degree", "betweenness", "closeness", "eigenvector"]:
            arr_cp = cp.zeros((num_sequences, max_seq_len, n_nodes), dtype=cp.float32)
            for i, seq in enumerate(data["features"][feat_name]):
                seq_cp = cp.asarray(seq)
                arr_cp[i, : seq_cp.shape[0], :] = seq_cp
            padded_features[feat_name] = cp.asnumpy(arr_cp)

        for feat_name in ["svd", "lsvd"]:
            embedding_dim = data["features"][feat_name][0].shape[-1]
            arr_cp = cp.zeros(
                (num_sequences, max_seq_len, n_nodes, embedding_dim), dtype=cp.float32
            )
            for i, seq in enumerate(data["features"][feat_name]):
                seq_cp = cp.asarray(seq)
                # If mismatch in embedding dim, fix it:
                if seq_cp.shape[-1] != embedding_dim:
                    # resize
                    resized_cp = cp.zeros(
                        (seq_cp.shape[0], seq_cp.shape[1], embedding_dim),
                        dtype=cp.float32,
                    )
                    min_dim = min(seq_cp.shape[-1], embedding_dim)
                    resized_cp[..., :min_dim] = seq_cp[..., :min_dim]
                    seq_cp = resized_cp
                arr_cp[i, : seq_cp.shape[0], :, :] = seq_cp
            padded_features[feat_name] = cp.asnumpy(arr_cp)

        # Pad graphs
        graphs_cp = cp.zeros(
            (num_sequences, max_seq_len, n_nodes, n_nodes), dtype=cp.float32
        )
        for i, graphs in enumerate(data["graphs"]):
            g_cp = cp.asarray(np.stack(graphs, axis=0), dtype=cp.float32)
            graphs_cp[i, : g_cp.shape[0]] = g_cp

        padded_graphs = cp.asnumpy(graphs_cp)
        sequence_lengths = np.array(data["sequence_lengths"])
        graph_types = data["graph_types"]

        validate_padded_shapes(
            data={"features": padded_features, "graphs": padded_graphs},
            num_sequences=num_sequences,
            max_seq_len=max_seq_len,
            n_nodes=n_nodes,
        )

        return {
            "features": padded_features,
            "graphs": padded_graphs,
            "sequence_lengths": sequence_lengths,
            "graph_types": graph_types,
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

                validate_padded_shapes(
                    data=data,
                    num_sequences=num_sequences,
                    max_seq_len=max_seq_len,
                    n_nodes=n_nodes,
                )

                hf.create_dataset(
                    "lengths", data=data["sequence_lengths"], compression="gzip"
                )

                if "graph_types" in data:
                    dt = h5py.special_dtype(vlen=str)
                    hf.create_dataset("graph_types", data=data["graph_types"], dtype=dt)

                adj_group = hf.create_group("adjacency")
                for seq_idx, graphs in enumerate(data["graphs"]):
                    adj_group.create_dataset(
                        f"sequence_{seq_idx}", data=graphs, compression="gzip"
                    )

                feat_group = hf.create_group("features")
                for feat_name in [
                    "degree",
                    "betweenness",
                    "closeness",
                    "eigenvector",
                    "svd",
                    "lsvd",
                ]:
                    if feat_name in data["features"]:
                        feat_group.create_dataset(
                            feat_name,
                            data=data["features"][feat_name],
                            compression="gzip",
                        )


def create_dataset(config: Optional[DatasetConfig] = None) -> Dict:
    if config is None:
        config = DatasetConfig.from_yaml()

    generator = DatasetGenerator(config)
    data = generator.generate_sequences()

    # Split and save the dataset
    split_data = generator._split_dataset(
        sequences=data["features"],
        graphs=data["graphs"],
        sequence_lengths=data["sequence_lengths"],
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
    print(f"  - Nodes per graph range: [{config.min_n}, {config.max_n}]")
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


def validate_feature_shapes(
    features: Dict[str, np.ndarray], seq_len: int, n_nodes: int
) -> None:
    for name in ["degree", "betweenness", "closeness", "eigenvector"]:
        if name not in features:
            raise ValueError(f"Missing centrality feature: {name}")
        feat = features[name]
        if not isinstance(feat, np.ndarray):
            raise ValueError(f"{name} is not a numpy array")
        if feat.shape != (seq_len, n_nodes):
            raise ValueError(
                f"{name} has invalid shape: {feat.shape}, expected {(seq_len, n_nodes)}"
            )

    for name in ["svd", "lsvd"]:
        if name not in features:
            raise ValueError(f"Missing embedding feature: {name}")
        feat = features[name]
        if not isinstance(feat, np.ndarray):
            raise ValueError(f"{name} is not a numpy array")
        if len(feat.shape) != 3:
            raise ValueError(f"{name} should be 3D array, got shape {feat.shape}")
        if feat.shape[:2] != (seq_len, n_nodes):
            raise ValueError(
                f"{name} has invalid sequence/node dimensions: {feat.shape[:2]}"
            )
        if feat.shape[2] < 2:
            raise ValueError(f"{name} embedding dimension too small: {feat.shape[2]}")


def validate_padded_shapes(
    data: Dict, num_sequences: int, max_seq_len: int, n_nodes: int
) -> None:
    if data["graphs"].shape != (num_sequences, max_seq_len, n_nodes, n_nodes):
        raise ValueError(
            f"Padded graphs have invalid shape: {data['graphs'].shape}, "
            f"expected {(num_sequences, max_seq_len, n_nodes, n_nodes)}"
        )

    for name in ["degree", "betweenness", "closeness", "eigenvector"]:
        feat = data["features"][name]
        expected_shape = (num_sequences, max_seq_len, n_nodes)
        if feat.shape != expected_shape:
            raise ValueError(
                f"Padded {name} has invalid shape: {feat.shape}, expected {expected_shape}"
            )

    for name in ["svd", "lsvd"]:
        feat = data["features"][name]
        if len(feat.shape) != 4:
            raise ValueError(
                f"Padded {name} should be 4D array, got shape {feat.shape}"
            )
        if feat.shape[:3] != (num_sequences, max_seq_len, n_nodes):
            raise ValueError(
                f"Padded {name} has invalid dimensions: {feat.shape}, "
                f"expected first 3 dims to be {(num_sequences, max_seq_len, n_nodes)}"
            )


if __name__ == "__main__":
    main()

# # Generate default dataset using default config files
# python -m synthetic_data.create_dataset

# # Generate custom dataset with specific config files
# python -m synthetic_data.create_dataset \
#     --dataset-config path/to/dataset_config.yaml \
#     --graph-config path/to/graph_config.yaml \
#     --output data/custom \
#     --graph-types BA ER \
#     --sequences 200

# # Generate only BA graphs
# python -m synthetic_data.create_dataset --graph-types BA --sequences 50
