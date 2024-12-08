# synthetic_data/create_dataset.py

"""
Dataset Generator for Graph Forecasting

This module creates synthetic graph datasets with labeled anomalies and global features.
The dataset is designed for a downstream task of forecasting the next m steps of the 6 global features.

Key Steps:
1. Generate synthetic graph sequences (BA, ER, NW) with fixed length (e.g., seq_len=200), so no padding is needed.
2. Compute node-level features (degree, betweenness, closeness, eigenvector, svd, lsvd).
3. Aggregate them to global features by averaging over nodes to get shape (seq_len, 6).
4. Since all sequences are of the same length, we simply stack all sequences.
5. Split into train/val/test sets.
6. Compute normalization parameters (mean, std) from the training set only, to prevent data leakage.
7. Apply the same normalization to train, val, and test splits.
8. Save final normalized data to HDF5 files for easy loading during model training.

Following best practices:
- Normalization is computed only from the training set.
- All sequences have the same length, thus no padding is required.
"""

import logging
import os
import sys
import h5py
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
            seq_len=graph_config_data["common"]["seq_len"],  # fixed sequence length
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
    """Generator for graph sequences dataset with fixed-length sequences (no padding needed)."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def _compute_features(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Extract and aggregate global features for each graph.
        Returns a (seq_len, 6) array with the order:
        [mean_degree, mean_betweenness, mean_eigenvector, mean_closeness, mean_svd, mean_lsvd]
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

    def generate_sequences(self) -> Dict:
        """Generate all sequences of graphs and their features without normalization.
        Since all sequences are the same length (seq_len), we can directly stack them.
        """
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
                    seq_len=self.config.seq_len,
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

        # Since all sequences have the same length, just stack them
        # features: (num_sequences, seq_len, 6)
        feat_cp = cp.stack([cp.asarray(f) for f in data["features"]], axis=0)
        stacked_features = cp.asnumpy(feat_cp)

        # graphs: (num_sequences, seq_len, n_nodes, n_nodes)
        graphs_cp = cp.stack(
            [cp.asarray(np.stack(g, axis=0)) for g in data["graphs"]], axis=0
        )
        stacked_graphs = cp.asnumpy(graphs_cp)

        sequence_lengths = np.array(data["sequence_lengths"])
        # All lengths must be equal to seq_len
        if not all(l == self.config.seq_len for l in sequence_lengths):
            raise ValueError("Not all sequences have the expected fixed length")

        return {
            "features": {"all": stacked_features},
            "graphs": stacked_graphs,
            "sequence_lengths": sequence_lengths,
            "graph_types": data["graph_types"],
            "change_points": data["change_points"],
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

    def _split_dataset(
        self,
        sequences: Dict[str, np.ndarray],
        graphs: np.ndarray,
        sequence_lengths: np.ndarray,
        change_points: List,
        graph_types: List[str],
    ) -> Dict:
        """Split dataset into train/val/test according to split_ratio."""
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


def compute_normalization_params(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization parameters (mean, std) from training set only.
    features shape: (N, T, 6)
    We compute mean and std across all sequences and timesteps in the training set.
    """
    feat_cp = cp.asarray(features)
    mean = cp.mean(feat_cp, axis=(0, 1), keepdims=True)  # shape (1,1,6)
    std = cp.std(feat_cp, axis=(0, 1), keepdims=True)  # shape (1,1,6)
    return cp.asnumpy(mean), cp.asnumpy(std)


def create_dataset(config: Optional[DatasetConfig] = None) -> Dict:
    """
    Create the dataset:
    - Generate sequences of length seq_len (no padding needed).
    - Split into train/val/test.
    - Compute normalization params (mean, std) from TRAIN only.
    - Normalize train/val/test using these parameters.
    - Save the normalized dataset.
    """
    if config is None:
        config = DatasetConfig.from_yaml()

    generator = DatasetGenerator(config)
    # Generate sequences, returns all sequences of equal length, unnormalized
    unnormalized_data = generator.generate_sequences()

    # Split dataset into train/val/test before normalization
    split_data = generator._split_dataset(
        sequences=unnormalized_data["features"],
        graphs=unnormalized_data["graphs"],
        sequence_lengths=unnormalized_data["sequence_lengths"],
        change_points=unnormalized_data["change_points"],
        graph_types=unnormalized_data["graph_types"],
    )

    # Compute normalization params from the training set only
    train_features = split_data["train"]["features"]["all"]  # shape (N_train, T, 6)
    mean, std = compute_normalization_params(train_features)

    # Apply normalization to train, val, and test sets
    # This ensures no data leakage and consistent scaling.
    for split_name in ["train", "val", "test"]:
        feats = split_data[split_name]["features"]["all"]
        normalized = (feats - mean) / (std + 1e-8)
        split_data[split_name]["features"]["all"] = normalized

    # Save normalized data
    generator._save_to_hdf5(split_data)
    return split_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph sequences dataset for forecasting tasks"
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
    print(f"  - Sequence length: {config.seq_len}")
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
