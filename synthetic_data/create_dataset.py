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
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graph_sequences import GraphConfig, GraphType, generate_graph_sequence
from src.changepoint import ChangePointDetector

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig(GraphConfig):
    """Configuration for dataset generation, extends GraphConfig."""

    # Dataset-specific parameters (not in GraphConfig)
    num_sequences: int = 20
    graph_types: List[str] = None
    threshold: float = 30.0
    epsilon: float = 0.8
    window_size: int = 5
    split_ratio: Dict[str, float] = None
    output_dir: str = "dataset"
    graph_params: Dict[str, Dict] = None

    @classmethod
    def from_yaml(
        cls, config_path: str = None, graph_config_path: str = None
    ) -> "DatasetConfig":
        """Create configuration from YAML files"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs/dataset_config.yaml"
        if graph_config_path is None:
            graph_config_path = Path(__file__).parent / "configs/graph_config.yaml"

        # Load dataset config
        with open(config_path, "r") as f:
            dataset_config = yaml.safe_load(f)["dataset"]

        # Load graph config
        with open(graph_config_path, "r") as f:
            graph_config_data = yaml.safe_load(f)

        # Get parameters for each graph type
        graph_params = {
            "ba": graph_config_data["ba"],
            "er": graph_config_data["er"],
            "nw": graph_config_data["nw"],
        }

        # Create base GraphConfig for first graph type
        graph_config = super().from_yaml(
            GraphType[dataset_config["graph_types"][0]], graph_config_path
        )

        return cls(
            # Inherit GraphConfig parameters
            graph_type=graph_config.graph_type,
            min_n=graph_config_data["common"]["min_n"],
            max_n=graph_config_data["common"]["max_n"],
            min_seq_length=graph_config_data["common"]["min_seq_length"],
            max_seq_length=graph_config_data["common"]["max_seq_length"],
            min_segment=graph_config_data["common"]["min_segment"],
            min_changes=graph_config_data["common"]["min_changes"],
            max_changes=graph_config_data["common"]["max_changes"],
            params=graph_config.params,
            # Add dataset-specific parameters
            num_sequences=dataset_config["num_sequences"],
            graph_types=dataset_config["graph_types"],
            threshold=dataset_config["threshold"],
            epsilon=dataset_config["epsilon"],
            window_size=dataset_config["window_size"],
            split_ratio=dataset_config["split_ratio"],
            output_dir=dataset_config["output_dir"],
            graph_params=graph_params,
        )


class DatasetGenerator:
    """Generator for graph anomaly detection datasets."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize dataset generator with configuration."""
        self.config = config or DatasetConfig()

    def _calculate_change_points(
        self, sequence_length: int, num_changes: int
    ) -> List[int]:
        """Randomly calculate change points within a sequence."""
        min_gap = sequence_length // (num_changes + 1)
        change_points = []
        for i in range(1, num_changes + 1):
            cp = i * min_gap + np.random.randint(-min_gap // 4, min_gap // 4)
            cp = max(1, min(sequence_length - 1, cp))
            change_points.append(cp)
        return sorted(change_points)

    def _compute_features(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Extract and combine all features for each graph."""
        try:
            # Initialize detector and get features properly
            detector = ChangePointDetector()
            detector.initialize(graphs)
            centralities = detector.extract_features()

            # Convert to feature matrix
            features = []
            for t in range(len(graphs)):
                graph_features = []
                for name in centralities:
                    # Take mean of each centrality measure to get a single value
                    feature_value = np.mean(centralities[name][t])
                    graph_features.append(feature_value)
                features.append(graph_features)

            return np.array(features)  # Shape: (sequence_length, num_features)

        except Exception as e:
            logger.error(f"Feature computation failed: {str(e)}")
            raise RuntimeError(f"Feature computation failed: {str(e)}")

    def _compute_anomaly_labels(
        self, features: np.ndarray, change_points: List[int]
    ) -> np.ndarray:
        """Compute binary anomaly labels using martingales."""
        try:
            # Compute martingales for each feature dimension
            martingales = []

            for feature_idx in range(features.shape[1]):
                feature_values = features[:, feature_idx]
                normalized_values = (feature_values - np.mean(feature_values)) / np.std(
                    feature_values
                )

                detector = ChangePointDetector()
                feature_2d = normalized_values.reshape(-1, 1)

                result = detector.martingale_test(
                    data=feature_2d,
                    threshold=self.config.threshold,
                    epsilon=self.config.epsilon,
                    reset=True,
                )
                martingales.append(result["martingales"])

            # Create labels: 1 for anomalies (within window of change points), 0 otherwise
            labels = np.zeros(len(features))
            w = self.config.window_size

            for cp in change_points:
                start_idx = max(0, cp - w)
                end_idx = min(len(features), cp + w + 1)
                labels[start_idx:end_idx] = 1

            return labels

        except Exception as e:
            logger.error(f"Label computation failed: {str(e)}")
            raise RuntimeError(f"Label computation failed: {str(e)}")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to have zero mean and unit variance."""
        flat = features.reshape(-1, features.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-8  # Prevent division by zero
        return (features - mean) / std

    def _split_dataset(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        change_points: List[List[int]],
        martingales: List[Dict],
        sequence_lengths: np.ndarray,
    ) -> Dict:
        """Split dataset into train, validation, and test sets."""
        num_sequences = len(sequences)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)

        train_end = int(self.config.split_ratio["train"] * num_sequences)
        val_end = train_end + int(self.config.split_ratio["val"] * num_sequences)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return {
            "train": {
                "sequences": sequences[train_idx],
                "labels": labels[train_idx],
                "change_points": [change_points[i] for i in train_idx],
                "martingales": [martingales[i] for i in train_idx],
                "sequence_lengths": sequence_lengths[train_idx],
            },
            "val": {
                "sequences": sequences[val_idx],
                "labels": labels[val_idx],
                "change_points": [change_points[i] for i in val_idx],
                "martingales": [martingales[i] for i in val_idx],
                "sequence_lengths": sequence_lengths[val_idx],
            },
            "test": {
                "sequences": sequences[test_idx],
                "labels": labels[test_idx],
                "change_points": [change_points[i] for i in test_idx],
                "martingales": [martingales[i] for i in test_idx],
                "sequence_lengths": sequence_lengths[test_idx],
            },
        }

    def _save_to_hdf5(self, data_split: Dict):
        """Save dataset splits to HDF5 files."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        for split_name, data in data_split.items():
            split_dir = os.path.join(self.config.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            # Pad change points to max length
            max_changes = max(len(cp) for cp in data["change_points"])
            padded_change_points = np.array(
                [cp + [-1] * (max_changes - len(cp)) for cp in data["change_points"]]
            )

            with h5py.File(os.path.join(split_dir, "data.h5"), "w") as hf:
                # Create main group for sequence data
                seq_group = hf.create_group("sequences")
                seq_group.create_dataset(
                    "features", data=data["sequences"], compression="gzip"
                )
                seq_group.create_dataset(
                    "labels", data=data["labels"], compression="gzip"
                )
                seq_group.create_dataset(
                    "lengths", data=data["sequence_lengths"], compression="gzip"
                )

                # Create group for change points
                cp_group = hf.create_group("change_points")
                cp_group.create_dataset(
                    "points", data=padded_change_points, compression="gzip"
                )
                cp_group.create_dataset(
                    "lengths",
                    data=[len(cp) for cp in data["change_points"]],
                    compression="gzip",
                )

                # Create group for martingales
                mart_group = hf.create_group("martingales")

                # Store martingales for each sequence
                for seq_idx, martingales in enumerate(data["martingales"]):
                    seq_group = mart_group.create_group(f"sequence_{seq_idx}")

                    # Store reset and cumulative martingales
                    for mart_type in ["reset", "cumulative"]:
                        type_group = seq_group.create_group(mart_type)

                        # Store martingales for each feature
                        for feat_name, mart_data in martingales[mart_type].items():
                            mart_values = np.array(
                                mart_data["martingales"], dtype=np.float64
                            )
                            type_group.create_dataset(
                                feat_name, data=mart_values, compression="gzip"
                            )

            logger.info(
                f"Saved {split_name} data: {data['sequences'].shape} sequences, "
                f"{data['labels'].shape} labels, {len(data['change_points'])} change points"
            )

    def _compute_martingales(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Compute both reset and cumulative martingales for the graph sequence.
        Consistent with visualization scripts implementation.

        Parameters:
            graphs (List[np.ndarray]): List of adjacency matrices

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with 'reset' and 'cumulative' martingales
        """
        detector = ChangePointDetector()
        detector.initialize(graphs)
        centralities = detector.extract_features()

        martingales_reset = {}
        martingales_cumulative = {}

        for name, values in centralities.items():
            # Normalize values
            values_array = np.array(values)
            normalized_values = (values_array - np.mean(values_array, axis=0)) / np.std(
                values_array, axis=0
            )

            # Compute martingales
            martingales_reset[name] = detector.martingale_test(
                data=normalized_values,
                threshold=self.config.threshold,
                epsilon=self.config.epsilon,
                reset=True,
            )

            cumulative_result = detector.martingale_test(
                data=normalized_values,
                threshold=self.config.threshold,
                epsilon=self.config.epsilon,
                reset=False,
            )

            # Convert to cumulative sum
            cumulative_values = np.array(cumulative_result["martingales"])
            cumulative_result["martingales"] = np.cumsum(cumulative_values)
            martingales_cumulative[name] = cumulative_result

        return {"reset": martingales_reset, "cumulative": martingales_cumulative}

    def generate_sequences(self) -> Dict:
        """Generate sequences for all specified graph types."""
        data = {
            "sequences": [],
            "labels": [],
            "change_points": [],
            "graph_types": [],
            "martingales": [],
            "sequence_lengths": [],
        }

        # Main progress bar for graph types
        for graph_type_str in tqdm(
            self.config.graph_types, desc="Graph Types", position=0
        ):
            try:
                graph_type = GraphType[graph_type_str.upper()]
                logger.info(
                    f"Generating {self.config.num_sequences} sequences for {graph_type.value}"
                )

                # Get parameters for this graph type
                graph_params = self.config.graph_params[graph_type_str.lower()]
                if not graph_params:
                    logger.error(
                        f"No parameters found for graph type: {graph_type_str}"
                    )
                    continue

                # Create a new GraphConfig for each graph type
                graph_config = GraphConfig(
                    graph_type=graph_type,
                    min_n=self.config.min_n,
                    max_n=self.config.max_n,
                    min_seq_length=self.config.min_seq_length,
                    max_seq_length=self.config.max_seq_length,
                    min_segment=self.config.min_segment,
                    min_changes=self.config.min_changes,
                    max_changes=self.config.max_changes,
                    params=graph_params,
                )

                # Nested progress bar for sequences of each type
                for _ in tqdm(
                    range(self.config.num_sequences),
                    desc=f"Sequences ({graph_type.value})",
                    position=1,
                    leave=False,
                ):
                    result = generate_graph_sequence(graph_config)

                    # Compute features and martingales directly from graphs
                    features = self._compute_features(result["graphs"])
                    martingales = self._compute_martingales(result["graphs"])

                    # Compute labels using martingale detection points
                    labels = self._compute_anomaly_labels(
                        features, result["change_points"]
                    )

                    data["sequences"].append(features)
                    data["labels"].append(labels)
                    data["change_points"].append(result["change_points"])
                    data["graph_types"].append(graph_type.value)
                    data["martingales"].append(martingales)
                    data["sequence_lengths"].append(len(result["graphs"]))

            except KeyError:
                logger.error(f"Invalid graph type: {graph_type_str}")
                continue

        # Clear the progress bars
        print("\n")

        # Convert to numpy arrays with padding
        max_length = max(data["sequence_lengths"])
        num_features = data["sequences"][0].shape[1]  # Number of features per timestep

        # Pad sequences and labels
        padded_sequences = np.zeros((len(data["sequences"]), max_length, num_features))
        padded_labels = np.zeros((len(data["labels"]), max_length))

        for i, (seq, lab, length) in enumerate(
            zip(data["sequences"], data["labels"], data["sequence_lengths"])
        ):
            padded_sequences[i, :length] = seq
            padded_labels[i, :length] = lab

        data["sequences"] = padded_sequences
        data["labels"] = padded_labels
        data["sequence_lengths"] = np.array(data["sequence_lengths"])

        return data


def create_dataset(config: Optional[DatasetConfig] = None) -> Dict:
    """Create a new dataset with the given configuration."""
    if config is None:
        config = DatasetConfig.from_yaml()

    generator = DatasetGenerator(config)
    data = generator.generate_sequences()

    # Split and save the dataset
    split_data = generator._split_dataset(
        data["sequences"],
        data["labels"],
        data["change_points"],
        data["martingales"],
        data["sequence_lengths"],
    )

    generator._save_to_hdf5(split_data)
    return split_data


def main():
    """Main entry point for dataset creation."""
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

    # Load configuration from YAML files
    config = DatasetConfig.from_yaml(
        config_path=args.dataset_config, graph_config_path=args.graph_config
    )

    # Override with command line arguments if provided
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
    print(f"  - Min changes: {config.min_changes}")
    print(f"  - Max changes: {config.max_changes}")
    print(f"  - Min segment length: {config.min_segment}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"\nGraph Parameters:")
    for graph_type in config.graph_types:
        print(f"  - {graph_type}: {config.graph_params[graph_type.lower()]}")
    print(f"\nDataset Parameters:")
    print(f"  - Threshold: {config.threshold}")
    print(f"  - Epsilon: {config.epsilon}")
    print(f"  - Window size: {config.window_size}")
    print(f"  - Split ratio: {config.split_ratio}")

    # Create and save dataset
    data = create_dataset(config)

    # Print dataset statistics
    for split_name, split_data in data.items():
        sequences = split_data["sequences"]
        labels = split_data["labels"]
        print(f"\n{split_name.capitalize()} set:")
        print(f"  - Sequences shape: {sequences.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Anomaly ratio: {np.mean(labels):.2%}")


if __name__ == "__main__":
    main()


# Usage examples in docstring:
"""
Usage Examples:
-------------------------------------------------
# Generate default dataset using default config files
python -m synthetic_data.create_dataset

# Generate custom dataset with specific config files
python -m synthetic_data.create_dataset \
    --dataset-config path/to/dataset_config.yaml \
    --graph-config path/to/graph_config.yaml \
    --output data/custom \
    --graph-types BA ER \
    --sequences 200

# Generate only BA graphs
python -m synthetic_data.create_dataset --graph-types BA --sequences 50
"""
