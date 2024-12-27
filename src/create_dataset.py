# src/create_dataset.py

"""
Dataset Generator for Graph Sequences with Link Prediction Features

This script creates a dataset of graph sequences optimized for link prediction:
- Generates sequences using the Stochastic Block Model (SBM)
- Extracts comprehensive link prediction features
- Handles both static and temporal features
- Saves the dataset in compressed HDF5 format
"""

import logging
import os
from typing import Dict, List
import yaml
import h5py
import numpy as np
import networkx as nx
from tqdm import tqdm

from graph.generator import GraphGenerator
from graph.features import (
    compute_link_prediction_features,
    compute_temporal_link_features,
    normalize_features,
)
from graph.params import SBMParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def generate_sbm_graph(n: int, sizes: List[int], p_in: float, p_out: float) -> nx.Graph:
    """Generate a Stochastic Block Model graph.

    Args:
        n: Total number of nodes
        sizes: List of block sizes (must sum to n)
        p_in: Probability of intra-block edges
        p_out: Probability of inter-block edges

    Returns:
        NetworkX graph with community structure
    """
    # Validate inputs
    if sum(sizes) != n:
        raise ValueError(f"Block sizes {sizes} must sum to n={n}")
    if not (0 <= p_in <= 1 and 0 <= p_out <= 1):
        raise ValueError(
            f"Probabilities must be in [0,1], got p_in={p_in}, p_out={p_out}"
        )

    # Create probability matrix
    num_blocks = len(sizes)
    p_matrix = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(p_matrix, p_in)

    try:
        return nx.stochastic_block_model(sizes, p_matrix)
    except Exception as e:
        raise RuntimeError(f"Failed to generate SBM graph: {str(e)}") from e


class DatasetGenerator:
    """Handles the generation of graph sequences and their features for link prediction"""

    def __init__(self, config_path: str):
        """Initialize with configuration file path"""
        self.config = self._load_config(config_path)
        self.graph_generator = GraphGenerator()

        # Register SBM model
        self._register_sbm_model()

    def _register_sbm_model(self):
        """Register the SBM model with the graph generator."""

        def sbm_metadata(params: Dict) -> Dict:
            """Generate metadata for SBM graph."""
            n_blocks = params["num_blocks"]
            block_sizes = [params["min_block_size"]] * n_blocks
            remaining_nodes = params["n"] - sum(block_sizes)

            # Distribute remaining nodes evenly
            for i in range(remaining_nodes):
                block_sizes[i % n_blocks] += 1

            # Generate community labels
            community_labels = []
            for block_idx, size in enumerate(block_sizes):
                community_labels.extend([block_idx] * size)

            # Convert params to a serializable format
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    serializable_params[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_params[key] = value.tolist()
                else:
                    serializable_params[key] = str(value)

            return {
                "community_labels": np.array(community_labels),
                "block_sizes": block_sizes,
                "params": serializable_params,
                "num_blocks": n_blocks,
                "edge_probabilities": {
                    "intra_block": params["initial_intra_prob"],
                    "inter_block": params["initial_inter_prob"],
                },
            }

        def sbm_param_mutation(params: Dict) -> Dict:
            """Custom mutation for SBM parameters."""
            new_params = params.copy()
            sbm_config = self.config["graph_params"]["sbm"]

            # Mutate intra-block probability
            new_params["initial_intra_prob"] = np.random.uniform(
                sbm_config["change_ranges"]["intra"]["min"],
                sbm_config["change_ranges"]["intra"]["max"],
            )

            # Mutate inter-block probability
            new_params["initial_inter_prob"] = np.random.uniform(
                sbm_config["change_ranges"]["inter"]["min"],
                sbm_config["change_ranges"]["inter"]["max"],
            )

            return new_params

        def sbm_generator(
            n: int,
            num_blocks: int,
            min_block_size: int,
            initial_intra_prob: float,
            initial_inter_prob: float,
            **kwargs,
        ) -> nx.Graph:
            """Generate SBM graph with filtered parameters."""
            # Calculate block sizes
            block_sizes = [min_block_size] * num_blocks
            remaining_nodes = n - sum(block_sizes)

            # Distribute remaining nodes evenly
            for i in range(remaining_nodes):
                block_sizes[i % num_blocks] += 1

            return generate_sbm_graph(
                n=n,
                sizes=block_sizes,
                p_in=initial_intra_prob,
                p_out=initial_inter_prob,
            )

        self.graph_generator.register_model(
            name="SBM",
            generator_func=sbm_generator,
            param_class=SBMParams,
            param_mutation_func=sbm_param_mutation,
            metadata_func=sbm_metadata,
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["dataset"]

        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration parameters"""
        required_fields = {
            "num_sequences",
            "seq_len",
            "min_segment",
            "min_changes",
            "max_changes",
            "graph_params",
            "features",
            "link_prediction",
            "split_ratio",
            "output",
        }

        if not all(field in config for field in required_fields):
            missing = required_fields - set(config.keys())
            raise ValueError(f"Missing required configuration fields: {missing}")

        if sum(config["split_ratio"].values()) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

    def _create_graph_params(self) -> SBMParams:
        """Create graph generation parameters from config"""
        sbm_config = self.config["graph_params"]["sbm"]

        # Ensure minimum segment length is reasonable
        min_segment = max(
            3, min(self.config["min_segment"], self.config["seq_len"] // 6)
        )
        max_changes = min(
            self.config["max_changes"],
            (self.config["seq_len"] - min_segment) // min_segment - 1,
        )
        min_changes = min(self.config["min_changes"], max_changes)

        return SBMParams(
            n=self.config["graph_params"]["nodes"],
            seq_len=self.config["seq_len"],
            min_segment=min_segment,  # Use adjusted min_segment
            min_changes=min_changes,  # Use adjusted min_changes
            max_changes=max_changes,  # Use adjusted max_changes
            num_blocks=sbm_config["num_blocks"],
            min_block_size=sbm_config["block_size_range"]["min"],
            max_block_size=sbm_config["block_size_range"]["max"],
            initial_intra_prob=sbm_config["edge_probabilities"]["intra_block"],
            initial_inter_prob=sbm_config["edge_probabilities"]["inter_block"],
            min_intra_prob=sbm_config["change_ranges"]["intra"]["min"],
            max_intra_prob=sbm_config["change_ranges"]["intra"]["max"],
            min_inter_prob=sbm_config["change_ranges"]["inter"]["min"],
            max_inter_prob=sbm_config["change_ranges"]["inter"]["max"],
        )

    def _extract_features(
        self, graphs: List[np.ndarray], metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """Extract all specified features from graph sequence"""
        features = {}
        feature_types = self.config["features"]["types"]

        # Extract static link prediction features
        static_feature_types = [
            ft
            for ft in feature_types
            if ft not in ["link_history", "temporal_stability", "cn_history"]
        ]

        if static_feature_types:
            static_features = compute_link_prediction_features(
                graphs,
                feature_types=static_feature_types,
                community_labels=(
                    metadata.get("community_labels", None)
                    if isinstance(metadata, dict)
                    else None
                ),
            )
            features.update(static_features)

        # Extract temporal features if specified
        temporal_feature_types = [
            ft
            for ft in feature_types
            if ft in ["link_history", "temporal_stability", "cn_history"]
        ]
        if temporal_feature_types:
            temporal_features = compute_temporal_link_features(
                graphs, window_size=3  # Use a reasonable default window size
            )
            features.update(temporal_features)

        # Normalize features if specified
        if self.config["features"]["normalize"]:
            features = normalize_features(features, method="standard")

        return features

    def generate_sequence(self) -> Dict:
        """Generate a single sequence with features"""
        # Generate graph sequence
        params = self._create_graph_params()
        sequence = self.graph_generator.generate_sequence("SBM", params)

        # Extract features
        features = self._extract_features(sequence["graphs"], sequence["metadata"])

        return {
            "adjacency": sequence["graphs"],
            "features": features,
            "change_points": sequence["change_points"],
            "metadata": sequence["metadata"],
            "params": params,
        }

    def create_dataset(self) -> None:
        """Create and save the full dataset"""
        logger.info("Starting dataset generation...")

        # Generate sequences
        sequences = []
        for i in tqdm(range(self.config["num_sequences"]), desc="Generating sequences"):
            try:
                sequence = self.generate_sequence()
                sequences.append(sequence)
            except Exception as e:
                logger.error(f"Failed to generate sequence {i}: {str(e)}")
                continue

        # Split sequences
        n_sequences = len(sequences)
        n_train = int(n_sequences * self.config["split_ratio"]["train"])
        n_val = int(n_sequences * self.config["split_ratio"]["val"])

        splits = {
            "train": sequences[:n_train],
            "val": sequences[n_train : n_train + n_val],
            "test": sequences[n_train + n_val :],
        }

        # Save dataset
        self._save_dataset(splits)

    def _save_dataset(self, splits: Dict[str, List[Dict]]) -> None:
        """Save the dataset in HDF5 format"""
        output_dir = self.config["output"]["dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "link_prediction_dataset.h5")

        logger.info(f"Saving dataset to {output_path}")

        with h5py.File(output_path, "w") as f:
            # Save each split
            for split_name, sequences in splits.items():
                split_group = f.create_group(split_name)

                for i, seq in enumerate(sequences):
                    seq_group = split_group.create_group(f"sequence_{i}")

                    # Save adjacency matrices
                    seq_group.create_dataset(
                        "adjacency",
                        data=seq["adjacency"],
                        compression=(
                            "gzip" if self.config["output"]["compression"] else None
                        ),
                    )

                    # Save features
                    feat_group = seq_group.create_group("features")
                    for feat_name, feat_data in seq["features"].items():
                        feat_group.create_dataset(
                            feat_name,
                            data=feat_data,
                            compression=(
                                "gzip" if self.config["output"]["compression"] else None
                            ),
                        )

                    # Save change points
                    seq_group.create_dataset("change_points", data=seq["change_points"])

                    # Save metadata
                    meta_group = seq_group.create_group("metadata")

                    # Use first metadata entry for community structure
                    first_meta = (
                        seq["metadata"][0]
                        if isinstance(seq["metadata"], list)
                        else seq["metadata"]
                    )

                    # Save community labels directly in metadata group
                    if "community_labels" in first_meta:
                        meta_group.create_dataset(
                            "community_labels", data=first_meta["community_labels"]
                        )

                    # Save block sizes
                    if "block_sizes" in first_meta:
                        meta_group.create_dataset(
                            "block_sizes", data=first_meta["block_sizes"]
                        )

                    # Save edge probabilities
                    if "edge_probabilities" in first_meta:
                        prob_group = meta_group.create_group("edge_probabilities")
                        for k, v in first_meta["edge_probabilities"].items():
                            prob_group.attrs[k] = v

                    # Save other parameters
                    if "params" in first_meta:
                        param_group = meta_group.create_group("params")
                        for k, v in first_meta["params"].items():
                            if isinstance(v, (list, np.ndarray)):
                                param_group.create_dataset(k, data=v)
                            else:
                                param_group.attrs[k] = v

                    # Save all metadata entries in a separate group if it's a list
                    if isinstance(seq["metadata"], list):
                        all_meta_group = meta_group.create_group("all_segments")
                        for i, meta in enumerate(seq["metadata"]):
                            segment_group = all_meta_group.create_group(f"segment_{i}")
                            for k, v in meta.items():
                                if isinstance(v, (np.ndarray, list)):
                                    segment_group.create_dataset(k, data=v)
                                else:
                                    segment_group.attrs[k] = str(v)

            # Save configuration
            config_group = f.create_group("config")
            for key, value in self.config.items():
                if isinstance(value, dict):
                    subgroup = config_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            subsubgroup = subgroup.create_group(subkey)
                            for subsubkey, subsubvalue in subvalue.items():
                                subsubgroup.attrs[subsubkey] = str(subsubvalue)
                        else:
                            subgroup.attrs[subkey] = str(subvalue)
                else:
                    config_group.attrs[key] = str(value)

        logger.info("Dataset saved successfully")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate link prediction dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset configuration file",
    )
    args = parser.parse_args()

    try:
        generator = DatasetGenerator(args.config)
        generator.create_dataset()
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
