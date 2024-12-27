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
import json

from graph.generator import GraphGenerator
from graph.features import (
    compute_link_prediction_features,
    compute_temporal_link_features,
    normalize_features,
    _get_positive_edges,
    _generate_negative_samples,
    _split_edges,
)
from graph.params import SBMParams

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _generate_sequence_worker(args: tuple) -> Dict:
    """Worker function for parallel sequence generation.

    Args:
        args: Tuple containing (config, idx)
            config: Configuration dictionary
            idx: Sequence index (unused, but needed for pool.map)

    Returns:
        Generated sequence or None if generation failed
    """
    try:
        config, _ = args
        generator = DatasetGenerator(config)
        return generator.generate_sequence()
    except Exception as e:
        logger.error(f"Failed to generate sequence: {str(e)}")
        return None


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
        if isinstance(config_path, str):
            self.config = self._load_config(config_path)
        else:
            self.config = config_path  # For multiprocessing case
        self.graph_generator = GraphGenerator()
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

    def _extract_features(self, graphs: List[np.ndarray], metadata: Dict) -> Dict:
        """Extract link prediction features from graph sequence.

        Args:
            graphs: List of adjacency matrices
            metadata: Graph metadata including community labels

        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            # Handle both list and dict metadata formats
            if isinstance(metadata, list):
                # Use first metadata entry for community labels
                community_labels = (
                    metadata[0].get("community_labels") if metadata else None
                )
            else:
                community_labels = metadata.get("community_labels")

            feature_config = self.config["features"]
            feature_types = feature_config["types"]

            # Extract static link prediction features
            static_features = compute_link_prediction_features(
                graphs=graphs,
                feature_types=[
                    ft for ft in feature_types if not ft.startswith("temporal_")
                ],
                community_labels=community_labels,
            )
            features.update(static_features)

            # Extract temporal features with configured window size
            temporal_features = compute_temporal_link_features(
                graphs=graphs, window_size=feature_config.get("window_size", 3)
            )
            features.update(temporal_features)

            # Normalize features if requested
            if feature_config.get("normalize", True):
                features = normalize_features(
                    features,
                    method=feature_config.get("normalization_method", "standard"),
                )

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}") from e

    def _generate_link_prediction_data(
        self, graphs: List[np.ndarray], metadata: Dict
    ) -> Dict:
        """Generate link prediction data including train/val/test splits.

        Args:
            graphs: List of adjacency matrices
            metadata: Graph metadata including community labels

        Returns:
            Dictionary containing edge splits and labels
        """
        try:
            logger.info("Starting link prediction data generation...")
            lp_config = self.config["link_prediction"]

            # Debug metadata
            logger.info(f"Metadata type: {type(metadata)}")
            logger.info(f"Metadata content: {metadata}")

            # Handle both list and dict metadata formats
            if isinstance(metadata, list):
                logger.info("Metadata is a list, using first entry")
                # Use first metadata entry for community labels
                community_labels = (
                    metadata[0].get("community_labels") if metadata else None
                )
            else:
                logger.info("Metadata is a dictionary")
                community_labels = metadata.get("community_labels")

            logger.info(
                f"Community labels shape: {community_labels.shape if community_labels is not None else None}"
            )

            # Get positive edges while preserving constraints
            logger.info("Getting positive edges...")
            positive_edges = _get_positive_edges(
                graphs=graphs,
                min_degree=lp_config.get("min_positive_degree", 0),
                preserve_connectivity=lp_config.get("preserve_connectivity", True),
            )
            logger.info(f"Positive edges shape: {positive_edges.shape}")
            logger.info(f"Positive edges type: {type(positive_edges)}")
            logger.info(f"First few positive edges: {positive_edges[:5]}")

            # Convert edges to tuples to ensure they're hashable
            logger.info("Converting positive edges to tuples...")
            try:
                positive_edges = [tuple(map(int, edge)) for edge in positive_edges]
                positive_edges = np.array(positive_edges)
                logger.info("Successfully converted positive edges to tuples")
            except Exception as e:
                logger.error(f"Failed to convert positive edges to tuples: {str(e)}")
                raise

            # Generate negative samples
            logger.info("Generating negative samples...")
            negative_edges = _generate_negative_samples(
                graphs=graphs,
                positive_edges=positive_edges,
                ratio=lp_config.get("negative_sampling_ratio", 1.0),
                strategy=lp_config.get("sampling_strategy", "random"),
                hard_negative=lp_config.get("hard_negative", False),
                community_labels=community_labels,
            )
            logger.info(f"Negative edges shape: {negative_edges.shape}")
            logger.info(f"Negative edges type: {type(negative_edges)}")
            logger.info(f"First few negative edges: {negative_edges[:5]}")

            # Convert negative edges to tuples as well
            logger.info("Converting negative edges to tuples...")
            try:
                negative_edges = [tuple(map(int, edge)) for edge in negative_edges]
                negative_edges = np.array(negative_edges)
                logger.info("Successfully converted negative edges to tuples")
            except Exception as e:
                logger.error(f"Failed to convert negative edges to tuples: {str(e)}")
                raise

            # Split edges into train/val/test sets
            logger.info("Splitting edges...")
            edge_splits = _split_edges(
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                method=lp_config.get("edge_split_method", "random"),
                community_labels=(
                    community_labels
                    if lp_config.get("edge_split_method") == "community_based"
                    else None
                ),
            )

            # Convert all edges in splits back to arrays for saving
            logger.info("Processing splits for saving...")
            processed_splits = {}
            for split_name, split_data in edge_splits.items():
                logger.info(f"Processing {split_name} split...")
                processed_splits[split_name] = {
                    "positive": np.array(
                        [list(edge) for edge in split_data["positive"]]
                    ),
                    "negative": np.array(
                        [list(edge) for edge in split_data["negative"]]
                    ),
                }

            result = {
                "splits": processed_splits,
                "metadata": {
                    "num_positive": len(positive_edges),
                    "num_negative": len(negative_edges),
                    "sampling_strategy": lp_config.get("sampling_strategy"),
                    "split_method": lp_config.get("edge_split_method"),
                },
            }
            logger.info("Link prediction data generation completed successfully")
            return result

        except Exception as e:
            logger.error(f"Link prediction data generation failed: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}", exc_info=True)
            raise RuntimeError(
                f"Link prediction data generation failed: {str(e)}"
            ) from e

    def generate_sequence(self) -> Dict:
        """Generate a single sequence with features"""
        # Generate graph sequence
        params = self._create_graph_params()
        sequence = self.graph_generator.generate_sequence("SBM", params)

        # Extract features
        features = self._extract_features(sequence["graphs"], sequence["metadata"])

        # Generate link prediction data
        link_pred_data = self._generate_link_prediction_data(
            sequence["graphs"], sequence["metadata"]
        )

        return {
            "graphs": sequence["graphs"],
            "features": features,
            "link_prediction": link_pred_data,
            "metadata": sequence["metadata"],
            "change_points": sequence["change_points"],
        }

    def create_dataset(self) -> None:
        """Create and save the full dataset"""
        logger.info("Starting dataset generation...")

        # Initialize multiprocessing
        import multiprocessing as mp

        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
        logger.info(f"Using {num_cores} CPU cores for parallel processing")

        # Create arguments for each sequence
        args = [(self.config, i) for i in range(self.config["num_sequences"])]

        # Create a pool of workers
        with mp.Pool(num_cores) as pool:
            # Generate sequences in parallel with progress bar
            from tqdm import tqdm

            sequences = []
            with tqdm(
                total=self.config["num_sequences"], desc="Generating sequences"
            ) as pbar:
                for result in pool.imap_unordered(_generate_sequence_worker, args):
                    if result is not None:
                        sequences.append(result)
                    pbar.update()

        # Verify we have enough sequences
        if not sequences:
            raise RuntimeError("No sequences were generated successfully")

        if len(sequences) < self.config["num_sequences"]:
            logger.warning(
                f"Only generated {len(sequences)}/{self.config['num_sequences']} sequences successfully"
            )

        # Split sequences into train/val/test
        n_sequences = len(sequences)
        n_train = int(n_sequences * self.config["split_ratio"]["train"])
        n_val = int(n_sequences * self.config["split_ratio"]["val"])

        splits = {
            "train": sequences[:n_train],
            "val": sequences[n_train : n_train + n_val],
            "test": sequences[n_train + n_val :],
        }

        # Save dataset
        output_path = os.path.join(
            self.config["output"]["dir"], "link_prediction_dataset.h5"
        )
        self.save_dataset(sequences, output_path)

        logger.info(f"Dataset generation complete. Total sequences: {len(sequences)}")
        logger.info(
            f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
        )
        logger.info(f"Dataset saved to {output_path}")

    def save_dataset(self, sequences: List[Dict], output_path: str) -> None:
        """Save the generated dataset to disk.

        Args:
            sequences: List of dictionaries containing graph sequences and features
            output_path: Path to save the dataset
        """
        try:
            # Check if we have any sequences to save
            if not sequences:
                raise ValueError("No sequences were generated successfully")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with h5py.File(output_path, "w") as f:
                # Create groups for different components
                seq_group = f.create_group("sequences")
                feat_group = f.create_group("features")
                lp_group = f.create_group("link_prediction")
                meta_group = f.create_group("metadata")

                for idx, seq_data in enumerate(sequences):
                    seq_name = f"sequence_{idx}"

                    # Save graph sequence
                    seq_subgroup = seq_group.create_group(seq_name)
                    for t, adj_matrix in enumerate(seq_data["graphs"]):
                        seq_subgroup.create_dataset(
                            f"graph_{t}",
                            data=adj_matrix,
                            compression=(
                                "gzip" if self.config["output"]["compression"] else None
                            ),
                        )

                    # Save features if requested
                    if self.config["output"]["save_features"]:
                        feat_subgroup = feat_group.create_group(seq_name)
                        for feat_name, feat_matrix in seq_data["features"].items():
                            feat_subgroup.create_dataset(
                                feat_name,
                                data=feat_matrix,
                                compression=(
                                    "gzip"
                                    if self.config["output"]["compression"]
                                    else None
                                ),
                            )

                    # Save link prediction data
                    lp_subgroup = lp_group.create_group(seq_name)
                    for split_name, split_data in seq_data["link_prediction"][
                        "splits"
                    ].items():
                        split_group = lp_subgroup.create_group(split_name)
                        for edge_type, edges in split_data.items():
                            split_group.create_dataset(edge_type, data=edges)

                    # Save metadata
                    meta_subgroup = meta_group.create_group(seq_name)
                    # Handle both list and dict metadata formats
                    if isinstance(seq_data["metadata"], list):
                        metadata = seq_data["metadata"][0]  # Use first metadata entry
                    else:
                        metadata = seq_data["metadata"]

                    # Save metadata with proper type handling
                    for key, value in metadata.items():
                        try:
                            if isinstance(value, (np.ndarray, list)):
                                meta_subgroup.create_dataset(key, data=value)
                            elif isinstance(value, (int, float, bool)):
                                meta_subgroup.attrs[key] = value
                            elif isinstance(value, str):
                                meta_subgroup.attrs[key] = value
                            elif isinstance(value, dict):
                                # Convert dict to JSON string
                                meta_subgroup.attrs[key] = json.dumps(value)
                            else:
                                # Convert other types to string
                                meta_subgroup.attrs[key] = str(value)
                        except Exception as e:
                            logger.warning(f"Failed to save metadata {key}: {str(e)}")
                            meta_subgroup.attrs[key] = str(value)

                # Save global metadata
                meta_group.attrs["num_sequences"] = len(sequences)
                meta_group.attrs["sequence_length"] = len(sequences[0]["graphs"])
                meta_group.attrs["num_nodes"] = sequences[0]["graphs"][0].shape[0]

                # Save configuration as JSON string
                meta_group.attrs["config"] = json.dumps(self.config)

            logger.info(f"Dataset saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise RuntimeError(f"Failed to save dataset: {str(e)}") from e


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
