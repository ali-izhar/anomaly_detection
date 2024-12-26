# synthetic_data/create_dataset.py

"""
Dataset Generator for Graph Sequences

Creates a dataset of graph sequences with their features:
- Generates sequences using various graph models (BA, ER, NW, SBM)
- Extracts link prediction features optimized for sparse graphs
- Saves adjacency matrices and features in compressed format
"""

import sys
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import yaml
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graphs import GraphType, generate_graph_sequence
from src.graph.params import BAParams, ERParams, NWParams, SBMParams
from extract_features import extract_features, LINK_FEATURES

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/dataset_generation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""

    num_sequences: int
    graph_types: List[str]
    features: Dict
    link_prediction: Dict
    split_ratio: Dict[str, float]
    output_dir: str
    save_format: str
    compression: bool
    graph_config: Optional[Dict] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "DatasetConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["dataset"]

            # Remove graph_config if it exists and create GraphConfig
            graph_config = config.pop("graph_config", None)
            dataset_config = cls(**config)
            dataset_config.graph_config = graph_config
            return dataset_config

    def validate(self):
        """Validate configuration parameters"""
        if self.num_sequences <= 0:
            raise ValueError("Number of sequences must be positive")

        # Validate graph types
        valid_graph_types = {"BA", "ER", "NW", "SBM"}
        invalid_types = set(self.graph_types) - valid_graph_types
        if invalid_types:
            raise ValueError(f"Unsupported graph types: {invalid_types}")

        if sum(self.split_ratio.values()) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

        if self.graph_config:
            # Validate common graph configuration
            required_fields = {"nodes"}
            if not required_fields.issubset(self.graph_config.keys()):
                raise ValueError(
                    f"Graph config missing required fields: {required_fields}"
                )

            # Validate BA-specific parameters if BA is in graph_types
            if "BA" in self.graph_types:
                ba_fields = {"edges", "min_edges", "max_edges"}
                if not ba_fields.issubset(self.graph_config.keys()):
                    raise ValueError(
                        f"Graph config missing BA-specific fields: {ba_fields}"
                    )
                if self.graph_config["min_edges"] > self.graph_config["max_edges"]:
                    raise ValueError("min_edges cannot be greater than max_edges")
                if self.graph_config["edges"] < 1:
                    raise ValueError("Initial edges must be positive")

            # Validate ER-specific parameters if ER is in graph_types
            if "ER" in self.graph_types:
                er_fields = {"initial_prob", "min_prob", "max_prob"}
                if not er_fields.issubset(self.graph_config.keys()):
                    raise ValueError(
                        f"Graph config missing ER-specific fields: {er_fields}"
                    )
                if not (0 <= self.graph_config["initial_prob"] <= 1):
                    raise ValueError("Initial probability must be between 0 and 1")

            # Validate NW-specific parameters if NW is in graph_types
            if "NW" in self.graph_types:
                nw_fields = {"k_nearest", "initial_prob", "min_prob", "max_prob"}
                if not nw_fields.issubset(self.graph_config.keys()):
                    raise ValueError(
                        f"Graph config missing NW-specific fields: {nw_fields}"
                    )
                if self.graph_config["k_nearest"] < 1:
                    raise ValueError("k_nearest must be positive")

            # Validate SBM-specific parameters if SBM is in graph_types
            if "SBM" in self.graph_types:
                sbm_fields = {
                    "num_blocks",
                    "min_block_size",
                    "max_block_size",
                    "initial_intra_prob",
                    "initial_inter_prob",
                    "min_intra_prob",
                    "max_intra_prob",
                    "min_inter_prob",
                    "max_inter_prob",
                }
                if not sbm_fields.issubset(self.graph_config.keys()):
                    raise ValueError(
                        f"Graph config missing SBM-specific fields: {sbm_fields}"
                    )
                if (
                    self.graph_config["min_block_size"]
                    > self.graph_config["max_block_size"]
                ):
                    raise ValueError(
                        "min_block_size cannot be greater than max_block_size"
                    )
                if not (0 <= self.graph_config["initial_intra_prob"] <= 1):
                    raise ValueError("initial_intra_prob must be between 0 and 1")
                if not (0 <= self.graph_config["initial_inter_prob"] <= 1):
                    raise ValueError("initial_inter_prob must be between 0 and 1")


def generate_sequence_with_features(
    args: Tuple[GraphType, Union[BAParams, ERParams, NWParams, SBMParams], Dict]
) -> Dict:
    """Generate a single sequence with optimized link prediction features"""
    graph_type, params, feature_config = args
    try:
        logger.info(f"Starting sequence generation with {graph_type.name} model")

        # Generate graph sequence
        logger.debug("Generating graph sequence...")
        sequence = generate_graph_sequence(graph_type, params)
        graphs = sequence["graphs"]
        logger.info(f"Generated sequence with {len(graphs)} graphs")
        logger.debug(
            f"Sequence metadata: {sequence['n']} nodes, {len(sequence['change_points'])} change points"
        )

        # Extract link prediction features
        logger.debug("Extracting link prediction features...")
        feature_types = feature_config.get("types", LINK_FEATURES)
        logger.debug(f"Feature types to extract: {feature_types}")
        features = extract_features(graphs, feature_types=feature_types)
        logger.info(f"Extracted features: {list(features.keys())}")

        result = {
            "adjacency": graphs,
            "features": features,
            "change_points": sequence["change_points"],
            "params": sequence["params"],
            "n_nodes": sequence["n"],
            "sequence_length": sequence["sequence_length"],
        }
        logger.debug("Successfully created sequence with features")
        return result

    except Exception as e:
        logger.error(f"Failed to generate sequence: {str(e)}", exc_info=True)
        raise


def create_dataset(
    config_path: str = "configs/dataset_config.yaml",
) -> Dict:
    """Create the full dataset with graph sequences and link prediction features"""
    try:
        logger.info("Starting dataset creation...")
        logger.debug(f"Using config file: {config_path}")

        # Load and validate configurations
        logger.debug("Loading configurations...")
        config = DatasetConfig.from_yaml(config_path)
        config.validate()
        logger.info(f"Loaded dataset config: {config}")

        # Create parameter objects for each graph type
        graph_params = {
            "BA": BAParams(
                n=config.graph_config["nodes"],
                seq_len=config.graph_config.get("seq_len", 50),
                min_segment=config.graph_config.get("min_segment", 10),
                min_changes=config.graph_config.get("min_changes", 1),
                max_changes=config.graph_config.get("max_changes", 2),
                initial_edges=config.graph_config["edges"],
                min_edges=config.graph_config["min_edges"],
                max_edges=config.graph_config["max_edges"],
                pref_exp=config.graph_config.get("preferential_exp", 1.0),
            ),
            "ER": ERParams(
                n=config.graph_config["nodes"],
                seq_len=config.graph_config.get("seq_len", 50),
                min_segment=config.graph_config.get("min_segment", 10),
                min_changes=config.graph_config.get("min_changes", 1),
                max_changes=config.graph_config.get("max_changes", 2),
                initial_prob=config.graph_config.get("initial_prob", 0.1),
                min_prob=config.graph_config.get("min_prob", 0.05),
                max_prob=config.graph_config.get("max_prob", 0.15),
            ),
            "NW": NWParams(
                n=config.graph_config["nodes"],
                seq_len=config.graph_config.get("seq_len", 50),
                min_segment=config.graph_config.get("min_segment", 10),
                min_changes=config.graph_config.get("min_changes", 1),
                max_changes=config.graph_config.get("max_changes", 2),
                k_nearest=config.graph_config.get("k_nearest", 4),
                initial_prob=config.graph_config.get("initial_prob", 0.1),
                min_prob=config.graph_config.get("min_prob", 0.05),
                max_prob=config.graph_config.get("max_prob", 0.15),
            ),
            "SBM": SBMParams(
                n=config.graph_config["nodes"],
                seq_len=config.graph_config.get("seq_len", 50),
                min_segment=config.graph_config.get("min_segment", 10),
                min_changes=config.graph_config.get("min_changes", 1),
                max_changes=config.graph_config.get("max_changes", 2),
                num_blocks=config.graph_config.get("num_blocks", 5),
                min_block_size=config.graph_config.get("min_block_size", 80),
                max_block_size=config.graph_config.get("max_block_size", 120),
                initial_intra_prob=config.graph_config.get("initial_intra_prob", 0.3),
                initial_inter_prob=config.graph_config.get("initial_inter_prob", 0.05),
                min_intra_prob=config.graph_config.get("min_intra_prob", 0.25),
                max_intra_prob=config.graph_config.get("max_intra_prob", 0.35),
                min_inter_prob=config.graph_config.get("min_inter_prob", 0.03),
                max_inter_prob=config.graph_config.get("max_inter_prob", 0.07),
            ),
        }

        # Prepare tasks for multiprocessing
        tasks = []
        for graph_type in config.graph_types:
            if graph_type not in graph_params:
                logger.warning(f"Skipping unsupported graph type: {graph_type}")
                continue
            params = graph_params[graph_type]
            tasks.extend(
                [(GraphType[graph_type], params, config.features)]
                * (config.num_sequences // len(config.graph_types))
            )

        n_workers = min(cpu_count(), len(tasks))
        logger.info(
            f"Generating {len(tasks)} graph sequences using {n_workers} workers"
        )

        # Generate sequences using multiprocessing
        with Pool(processes=n_workers) as pool:
            sequences = []
            for result in tqdm(
                pool.imap(generate_sequence_with_features, tasks),
                total=len(tasks),
                desc="Generating sequences",
            ):
                sequences.append(result)
                logger.debug(f"Generated sequence {len(sequences)}/{len(tasks)}")

        logger.info(f"Successfully generated {len(sequences)} sequences")

        # Split sequences into train/val/test
        n_sequences = len(sequences)
        n_train = int(n_sequences * config.split_ratio["train"])
        n_val = int(n_sequences * config.split_ratio["val"])

        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train : n_train + n_val]
        test_sequences = sequences[n_train + n_val :]

        logger.info(
            f"Split sizes - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}"
        )

        dataset = {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences,
            "config": {
                "graph_params": graph_params,
                "features": config.features,
                "link_prediction": config.link_prediction,
            },
        }

        # Save dataset
        logger.info("Saving dataset...")
        save_dataset(dataset, config)
        logger.info("Dataset creation completed successfully")
        return dataset

    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}", exc_info=True)
        raise


def save_dataset(dataset: Dict, config: DatasetConfig) -> None:
    """Save the dataset to disk in compressed format"""
    try:
        logger.info(f"Saving dataset to {config.output_dir}")
        os.makedirs(config.output_dir, exist_ok=True)

        if config.save_format == "h5":
            output_path = os.path.join(config.output_dir, "ba_dataset.h5")
            logger.debug(f"Saving in H5 format to {output_path}")

            with h5py.File(output_path, "w") as f:
                for split_name in ["train", "val", "test"]:
                    logger.debug(f"Processing {split_name} split...")
                    split_group = f.create_group(split_name)
                    sequences = dataset[split_name]

                    for i, seq in enumerate(sequences):
                        logger.debug(f"Saving sequence {i} in {split_name} split")
                        seq_group = split_group.create_group(f"sequence_{i}")

                        # Save adjacency matrices
                        seq_group.create_dataset(
                            "adjacency",
                            data=seq["adjacency"],
                            compression="gzip" if config.compression else None,
                        )

                        # Save features
                        feat_group = seq_group.create_group("features")
                        for feat_name, feat_matrices in seq["features"].items():
                            feat_data = np.array([m.toarray() for m in feat_matrices])
                            feat_group.create_dataset(
                                feat_name,
                                data=feat_data,
                                compression="gzip" if config.compression else None,
                            )

                        # Save metadata
                        seq_group.create_dataset(
                            "change_points", data=seq["change_points"]
                        )
                        seq_group.create_dataset("n_nodes", data=seq["n_nodes"])
                        seq_group.create_dataset(
                            "sequence_length", data=seq["sequence_length"]
                        )
                        seq_group.attrs["params"] = str(seq["params"])

                # Save configurations
                logger.debug("Saving configuration metadata...")
                config_group = f.create_group("config")
                config_group.attrs["graph_config"] = str(
                    dataset["config"]["graph"].__dict__
                )
                config_group.attrs["feature_config"] = str(
                    dataset["config"]["features"]
                )
                config_group.attrs["link_prediction_config"] = str(
                    dataset["config"]["link_prediction"]
                )

        elif config.save_format == "npz":
            logger.debug("Saving in NPZ format")
            for split_name in ["train", "val", "test"]:
                output_path = os.path.join(
                    config.output_dir, f"ba_dataset_{split_name}.npz"
                )
                logger.debug(f"Processing {split_name} split to {output_path}")

                sequences = dataset[split_name]
                adjacency = np.array([seq["adjacency"] for seq in sequences])
                features = {
                    feat_name: np.array(
                        [
                            [m.toarray() for m in seq["features"][feat_name]]
                            for seq in sequences
                        ]
                    )
                    for feat_name in sequences[0]["features"].keys()
                }

                metadata = {
                    "change_points": [seq["change_points"] for seq in sequences],
                    "n_nodes": [seq["n_nodes"] for seq in sequences],
                    "sequence_length": [seq["sequence_length"] for seq in sequences],
                    "params": [str(seq["params"]) for seq in sequences],
                }

                save_func = np.savez_compressed if config.compression else np.savez
                save_func(
                    output_path,
                    adjacency=adjacency,
                    features=features,
                    metadata=metadata,
                    config=str(dataset["config"]),
                )

        logger.info("Dataset saved successfully")

    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate BA graph sequence dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset config file",
    )
    args = parser.parse_args()

    try:
        # Create dataset
        dataset = create_dataset(args.config)

        # Print summary
        print("\nDataset Summary:")
        print("-" * 50)
        for split_name, sequences in dataset.items():
            if split_name != "config":
                print(f"\n{split_name.upper()} Split:")
                print(f"  Number of sequences: {len(sequences)}")
                if sequences:
                    print(f"  Sequence length: {sequences[0]['sequence_length']}")
                    print(f"  Number of nodes: {sequences[0]['n_nodes']}")
                    print(f"  Feature types: {list(sequences[0]['features'].keys())}")
                    print(
                        f"  Average change points per sequence: "
                        f"{np.mean([len(seq['change_points']) for seq in sequences]):.2f}"
                    )

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
