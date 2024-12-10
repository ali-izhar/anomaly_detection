# synthetic_data/create_dataset.py

"""
Dataset Generator

Creates a dataset of graph sequences with their features:
- Generates sequences for each graph type (BA, ER, NW)
- Extracts node-level or global features
- Saves adjacency matrices and features for each timestep
"""

import sys
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import yaml
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graphs import GraphConfig, GraphType, generate_graph_sequence
from concat import concat_features

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""

    num_sequences: int
    graph_types: List[str]
    features: Dict
    split_ratio: Dict[str, float]
    output_dir: str
    save_format: str
    compression: bool

    @classmethod
    def from_yaml(cls, config_path: str) -> "DatasetConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["dataset"]
        return cls(**config)


def generate_sequence_with_features(args: Tuple[GraphType, GraphConfig, Dict]) -> Dict:
    """Generate a single sequence with features (for multiprocessing)"""
    graph_type, graph_config, feature_config = args
    try:
        # Generate graph sequence
        sequence = generate_graph_sequence(graph_config)
        graphs = sequence["graphs"]

        # Extract and concatenate features
        combined, norm_params = concat_features(
            graphs,
            feature_types=feature_config.get("types"),
            use_node_features=feature_config.get("use_node_features", True),
            normalize=feature_config.get("normalize", True),
            include_adjacency=feature_config.get("include_adjacency", False),
        )

        return {
            "adjacency": graphs,
            "features": combined,
            "norm_params": norm_params,
            "change_points": sequence["change_points"],
            "graph_type": graph_type.value,
            "params": sequence["params"],
        }

    except Exception as e:
        logger.error(f"Failed to generate sequence: {str(e)}")
        raise


def create_dataset(
    config_path: str = "configs/dataset_config.yaml",
    graph_config_path: str = "configs/graph_config.yaml",
) -> Dict:
    """Create the full dataset with all sequences and features"""
    try:
        # Load configurations
        config = DatasetConfig.from_yaml(config_path)

        # Prepare tasks for multiprocessing
        tasks = []
        for graph_type_str in config.graph_types:
            graph_type = GraphType[graph_type_str]
            graph_config = GraphConfig.from_yaml(graph_type, graph_config_path)

            for _ in range(config.num_sequences):
                tasks.append((graph_type, graph_config, config.features))

        # Generate sequences using multiprocessing
        n_workers = min(cpu_count(), len(tasks))
        logger.info(f"Generating {len(tasks)} sequences using {n_workers} workers")

        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(generate_sequence_with_features, tasks),
                    total=len(tasks),
                    desc="Generating sequences",
                )
            )

        # Organize results by graph type
        dataset = {
            graph_type: {"sequences": [], "change_points": [], "params": []}
            for graph_type in config.graph_types
        }

        for result in results:
            # Get the graph type from the enum value mapping
            if result["graph_type"] == GraphType.BA.value:
                graph_type = "BA"
            elif result["graph_type"] == GraphType.ER.value:
                graph_type = "ER"
            elif result["graph_type"] == GraphType.NW.value:
                graph_type = "NW"
            else:
                raise ValueError(f"Unknown graph type: {result['graph_type']}")

            dataset[graph_type]["sequences"].append(
                {"adjacency": result["adjacency"], "features": result["features"]}
            )
            dataset[graph_type]["change_points"].append(result["change_points"])
            dataset[graph_type]["params"].append(result["params"])

        # Save dataset
        save_dataset(dataset, config)
        return dataset

    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


def save_dataset(dataset: Dict, config: DatasetConfig) -> None:
    """Save the dataset to disk"""
    os.makedirs(config.output_dir, exist_ok=True)

    if config.save_format == "h5":
        with h5py.File(os.path.join(config.output_dir, "dataset.h5"), "w") as f:
            for graph_type in dataset.keys():
                group = f.create_group(graph_type)

                # Save sequences
                seq_group = group.create_group("sequences")
                for i, seq in enumerate(dataset[graph_type]["sequences"]):
                    seq_group.create_dataset(
                        f"seq_{i}/adjacency",
                        data=seq["adjacency"],
                        compression="gzip" if config.compression else None,
                    )
                    seq_group.create_dataset(
                        f"seq_{i}/features",
                        data=seq["features"],
                        compression="gzip" if config.compression else None,
                    )

                # Save change points (create separate dataset for each sequence)
                cp_group = group.create_group("change_points")
                for i, cp in enumerate(dataset[graph_type]["change_points"]):
                    cp_group.create_dataset(
                        f"seq_{i}",
                        data=np.array(cp, dtype=np.int32),
                        compression="gzip" if config.compression else None,
                    )

                # Save parameters (store as JSON strings)
                param_group = group.create_group("params")
                for i, params in enumerate(dataset[graph_type]["params"]):
                    # Convert params to string representation
                    param_str = str(params)
                    param_group.create_dataset(
                        f"seq_{i}", data=param_str, dtype=h5py.special_dtype(vlen=str)
                    )

    elif config.save_format == "npz":
        for graph_type in dataset.keys():
            # Convert sequences to arrays
            sequences = dataset[graph_type]["sequences"]
            adjacency = np.array([seq["adjacency"] for seq in sequences])
            features = np.array([seq["features"] for seq in sequences])

            # Save as compressed npz
            save_func = np.savez_compressed if config.compression else np.savez
            save_func(
                os.path.join(config.output_dir, f"{graph_type}_dataset.npz"),
                adjacency=adjacency,
                features=features,
                change_points=np.array(
                    dataset[graph_type]["change_points"], dtype=object
                ),
                params=np.array(dataset[graph_type]["params"], dtype=object),
            )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate graph sequence dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset config file",
    )
    parser.add_argument(
        "--graph-config",
        type=str,
        default="configs/graph_config.yaml",
        help="Path to graph config file",
    )
    args = parser.parse_args()

    try:
        # Create dataset
        dataset = create_dataset(args.config, args.graph_config)

        # Print summary
        print("\nDataset Summary:")
        print("-" * 50)
        for graph_type, data in dataset.items():
            print(f"\n{graph_type} Sequences:")
            print(f"  Number of sequences: {len(data['sequences'])}")
            print(f"  Sequence length: {len(data['sequences'][0]['adjacency'])}")
            print(f"  Feature shape: {data['sequences'][0]['features'].shape}")
            print(f"  Number of change points: {len(data['change_points'])}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
