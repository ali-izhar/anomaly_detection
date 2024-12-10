"""
Main Dataset Generation Script

Creates three different datasets:
1. Node-level features with adjacency matrices
2. Global features with adjacency matrices
3. Combined features with adjacency matrices
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import shutil
from typing import Dict

import yaml
from create_dataset import create_dataset
from inspect_dataset import DatasetInspector

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_config_variant(
    base_config_path: str, output_dir: str, feature_config: Dict, variant_name: str
) -> str:
    """Create a variant of the config file with specific feature settings."""
    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify feature settings
    config["dataset"]["features"] = feature_config
    config["dataset"]["output_dir"] = os.path.join(output_dir, variant_name)

    # Save variant config
    os.makedirs(output_dir, exist_ok=True)
    variant_path = os.path.join(output_dir, f"dataset_config_{variant_name}.yaml")

    with open(variant_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return variant_path


def generate_datasets(
    base_config: str = "configs/dataset_config.yaml",
    graph_config: str = "configs/graph_config.yaml",
    output_dir: str = "datasets",
):
    """Generate three variants of the dataset."""
    try:
        # Define feature configurations for each variant
        variants = {
            "node_level": {
                "use_node_features": True,
                "include_adjacency": False,
                "normalize": True,
                "types": [
                    "degree",
                    "betweenness",
                    "eigenvector",
                    "closeness",
                    "svd",
                    "lsvd",
                ],
            },
            "global": {
                "use_node_features": False,
                "include_adjacency": True,
                "normalize": True,
                "types": [
                    "degree",
                    "betweenness",
                    "eigenvector",
                    "closeness",
                    "svd",
                    "lsvd",
                ],
            },
            "combined": {
                "use_node_features": True,
                "include_adjacency": True,
                "normalize": True,
                "types": [
                    "degree",
                    "betweenness",
                    "eigenvector",
                    "closeness",
                    "svd",
                    "lsvd",
                ],
            },
        }

        # Generate each variant
        for variant_name, feature_config in variants.items():
            logger.info(f"\nGenerating {variant_name} dataset...")

            # Create variant config
            variant_config = create_config_variant(
                base_config, output_dir, feature_config, variant_name
            )

            # Generate dataset
            dataset = create_dataset(
                config_path=variant_config, graph_config_path=graph_config
            )

            # Inspect and save analysis
            inspector = DatasetInspector(
                os.path.join(output_dir, variant_name, "dataset.h5")
            )

            # Create analysis directory
            analysis_dir = os.path.join(output_dir, variant_name, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # Generate plots
            inspector.plot_feature_distributions(
                save_path=os.path.join(analysis_dir, "feature_distributions.png")
            )
            inspector.plot_change_point_distribution(
                save_path=os.path.join(analysis_dir, "change_point_distribution.png")
            )

            # Save statistics
            stats = inspector.get_basic_stats()
            cp_stats = inspector.analyze_change_points()

            with open(os.path.join(analysis_dir, "statistics.txt"), "w") as f:
                f.write("Dataset Statistics:\n")
                f.write("-" * 50 + "\n")
                for graph_type, graph_stats in stats.items():
                    f.write(f"\n{graph_type}:\n")
                    for key, value in graph_stats.items():
                        f.write(f"  {key}: {value}\n")

                f.write("\nChange Point Analysis:\n")
                f.write("-" * 50 + "\n")
                for graph_type, cp_stat in cp_stats.items():
                    f.write(f"\n{graph_type}:\n")
                    for key, value in cp_stat.items():
                        f.write(f"  {key}: {value:.2f}\n")

            logger.info(f"Completed {variant_name} dataset generation")

        logger.info("\nAll datasets generated successfully!")

    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple graph sequence datasets"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to base dataset config",
    )
    parser.add_argument(
        "--graph-config",
        type=str,
        default="configs/graph_config.yaml",
        help="Path to graph config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets",
        help="Output directory for all datasets",
    )
    args = parser.parse_args()

    try:
        generate_datasets(
            base_config=args.base_config,
            graph_config=args.graph_config,
            output_dir=args.output,
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
