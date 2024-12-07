# synthetic_data/inspect_dataset.py

"""
Dataset Inspector

This script provides detailed analysis and visualization of the generated graph sequence datasets,
helping understand the data structure, feature distributions, and sequence characteristics.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetInspector:
    """Analyzes and visualizes the structure and statistics of graph sequence datasets."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.splits = ["train", "val", "test"]
        self.feature_types = [
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "svd",
            "lsvd",
        ]

    def load_split(self, split: str) -> Dict[str, Any]:
        """Load a specific dataset split."""
        path = self.dataset_dir / split / "data.h5"
        if not path.exists():
            raise FileNotFoundError(f"No data found at {path}")

        with h5py.File(path, "r") as f:
            data = {
                "sequence_lengths": f["lengths"][:],
                "graph_types": f["graph_types"][:] if "graph_types" in f else None,
                "features": {},
                "adjacency": {},
                "change_points": [],
            }

            # Load features
            feat_group = f["features"]
            for feat in self.feature_types:
                if feat in feat_group:
                    data["features"][feat] = feat_group[feat][:]

            # Load adjacency matrices
            adj_group = f["adjacency"]
            for i in range(len(data["sequence_lengths"])):
                data["adjacency"][i] = adj_group[f"sequence_{i}"][:]

            # Load change points
            if "change_points" in f:
                cp_group = f["change_points"]
                for i in range(len(data["sequence_lengths"])):
                    data["change_points"].append(cp_group[f"sequence_{i}"][:])

        return data

    def analyze_dataset(self) -> Dict[str, Dict]:
        """Perform comprehensive analysis of the dataset."""
        analysis = {}

        for split in self.splits:
            try:
                data = self.load_split(split)
                analysis[split] = self._analyze_split(data)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")
                continue

        return analysis

    def _analyze_split(self, data: Dict) -> Dict:
        """Analyze a single dataset split."""
        num_sequences = len(data["sequence_lengths"])

        # Basic statistics
        stats = {
            "num_sequences": num_sequences,
            "sequence_lengths": {
                "min": int(np.min(data["sequence_lengths"])),
                "max": int(np.max(data["sequence_lengths"])),
                "mean": float(np.mean(data["sequence_lengths"])),
                "std": float(np.std(data["sequence_lengths"])),
            },
            "feature_shapes": {},
            "change_points": {
                "counts": [len(cp) for cp in data["change_points"]],
                "positions": [cp.tolist() for cp in data["change_points"]],
            },
        }

        # Feature analysis
        for feat_name, feat_data in data["features"].items():
            stats["feature_shapes"][feat_name] = feat_data.shape
            stats[f"{feat_name}_stats"] = {
                "mean": float(np.mean(feat_data)),
                "std": float(np.std(feat_data)),
                "min": float(np.min(feat_data)),
                "max": float(np.max(feat_data)),
            }

        return stats

    def print_analysis(self, analysis: Dict[str, Dict]):
        """Print detailed analysis results."""
        print("\n=== Dataset Analysis ===\n")

        for split, stats in analysis.items():
            print(f"\n{split.upper()} SET:")
            print("-" * 50)

            # Basic statistics
            print(f"\nSequences: {stats['num_sequences']}")
            print("\nSequence Lengths:")
            for k, v in stats["sequence_lengths"].items():
                print(f"  {k}: {v}")

            # Feature shapes
            print("\nFeature Shapes:")
            for feat, shape in stats["feature_shapes"].items():
                print(f"  {feat}: {shape}")
                print("    Interpretation:")
                self._explain_shape(feat, shape)

            # Change points
            print("\nChange Points:")
            cp_counts = stats["change_points"]["counts"]
            print(f"  Average count: {np.mean(cp_counts):.1f}")
            print(f"  Range: {min(cp_counts)}-{max(cp_counts)}")

    def _explain_shape(self, feature: str, shape: tuple):
        """Explain what each dimension in the shape means."""
        if len(shape) == 3:  # Centrality features
            print(f"    - {shape[0]} sequences in the batch")
            print(f"    - Each sequence has {shape[1]} timesteps")
            print(f"    - {shape[2]} nodes per graph")
        elif len(shape) == 4:  # Embedding features
            print(f"    - {shape[0]} sequences in the batch")
            print(f"    - Each sequence has {shape[1]} timesteps")
            print(f"    - {shape[2]} nodes per graph")
            print(f"    - {shape[3]} dimensional embedding per node")

    def visualize_statistics(
        self, analysis: Dict[str, Dict], output_dir: str = "analysis"
    ):
        """Create visualizations of dataset statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sequence length distribution
        plt.figure(figsize=(10, 6))
        for split in analysis:
            lengths = analysis[split]["sequence_lengths"]
            plt.hist(lengths, alpha=0.5, label=split)
        plt.title("Sequence Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(output_dir / "sequence_lengths.png")
        plt.close()

        # Change point distribution
        plt.figure(figsize=(10, 6))
        for split in analysis:
            cp_counts = analysis[split]["change_points"]["counts"]
            plt.hist(cp_counts, alpha=0.5, label=split)
        plt.title("Change Points per Sequence")
        plt.xlabel("Number of Change Points")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(output_dir / "change_points.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect graph sequence dataset")
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset", help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory for analysis outputs",
    )
    args = parser.parse_args()

    inspector = DatasetInspector(args.dataset_dir)
    analysis = inspector.analyze_dataset()
    inspector.print_analysis(analysis)
    inspector.visualize_statistics(analysis, args.output_dir)


if __name__ == "__main__":
    main()
