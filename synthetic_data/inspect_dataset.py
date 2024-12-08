# synthetic_data/inspect_dataset.py

"""
Dataset Inspector

This script provides detailed analysis and basic assertions on the generated graph sequence datasets.
We check:
- That all sequences have the expected fixed length.
- That normalization in the training set is approximately correct (mean ~0, std ~1).
- Basic statistics and structure of the dataset.

If any major discrepancy is found, assertions or warnings will be raised.
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
    """Analyzes and inspects the structure and statistics of graph sequence datasets."""

    def __init__(self, dataset_dir: str = "dataset", expected_seq_len: int = 200):
        self.dataset_dir = Path(dataset_dir)
        self.splits = ["train", "val", "test"]
        self.expected_seq_len = expected_seq_len

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
                "graphs": None,
                "change_points": [],
            }

            # Load features (expecting features/all dataset)
            feat_group = f["features"]
            if "all" not in feat_group:
                raise ValueError("Features dataset 'all' not found")
            all_features = feat_group["all"][:]  # shape: (num_sequences, seq_len, 6)
            data["features"]["all"] = all_features

            # Load graphs
            adj_group = f["adjacency"]
            num_sequences = len(data["sequence_lengths"])
            first_graph = adj_group["sequence_0"][:]
            data["graphs"] = np.zeros_like(all_features, dtype=np.float32)
            # Actually, all_features has shape (N,T,6), graphs has shape (N,T,N_nodes,N_nodes)
            # Determine shapes from first_graph:
            max_seq_len, n_nodes, _ = first_graph.shape
            graphs_all = np.zeros(
                (num_sequences, max_seq_len, n_nodes, n_nodes), dtype=np.float32
            )
            graphs_all[0] = first_graph
            for i in range(1, num_sequences):
                graphs_all[i] = adj_group[f"sequence_{i}"][:]
            data["graphs"] = graphs_all

            # Load change points if available
            if "change_points" in f:
                cp_group = f["change_points"]
                for i in range(num_sequences):
                    data["change_points"].append(cp_group[f"sequence_{i}"][:])

        return data

    def analyze_dataset(self) -> Dict[str, Dict]:
        """Perform comprehensive analysis of the dataset."""
        analysis = {}

        for split in self.splits:
            try:
                data = self.load_split(split)
                analysis[split] = self._analyze_split(split, data)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")
                continue

        # After analysis, run assertions across splits if train split is available
        if "train" in analysis:
            self._assert_normalization(analysis["train"], split="train")

        return analysis

    def _analyze_split(self, split: str, data: Dict) -> Dict:
        """Analyze a single dataset split."""
        num_sequences = len(data["sequence_lengths"])
        seq_lengths = data["sequence_lengths"]

        # Assert all sequences have the expected fixed length
        if not np.all(seq_lengths == self.expected_seq_len):
            raise AssertionError(
                f"Not all sequences in {split} split have length {self.expected_seq_len}"
            )

        # Basic statistics
        stats = {
            "num_sequences": num_sequences,
            "sequence_lengths": {
                "min": int(np.min(seq_lengths)),
                "max": int(np.max(seq_lengths)),
                "mean": float(np.mean(seq_lengths)),
                "std": float(np.std(seq_lengths)),
            },
            "change_points": {
                "counts": [len(cp) for cp in data["change_points"]],
            },
        }

        # Feature stats
        all_features = data["features"]["all"]  # (N, T, 6)
        stats["features_stats"] = self._compute_feature_stats(all_features)

        return stats

    def _compute_feature_stats(self, features: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics (mean/std/min/max) for the entire features array."""
        return {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
        }

    def _assert_normalization(self, train_stats: Dict, split: str):
        """
        Since the train set should be normalized (mean ~0, std ~1),
        we assert that it is reasonably close to these values.
        """
        feats = train_stats["features_stats"]
        mean_val = feats["mean"]
        std_val = feats["std"]

        # We allow some tolerance, say mean within 0.1 and std within (0.9,1.1)
        if abs(mean_val) > 0.1:
            logger.warning(
                f"WARNING: {split} features mean is {mean_val:.3f}, expected ~0. Check normalization."
            )
        if not (0.9 < std_val < 1.1):
            logger.warning(
                f"WARNING: {split} features std is {std_val:.3f}, expected ~1. Check normalization."
            )

    def print_analysis(self, analysis: Dict[str, Dict]):
        """Print detailed analysis results."""
        print("\n=== Dataset Analysis ===\n")

        for split, stats in analysis.items():
            print(f"\n{split.upper()} SET:")
            print("-" * 50)

            # Basic statistics
            print(f"\nSequences: {stats['num_sequences']}")
            print("Sequence Lengths:")
            for k, v in stats["sequence_lengths"].items():
                print(f"  {k}: {v}")

            # Feature stats
            feats = stats["features_stats"]
            print("\nFeatures (all) Stats:")
            for k, v in feats.items():
                print(f"  {k}: {v:.4f}")

            # Change points
            cp_counts = stats["change_points"]["counts"]
            if cp_counts:
                print("\nChange Points:")
                print(f"  Average count: {np.mean(cp_counts):.1f}")
                print(f"  Range: {min(cp_counts)} - {max(cp_counts)}")

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
            # lengths dict has min, max, mean, std, but not raw distribution
            # Since all sequences have the same length now, this might be unnecessary.
            # If needed, we must load data again or store full lengths from somewhere.
            # For now, we skip a hist if all have same length.
            # Just a bar chart indicating that all sequences have length expected_seq_len.
            plt.bar(split, lengths["mean"], yerr=lengths["std"], alpha=0.5)
        plt.title("Sequence Length Summary")
        plt.ylabel("Length (with std as error bar)")
        plt.savefig(output_dir / "sequence_lengths.png")
        plt.close()

        # Change point distribution
        plt.figure(figsize=(10, 6))
        for split in analysis:
            cp_counts = analysis[split]["change_points"]["counts"]
            if cp_counts:
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
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=200,
        help="Expected sequence length for assertions",
    )
    args = parser.parse_args()

    inspector = DatasetInspector(
        args.dataset_dir, expected_seq_len=args.expected_seq_len
    )
    analysis = inspector.analyze_dataset()
    inspector.print_analysis(analysis)
    inspector.visualize_statistics(analysis, args.output_dir)


if __name__ == "__main__":
    main()
