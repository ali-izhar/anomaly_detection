# tests/visualize_ba_martingales.py

"""
Barabási-Albert Graph Martingale Visualization Script

This script demonstrates the martingale-based change point detection for BA graphs.
It generates BA graphs, computes martingales, and visualizes the results.

Usage:
    1. Update the BA_MARTINGALE_CONFIG if needed
    2. Run the script: python tests/visualize_ba_martingales.py
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_ba_graphs import generate_ba_graphs
from src.changepoint import ChangePointDetector
from visualize_martingales import MartingaleVisualizer

# -----------------------------------------------------------------------------#
#                              USER CONFIGURATION                              #
# -----------------------------------------------------------------------------#

BA_MARTINGALE_CONFIG = {
    "threshold": 30,  # Detection threshold for martingales
    "epsilon": 0.8,  # Sensitivity parameter for martingale computation
    "output_dir": "martingale_outputs",  # Directory to save results
}

# -----------------------------------------------------------------------------#
#                           IMPLEMENTATION DETAILS                             #
# -----------------------------------------------------------------------------#


@dataclass
class BAMartingaleAnalyzer:
    """Analyzer for BA graph martingales."""

    threshold: float = BA_MARTINGALE_CONFIG["threshold"]
    epsilon: float = BA_MARTINGALE_CONFIG["epsilon"]
    output_dir: str = BA_MARTINGALE_CONFIG["output_dir"]

    def compute_martingales(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Compute both reset and cumulative martingales for the graph sequence."""
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
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=True,
            )

            cumulative_result = detector.martingale_test(
                data=normalized_values,
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=False,
            )

            # Convert to cumulative sum
            cumulative_values = np.array(cumulative_result["martingales"])
            cumulative_result["martingales"] = np.cumsum(cumulative_values)
            martingales_cumulative[name] = cumulative_result

        return {"reset": martingales_reset, "cumulative": martingales_cumulative}

    def analyze_and_visualize(self) -> None:
        """Run the complete analysis pipeline."""
        print("\nGenerating BA graphs...")
        result = generate_ba_graphs()
        graphs = result["graphs"]
        change_points = result["change_points"]

        print("Computing martingales...")
        martingales = self.compute_martingales(graphs)

        print("Creating visualizations...")
        visualizer = MartingaleVisualizer(
            graphs=graphs,
            change_points=change_points,
            martingales=martingales,
            graph_type="BA",
            threshold=self.threshold,
            epsilon=self.epsilon,
            output_dir=self.output_dir,
        )

        visualizer.create_dashboard()
        visualizer.save_results()
        print(f"\nResults saved to: {self.output_dir}/")


# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#


def main():
    """Main entry point for BA graph martingale analysis."""
    print("\nStarting BA Graph Martingale Analysis")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Threshold: {BA_MARTINGALE_CONFIG['threshold']}")
    print(f"  - Epsilon: {BA_MARTINGALE_CONFIG['epsilon']}")
    print(f"  - Output directory: {BA_MARTINGALE_CONFIG['output_dir']}")

    # Create analyzer and run analysis
    analyzer = BAMartingaleAnalyzer()
    analyzer.analyze_and_visualize()


if __name__ == "__main__":
    main()
