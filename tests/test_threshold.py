# tests/test_threshold.py

"""
Threshold Testing Module for Martingale Analysis

This module tests different threshold values for martingale-based change point detection
on BA graphs. It allows comparing how different thresholds affect the sum and average
martingale models.
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
#                              TEST CONFIGURATION                              #
# -----------------------------------------------------------------------------#

THRESHOLD_TEST_CONFIG = {
    "thresholds": {
        "low": 15,      # Lower sensitivity
        "medium": 30,   # Default sensitivity
        "high": 45,     # Higher sensitivity
    },
    "epsilon": 0.8,
    "output_dir": "threshold_test_outputs",
}


@dataclass
class ThresholdTester:
    """Tester for different martingale threshold values."""
    
    thresholds: Dict[str, float] = None
    epsilon: float = THRESHOLD_TEST_CONFIG["epsilon"]
    output_dir: str = THRESHOLD_TEST_CONFIG["output_dir"]

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = THRESHOLD_TEST_CONFIG["thresholds"]

    def compute_martingales(
        self, graphs: List[np.ndarray], threshold: float
    ) -> Dict[str, Any]:
        """Compute martingales with specific threshold."""
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
                threshold=threshold,
                epsilon=self.epsilon,
                reset=True,
            )

            cumulative_result = detector.martingale_test(
                data=normalized_values,
                threshold=threshold,
                epsilon=self.epsilon,
                reset=False,
            )

            # Convert to cumulative sum
            cumulative_values = np.array(cumulative_result["martingales"])
            cumulative_result["martingales"] = np.cumsum(cumulative_values)
            martingales_cumulative[name] = cumulative_result

        return {"reset": martingales_reset, "cumulative": martingales_cumulative}

    def run_threshold_tests(self) -> None:
        """Run tests with different threshold values."""
        print("\nGenerating BA graphs for threshold testing...")
        result = generate_ba_graphs()
        graphs = result["graphs"]
        change_points = result["change_points"]

        print("\nTesting different threshold values...")
        for threshold_name, threshold_value in self.thresholds.items():
            print(f"\nAnalyzing with {threshold_name} threshold ({threshold_value})...")
            
            # Compute martingales with current threshold
            martingales = self.compute_martingales(graphs, threshold_value)

            # Create visualizations
            output_subdir = f"{self.output_dir}/threshold_{threshold_value}"
            visualizer = MartingaleVisualizer(
                graphs=graphs,
                change_points=change_points,
                martingales=martingales,
                graph_type=f"BA (tau={threshold_value})",
                threshold=threshold_value,
                epsilon=self.epsilon,
                output_dir=output_subdir,
            )

            visualizer.create_dashboard()
            visualizer.save_results()
            print(f"Results saved to: {output_subdir}/")


# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#

def main():
    """Main entry point for threshold testing."""
    print("\nStarting Threshold Testing for Martingale Analysis")
    print("------------------------------------------------")
    print("Configuration:")
    print(f"  - Thresholds: {THRESHOLD_TEST_CONFIG['thresholds']}")
    print(f"  - Epsilon: {THRESHOLD_TEST_CONFIG['epsilon']}")
    print(f"  - Output directory: {THRESHOLD_TEST_CONFIG['output_dir']}")

    # Create tester and run tests
    tester = ThresholdTester()
    tester.run_threshold_tests()


if __name__ == "__main__":
    main()
