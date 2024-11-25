"""
Erdős-Rényi Graph Martingale Visualization Script

This script demonstrates the martingale-based change point detection for ER graphs.
It generates ER graphs, computes martingales, and visualizes the results.

Usage:
    1. Update the ER_MARTINGALE_CONFIG if needed
    2. Run the script: python tests/visualize_er_martingales.py
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_er_graphs import generate_er_graphs
from src.changepoint import ChangePointDetector
from visualize_martingales import MartingaleVisualizer

# -----------------------------------------------------------------------------#
#                              USER CONFIGURATION                               #
# -----------------------------------------------------------------------------#

ER_MARTINGALE_CONFIG = {
    "threshold": 30,  # Detection threshold for martingales
    "epsilon": 0.8,   # Sensitivity parameter for martingale computation
    "output_dir": "martingale_outputs"  # Directory to save results
}

# -----------------------------------------------------------------------------#
#                           IMPLEMENTATION DETAILS                              #
# -----------------------------------------------------------------------------#

@dataclass
class ERMartingaleAnalyzer:
    """Analyzer for ER graph martingales."""
    
    threshold: float = ER_MARTINGALE_CONFIG["threshold"]
    epsilon: float = ER_MARTINGALE_CONFIG["epsilon"]
    output_dir: str = ER_MARTINGALE_CONFIG["output_dir"]
    
    def compute_martingales(
        self,
        graphs: List[np.ndarray],
        change_points: List[int]
    ) -> Dict[str, Any]:
        """Compute both reset and cumulative martingales for the graph sequence."""
        detector = ChangePointDetector()
        detector.initialize(graphs)
        centralities = detector.extract_features()
        
        martingales_reset = {}
        martingales_cumulative = {}
        
        for name, values in centralities.items():
            # Normalize values
            values_array = np.array(values)
            normalized_values = (values_array - np.mean(values_array, axis=0)) / np.std(values_array, axis=0)
            
            # Compute martingales
            martingales_reset[name] = detector.martingale_test(
                data=normalized_values,
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=True
            )
            
            cumulative_result = detector.martingale_test(
                data=normalized_values,
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=False
            )
            
            # Convert to cumulative sum
            cumulative_values = np.array(cumulative_result["martingales"])
            cumulative_result["martingales"] = np.cumsum(cumulative_values)
            martingales_cumulative[name] = cumulative_result
        
        return {
            "reset": martingales_reset,
            "cumulative": martingales_cumulative
        }
    
    def analyze_and_visualize(self) -> None:
        """Run the complete analysis pipeline."""
        # Generate ER graphs
        print("\nGenerating ER graphs...")
        result = generate_er_graphs()
        graphs = result["graphs"]
        change_points = result["change_points"]
        
        # Compute martingales
        print("Computing martingales...")
        martingales = self.compute_martingales(graphs, change_points)
        
        # Create visualizer and generate plots
        print("Creating visualizations...")
        visualizer = MartingaleVisualizer(
            graphs=graphs,
            change_points=change_points,
            martingales=martingales,
            graph_type="ER",
            output_dir=self.output_dir
        )
        
        # Create visualization and save results
        visualizer.create_dashboard()
        visualizer.save_results()
        print(f"\nResults saved to: {self.output_dir}/")

# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#

def main():
    """Main entry point for ER graph martingale analysis."""
    print("\nStarting ER Graph Martingale Analysis")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Threshold: {ER_MARTINGALE_CONFIG['threshold']}")
    print(f"  - Epsilon: {ER_MARTINGALE_CONFIG['epsilon']}")
    print(f"  - Output directory: {ER_MARTINGALE_CONFIG['output_dir']}")
    
    # Create analyzer and run analysis
    analyzer = ERMartingaleAnalyzer()
    analyzer.analyze_and_visualize()


if __name__ == "__main__":
    main()
