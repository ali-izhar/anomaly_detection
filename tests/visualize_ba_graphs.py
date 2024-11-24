# tests/visualize_ba_graphs.py

"""Barabási-Albert Graph Visualization Script"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_ba_graphs import generate_ba_graphs
from visualize_graphs import GraphVisualizer


def main():
    result = generate_ba_graphs()
    visualizer = GraphVisualizer(
        graphs=result["graphs"], change_points=result["change_points"], graph_type="BA"
    )
    
    visualizer.create_dashboard()


if __name__ == "__main__":
    main()
