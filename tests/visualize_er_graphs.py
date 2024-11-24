# tests/visualize_er_graphs.py

"""Erdős-Rényi Graph Visualization Script"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_er_graphs import generate_er_graphs
from visualize_graphs import GraphVisualizer


def main():
    result = generate_er_graphs()
    visualizer = GraphVisualizer(
        graphs=result["graphs"], change_points=result["change_points"], graph_type="ER"
    )

    visualizer.create_dashboard()


if __name__ == "__main__":
    main()
