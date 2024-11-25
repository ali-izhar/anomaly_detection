# tests/visualize_nw_graphs.py

"""Newman-Watts Small-World Graph Visualization Script"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_nw_graphs import generate_nw_graphs
from visualize_graphs import GraphVisualizer


def main():
    result = generate_nw_graphs()
    visualizer = GraphVisualizer(
        graphs=result["graphs"], change_points=result["change_points"], graph_type="NW"
    )

    visualizer.create_dashboard()


if __name__ == "__main__":
    main()
