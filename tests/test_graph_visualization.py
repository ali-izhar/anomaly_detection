# tests/test_graph_visualization.py

"""
Test the visualization module.

Tests cover:
1. Basic visualization functionality
2. Different graph types and layouts
3. Sequence visualization
4. Evolution plots with metrics
5. Style customization
"""

import pytest
import numpy as np
import networkx as nx
import matplotlib

# Use Agg backend for testing (non-interactive)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing."""
    plt.switch_backend("Agg")
    yield
    plt.close("all")


@pytest.fixture
def sample_graphs():
    """Generate sample graphs for testing."""
    return {
        "empty": nx.Graph(),
        "single": nx.Graph([(0, 0)]),
        "path": nx.path_graph(5),
        "complete": nx.complete_graph(5),
        "random": nx.gnp_random_graph(10, 0.3, seed=42),
        "grid": nx.grid_2d_graph(3, 3),
    }


@pytest.fixture
def visualizer():
    """Create visualizer instance."""
    return NetworkVisualizer()


@pytest.fixture
def graph_sequence():
    """Generate a sequence of evolving graphs."""
    generator = GraphGenerator("ba")
    config = {
        "n": 20,
        "seq_len": 5,
        "min_segment": 2,
        "min_changes": 1,
        "max_changes": 2,
        "m": 2,
        "min_m": 1,
        "max_m": 3,
        "seed": 42,
    }
    result = generator.generate_sequence(config)
    return result


class TestBasicVisualization:
    """Test basic visualization functionality."""

    def test_adjacency_plot(self, visualizer, sample_graphs):
        """Test adjacency matrix plotting."""
        for name, graph in sample_graphs.items():
            adj_matrix = nx.to_numpy_array(graph)
            fig, ax = visualizer.plot_adjacency(adj_matrix, title=f"Adjacency - {name}")
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            plt.close(fig)

    @pytest.mark.parametrize(
        "layout", ["spring", "circular", "spectral", "random", "shell"]
    )
    def test_network_layouts(self, visualizer, sample_graphs, layout):
        """Test different network layouts."""
        for name, graph in sample_graphs.items():
            if graph.number_of_nodes() > 0:  # Skip empty graph
                # Set layout-specific parameters
                layout_params = {}
                if layout == "spring":
                    layout_params = {"k": 0.5}  # k parameter only for spring layout
                elif layout == "shell":
                    layout_params = {"nlist": [range(graph.number_of_nodes())]}

                fig, ax = visualizer.plot_network(
                    graph,
                    title=f"{name} - {layout}",
                    layout=layout,
                    layout_params=layout_params,
                )
                assert isinstance(fig, plt.Figure)
                assert isinstance(ax, plt.Axes)
                plt.close(fig)

    def test_style_customization(self, sample_graphs):
        """Test style customization."""
        custom_style = {
            "node_size": 500,
            "node_color": "red",
            "edge_color": "blue",
            "font_size": 12,
            "width": 2,
            "alpha": 0.5,
            "cmap": "coolwarm",
        }
        visualizer = NetworkVisualizer(style=custom_style)
        graph = sample_graphs["random"]

        # Test with spring layout and its specific parameters
        fig, ax = visualizer.plot_network(
            graph, title="Custom Style", layout="spring", layout_params={"k": 0.5}
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, ax = visualizer.plot_adjacency(nx.to_numpy_array(graph))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSequenceVisualization:
    """Test sequence visualization functionality."""

    def test_graph_sequence_network(self, visualizer, graph_sequence):
        """Test graph sequence visualization as networks."""
        fig, axes = visualizer.plot_graph_sequence(
            graph_sequence["graphs"],
            n_cols=3,
            plot_type="network",
            layout="spring",
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) > 0
        plt.close(fig)

    def test_graph_sequence_adjacency(self, visualizer, graph_sequence):
        """Test graph sequence visualization as adjacency matrices."""
        fig, axes = visualizer.plot_graph_sequence(
            graph_sequence["graphs"],
            n_cols=3,
            plot_type="adjacency",
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) > 0
        plt.close(fig)

    def test_custom_titles(self, visualizer, graph_sequence):
        """Test sequence visualization with custom titles."""
        titles = [f"Time {t}" for t in range(len(graph_sequence["graphs"]))]
        fig, axes = visualizer.plot_graph_sequence(
            graph_sequence["graphs"],
            titles=titles,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEvolutionVisualization:
    """Test evolution visualization functionality."""

    def test_evolution_with_metrics(self, visualizer, graph_sequence):
        """Test evolution plot with metrics."""
        # Generate some mock metrics
        n_graphs = len(graph_sequence["graphs"])
        metrics = {
            "clustering": np.random.rand(n_graphs),
            "modularity": np.random.rand(n_graphs),
        }

        fig, axes = visualizer.plot_graph_evolution(
            graph_sequence["graphs"],
            change_points=graph_sequence["change_points"],
            metrics=metrics,
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2  # Metrics plot + density plot
        plt.close(fig)

    def test_evolution_without_metrics(self, visualizer, graph_sequence):
        """Test evolution plot without metrics."""
        fig, axes = visualizer.plot_graph_evolution(
            graph_sequence["graphs"],
            change_points=graph_sequence["change_points"],
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 1  # Only density plot
        plt.close(fig)


class TestEdgeCases:
    """Test visualization edge cases."""

    def test_single_node_visualization(self, visualizer):
        """Test visualization of single-node graph."""
        graph = nx.Graph([(0, 0)])
        fig, ax = visualizer.plot_network(graph)
        plt.close(fig)

        fig, ax = visualizer.plot_adjacency(nx.to_numpy_array(graph))
        plt.close(fig)

    def test_empty_graph_visualization(self, visualizer):
        """Test visualization of empty graph."""
        graph = nx.Graph()
        fig, ax = visualizer.plot_network(graph)
        plt.close(fig)

        fig, ax = visualizer.plot_adjacency(nx.to_numpy_array(graph))
        plt.close(fig)

    def test_large_graph_visualization(self, visualizer):
        """Test visualization of large graph."""
        graph = nx.barabasi_albert_graph(100, 2, seed=42)
        fig, ax = visualizer.plot_network(graph)
        plt.close(fig)

        fig, ax = visualizer.plot_adjacency(nx.to_numpy_array(graph))
        plt.close(fig)
