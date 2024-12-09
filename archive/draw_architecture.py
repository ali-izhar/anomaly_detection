import matplotlib.pyplot as plt
import numpy as np


def plot_high_level_architecture():
    """Create a professional architecture diagram with consistent visual elements."""
    plt.figure(figsize=(20, 12))
    ax = plt.gca()

    colors = {
        "graph": "#4B6BF5",  # Blue - for graph structure
        "features": "#45A247",  # Green - for features
        "learned": "#FF4B4B",  # Red - for learned representations
        "predict": "#FFA500",  # Orange - for predictions
        "arrow": "#666666",  # Gray - for connections
        "text": "#2F4F4F",  # Dark slate gray - for text
    }

    # Component positions
    x_positions = {"input": 0.15, "process": 0.45, "learned": 0.75, "output": 0.9}
    y_positions = {"graph": 0.7, "features": 0.3}
    middle_y = (y_positions["graph"] + y_positions["features"]) / 2

    # 1. Graph Structure Input
    draw_graph_input(
        ax,
        x_positions["input"],
        y_positions["graph"],
        colors["graph"],
        "Graph Structure",
        colors,
    )

    # 2. GNN Block
    draw_gnn_block(
        ax,
        x_positions["process"],
        y_positions["graph"],
        colors["graph"],
        "Graph Neural Network",
        colors,
    )

    # 3. Feature Input
    draw_feature_input(
        ax,
        x_positions["input"],
        y_positions["features"],
        colors["features"],
        "Graph Features",
        colors,
    )

    # 4. LSTM Block
    draw_temporal_block(
        ax,
        x_positions["process"],
        y_positions["features"],
        colors["features"],
        "Temporal Learning",
        colors,
    )

    # 5. Learned Representation
    draw_learned_space(
        ax,
        x_positions["learned"],
        middle_y,
        colors["learned"],
        "Joint Representation",
        colors,
    )

    # 6. Prediction Output
    draw_prediction_block(
        ax, x_positions["output"], middle_y, colors["predict"], "Predictions", colors
    )

    # Add arrows and annotations
    add_connections(ax, x_positions, y_positions, middle_y, colors)
    add_component_descriptions(ax, x_positions, y_positions, middle_y, colors)

    # Add title and framework
    plt.title(
        "Graph-Temporal Anomaly Detection Framework",
        fontsize=16,
        pad=20,
        color=colors["text"],
    )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_graph_input(ax, x, y, color, text, colors):
    """Draw graph input representation with adjacency matrix and graph."""
    width, height = 0.15, 0.15

    # Draw matrix-like structure
    matrix_size = 5
    cell_size = width / matrix_size
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:  # Show connectivity
                alpha = 0.2 if np.random.rand() > 0.7 else 0
                rect = plt.Rectangle(
                    (x - width / 2 + i * cell_size, y - height / 2 + j * cell_size),
                    cell_size,
                    cell_size,
                    facecolor=color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

    # Draw border
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="none",
        edgecolor=color,
    )
    ax.add_patch(rect)

    # Add label
    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def draw_gnn_block(ax, x, y, color, text, colors):
    """Draw GNN block with message passing visualization."""
    width, height = 0.15, 0.15

    # Main container
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)

    # Draw message passing visualization
    nodes = [(0, 0), (0.5, 0.7), (1, 0), (0.5, -0.7)]
    scale = 0.05

    # Draw edges
    for i, (x1, y1) in enumerate(nodes):
        for j, (x2, y2) in enumerate(nodes):
            if i < j:
                ax.plot(
                    [x + x1 * scale, x + x2 * scale],
                    [y + y1 * scale, y + y2 * scale],
                    color=color,
                    alpha=0.3,
                )

    # Draw nodes
    for nx, ny in nodes:
        ax.scatter(x + nx * scale, y + ny * scale, color=color, s=100)

    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def draw_feature_input(ax, x, y, color, text, colors):
    """Draw feature input block with centrality measures visualization."""
    width, height = 0.15, 0.15

    # Main container
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)

    # Draw feature bars representing different centrality measures
    features = ["Deg", "Btw", "Eig", "Cls", "SVD", "LSVD"]
    n_features = len(features)
    bar_width = width * 0.8 / n_features

    for i, feat in enumerate(features):
        # Random height for visualization
        bar_height = np.random.uniform(0.02, 0.08)
        bar_x = x - width * 0.4 + i * bar_width

        # Draw bar
        rect = plt.Rectangle(
            (bar_x, y - bar_height / 2),
            bar_width * 0.8,
            bar_height,
            facecolor=color,
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Add feature label
        ax.text(
            bar_x + bar_width * 0.4,
            y - height * 0.3,
            feat,
            ha="center",
            va="top",
            fontsize=6,
            color=color,
            rotation=45,
        )

    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def draw_temporal_block(ax, x, y, color, text, colors):
    """Draw temporal learning block with LSTM-style gates."""
    width, height = 0.15, 0.15

    # Main container
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)

    # Draw LSTM components
    cell_width = width * 0.2

    # Cell state line
    ax.plot(
        [x - width * 0.4, x + width * 0.4],
        [y + height * 0.2] * 2,
        color=color,
        linestyle="--",
        alpha=0.5,
    )

    # Gates
    gate_positions = [
        (x - width * 0.25, "f"),  # Forget gate
        (x, "i"),  # Input gate
        (x + width * 0.25, "o"),  # Output gate
    ]

    for gate_x, gate_label in gate_positions:
        # Gate circle
        circle = plt.Circle(
            (gate_x, y - height * 0.1),
            cell_width * 0.4,
            facecolor="white",
            edgecolor=color,
        )
        ax.add_patch(circle)
        # Gate label
        ax.text(
            gate_x,
            y - height * 0.1,
            gate_label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def draw_learned_space(ax, x, y, color, text, colors):
    """Draw learned representation space with embedded points."""
    width, height = 0.15, 0.15

    # Main container
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)

    # Draw embedded points
    np.random.seed(42)
    n_points = 20
    points_x = np.random.normal(x, width * 0.2, n_points)
    points_y = np.random.normal(y, height * 0.2, n_points)

    # Draw connections between close points
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.sqrt(
                (points_x[i] - points_x[j]) ** 2 + (points_y[i] - points_y[j]) ** 2
            )
            if dist < width * 0.15:
                ax.plot(
                    [points_x[i], points_x[j]],
                    [points_y[i], points_y[j]],
                    color=color,
                    alpha=0.1,
                )

    # Draw points
    ax.scatter(points_x, points_y, color=color, s=30, alpha=0.5)

    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def draw_prediction_block(ax, x, y, color, text, colors):
    """Draw prediction block with forecasting visualization."""
    width, height = 0.15, 0.15

    # Main container
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)

    # Draw prediction curve
    t = np.linspace(-width * 0.4, width * 0.4, 100)
    curve = 0.05 * np.sin(10 * t) * np.exp(-abs(t))
    ax.plot(t + x, curve + y, color=color, linewidth=2)

    # Draw confidence interval
    ax.fill_between(t + x, curve + y - 0.02, curve + y + 0.02, color=color, alpha=0.2)

    ax.text(x, y - height / 1.5, text, ha="center", va="top", color=colors["text"])


def add_connections(ax, x_positions, y_positions, middle_y, colors):
    """Add connecting arrows between components."""

    def add_arrow(start_x, start_y, end_x, end_y, color, label=""):
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xycoords="data",
            xytext=(start_x, start_y),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color, lw=2
            ),
        )
        if label:
            ax.text(
                (start_x + end_x) / 2,
                (start_y + end_y) / 2 + 0.02,
                label,
                ha="center",
                va="bottom",
                color=color,
            )

    # Graph structure path
    add_arrow(
        x_positions["input"] + 0.08,
        y_positions["graph"],
        x_positions["process"] - 0.08,
        y_positions["graph"],
        colors["graph"],
        "Structure",
    )

    # Feature path
    add_arrow(
        x_positions["input"] + 0.08,
        y_positions["features"],
        x_positions["process"] - 0.08,
        y_positions["features"],
        colors["features"],
        "Features",
    )

    # Converging paths
    add_arrow(
        x_positions["process"] + 0.08,
        y_positions["graph"],
        x_positions["learned"] - 0.08,
        middle_y,
        colors["learned"],
        "Embed",
    )
    add_arrow(
        x_positions["process"] + 0.08,
        y_positions["features"],
        x_positions["learned"] - 0.08,
        middle_y,
        colors["learned"],
        "Encode",
    )

    # Prediction path
    add_arrow(
        x_positions["learned"] + 0.08,
        middle_y,
        x_positions["output"] - 0.08,
        middle_y,
        colors["predict"],
        "Forecast",
    )


def add_component_descriptions(ax, x_positions, y_positions, middle_y, colors):
    """Add descriptive annotations for each component."""
    descriptions = [
        (
            x_positions["input"],
            y_positions["graph"] + 0.15,
            "Graph Structure\n• Adjacency Matrix\n• Node Connections",
            colors["graph"],
        ),
        (
            x_positions["process"],
            y_positions["graph"] + 0.15,
            "Message Passing\n• Node Updates\n• Edge Features",
            colors["graph"],
        ),
        (
            x_positions["input"],
            y_positions["features"] - 0.15,
            "Graph Features\n• Centrality Measures\n• Global Properties",
            colors["features"],
        ),
        (
            x_positions["process"],
            y_positions["features"] - 0.15,
            "Temporal Patterns\n• Sequential Learning\n• State Tracking",
            colors["features"],
        ),
        (
            x_positions["learned"],
            middle_y + 0.15,
            "Joint Embedding\n• Combined Features\n• Temporal Context",
            colors["learned"],
        ),
        (
            x_positions["output"],
            middle_y + 0.15,
            "Predictions\n• Future States\n• Anomaly Scores",
            colors["predict"],
        ),
    ]

    for x, y, text, color in descriptions:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white", edgecolor=color, alpha=0.9, boxstyle="round,pad=0.5"
            ),
            fontsize=8,
            color=colors["text"],
        )


if __name__ == "__main__":
    plot_high_level_architecture()
