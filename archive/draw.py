import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.changepoint import ChangePointDetector


def generate_change_point_data(n_points=200, change_point=100, seed=42):
    """Generate synthetic data with a change point"""
    np.random.seed(seed)

    # Generate two different normal distributions
    x1 = np.random.normal(0, 1, change_point)  # N(0,1) before change point
    x2 = np.random.normal(3, 1, n_points - change_point)  # N(3,1) after change point

    # Combine the sequences
    sequence = np.concatenate([x1, x2])
    time = np.arange(n_points)

    return time, sequence, change_point


def plot_change_point_analysis():
    """Visualize the change point data and its statistical properties"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    # Generate data
    time, sequence, change_point = generate_change_point_data()

    # 1. Time Series Plot (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, sequence, color="#4B6BF5", label="Sequence")
    ax1.axvline(x=change_point, color="#FF4B4B", linestyle="--", label="Change Point")
    ax1.set_title("Time Series with Change Point")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.text(25, -2.5, r"$X_{1:k} \sim N(0,1)$", fontsize=10)
    ax1.text(change_point + 25, -2.5, r"$X_{k+1:n} \sim N(2,1)$", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Distribution Analysis (bottom row)
    # 2.1 Density Plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    before_change = sequence[:change_point]
    after_change = sequence[change_point:]
    x_range = np.linspace(min(sequence), max(sequence), 200)
    kde_before = stats.gaussian_kde(before_change)
    kde_after = stats.gaussian_kde(after_change)

    ax2.plot(
        x_range, kde_before(x_range), color="#4B6BF5", label="Distribution before k"
    )
    ax2.plot(x_range, kde_after(x_range), color="#FF4B4B", label="Distribution after k")
    ax2.set_title("Density Distributions")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2.2 QQ Plot (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    stats.probplot(before_change, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_color("#4B6BF5")
    ax3.get_lines()[1].set_color("#4B6BF5")
    stats.probplot(after_change, dist="norm", plot=ax3)
    ax3.get_lines()[2].set_color("#FF4B4B")
    ax3.get_lines()[3].set_color("#FF4B4B")
    ax3.set_title("Q-Q Plot")

    plt.tight_layout()
    plt.show()


def plot_martingale_mechanism():
    """Visualize the martingale detection mechanism and its application"""
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

    # 1. Top row with Martingale Property and P-value Transformation
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1, 1])

    # 1.1 Martingale Property (top left)
    ax1 = fig.add_subplot(gs_top[0])
    np.random.seed(42)
    n_steps = 100
    n_paths = 5
    paths = np.random.normal(0, 0.1, (n_paths, n_steps)).cumsum(axis=1)
    paths = paths - paths[:, 0:1]
    time = np.arange(n_steps)

    for i in range(n_paths):
        ax1.plot(time, paths[i], alpha=0.5, label=f"Path {i+1}")
    mean_path = paths.mean(axis=0)
    ax1.plot(time, mean_path, "k--", linewidth=2, label="Expected Value")
    ax1.axhline(y=0, color="r", linestyle=":", label="Initial Value")

    ax1.set_title("Martingale Property:\n" + r"$\mathbb{E}[M_{n+1}|M_1,...,M_n] = M_n$")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1.2 P-value Transformation (top right)
    ax2 = fig.add_subplot(gs_top[1])
    x = np.linspace(0.001, 1, 1000)
    epsilons = [0.7, 0.5, 0.3]
    colors = ["#4B6BF5", "#45A247", "#FF4B4B"]

    # Plot transformation curves
    for eps, color in zip(epsilons, colors):
        y = eps * x ** (eps - 1)
        ax2.plot(
            x,
            y,
            color=color,
            label=f'ε={eps} ({"more" if eps > 0.5 else "less" if eps < 0.5 else "medium"} sensitive)',
        )

    # Add significance level line and annotation
    ax2.axvline(x=0.05, color="r", linestyle=":", label="Significance Level")
    ax2.annotate(
        "Significance\nLevel (α=0.05)",
        xy=(0.05, 0),
        xytext=(0.15, 2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="red"),
    )

    # Add p-value effect annotations
    ax2.annotate(
        "Low p-value\nIncreases Mn",
        xy=(0.02, eps * 0.02 ** (eps - 1)),
        xytext=(0.2, 8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
    )
    ax2.annotate(
        "High p-value\nDecreases Mn",
        xy=(0.8, eps * 0.8 ** (eps - 1)),
        xytext=(0.6, 4),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )

    ax2.set_title("P-value Transformation:\n" + r"$\epsilon \cdot p_n^{(\epsilon-1)}$")
    ax2.set_xlabel("p-value")
    ax2.set_ylabel("Multiplier")
    ax2.set_ylim(0, 10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2. Combined Data and Martingale Plot (bottom)
    ax3 = fig.add_subplot(gs[1])
    time, sequence, change_point = generate_change_point_data()

    # Plot the original data with lower opacity
    ax3.plot(time, sequence, color="blue", alpha=0.2, label="Data", zorder=1)

    # Highlight the change in distribution
    before_mean = np.mean(sequence[:change_point])
    after_mean = np.mean(sequence[change_point:])
    ax3.axhline(
        y=before_mean,
        color="blue",
        linestyle="--",
        alpha=0.3,
        xmax=change_point / len(time),
    )
    ax3.axhline(
        y=after_mean,
        color="blue",
        linestyle="--",
        alpha=0.3,
        xmin=change_point / len(time),
    )

    # Add change point line
    ax3.axvline(
        x=change_point, color="black", linestyle="--", label="Change Point", zorder=2
    )

    # Highlight points around change point
    window = 5  # Number of points to highlight before and after change point
    highlight_indices = np.arange(change_point - window, change_point + window + 1)
    ax3.scatter(
        highlight_indices,
        sequence[highlight_indices],
        color="red",
        s=50,
        alpha=0.6,
        label="Points near change",
        zorder=3,
    )

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Data Value", color="blue")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax3.grid(True, alpha=0.3)

    # Add martingale values on secondary y-axis
    ax3_twin = ax3.twinx()

    # Prepare and normalize data
    data = sequence.reshape(-1, 1)
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    detector = ChangePointDetector()

    # Plot martingale values for different epsilons
    for eps, color in zip(epsilons, colors):
        martingale_results = detector.martingale_test(
            data=normalized_data, threshold=30, epsilon=eps, reset=True
        )
        martingale_values = np.array(martingale_results["martingales"])
        martingale_values = np.clip(martingale_values, 1e-10, None)
        ax3_twin.semilogy(
            time,
            martingale_values,
            color=color,
            label=f"Martingale (ε={eps})",
            zorder=4,
        )

    ax3_twin.set_ylabel("Martingale Value (Mn) - Log Scale", color="red")
    ax3_twin.tick_params(axis="y", labelcolor="red")

    # Add text annotations
    ax3.text(
        25,
        -2,
        "Normal Data Region\nHigh p-values\nMn stable/decreasing",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax3.text(
        change_point + 25,
        -2,
        "Unusual Data Region\nLow p-values\nMn increasing",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax3.set_title("Martingale Sequence Detection on Change Point Data")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_change_point_analysis()
    plot_martingale_mechanism()
