# src/utils/sd_visualizer.py

"""Visualization tools for synthetic data analysis."""

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns  # type: ignore
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class SyntheticDataVisualizer:
    """Visualization tools for temporal graph analysis and change detection."""

    @staticmethod
    def plot_graph_samples(
        graphs: List[np.ndarray],
        indices: List[int],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        graph_type: str = "Graph",
        description: str = "",
    ) -> None:
        """Visualize graph structure evolution over time.
        Create a grid layout of graph snapshots G1, ..., Gk showing topological changes in the network.
        """
        if not graphs or not indices:
            logger.error("Empty graph or index sequence provided")
            raise ValueError("Empty graph or index sequence")
        if len(graphs) != len(indices):
            logger.error(
                f"Dimension mismatch: {len(graphs)} graphs vs {len(indices)} indices"
            )
            raise ValueError("Number of graphs and indices must match")

        try:
            num_graphs = len(graphs)
            cols = min(3, num_graphs)
            rows = (num_graphs + cols - 1) // cols

            logger.info(f"Creating {rows}x{cols} grid for {num_graphs} graph samples")
            plt.figure(figsize=(cols * 3, rows * 3))

            if description:
                plt.figtext(0.5, 0.95, description, ha="center", fontsize=10)
                logger.debug(f"Added description: {description}")

            for i, (graph, idx) in enumerate(zip(graphs, indices), 1):
                logger.debug(f"Plotting graph {i}/{num_graphs} (index {idx})")
                plt.subplot(rows, cols, i)
                G = nx.from_numpy_array(graph)
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, node_size=30, with_labels=False, edge_color="gray")
                plt.title(f"{graph_type} {idx}", fontsize=8)
                plt.axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.93])

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved graph samples plot to {save_path}")

            if show:
                logger.debug("Displaying plot")
                plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Graph visualization failed: {str(e)}")
            raise RuntimeError(f"Graph visualization failed: {str(e)}")

    @staticmethod
    def plot_centrality_shapes(
        centralities: Dict[str, List[List[float]]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Log dimensionality of centrality measures."""
        try:
            logger.info("Computing centrality shapes")
            shapes = {
                name: (len(cent), len(cent[0]) if cent else 0)
                for name, cent in centralities.items()
            }

            for name, shape in shapes.items():
                logger.debug(f"{name.upper()} Centrality: {shape}")

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open("w") as f:
                    for name, shape in shapes.items():
                        f.write(f"{name.upper()} Centrality: {shape}\n")
                logger.info(f"Saved centrality shapes to {save_path}")

        except Exception as e:
            logger.error(f"Shape logging failed: {str(e)}")
            raise RuntimeError(f"Shape logging failed: {str(e)}")

    @staticmethod
    def plot_martingale_results(
        martingales: Dict[str, Dict[str, Any]],
        title: str,
        output_dir: Union[str, Path],
        save: bool = False,
        show: bool = False,
    ) -> None:
        """Visualize martingale sequences for change detection."""
        try:
            logger.info(f"Plotting martingale results: {title}")
            centrality_martingales = {
                k: v["martingales"] for k, v in martingales.items()
            }
            M_sum = sum(centrality_martingales.values())
            M_avg = M_sum / len(centrality_martingales)

            plt.figure(figsize=(10, 6))
            plt.plot(M_sum, label="Martingale Sum", linewidth=1.5)
            plt.plot(M_avg, label="Martingale Average", linewidth=1.5)

            for name, values in centrality_martingales.items():
                plt.plot(values, label=name.capitalize(), linewidth=1)
                logger.debug(f"Added {name} martingale sequence")

            plt.legend(loc="upper left", fontsize=8)
            plt.title(title, fontsize=12)
            plt.xlabel("Time Instants", fontsize=10)
            plt.ylabel("Martingale Values", fontsize=10)
            plt.grid(True)

            if save:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{title.replace(' ', '_')}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved martingale plot to {output_path}")

            if show:
                logger.debug("Displaying plot")
                plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Martingale visualization failed: {str(e)}")
            raise RuntimeError(f"Martingale visualization failed: {str(e)}")

    @staticmethod
    def plot_change_point_detection(
        martingales: Dict[str, Dict[str, Any]],
        shap_values: np.ndarray,
        time_range: Tuple[int, int],
        output_dir: Union[str, Path],
        save: bool = False,
        show: bool = False,
    ) -> None:
        """Visualize change point detection results with SHAP explanations.
        Create a plot showing the martingale sequences and detection thresholds with
        SHAP values showing feature importance over time."""
        try:
            a, b = time_range
            logger.info(
                f"Creating change point detection plot for time range [{a}, {b}]"
            )

            plt.figure(figsize=(10, 8))
            x_range = range(a, b)
            colors = sns.color_palette("husl", len(martingales) + 2)

            # Plot martingales
            logger.debug("Plotting martingale sequences")
            plt.subplot(2, 1, 1)
            M_sum = sum(m["martingales"] for m in martingales.values())
            M_avg = M_sum / len(martingales)

            plt.plot(x_range, M_sum[a:b], color=colors[0], label="Sum", linewidth=1.5)
            plt.plot(
                x_range, M_avg[a:b], color=colors[1], label="Average", linewidth=1.5
            )

            for i, (label, m) in enumerate(martingales.items()):
                plt.plot(
                    x_range,
                    m["martingales"][a:b],
                    color=colors[i + 2],
                    label=label.capitalize(),
                    linewidth=1,
                )

            plt.legend(fontsize=8)
            plt.title("Martingale Measures", fontsize=10)
            plt.xlabel("Time", fontsize=8)
            plt.ylabel("Martingale Values", fontsize=8)
            plt.grid(True)

            # Plot SHAP values
            logger.debug("Plotting SHAP values")
            plt.subplot(2, 1, 2)
            for i, label in enumerate(martingales.keys()):
                plt.plot(
                    x_range,
                    shap_values[:, i][a:b],
                    color=colors[i + 2],
                    label=label.capitalize(),
                    linewidth=1,
                )

            plt.legend(fontsize=8)
            plt.title("SHAP Values over Time", fontsize=10)
            plt.xlabel("Time", fontsize=8)
            plt.ylabel("Feature Importance", fontsize=8)
            plt.grid(True)

            plt.tight_layout()

            if save:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"change_point_detection.png"
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved change point detection plot to {save_path}")

            if show:
                logger.debug("Displaying plot")
                plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Change point visualization failed: {str(e)}")
            raise RuntimeError(f"Change point visualization failed: {str(e)}")

    @staticmethod
    def plot_centrality_distribution(
        centralities: Dict[str, List[List[float]]],
        title: str,
        output_dir: Union[str, Path],
        save: bool = False,
        show: bool = False,
    ) -> None:
        """Visualize distribution of centrality measures.
        For each centrality measure, plot the kernel density estimate of the flattened values.
        """
        try:
            logger.info(f"Plotting centrality distributions: {title}")
            plt.figure(figsize=(10, 6))

            for name, values in centralities.items():
                flat_values = [v for graph in values for v in graph]
                sns.kdeplot(flat_values, label=name.capitalize(), fill=True)
                logger.debug(f"Added distribution for {name} centrality")

            plt.legend(title="Centrality Measures", fontsize=8)
            plt.title(title, fontsize=12)
            plt.xlabel("Centrality Value", fontsize=10)
            plt.ylabel("Density", fontsize=10)
            plt.grid(True)

            if save:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"{title.replace(' ', '_')}.png"
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved distribution plot to {save_path}")

            if show:
                logger.debug("Displaying plot")
                plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Distribution visualization failed: {str(e)}")
            raise RuntimeError(f"Distribution visualization failed: {str(e)}")

    @staticmethod
    def plot_comprehensive_dashboard(
        sample_graphs: List[np.ndarray],
        sample_indices: List[int],
        graph_type: str,
        description: str,
        martingales_detect: Dict[str, Dict[str, Any]],
        martingales_no_detect: Dict[str, Dict[str, Any]],
        shap_values: np.ndarray,
        centralities: Dict[str, List[List[float]]],
        config: Any,
    ) -> None:
        """Create a comprehensive visualization dashboard."""
        try:
            logger.info(f"Creating comprehensive dashboard for {graph_type}")
            fig = plt.figure(figsize=(20, 24))
            gs = fig.add_gridspec(
                4, 5, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3
            )

            # Row 0: Sample Graphs
            logger.debug("Plotting sample graphs")
            for i, (graph, idx) in enumerate(
                zip(sample_graphs[:5], sample_indices[:5])
            ):
                ax = fig.add_subplot(gs[0, i])
                G = nx.from_numpy_array(graph)
                pos = nx.spring_layout(G, seed=42)
                nx.draw(
                    G, pos, node_size=20, with_labels=False, edge_color="gray", ax=ax
                )
                ax.set_title(f"{graph_type} {idx}", fontsize=8)
                ax.axis("off")

            # Row 1: Martingale Results
            logger.debug("Plotting martingale results")
            ax1 = fig.add_subplot(gs[1, :3])
            for cent, result in martingales_detect.items():
                ax1.plot(result["martingales"], label=cent.capitalize())
            ax1.set_title("Martingale Values with Detection", fontsize=10)
            ax1.set_xlabel("Time", fontsize=8)
            ax1.set_ylabel("Martingale Values", fontsize=8)
            ax1.legend(fontsize=6)
            ax1.grid(True)

            ax2 = fig.add_subplot(gs[1, 3:])
            for cent, result in martingales_no_detect.items():
                ax2.plot(result["martingales"], label=cent.capitalize())
            ax2.set_title("Martingale Values without Detection", fontsize=10)
            ax2.set_xlabel("Time", fontsize=8)
            ax2.set_ylabel("Martingale Values", fontsize=8)
            ax2.legend(fontsize=6)
            ax2.grid(True)

            # Row 2: SHAP Analysis
            logger.debug("Plotting SHAP analysis")
            ax3 = fig.add_subplot(gs[2, :3])
            for i, cent in enumerate(martingales_detect.keys()):
                ax3.plot(shap_values[:, i], label=cent.capitalize())
            ax3.set_title("SHAP Values over Time", fontsize=10)
            ax3.set_xlabel("Time", fontsize=8)
            ax3.set_ylabel("Feature Importance", fontsize=8)
            ax3.legend(
                fontsize=5,
                loc="upper right",
                title="Centrality Measures",
                title_fontsize=6,
                framealpha=0.8,
            )
            ax3.grid(True)

            ax4 = fig.add_subplot(gs[2, 3:])

            # Normalize SHAP values for better visualization
            normalized_shap = (shap_values - np.min(shap_values)) / (
                np.max(shap_values) - np.min(shap_values)
            )

            # Create heatmap with improved settings
            sns.heatmap(
                normalized_shap.T,
                ax=ax4,
                cmap="RdBu_r",
                center=0.5,
                xticklabels=50,
                yticklabels=[name.capitalize() for name in martingales_detect.keys()],
                cbar_kws={
                    "label": "Relative Feature Importance",
                    "orientation": "horizontal",
                },
            )

            ax4.set_title(
                "Feature Importance Over Time\n(Red = High Impact, Blue = Low Impact)",
                fontsize=10,
            )
            ax4.set_xlabel("Time Steps", fontsize=8)
            ax4.set_ylabel("")

            # Move y-axis labels to the right
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.tick_right()

            # Rotate x-axis labels for better readability
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
            # Keep y-axis labels horizontal
            plt.setp(ax4.get_yticklabels(), rotation=0)

            # Row 3: Change Points and Distributions
            logger.debug("Plotting change points and distributions")
            ax5 = fig.add_subplot(gs[3, :3])
            change_points = martingales_detect.get("svd", {}).get(
                "change_detected_instant", []
            )

            M_sum = sum(m["martingales"] for m in martingales_detect.values())
            M_avg = M_sum / len(martingales_detect)

            ax5.plot(M_sum, color="blue", alpha=0.3, label="Martingale Sum")
            ax5.plot(M_avg, color="green", alpha=0.3, label="Martingale Average")

            y_max = max(M_sum) if len(M_sum) > 0 else 1
            for cp in change_points:
                ax5.axvline(
                    cp,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Change at t={cp}",
                )
                logger.debug(f"Marked change point at t={cp}")

            ax5.set_title("Detected Change Points", fontsize=10)
            ax5.set_xlabel("Time", fontsize=8)
            ax5.set_ylabel("Martingale Value", fontsize=8)
            if change_points:
                ax5.legend(fontsize=6)
            ax5.grid(True)

            ax6 = fig.add_subplot(gs[3, 3:])

            for name, values in centralities.items():
                flat_values = [v for graph in values for v in graph]

                # Remove extreme outliers using percentiles
                q1, q3 = np.percentile(flat_values, [1, 99])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_values = [
                    v for v in flat_values if lower_bound <= v <= upper_bound
                ]

                # Normalize the values to [0,1] range for each centrality measure
                min_val = min(filtered_values)
                max_val = max(filtered_values)
                normalized_values = [
                    (v - min_val) / (max_val - min_val) for v in filtered_values
                ]

                # Plot with improved settings
                sns.kdeplot(
                    data=normalized_values,
                    label=f"{name.capitalize()} [{min_val:.2e}, {max_val:.2e}]",
                    ax=ax6,
                    fill=True,
                    alpha=0.4,
                    bw_adjust=0.5,  # Slightly reduce bandwidth for better detail
                    common_norm=True,  # Use common normalization for fair comparison
                )

            ax6.set_title(
                "Normalized Centrality Distributions\n(Original Range Shown in Labels)",
                fontsize=10,
            )
            ax6.set_xlabel("Normalized Centrality Value (0-1 Scale)", fontsize=8)
            ax6.set_ylabel("Density", fontsize=8)

            # Improve legend with value ranges
            ax6.legend(
                title="Centrality Measures [min, max]",
                fontsize=6,
                title_fontsize=8,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

            # Add grid and set limits
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(-0.05, 1.05)  # Add small padding around 0-1 range
            ax6.set_ylim(bottom=0)

            # Add text explaining the normalization
            ax6.text(
                0.98,
                0.02,
                "Note: Each measure normalized to [0,1] range\nfor shape comparison",
                fontsize=6,
                ha="right",
                va="bottom",
                transform=ax6.transAxes,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

            # Add overall title and description
            fig.suptitle(
                f"Comprehensive Analysis Dashboard: {graph_type}", fontsize=16, y=0.95
            )
            if description:
                fig.text(
                    0.5, 0.92, description, ha="center", fontsize=10, style="italic"
                )
                logger.debug(f"Added description: {description}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.91])

            if config.visualization.save_plots:
                save_path = (
                    Path(config.paths.output.dir) / f"dashboard_{graph_type}.png"
                )
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved comprehensive dashboard to {save_path}")

            logger.debug("Displaying dashboard")
            plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Comprehensive dashboard creation failed: {str(e)}")
            raise RuntimeError(f"Comprehensive dashboard failed: {str(e)}")

    @staticmethod
    def print_change_points(
        martingale_result: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Log detected change points with timestamps."""
        try:
            change_points = martingale_result.get("change_detected_instant", [])
            message = f"Change points detected at: {change_points}"
            logger.info(message)

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open("w") as f:
                    f.write(f"{message}\n")
                logger.debug(f"Saved change points to {save_path}")

        except Exception as e:
            logger.error(f"Change point logging failed: {str(e)}")
            raise RuntimeError(f"Change point logging failed: {str(e)}")
