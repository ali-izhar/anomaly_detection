#!/usr/bin/env python
"""Run Martingale Detection on MIT Reality Dataset"""

import pandas as pd
import networkx as nx
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from typing import List, Dict, Any

# Add parent directory to path to allow imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint import BettingFunctionConfig, ChangePointDetector, DetectorConfig
from src.graph import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.utils import (
    normalize_features,
    normalize_predictions,
    OutputManager,
    prepare_result_data,
)
from src.predictor import PredictorFactory

logger = logging.getLogger(__name__)


def read_proximity_csv(
    file_path: str,
    drop_na: bool = True,
    save_cleaned: bool = False,
    output_dir: str = "results/mit_reality/data",
) -> pd.DataFrame:
    """Load and inspect the Proximity.csv file.
    Expected columns: user.id, remote.user.id.if.known, time, prob2.

    Args:
        file_path: Path to the input CSV file
        drop_na: Whether to drop rows with NaN values
        save_cleaned: Whether to save the cleaned DataFrame for future use
        output_dir: Directory to save the cleaned data

    Returns:
        Cleaned DataFrame with standardized column names
    """
    logger.info(f"Loading data from {file_path}")

    try:
        # Load the file (comma-delimited) and normalize headers
        df = pd.read_csv(file_path, header=0)

        # Clean column names: strip whitespace, quotes, BOM
        df.columns = (
            df.columns.str.strip().str.strip('"').str.replace("\ufeff", "", regex=False)
        )

        # Rename columns to expected names
        df = df.rename(
            columns={
                "user.id": "user_id",
                "remote.user.id.if.known": "remote_user_id",
                "time": "timestamp",
                "prob2": "probability",
            }
        )

        # Clean and parse timestamp column into datetime
        df["timestamp"] = df["timestamp"].astype(str).str.strip().str.strip('"')
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Convert numeric columns, coercing errors to NaN
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df["remote_user_id"] = pd.to_numeric(df["remote_user_id"], errors="coerce")
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

        # Remove records with invalid timestamps
        invalid_timestamps = df["timestamp"].isna().sum()
        if invalid_timestamps > 0:
            logger.warning(
                f"Removed {invalid_timestamps} records with invalid timestamps"
            )

        # Add date column for easier grouping
        df["date"] = df["timestamp"].dt.date

        if drop_na:
            df = df.dropna(subset=["timestamp", "probability"])
            df.reset_index(drop=True, inplace=True)

        # Log data stats
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique users: {df['user_id'].nunique()}")
        logger.info(f"Loaded {len(df)} proximity records")

        # Save cleaned DataFrame
        if save_cleaned:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Generate filename with timestamp to avoid overwriting
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                clean_csv_path = os.path.join(
                    output_dir, f"cleaned_proximity_{timestamp}.csv"
                )

                # Save as CSV for broader compatibility
                df.to_csv(clean_csv_path, index=False)
                logger.info(f"Saved cleaned DataFrame to {clean_csv_path}")

                # Check if pyarrow is available before trying to save as parquet
                has_parquet_support = False
                try:
                    import pyarrow

                    has_parquet_support = True
                except ImportError:
                    has_parquet_support = False

                # Only attempt to save as parquet if support is available
                if has_parquet_support:
                    clean_parquet_path = os.path.join(
                        output_dir, f"cleaned_proximity_{timestamp}.parquet"
                    )
                    df.to_parquet(clean_parquet_path, index=False)
                    logger.info(
                        f"Saved cleaned DataFrame to {clean_parquet_path} (efficient format)"
                    )
                else:
                    logger.info(
                        "Parquet format not saved - install pyarrow or fastparquet for more efficient storage"
                    )

                # Save a minimal metadata file with basic info
                metadata = {
                    "original_file": file_path,
                    "cleaned_date": timestamp,
                    "row_count": len(df),
                    "unique_users": df["user_id"].nunique(),
                    "date_range": [
                        df["date"].min().isoformat(),
                        df["date"].max().isoformat(),
                    ],
                    "columns": list(df.columns),
                }

                # Write metadata to JSON
                meta_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
                with open(meta_path, "w") as meta_file:
                    json.dump(metadata, meta_file, indent=2)
                logger.info(f"Saved data metadata to {meta_path}")

            except Exception as save_err:
                logger.error(f"Error saving cleaned data: {str(save_err)}")
                logger.info("Continuing with processing, but cleaned data not saved")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_daily_graphs(
    df: pd.DataFrame,
    threshold: float = 0.3,
    save_graphs: bool = False,
    output_dir: str = "results/mit_reality/daily_graphs",
    save_adjacency: bool = False,
) -> dict:
    """Create daily graph snapshots from the proximity DataFrame.
    Only include edges with probability >= threshold.
    Skip days with too few records or too few edges.
    Returns a dict mapping 'YYYY-MM-DD' to NetworkX Graphs.

    Args:
        df: DataFrame containing proximity data
        threshold: Minimum probability for an edge to be included
        save_graphs: Whether to save graphs as visualizations
        output_dir: Directory for saving graphs
        save_adjacency: Whether to save adjacency matrices as CSVs
    """
    logger.info("Creating daily graph snapshots")

    # Copy and extract date
    df = df.copy()

    # If date column doesn't exist, create it
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date

    # Get sorted list of unique dates
    dates = sorted(df["date"].unique())

    # Gather all users for consistent node sets
    all_users = set(df["user_id"].unique()) | set(df["remote_user_id"].unique())
    logger.info(f"Identified {len(all_users)} unique users across all days")

    daily_graphs = {}
    skipped_days = 0

    # Create a dictionary for more efficient graph generation
    user_pairs_by_date = {}

    # Group data by date for faster processing
    logger.info("Preprocessing data by date for faster graph generation")
    for date in dates:
        # Subset day's data
        day_df = df[df["date"] == date]

        # Skip days with too few interactions
        if len(day_df) <= 5:
            skipped_days += 1
            logger.debug(f"Skipping date {date} with insufficient interactions")
            continue

        # Filter by probability threshold
        day_df = day_df[day_df["probability"] >= threshold]

        # Extract user pairs for faster graph creation
        if len(day_df) > 0:
            user_pairs = day_df[["user_id", "remote_user_id"]].values
            date_str = date.isoformat()
            user_pairs_by_date[date_str] = user_pairs

    # Create graphs in parallel if possible
    logger.info("Generating graphs from interactions")
    for date_str, user_pairs in user_pairs_by_date.items():
        # Initialize graph and add all users as nodes for consistent node sets
        G = nx.Graph()
        G.add_nodes_from(all_users)

        # Add edges more efficiently using add_edges_from
        G.add_edges_from(user_pairs)

        # Skip graphs with too few edges
        if G.number_of_edges() <= 1:
            skipped_days += 1
            logger.debug(f"Skipping date {date_str} with insufficient edges")
            continue

        # Store graph with ISO date key
        daily_graphs[date_str] = G

        # Optional debug logging for each graph
        logger.debug(
            f"Created graph for {date_str}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    logger.info(
        f"Created {len(daily_graphs)} daily graphs, skipped {skipped_days} days with insufficient data"
    )

    # Save graphs if requested
    if save_graphs:
        _save_graphs(daily_graphs, os.path.dirname(output_dir), save_adjacency)

    return daily_graphs


def create_daily_adjacency_matrices(
    df: pd.DataFrame,
    threshold: float = 0.3,
    save_matrices: bool = False,
    output_dir: str = "results/mit_reality/adjacency_matrices",
) -> tuple:
    """Create daily adjacency matrices from the proximity DataFrame.
    A more algorithm-friendly version that returns NumPy arrays directly.

    Args:
        df: DataFrame containing proximity data
        threshold: Minimum probability for an edge to be included
        save_matrices: Whether to save adjacency matrices to files
        output_dir: Directory for saving matrices

    Returns:
        Tuple of (adjacency_matrices, dates) where:
            - adjacency_matrices is a list of NumPy arrays
            - dates is a list of date strings in ISO format
    """
    # First create the NetworkX graphs
    daily_graphs = create_daily_graphs(df, threshold=threshold)

    # Extract sorted dates
    dates = sorted(daily_graphs.keys())

    # Convert each graph to a NumPy adjacency matrix
    adjacency_matrices = []
    for date in dates:
        adj_matrix = nx.to_numpy_array(daily_graphs[date])
        adjacency_matrices.append(adj_matrix)

    # Save matrices if requested
    if save_matrices and adjacency_matrices:
        os.makedirs(output_dir, exist_ok=True)
        for i, date in enumerate(dates):
            np.savetxt(
                os.path.join(output_dir, f"{date}.csv"),
                adjacency_matrices[i],
                delimiter=",",
            )
        logger.info(
            f"Saved {len(adjacency_matrices)} adjacency matrices to {output_dir}"
        )

    return adjacency_matrices, dates


def extract_network_features(
    adjacency_matrices: List[np.ndarray],
    feature_names: List[str] = None,
    normalize: bool = True,
    save_features: bool = False,
    output_dir: str = "results/mit_reality/features",
) -> Dict[str, Any]:
    """Extract network features from adjacency matrices using NetworkFeatureExtractor.

    This function replicates the feature extraction process from the main algorithm.py
    to ensure compatibility with the change point detection algorithm.

    Args:
        adjacency_matrices: List of adjacency matrices as NumPy arrays
        feature_names: List of feature names to extract. If None, uses a default set
        normalize: Whether to normalize features
        save_features: Whether to save features to CSV
        output_dir: Directory to save feature CSVs

    Returns:
        Dictionary containing extracted features:
          - features_numeric: NumPy array of numeric features (samples x features)
          - features_raw: List of dictionaries with all raw features
          - features_normalized: Normalized features if normalize=True
          - feature_means: Mean values for each feature (for normalization)
          - feature_stds: Standard deviations for each feature (for normalization)
    """
    logger.info(f"Extracting network features from {len(adjacency_matrices)} graphs")

    # Default features if none specified (based on algorithm.yaml defaults)
    if feature_names is None:
        feature_names = [
            "mean_degree",  # Average node degree
            "density",  # Graph density
            "mean_clustering",  # Average clustering coefficient
            "mean_betweenness",  # Average betweenness centrality
            "mean_eigenvector",  # Average eigenvector centrality
            "mean_closeness",  # Average closeness centrality
            "max_singular_value",  # Largest singular value
            "min_nonzero_laplacian",  # Smallest non-zero Laplacian eigenvalue
        ]

    feature_extractor = NetworkFeatureExtractor()
    features_raw = []
    features_numeric = []

    # Process each adjacency matrix
    for adj_matrix in adjacency_matrices:
        # Convert adjacency matrix to graph
        graph = adjacency_to_graph(adj_matrix)

        # Extract features
        raw_features = feature_extractor.get_features(graph)
        numeric_features = feature_extractor.get_numeric_features(graph)

        # Store results
        features_raw.append(raw_features)
        features_numeric.append([numeric_features[name] for name in feature_names])

    # Convert to numpy array
    features_array = np.array(features_numeric)

    # Normalize features if requested
    features_normalized = None
    feature_means = None
    feature_stds = None

    if normalize:
        features_normalized, feature_means, feature_stds = normalize_features(
            features_array
        )
        logger.debug(
            f"Normalized features with means: {feature_means[:3]}{'...' if len(feature_means) > 3 else ''}"
        )
        logger.debug(
            f"Normalized features with std devs: {feature_stds[:3]}{'...' if len(feature_stds) > 3 else ''}"
        )

    # Save features if requested
    if save_features:
        os.makedirs(output_dir, exist_ok=True)

        # Save raw numeric features
        np.savetxt(
            os.path.join(output_dir, "features_numeric.csv"),
            features_array,
            delimiter=",",
            header=",".join(feature_names),
            comments="",
        )

        # Save normalized features if available
        if features_normalized is not None:
            np.savetxt(
                os.path.join(output_dir, "features_normalized.csv"),
                features_normalized,
                delimiter=",",
                header=",".join(feature_names),
                comments="",
            )

            # Save normalization parameters
            np.savetxt(
                os.path.join(output_dir, "feature_means.csv"),
                feature_means,
                delimiter=",",
                header=",".join(feature_names),
                comments="",
            )

            np.savetxt(
                os.path.join(output_dir, "feature_stds.csv"),
                feature_stds,
                delimiter=",",
                header=",".join(feature_names),
                comments="",
            )

        logger.info(f"Saved extracted features to {output_dir}")

    # Return all results
    result = {
        "features_numeric": features_array,
        "features_raw": features_raw,
        "feature_names": feature_names,
    }

    if normalize:
        result.update(
            {
                "features_normalized": features_normalized,
                "feature_means": feature_means,
                "feature_stds": feature_stds,
            }
        )

    return result


def _save_graphs(
    daily_graphs: dict,
    base_output_dir: str = "results/mit_reality",
    save_adjacency: bool = False,
    visualize: bool = True,
) -> None:
    """
    Save graph visualizations and optionally adjacency matrices.

    Args:
        daily_graphs: Dictionary mapping dates to NetworkX graphs
        base_output_dir: Directory to save outputs
        save_adjacency: Whether to save adjacency matrices as CSVs
        visualize: Whether to create visualizations
    """
    if not daily_graphs:
        logger.warning("No graphs to save")
        return

    logger.info(f"Saving graphs to {base_output_dir}")

    # Create output directories
    viz_dir = os.path.join(base_output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    if save_adjacency:
        adj_dir = os.path.join(base_output_dir, "adjacency_matrices")
        os.makedirs(adj_dir, exist_ok=True)

    # Track progress
    viz_count = 0
    adj_count = 0

    # Try to use a fast layout algorithm that can be reused
    base_positions = None

    # Set publication-quality figure parameters
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "figure.figsize": (8, 8),
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        }
    )

    edge_color = "#d3d3d3"  # Light gray
    cmap = plt.cm.viridis

    for date, G in daily_graphs.items():
        # Save adjacency matrix as CSV
        if save_adjacency:
            adj_matrix = nx.to_numpy_array(G)
            adj_path = os.path.join(adj_dir, f"{date}.csv")
            np.savetxt(adj_path, adj_matrix, delimiter=",")
            adj_count += 1

        # Visualize graph if requested and not too large
        if visualize and G.number_of_nodes() <= 100:
            # Create figure and axis objects for more control
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

            # Generate or reuse layout (layouts are computationally expensive)
            if base_positions is None:
                pos = nx.spring_layout(G, seed=42, k=0.3)  # k controls spacing
                base_positions = pos.copy()  # Save for reuse
            else:
                # Reuse previously calculated positions and refine
                pos = nx.spring_layout(
                    G, pos=base_positions, seed=42, iterations=50, k=0.3
                )

            # Calculate node degrees for size and color mapping
            node_degrees = dict(G.degree())
            node_sizes = [30 + 20 * node_degrees[node] for node in G.nodes()]

            # Use degree for color mapping
            node_colors = [node_degrees[node] for node in G.nodes()]

            # Draw network with enhanced visual styling
            # Draw edges first (underneath)
            nx.draw_networkx_edges(
                G, pos, alpha=0.3, edge_color=edge_color, width=0.8, ax=ax
            )

            # Draw nodes with size based on degree and color based on degree
            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=cmap,
                alpha=0.85,
                linewidths=1.0,
                edgecolors="white",
                ax=ax,
            )

            # Create title with date information
            formatted_date = date.replace("-", "/")
            plt.title(f"Proximity Network: {formatted_date}", fontweight="bold", pad=20)

            # Add a colorbar to show degree mapping
            if len(G.nodes()) > 5:  # Only add colorbar if enough nodes
                cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, pad=0.05)
                cbar.set_label("Node Degree", fontsize=12)

            # Add network statistics as text
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            density = nx.density(G)
            stats_text = (
                f"Nodes: {G.number_of_nodes()}\n"
                f"Edges: {G.number_of_edges()}\n"
                f"Avg. Degree: {avg_degree:.2f}\n"
                f"Density: {density:.4f}"
            )
            plt.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
            )

            # Remove axis ticks and labels for cleaner look
            ax.set_axis_off()

            # Add a border around the plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.0)

            # Tight layout and save with high quality
            plt.tight_layout()
            viz_path = os.path.join(viz_dir, f"{date}.png")
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")

            # Save vectorized PDF version for publication
            pdf_path = os.path.join(viz_dir, f"{date}.pdf")
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

            plt.close(fig)
            viz_count += 1

    if visualize:
        logger.info(
            f"Saved {viz_count} publication-quality graph visualizations to {viz_dir}"
        )
    if save_adjacency:
        logger.info(f"Saved {adj_count} adjacency matrices to {adj_dir}")


def create_grid(
    daily_graphs: dict,
    output_dir: str = "results/mit_reality/visualizations",
    num_days: int = 9,
    specific_dates: list = None,
    figsize: tuple = (9, 8),  # Even smaller figure size
):
    """
    Create a compact, well-organized grid visualization of network graphs with LaTeX styling.

    Args:
        daily_graphs: Dictionary mapping dates to NetworkX graphs
        output_dir: Directory to save the grid visualization
        num_days: Number of days to include in the grid (max 9)
        specific_dates: List of specific dates to include (if None, evenly sample days)
        figsize: Size of the figure in inches

    Returns:
        Path to the saved visualization
    """
    if not daily_graphs:
        logger.warning("No graphs available to create grid visualization")
        return None

    # Limit to maximum 9 days (3x3 grid)
    num_days = min(num_days, 9)

    # Sort dates chronologically
    all_dates = sorted(daily_graphs.keys())

    # Select dates to display
    if specific_dates:
        # Filter to only include dates that exist in the daily_graphs
        display_dates = [d for d in specific_dates if d in daily_graphs]
        # Limit to the requested number
        display_dates = display_dates[:num_days]
    else:
        # Evenly sample days across the date range
        if len(all_dates) <= num_days:
            display_dates = all_dates
        else:
            # Calculate indices for evenly spaced sampling
            indices = np.linspace(0, len(all_dates) - 1, num_days, dtype=int)
            display_dates = [all_dates[i] for i in indices]

    # Check if we have enough dates
    if not display_dates:
        logger.warning("No valid dates to display in grid")
        return None

    # Use seaborn-v0_8-paper style as base for LaTeX-like appearance
    try:
        plt.style.use("seaborn-v0_8-paper")
    except:
        try:
            plt.style.use("seaborn-paper")  # Fallback for newer seaborn
        except:
            logger.warning("Seaborn paper style not found, using default style")

    # Create figure and set LaTeX-style parameters
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 8,  # Even smaller base font
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,  # Minimal padding
            "figure.figsize": figsize,
            "axes.grid": True,
            "grid.alpha": 0.15,  # Subtle grid
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Create the figure with tighter spacing
    # We'll use gridspec for more control over the layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3,
        4,  # 3 rows, 4 columns (3 for plots, 1 for colorbar)
        width_ratios=[1, 1, 1, 0.05],  # Last column is narrow for colorbar
        wspace=0.05,  # Very tight spacing between columns
        hspace=0.3,  # Increased vertical spacing between rows for better date separation
    )

    # Create axes for plots (3x3 grid)
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Create axis for colorbar
    cbar_ax = fig.add_subplot(gs[:, 3])  # Spans all rows in the last column

    # Define a professional colormap
    cmap = plt.cm.viridis
    edge_color = "#404040"  # Darker gray for better contrast

    # Calculate positions once for all networks using a reference graph
    # Use the graph with the most edges as reference for better layout
    reference_idx = 0
    max_edges = 0
    for i, date in enumerate(display_dates):
        edges = daily_graphs[date].number_of_edges()
        if edges > max_edges:
            max_edges = edges
            reference_idx = i

    reference_graph = daily_graphs[display_dates[reference_idx]]
    base_positions = nx.spring_layout(reference_graph, seed=42, k=0.25, iterations=100)

    # Find max degree across all graphs for consistent color scaling
    max_degree = 0
    for date in display_dates:
        G = daily_graphs[date]
        if G.number_of_nodes() > 0:
            max_degree = max(max_degree, max(dict(G.degree()).values(), default=0))

    # Create visualizations
    for i, date in enumerate(display_dates):
        if i >= len(axes):
            break

        # Get the current axis
        ax = axes[i]

        # Get the graph for this date
        G = daily_graphs[date]

        # Turn on grid with subtle lines
        ax.grid(True, alpha=0.12, linestyle="-", linewidth=0.2)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.spines["left"].set_linewidth(0.4)

        # Set up position with careful adjustments for consistency
        pos = nx.spring_layout(
            G,
            pos=base_positions,
            seed=42,
            iterations=50,
            k=0.25,
            weight=None,
            scale=0.85,  # Slightly smaller to leave room for labels
        )

        # Calculate node properties based on degree
        node_degrees = dict(G.degree())
        # More compact node sizing with cap for very large nodes
        node_sizes = [8 + 6 * min(node_degrees[node], 15) for node in G.nodes()]
        node_colors = [node_degrees[node] for node in G.nodes()]

        # Draw edges with better contrast
        nx.draw_networkx_edges(
            G, pos, alpha=0.25, edge_color=edge_color, width=0.3, ax=ax
        )

        # Draw nodes with improved visual style
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            vmin=0,
            vmax=max_degree,
            alpha=0.9,
            linewidths=0.2,
            edgecolors="white",
            ax=ax,
        )

        # Format date more cleanly
        formatted_date = date.replace("-", "/")
        ax.set_title(formatted_date, fontsize=9, pad=2)

        # Set axis limits to ensure grid is properly sized and maximizes space
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

        # Remove tick labels to save space
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        # Add compact network statistics in the corner
        avg_degree = sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
        density = nx.density(G)

        # Place stats directly inside the subplot at bottom left
        stats_text = (
            f"$N = {G.number_of_nodes()}$, "
            f"$E = {G.number_of_edges()}$\n"
            f"$\\bar{{d}} = {avg_degree:.1f}$, "
            f"$\\rho = {density:.3f}$"
        )

        # Position the text box inside the subplot
        ax.text(
            0.02,  # Position in bottom-left inside the plot
            0.02,
            stats_text,
            transform=ax.transAxes,  # Use axes coordinates (0-1)
            fontsize=7.5,  # Slightly smaller to fit inside
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#dddddd",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    # Hide any unused axes
    for j in range(len(display_dates), len(axes)):
        axes[j].set_visible(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_degree))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Node Degree", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, pad=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f"network_grid_{timestamp}.png")
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")

    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, f"network_grid_{timestamp}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    plt.close(fig)

    logger.info(
        f"Created grid visualization of {len(display_dates)} days at {grid_path}"
    )
    return grid_path


def process_mit_reality_dataset(
    file_path: str,
    probability_threshold: float = 0.3,
    feature_names: List[str] = None,
    save_results: bool = True,
    output_dir: str = "results/mit_reality",
    visualize: bool = False,
    run_detection: bool = True,
    enable_prediction: bool = True,  # Added prediction parameter
    detection_config: Dict = None,
) -> Dict[str, Any]:
    """Process MIT Reality dataset from raw file to extracted features.

    This function performs the complete pipeline:
    1. Load and clean the dataset
    2. Create daily graphs
    3. Extract features
    4. Optionally run change point detection
    5. Optionally visualize and save results

    Args:
        file_path: Path to MIT Reality Proximity.csv
        probability_threshold: Threshold for edge probability
        feature_names: List of features to extract
        save_results: Whether to save results to disk
        output_dir: Base directory for saving results
        visualize: Whether to generate visualizations
        run_detection: Whether to run change point detection
        enable_prediction: Whether to use prediction for enhanced detection
        detection_config: Optional configuration for detection

    Returns:
        Dictionary with processing results
    """
    # Create output directories
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and clean dataset
    logger.info(f"Processing MIT Reality dataset from {file_path}")
    df = read_proximity_csv(file_path, save_cleaned=save_results)

    # Step 2: Create daily graphs
    daily_graphs = create_daily_graphs(
        df,
        threshold=probability_threshold,
        save_graphs=visualize and save_results,
        output_dir=os.path.join(output_dir, "daily_graphs"),
    )

    # Step 3: Create adjacency matrices
    adj_matrices, dates = create_daily_adjacency_matrices(
        df,
        threshold=probability_threshold,
        save_matrices=save_results,
        output_dir=os.path.join(output_dir, "adjacency_matrices"),
    )

    # Step 4: Extract features
    features = extract_network_features(
        adj_matrices,
        feature_names=feature_names,
        normalize=True,
        save_features=save_results,
        output_dir=os.path.join(output_dir, "features"),
    )

    # Define known periods/events based on Reality Mining research
    # These are potential change points to evaluate against
    known_events = {
        # From Reality Mining research, days are mapped to dates
        "NewYear": {
            "start_idx": 81,
            "end_idx": 141,
            "description": "New Year holiday period (potential structural changes)",
            "date_around": "2009-01-01",  # Approximate date of interest
        },
        "ColumbusDay": {
            "start_idx": 1,
            "end_idx": 60,
            "description": "Columbus Day / Fall break (potential structural changes)",
            "date_around": "2008-10-12",  # Approximate date of interest
        },
        # Add Thanksgiving, Christmas, Spring Break, End of Semester
        "Thanksgiving": {
            "date_around": "2008-11-26",
            "description": "Thanksgiving holiday (potential structural changes)",
        },
        "Christmas": {
            "date_around": "2008-12-25",
            "description": "Christmas holiday (potential structural changes)",
        },
        "SpringBreak": {
            "date_around": "2009-03-15",
            "description": "Spring break (potential structural changes)",
        },
        "EndOfSemester": {
            "date_around": "2009-05-15",
            "description": "End of spring semester (potential structural changes)",
        },
    }

    # Map dates to indices
    date_to_idx = {date: i for i, date in enumerate(dates)}

    # Find the closest date for each event
    for event_name, event_info in known_events.items():
        if "date_around" in event_info:
            target_date = event_info["date_around"]
            # Find the closest date in our dataset
            closest_date = min(
                dates,
                key=lambda x: abs(pd.to_datetime(x) - pd.to_datetime(target_date)),
            )
            # Store the index
            known_events[event_name]["closest_idx"] = date_to_idx[closest_date]
            known_events[event_name]["closest_date"] = closest_date
            logger.info(
                f"Event {event_name} mapped to date {closest_date} (index {date_to_idx[closest_date]})"
            )

    # Create a list of potential change points for evaluation
    potential_change_points = [
        event_info.get("closest_idx")
        for event_name, event_info in known_events.items()
        if "closest_idx" in event_info
    ]
    potential_change_points = sorted(
        list(set(potential_change_points))
    )  # Remove duplicates

    logger.info(
        f"Identified {len(potential_change_points)} potential change points for evaluation"
    )
    logger.info(f"Potential change points at indices: {potential_change_points}")
    logger.info(f"Corresponding dates: {[dates[cp] for cp in potential_change_points]}")

    # Step 5: Run change point detection if requested
    detection_results = None
    if run_detection:
        logger.info("Running change point detection")
        detection_dir = os.path.join(output_dir, "detection")

        # Create a complete configuration by combining default with user-provided
        # This ensures we get all the necessary parameters for proper detection
        n_timesteps = len(adj_matrices)
        default_config = {
            "trials": {
                "n_trials": 3,
                "random_seeds": 42,
            },
            "detection": {
                # From Reality Mining config: lower threshold for more sensitive detection
                "threshold": 20.0,  # Using 20.0 instead of 60.0 for more sensitive detection
                "batch_size": 1000,
                "reset": True,
                "reset_on_traditional": True,
                "max_window": None,
                "prediction_horizon": 5,
                "sequence": {
                    "length": n_timesteps,
                },
                "betting_func_config": {
                    "name": "mixture",
                    "power": {"epsilon": 0.7},
                    "exponential": {"lambd": 1.0},
                    "mixture": {"epsilons": [0.7, 0.8, 0.9]},
                    "constant": {},
                    "beta": {"a": 0.5, "b": 1.5},
                    "kernel": {"bandwidth": 0.1, "past_pvalues": []},
                },
                "distance": {
                    "measure": "mahalanobis",
                    "p": 2.0,  # Make sure p is included here
                },
            },
            "model": {
                "type": "multiview",
                "network": "sbm",  # Added for compatibility
                "predictor": {
                    "type": "graph",  # Default graph predictor
                    "config": {
                        "n_history": 5,
                        "alpha": 0.8,
                        "gamma": 0.5,
                        "beta_init": 0.5,
                        "enforce_connectivity": True,
                        "adaptive": True,
                    },
                },
            },
            "features": features["feature_names"],
            "params": {
                "seq_len": n_timesteps,
            },
            # Add known change points for evaluation
            "evaluation": {
                "true_change_points": potential_change_points,
                "events": known_events,
            },
            "execution": {
                "enable_prediction": enable_prediction,
                "save_csv": save_results,
            },
        }

        # Merge configurations recursively
        final_config = default_config.copy()
        if detection_config:
            # Recursively merge nested dictionaries
            def deep_merge(d1, d2):
                """Recursively merge dictionaries."""
                for key, value in d2.items():
                    if (
                        key in d1
                        and isinstance(d1[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(d1[key], value)
                    else:
                        d1[key] = value
                return d1

            # Apply the recursive merge
            final_config = deep_merge(final_config, detection_config)

        detection_results = run_change_detection(
            features["features_numeric"],
            features["features_normalized"],
            features["feature_names"],
            adj_matrices,
            dates,
            config=final_config,  # Use the merged config
            output_dir=detection_dir,
            save_csv=save_results,
            enable_prediction=enable_prediction,  # Pass the prediction flag
        )

        # Add known events information to results
        detection_results["known_events"] = known_events
        detection_results["potential_change_points"] = potential_change_points

        # Evaluate detected change points against known events
        if (
            "change_points" in detection_results
            and "traditional" in detection_results["change_points"]
        ):
            detected_cps = detection_results["change_points"]["traditional"]
            evaluation_results = evaluate_change_points(
                detected_cps, potential_change_points, tolerance=10
            )
            detection_results["evaluation"] = evaluation_results

            # Log evaluation results
            logger.info(f"Evaluation results: {evaluation_results}")

            # Map detected change points to events
            cp_to_events = map_change_points_to_events(
                detected_cps, known_events, dates, tolerance=10
            )
            detection_results["change_points_to_events"] = cp_to_events

            # Log mapped events
            for cp, events in cp_to_events.items():
                event_names = [e["name"] for e in events]
                logger.info(
                    f"Change point at {cp} (date: {dates[cp]}) might be related to events: {event_names}"
                )

            # Evaluate horizon change points if available
            if "horizon" in detection_results["change_points"]:
                horizon_cps = detection_results["change_points"]["horizon"]
                horizon_evaluation = evaluate_change_points(
                    horizon_cps, potential_change_points, tolerance=10
                )
                detection_results["horizon_evaluation"] = horizon_evaluation

                # Map horizon change points to events
                horizon_cp_to_events = map_change_points_to_events(
                    horizon_cps, known_events, dates, tolerance=10
                )
                detection_results["horizon_change_points_to_events"] = (
                    horizon_cp_to_events
                )

                # Compare traditional vs horizon detection
                if horizon_cps and detected_cps:
                    # For each actual event, compute the detection latency difference
                    latency_comparison = {}
                    for cp in potential_change_points:
                        # Find closest traditional CP
                        closest_trad = (
                            min(detected_cps, key=lambda x: abs(x - cp))
                            if detected_cps
                            else None
                        )
                        trad_latency = (
                            closest_trad - cp
                            if closest_trad and closest_trad >= cp
                            else None
                        )

                        # Find closest horizon CP
                        closest_horizon = (
                            min(horizon_cps, key=lambda x: abs(x - cp))
                            if horizon_cps
                            else None
                        )
                        horizon_latency = (
                            closest_horizon - cp
                            if closest_horizon and closest_horizon >= cp
                            else None
                        )

                        if trad_latency is not None and horizon_latency is not None:
                            latency_diff = trad_latency - horizon_latency
                            latency_comparison[cp] = {
                                "traditional_latency": trad_latency,
                                "horizon_latency": horizon_latency,
                                "latency_difference": latency_diff,
                                "date": dates[cp],
                                "improvement": "Yes" if latency_diff > 0 else "No",
                            }

                    if latency_comparison:
                        detection_results["latency_comparison"] = latency_comparison
                        # Log latency comparison
                        logger.info("Latency comparison (traditional vs horizon):")
                        for cp, comparison in latency_comparison.items():
                            logger.info(
                                f"  Event at {comparison['date']}: "
                                f"Traditional={comparison['traditional_latency']} days, "
                                f"Horizon={comparison['horizon_latency']} days, "
                                f"Improvement={comparison['improvement']}"
                            )

        logger.info("Change point detection completed")

    # Step 6: Generate grid visualization if requested
    if visualize and save_results:
        grid_path = create_grid(
            daily_graphs,
            output_dir=os.path.join(output_dir, "visualizations"),
            num_days=9,
        )

        if grid_path:
            logger.info(f"Created grid visualization at {grid_path}")

    # Return complete results
    results = {
        "adjacency_matrices": adj_matrices,
        "dates": dates,
        "features": features,
        "known_events": known_events,
        "potential_change_points": potential_change_points,
    }

    if visualize:
        results["daily_graphs"] = daily_graphs

    if detection_results:
        results["detection"] = detection_results

    logger.info(f"MIT Reality dataset processing complete: {len(dates)} days analyzed")
    return results


def evaluate_change_points(detected_cps, true_cps, tolerance=10):
    """Evaluate detected change points against known/potential change points.

    Args:
        detected_cps: List of detected change point indices
        true_cps: List of known/potential change point indices
        tolerance: Number of timesteps to consider a match

    Returns:
        Dict with evaluation metrics
    """
    if not true_cps:
        return {"precision": 0, "recall": 0, "f1": 0, "matches": []}

    matches = []
    for cp in detected_cps:
        # Find the closest true change point
        closest_true_cp = min(true_cps, key=lambda x: abs(x - cp))
        distance = abs(closest_true_cp - cp)

        # Consider it a match if within tolerance
        if distance <= tolerance:
            matches.append((cp, closest_true_cp, distance))

    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(detected_cps) - true_positives
    false_negatives = len(true_cps) - true_positives

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def map_change_points_to_events(change_points, known_events, dates, tolerance=10):
    """Map detected change points to known events.

    Args:
        change_points: List of detected change point indices
        known_events: Dictionary of known events with their information
        dates: List of dates corresponding to each timestep
        tolerance: Number of timesteps to consider a match

    Returns:
        Dict mapping each change point to relevant events
    """
    cp_to_events = {}

    for cp in change_points:
        cp_to_events[cp] = []
        cp_date = pd.to_datetime(dates[cp])

        for event_name, event_info in known_events.items():
            if "closest_idx" in event_info:
                event_idx = event_info["closest_idx"]
                distance = abs(event_idx - cp)

                if distance <= tolerance:
                    cp_to_events[cp].append(
                        {
                            "name": event_name,
                            "distance": distance,
                            "description": event_info.get("description", ""),
                            "date": dates[event_idx],
                        }
                    )

    return cp_to_events


def run_change_detection(
    features_numeric,
    features_normalized,
    feature_names,
    adjacency_matrices,
    dates,
    config=None,
    output_dir="results/mit_reality/detection",
    n_trials=3,  # Increased to 3 trials by default for better aggregation
    random_seed=42,
    detection_threshold=20.0,  # Changed from 60.0 to 20.0 (from Reality Mining config)
    betting_function="mixture",
    distance_measure="mahalanobis",
    enable_prediction=True,  # Changed to True to enable prediction
    save_csv=True,
):
    """Run change point detection on extracted features.

    Args:
        features_numeric: Raw numeric features
        features_normalized: Normalized features
        feature_names: Names of features
        adjacency_matrices: List of adjacency matrices
        dates: List of dates corresponding to each graph
        config: Optional config dict (will use default if None)
        output_dir: Output directory for results
        n_trials: Number of detection trials to run
        random_seed: Random seed for reproducibility
        detection_threshold: Detection threshold (20.0 from Reality Mining config)
        betting_function: Betting function to use
        distance_measure: Distance measure for detection
        enable_prediction: Whether to use prediction
        save_csv: Whether to save results to CSV

    Returns:
        Dict with detection results
    """
    logger.info("Running change point detection")

    # Get the actual number of timesteps
    n_timesteps = len(adjacency_matrices)
    logger.info(f"Working with {n_timesteps} timesteps of data")

    # Extract potential_change_points and known_events from config if available
    potential_change_points = (
        config.get("evaluation", {}).get("true_change_points", []) if config else []
    )
    known_events = config.get("evaluation", {}).get("events", {}) if config else {}

    # Create default config if none provided
    if config is None:
        config = {
            "trials": {
                "n_trials": n_trials,
                "random_seeds": random_seed,
            },
            "detection": {
                "threshold": detection_threshold,  # Using threshold from Reality Mining
                "batch_size": 1000,
                "reset": True,
                "reset_on_traditional": True,
                "max_window": None,
                "prediction_horizon": 5,
                "sequence": {
                    "length": n_timesteps,
                },
                "betting_func_config": {
                    "name": betting_function,
                    "power": {"epsilon": 0.7},
                    "exponential": {"lambd": 1.0},
                    "mixture": {"epsilons": [0.7, 0.8, 0.9]},
                    "constant": {},
                    "beta": {"a": 0.5, "b": 1.5},
                    "kernel": {"bandwidth": 0.1, "past_pvalues": []},
                },
                "distance": {
                    "measure": distance_measure,
                    "p": 2.0,  # Make sure p is included here
                },
            },
            "model": {
                "type": "multiview",
                "network": "sbm",  # Placeholder for compatibility
                "predictor": {
                    "type": "graph",  # Default graph predictor
                    "config": {
                        "n_history": 5,
                        "alpha": 0.8,
                        "gamma": 0.5,
                        "omega": None,
                        "beta_init": 0.5,
                        "enforce_connectivity": True,
                        "adaptive": True,
                        "optimization_iterations": 3,
                        "threshold": 0.5,
                    },
                },
            },
            "features": feature_names,
            "output": {
                "directory": output_dir,
                "prefix": "",
                "save_predictions": True,
                "save_features": True,
                "save_martingales": True,
                "visualization": {
                    "enabled": True,
                    "skip_shap": False,
                },
            },
            "execution": {
                "enable_prediction": enable_prediction,
                "save_csv": save_csv,
            },
            "params": {
                "seq_len": n_timesteps,
            },
            "evaluation": {
                "true_change_points": potential_change_points,
                "events": known_events,
            },
        }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize predictor
    predictor = None
    predicted_features = None
    predicted_normalized = None

    if enable_prediction:
        try:
            logger.info("Initializing graph predictor")
            predictor_config = config["model"]["predictor"]
            predictor = PredictorFactory.create(
                predictor_config["type"],
                predictor_config["config"],
            )

            # Generate predictions
            logger.info("Generating graph predictions")
            predicted_graphs = generate_predictions(
                adjacency_matrices, predictor, config
            )

            if predicted_graphs:
                # Process predictions into features
                logger.info("Processing predictions into features")
                predicted_features = process_predictions(
                    predicted_graphs, feature_names
                )

                # Normalize predicted features using the same parameters as original features
                _, feature_means, feature_stds = normalize_features(features_numeric)
                predicted_normalized = normalize_predictions(
                    predicted_features, feature_means, feature_stds
                )
                logger.info(
                    f"Generated predictions for {len(predicted_normalized)} timesteps"
                )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.info("Continuing without prediction")
            predicted_normalized = None
    else:
        logger.info("Prediction disabled")

    # Determine number of trials and seeds
    n_trials = config["trials"]["n_trials"]
    base_seed = config["trials"]["random_seeds"]

    # Handle random seeds
    if base_seed is None:
        # Generate completely random seeds
        random_seeds = np.random.randint(0, 2**31 - 1, size=n_trials)
    elif isinstance(base_seed, (int, float)):
        # Generate deterministic sequence of seeds from base seed
        rng = np.random.RandomState(int(base_seed))
        random_seeds = rng.randint(0, 2**31 - 1, size=n_trials)
    else:
        # Use provided list of seeds
        random_seeds = np.array(base_seed)

    logger.info(f"Running {n_trials} detection trials with varying algorithm seeds")

    # Run individual trials
    individual_results = []
    for trial_idx, seed in enumerate(random_seeds):
        if trial_idx >= n_trials:
            break

        # Convert seed to integer if needed
        int_seed = int(seed) if seed is not None else None

        logger.info(f"Running trial {trial_idx + 1}/{n_trials} with seed {int_seed}")

        # Create different seeds for various components from the main seed
        detector_seed = int_seed
        strangeness_seed = (int_seed + 1) % (2**31 - 1)
        pvalue_seed = (int_seed + 2) % (2**31 - 1)
        betting_seed = (int_seed + 3) % (2**31 - 1)

        # Get betting function config
        betting_func_name = config["detection"]["betting_func_config"]["name"]
        betting_func_params = config["detection"]["betting_func_config"].get(
            betting_func_name, {}
        )

        # Create proper BettingFunctionConfig
        betting_func_config = BettingFunctionConfig(
            name=betting_func_name,
            params=betting_func_params,
            random_seed=betting_seed,
        )

        # Create detector config
        detector_config = DetectorConfig(
            method=config["model"]["type"],
            threshold=config["detection"]["threshold"],
            history_size=config["model"]["predictor"]["config"]["n_history"],
            batch_size=config["detection"]["batch_size"],
            reset=config["detection"]["reset"],
            reset_on_traditional=config["detection"].get("reset_on_traditional", False),
            max_window=config["detection"]["max_window"],
            betting_func_config=betting_func_config,
            distance_measure=config["detection"]["distance"]["measure"],
            distance_p=config["detection"]["distance"]["p"],
            random_state=detector_seed,
            strangeness_seed=strangeness_seed,
            pvalue_seed=pvalue_seed,
        )

        # Initialize detector
        detector = ChangePointDetector(detector_config)

        try:
            # Run detection with predictions if available
            detection_result = detector.run(
                data=features_normalized,
                predicted_data=predicted_normalized,  # Pass predictions if available
                reset_state=True,
            )

            if detection_result is None:
                logger.warning(
                    f"Trial {trial_idx + 1}/{n_trials} failed: No detection result"
                )
                continue

            individual_results.append(detection_result)

            # Log key results from this trial
            if "traditional_change_points" in detection_result:
                trad_cp = detection_result.get("traditional_change_points", [])
                horizon_cp = detection_result.get("horizon_change_points", [])
                logger.info(
                    f"Trial {trial_idx + 1} results: Traditional CPs: {trad_cp}"
                )

                # Map to dates for better understanding
                if trad_cp:
                    date_cps = [dates[cp] for cp in trad_cp if cp < len(dates)]
                    logger.info(f"Change points detected on dates: {date_cps}")

                # Log horizon change points (from prediction) if available
                if horizon_cp:
                    horizon_date_cps = [
                        dates[cp] for cp in horizon_cp if cp < len(dates)
                    ]
                    logger.info(
                        f"Horizon CPs: {horizon_cp} (dates: {horizon_date_cps})"
                    )

        except Exception as e:
            logger.error(f"Trial {trial_idx + 1}/{n_trials} failed: {str(e)}")
            continue

    if not individual_results:
        raise RuntimeError("All detection trials failed")

    logger.info(f"Completed {len(individual_results)}/{n_trials} trials successfully")

    # Use the first trial's results for visualization
    aggregated_results = individual_results[0].copy()

    # Combine results - match algorithm.py structure exactly
    trial_results = {
        "individual_trials": individual_results,
        "aggregated": aggregated_results,
        "random_seeds": random_seeds.tolist(),
    }

    # Create a sequence result for compatibility with prepare_result_data
    sequence_result = {
        "graphs": adjacency_matrices,
        "change_points": config.get("evaluation", {}).get(
            "true_change_points", []
        ),  # Use potential change points from config
        "dates": dates,
    }

    # Prepare full results data using the same approach as algorithm.py
    results = prepare_result_data(
        sequence_result,
        features_numeric,
        None,  # Skip features_raw
        (
            predicted_graphs if enable_prediction else None
        ),  # Include predicted graphs if available
        trial_results,
        predictor,  # Include predictor
        config,
    )

    # Add dates to results for reference
    results["dates"] = dates

    # Export to CSV if requested - match algorithm.py approach exactly
    if save_csv and aggregated_results:
        try:
            # Define CSV output directory
            csv_output_dir = os.path.join(output_dir, "csv")
            os.makedirs(csv_output_dir, exist_ok=True)

            # Initialize output manager
            output_manager = OutputManager(csv_output_dir, config)

            # Important: Use the same structure as algorithm.py for export
            output_manager.export_to_csv(
                trial_results["aggregated"],  # Use the same structure as algorithm.py
                sequence_result[
                    "change_points"
                ],  # Pass potential change points as true change points for evaluation
                individual_trials=trial_results["individual_trials"],
            )

            logger.info(f"Exported detection results to {csv_output_dir}")

        except Exception as e:
            logger.error(f"Failed to export results to CSV: {str(e)}")

    # Map change points to dates for easier interpretation
    if "change_points" in results and "traditional" in results["change_points"]:
        trad_cp = results["change_points"]["traditional"]
        date_cps = [dates[cp] for cp in trad_cp if cp < len(dates)]
        results["change_points"]["traditional_dates"] = date_cps

        logger.info(f"Final detected change points: {trad_cp}")
        logger.info(f"Corresponding dates: {date_cps}")

        # Map horizon change points if available
        if "horizon" in results["change_points"]:
            horizon_cp = results["change_points"]["horizon"]
            horizon_date_cps = [dates[cp] for cp in horizon_cp if cp < len(dates)]
            results["change_points"]["horizon_dates"] = horizon_date_cps

            logger.info(f"Final horizon change points: {horizon_cp}")
            logger.info(f"Corresponding dates: {horizon_date_cps}")

    return results


def generate_predictions(graphs, predictor, config):
    """Generate predictions for the graph sequence.

    Args:
        graphs: List of adjacency matrices
        predictor: Initialized predictor instance
        config: Configuration dictionary

    Returns:
        List of predicted future adjacency matrices
    """
    predicted_graphs = []
    horizon = config["detection"]["prediction_horizon"]
    history_size = config["model"]["predictor"]["config"]["n_history"]

    for t in range(len(graphs)):
        history_start = max(0, t - history_size)
        history = [{"adjacency": g} for g in graphs[history_start:t]]

        if t >= history_size:
            try:
                predictions = predictor.predict(history, horizon=horizon)
                predicted_graphs.append(predictions)
            except Exception as e:
                logger.warning(f"Prediction failed at timestep {t}: {str(e)}")
                # If prediction fails, use the last graph repeated
                if t > 0:
                    last_graph = graphs[t - 1]
                    predicted_graphs.append([last_graph] * horizon)

    return predicted_graphs


def process_predictions(predicted_graphs, feature_names):
    """Process predictions into feature space.

    Args:
        predicted_graphs: List of predicted adjacency matrices
        feature_names: Names of features to extract

    Returns:
        numpy array of predicted features
    """
    feature_extractor = NetworkFeatureExtractor()
    predicted_features = []

    for predictions in predicted_graphs:
        timestep_features = []
        for pred_adj in predictions:
            graph = adjacency_to_graph(pred_adj)
            numeric_features = feature_extractor.get_numeric_features(graph)
            feature_vector = [numeric_features[name] for name in feature_names]
            timestep_features.append(feature_vector)
        predicted_features.append(timestep_features)

    predicted_array = np.array(predicted_features)
    return predicted_array


def main():
    file_path = "archive/data/Proximity.csv"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Process the MIT Reality dataset with the integrated pipeline
    results = process_mit_reality_dataset(
        file_path,
        probability_threshold=0.3,
        save_results=True,
        visualize=False,
        run_detection=True,
        enable_prediction=True,  # Enable prediction
        detection_config={
            "trials": {
                "n_trials": 5,  # Run 3 trials for proper aggregation
                "random_seeds": 42,  # Use consistent seed for reproducibility
            },
            "detection": {
                "threshold": 20.0,  # Lower threshold from Reality Mining config (was 60.0)
                "batch_size": 1000,
                "reset": True,
                "reset_on_traditional": True,
                "betting_func_config": {
                    "name": "mixture",  # Use mixture betting function
                    "mixture": {
                        "epsilons": [0.7, 0.8, 0.9],  # Default epsilons
                    },
                },
                "distance": {
                    "measure": "mahalanobis",  # Use Mahalanobis distance
                    "p": 2.0,
                },
            },
            "model": {
                "type": "multiview",
                "predictor": {
                    "type": "graph",  # Default graph predictor
                    "config": {
                        "n_history": 5,  # Smaller history window
                        "alpha": 0.8,
                        "gamma": 0.5,
                        "beta_init": 0.5,
                        "enforce_connectivity": True,
                        "adaptive": True,
                    },
                },
            },
            "execution": {
                "enable_prediction": True,
            },
        },
    )

    logger.info(f"Processed {len(results['dates'])} days of data")
    logger.info(
        f"Extracted {results['features']['features_numeric'].shape[1]} features"
    )

    # Print detection results if available
    if "detection" in results and "change_points" in results["detection"]:
        if "traditional_dates" in results["detection"]["change_points"]:
            trad_cps = results["detection"]["change_points"]["traditional_dates"]
            logger.info(f"Detected traditional change points on dates: {trad_cps}")

        if "horizon_dates" in results["detection"]["change_points"]:
            horizon_cps = results["detection"]["change_points"]["horizon_dates"]
            logger.info(f"Detected horizon change points on dates: {horizon_cps}")

        # Print evaluation results if available
        if "evaluation" in results["detection"]:
            eval_results = results["detection"]["evaluation"]
            logger.info(f"Traditional CP Evaluation:")
            logger.info(f"  Precision: {eval_results['precision']:.2f}")
            logger.info(f"  Recall: {eval_results['recall']:.2f}")
            logger.info(f"  F1 Score: {eval_results['f1']:.2f}")

        if "horizon_evaluation" in results["detection"]:
            horizon_eval = results["detection"]["horizon_evaluation"]
            logger.info(f"Horizon CP Evaluation:")
            logger.info(f"  Precision: {horizon_eval['precision']:.2f}")
            logger.info(f"  Recall: {horizon_eval['recall']:.2f}")
            logger.info(f"  F1 Score: {horizon_eval['f1']:.2f}")

        # Print latency comparison if available
        if "latency_comparison" in results["detection"]:
            logger.info("Latency Comparison Summary:")
            improvements = [
                c
                for c in results["detection"]["latency_comparison"].values()
                if c["improvement"] == "Yes"
            ]
            logger.info(
                f"  Events with improved latency: {len(improvements)}/{len(results['detection']['latency_comparison'])}"
            )

        # Print mapped events
        if "change_points_to_events" in results["detection"]:
            logger.info("Change points mapped to events:")
            for cp, events in results["detection"]["change_points_to_events"].items():
                cp_date = results["dates"][cp]
                event_names = [e["name"] for e in events] if events else ["Unknown"]
                logger.info(f"  CP {cp} ({cp_date}): {', '.join(event_names)}")


if __name__ == "__main__":
    main()
