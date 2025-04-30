#!/usr/bin/env python
"""Martingale Detection on MIT Reality Dataset"""

import pandas as pd
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import json

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


def main():
    file_path = "archive/data/Proximity.csv"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Read and clean proximity file, saving the cleaned version
    df = read_proximity_csv(file_path, save_cleaned=True)

    # Create daily graphs without saving by default
    daily_graphs = create_daily_graphs(df, save_graphs=False, save_adjacency=False)
    logger.info(f"Generated {len(daily_graphs)} daily graphs")

    # Create a grid visualization of selected days
    if daily_graphs:
        create_grid(daily_graphs, num_days=9)


if __name__ == "__main__":
    main()
