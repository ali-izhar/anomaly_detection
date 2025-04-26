# src/utils/plot_dist_bettings.py

"""
Script to create comparative plots for different combinations of
betting functions and distance measures from algorithm results.
"""

import os
import sys
import glob
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_combination_results(combinations_dir):
    """
    Load results from all combination directories.

    Args:
        combinations_dir: Path to the directory containing all combinations

    Returns:
        Dictionary of results data frames and metadata
    """
    logger.info(f"Loading results from {combinations_dir}")

    # Find all combination directories
    combination_dirs = glob.glob(os.path.join(combinations_dir, "*_*"))

    # Filter out any directories that don't follow the betting_distance naming pattern
    combination_dirs = [
        d for d in combination_dirs if os.path.basename(d).count("_") == 1
    ]

    if not combination_dirs:
        logger.error(f"No combination directories found in {combinations_dir}")
        return None

    # Prepare data structure for results
    results = {
        "martingales": {},
        "change_points": {},
        "detection_frequencies": {},
        "metadata": {},
        "configs": {},
        "detection_summaries": {},
    }

    # Initialize true change points if we need to extract them manually
    true_cp_positions = None

    # Check for shared_data directory that might contain true change points
    shared_data_dir = os.path.join(combinations_dir, "shared_data")
    if os.path.exists(shared_data_dir):
        logger.info(f"Found shared_data directory: {shared_data_dir}")

        # Try to load graph_sequence.pkl which contains true change points
        graph_sequence_path = os.path.join(shared_data_dir, "graph_sequence.pkl")
        if os.path.exists(graph_sequence_path):
            try:
                import pickle

                with open(graph_sequence_path, "rb") as f:
                    shared_data = pickle.load(f)
                    if "true_change_points" in shared_data:
                        true_cp_positions = shared_data["true_change_points"]
                        logger.info(
                            f"Loaded true change points from shared data: {true_cp_positions}"
                        )
                        # Create a DataFrame for the true change points
                        results["change_points"]["true"] = pd.DataFrame(
                            {"Position": true_cp_positions}
                        )
            except Exception as e:
                logger.warning(f"Could not load shared graph sequence data: {str(e)}")

    # Load results from each combination directory
    for comb_dir in combination_dirs:
        dir_basename = os.path.basename(comb_dir)
        logger.info(f"Processing combination: {dir_basename}")

        # Extract betting function and distance from directory name
        match = re.match(r"([^_]+)_([^_]+)", dir_basename)
        if not match:
            logger.warning(
                f"Could not parse combination from directory name: {dir_basename}"
            )
            continue

        betting_func, distance = match.groups()
        combination_key = f"{betting_func}_{distance}"

        # Load configuration
        config_path = os.path.join(comb_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                results["configs"][combination_key] = yaml.safe_load(f)

        # Find the Excel results file directly in the combination directory
        excel_files = glob.glob(os.path.join(comb_dir, "detection_results.xlsx"))

        if not excel_files:
            logger.warning(f"No detection_results.xlsx found in {comb_dir}")
            continue

        excel_path = excel_files[0]
        logger.info(f"Found Excel results file: {excel_path}")

        try:
            # Load the Excel file and extract data from different sheets
            xls = pd.ExcelFile(excel_path)
            sheet_names = xls.sheet_names
            logger.info(f"Excel file contains sheets: {sheet_names}")

            # Load data from each sheet based on the actual sheet names
            if "Trial1" in sheet_names:
                # Trial1 typically contains martingale values over time
                df = pd.read_excel(excel_path, sheet_name="Trial1")

                # Check if this has Time/Step and martingale value columns
                if len(df.columns) >= 2:
                    results["martingales"][combination_key] = df
                    logger.info(
                        f"Loaded martingale data for {combination_key} from Trial1 sheet"
                    )

                    # If true change points were in the data, add them to the dataframe
                    if (
                        true_cp_positions is not None
                        and "true_change_point" not in df.columns
                    ):
                        df["true_change_point"] = 0
                        for cp in true_cp_positions:
                            if cp < len(df):
                                df.loc[cp, "true_change_point"] = 1
                        logger.info(
                            f"Added true change points from shared data to martingale data"
                        )

            if "Detection Summary" in sheet_names:
                # Detection Summary would contain detection rate, false positives, etc.
                df = pd.read_excel(excel_path, sheet_name="Detection Summary")
                results["detection_summaries"][combination_key] = df
                logger.info(f"Loaded detection summary for {combination_key}")

                # Try to find info about detected change points in this sheet
                # Look for columns that might contain change point positions
                cp_cols = [
                    col
                    for col in df.columns
                    if any(
                        term in col.lower()
                        for term in ["change", "position", "detection", "cp"]
                    )
                ]

                if cp_cols:
                    # Extract detected change points
                    cp_df = df[cp_cols].copy()
                    results["change_points"][combination_key] = cp_df
                    logger.info(
                        f"Extracted change points for {combination_key} from Detection Summary"
                    )

                    # If we find a column with "true" and "change point" or similar, use as true change points
                    true_cp_cols = [col for col in cp_cols if "true" in col.lower()]
                    if (
                        true_cp_cols
                        and "true" not in results["change_points"]
                        and true_cp_positions is None
                    ):
                        true_cp_df = df[true_cp_cols].copy()
                        results["change_points"]["true"] = true_cp_df
                        logger.info(
                            f"Extracted true change points from Detection Summary"
                        )

            if "Detection Details" in sheet_names:
                # Detection Details would contain more granular info about each detection
                df = pd.read_excel(excel_path, sheet_name="Detection Details")

                # This could contain detection frequencies or detailed change point info
                # Check for frequency-like columns
                freq_cols = [
                    col
                    for col in df.columns
                    if any(
                        term in col.lower() for term in ["freq", "count", "occurrence"]
                    )
                ]

                if freq_cols and len(df) > 0:
                    # If we have position/time and frequency columns, this is likely detection frequency data
                    pos_cols = [
                        col
                        for col in df.columns
                        if any(
                            term in col.lower()
                            for term in ["position", "time", "step", "timestep"]
                        )
                    ]

                    if pos_cols:
                        # Create a dataframe with position and frequency columns
                        freq_df = df[[pos_cols[0], freq_cols[0]]].copy()
                        freq_df.columns = [
                            "Position",
                            "Frequency",
                        ]  # Standardize column names
                        results["detection_frequencies"][combination_key] = freq_df
                        logger.info(
                            f"Extracted detection frequencies for {combination_key} from Detection Details"
                        )

        except Exception as e:
            logger.error(f"Error processing Excel file {excel_path}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

    # If we haven't found any martingale data yet, try to infer it from sheet data
    if not results["martingales"]:
        logger.warning(
            "No martingale data found in expected sheets, trying to infer from available data"
        )

        # For each Excel file, try to find a sheet with time series data that could be martingales
        for comb_dir in combination_dirs:
            dir_basename = os.path.basename(comb_dir)
            match = re.match(r"([^_]+)_([^_]+)", dir_basename)
            if not match:
                continue

            betting_func, distance = match.groups()
            combination_key = f"{betting_func}_{distance}"

            result_dirs = glob.glob(os.path.join(comb_dir, "*"))
            for result_dir in result_dirs:
                if not os.path.isdir(result_dir):
                    continue

                excel_files = glob.glob(
                    os.path.join(result_dir, "detection_results.xlsx")
                )
                if not excel_files:
                    continue

                excel_path = excel_files[0]

                try:
                    # Try each sheet
                    xls = pd.ExcelFile(excel_path)
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(excel_path, sheet_name=sheet_name)

                        # If the dataframe has at least 2 columns and more than 10 rows,
                        # and the first column is numeric (could be a timestep),
                        # it might be martingale data
                        if (
                            len(df.columns) >= 2
                            and len(df) > 10
                            and pd.api.types.is_numeric_dtype(df.iloc[:, 0])
                        ):
                            # Check if second column is also numeric (could be martingale values)
                            if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                                results["martingales"][combination_key] = df
                                logger.info(
                                    f"Inferred martingale data for {combination_key} from sheet {sheet_name}"
                                )
                                break
                except Exception as e:
                    logger.error(
                        f"Error trying to infer martingale data from {excel_path}: {str(e)}"
                    )

    # Check if we need to manually create change points from config
    if "true" not in results["change_points"]:
        logger.warning(
            "No true change points found in data, trying to extract from configuration"
        )

        # Try to get change points from the first config that might have them
        for combination, config in results["configs"].items():
            if "model" in config and "network" in config["model"]:
                network_params = config.get("network_params", {})
                if "change_points" in network_params:
                    true_cps = network_params["change_points"]
                    results["change_points"]["true"] = pd.DataFrame(
                        {"Position": true_cps}
                    )
                    logger.info(
                        f"Extracted true change points from configuration: {true_cps}"
                    )
                    break

    # If we still don't have change points, we may need to manually set them based on domain knowledge
    if "true" not in results["change_points"]:
        # This would be a fallback if we know the true change points from domain knowledge
        # For now, we'll leave it empty but provide a warning
        logger.warning(
            "No true change points could be found or inferred. Some visualizations may be incomplete."
        )

    return results


def create_martingale_comparison_plot(results, output_dir):
    """
    Create a publication-ready comparison plot of martingale values for different combinations.

    Args:
        results: Dictionary of results data frames
        output_dir: Directory to save the plot
    """
    if not results["martingales"]:
        logger.warning("No martingale data available for plotting")
        return

    logger.info("Creating compact martingale comparison plot")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-quality plot style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.size": 8,  # Smaller font size
            "axes.labelsize": 9,  # Smaller axis labels
            "axes.titlesize": 10,  # Smaller title
            "xtick.labelsize": 8,  # Smaller tick labels
            "ytick.labelsize": 8,
            "legend.fontsize": 7,  # Smaller legend text
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    # Create figure with compact dimensions for publication (journal column width)
    # Most journals use columns ~3.5 inches wide
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Define a consistent, visually distinct color palette
    # Use a perceptually uniform colormap for better distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(results["martingales"])))

    # Get threshold if available
    threshold = None
    for combination, config in results["configs"].items():
        if "detection" in config and "threshold" in config["detection"]:
            threshold = float(config["detection"]["threshold"])
            break

    # Plot each combination's martingale values with improved styling
    i = 0
    for combination, df in results["martingales"].items():
        betting_func, distance = combination.split("_")
        label = f"{betting_func} + {distance}"

        # First check if we have the expected martingale columns for this dataset
        if "traditional_sum_martingales" in df.columns:
            x_col = "timestep" if "timestep" in df.columns else df.columns[0]
            y_col = "traditional_sum_martingales"

            ax.plot(
                df[x_col],
                df[y_col],
                label=label,
                color=colors[i],
                linewidth=1.0,
                alpha=0.8,
            )

            # Add arrow and annotation when martingale crosses threshold
            if threshold is not None:
                values = df[y_col].values
                # Find first crossing point
                for j in range(1, len(values)):
                    if values[j - 1] <= threshold < values[j]:
                        # Get exact timestep and value
                        timestep = df[x_col].iloc[j]
                        value = values[j]
                        # Add arrow and annotation - placing boxes alternating on left/right
                        # Place odd-indexed items on the left, even-indexed on the right
                        if i % 2 == 0:  # Even index - place on right side
                            x_offset = 7
                        else:  # Odd index - place on left side
                            x_offset = -7

                        # Calculate a safe y position that doesn't go outside the plot
                        max_val = max(values)
                        y_pos = min(
                            value + max_val * 0.15, max_val * 0.9
                        )  # Cap at 90% of max height

                        ax.annotate(
                            f"t={timestep}",
                            xy=(timestep, value),
                            xytext=(
                                timestep + x_offset,
                                y_pos,
                            ),  # Use bounded y position
                            arrowprops=dict(
                                facecolor=colors[i],
                                shrink=0.05,
                                width=0.5,  # Much thinner arrow
                                headwidth=2,  # Smaller arrowhead width
                                headlength=2,  # Shorter arrowhead
                                alpha=0.8,
                                edgecolor="none",  # Remove outline
                            ),
                            fontsize=6,
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                fc="white",
                                ec=colors[i],
                                alpha=0.8,
                            ),  # Increased padding
                            horizontalalignment=(
                                "center" if i % 2 == 0 else "right"
                            ),  # Adjust text alignment
                        )
                        break

        elif "horizon_sum_martingales" in df.columns:
            x_col = "timestep" if "timestep" in df.columns else df.columns[0]
            y_col = "horizon_sum_martingales"

            ax.plot(
                df[x_col],
                df[y_col],
                label=label,
                color=colors[i],
                linewidth=1.0,
                alpha=0.8,
            )

            # Add arrow and annotation when martingale crosses threshold
            if threshold is not None:
                values = df[y_col].values
                # Find first crossing point
                for j in range(1, len(values)):
                    if values[j - 1] <= threshold < values[j]:
                        # Get exact timestep and value
                        timestep = df[x_col].iloc[j]
                        value = values[j]
                        # Add arrow and annotation with alternating left/right positions
                        if i % 2 == 0:  # Even index - place on right side
                            x_offset = 7
                        else:  # Odd index - place on left side
                            x_offset = -7

                        # Calculate a safe y position that doesn't go outside the plot
                        max_val = max(values)
                        y_pos = min(
                            value + max_val * 0.15, max_val * 0.9
                        )  # Cap at 90% of max height

                        ax.annotate(
                            f"t={timestep}",
                            xy=(timestep, value),
                            xytext=(
                                timestep + x_offset,
                                y_pos,
                            ),  # Use bounded y position
                            arrowprops=dict(
                                facecolor=colors[i],
                                shrink=0.05,
                                width=0.5,  # Much thinner arrow
                                headwidth=2,  # Smaller arrowhead width
                                headlength=2,  # Shorter arrowhead
                                alpha=0.8,
                                edgecolor="none",  # Remove outline
                            ),
                            fontsize=6,
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                fc="white",
                                ec=colors[i],
                                alpha=0.8,
                            ),  # Increased padding
                            horizontalalignment=(
                                "center" if i % 2 == 0 else "right"
                            ),  # Adjust text alignment
                        )
                        break

        else:
            martingale_cols = [
                col for col in df.columns if "martingale" in str(col).lower()
            ]

            if martingale_cols:
                x_col = "timestep" if "timestep" in df.columns else df.columns[0]
                y_col = martingale_cols[0]

                ax.plot(
                    df[x_col],
                    df[y_col],
                    label=label,
                    color=colors[i],
                    linewidth=1.0,
                    alpha=0.8,
                )

                # Add arrow and annotation when martingale crosses threshold
                if threshold is not None:
                    values = df[y_col].values
                    # Find first crossing point
                    for j in range(1, len(values)):
                        if values[j - 1] <= threshold < values[j]:
                            # Get exact timestep and value
                            timestep = df[x_col].iloc[j]
                            value = values[j]
                            # Add arrow and annotation with alternating left/right positions
                            if i % 2 == 0:  # Even index - place on right side
                                x_offset = 7
                            else:  # Odd index - place on left side
                                x_offset = -7

                            # Calculate a safe y position that doesn't go outside the plot
                            max_val = max(values)
                            y_pos = min(
                                value + max_val * 0.15, max_val * 0.9
                            )  # Cap at 90% of max height

                            ax.annotate(
                                f"t={timestep}",
                                xy=(timestep, value),
                                xytext=(
                                    timestep + x_offset,
                                    y_pos,
                                ),  # Use bounded y position
                                arrowprops=dict(
                                    facecolor=colors[i],
                                    shrink=0.05,
                                    width=0.5,  # Much thinner arrow
                                    headwidth=2,  # Smaller arrowhead width
                                    headlength=2,  # Shorter arrowhead
                                    alpha=0.8,
                                    edgecolor="none",  # Remove outline
                                ),
                                fontsize=6,
                                bbox=dict(
                                    boxstyle="round,pad=0.4",
                                    fc="white",
                                    ec=colors[i],
                                    alpha=0.8,
                                ),  # Increased padding
                                horizontalalignment=(
                                    "center" if i % 2 == 0 else "right"
                                ),  # Adjust text alignment
                            )
                            break
            else:
                logger.warning(f"No martingale column found for {combination}")
                continue

        i += 1

    # Add true change points from the data
    for combination, df in results["martingales"].items():
        if "true_change_point" in df.columns:
            true_cp_rows = df[df["true_change_point"] == 1]
            if not true_cp_rows.empty:
                true_cp_timesteps = true_cp_rows["timestep"].values

                for cp in true_cp_timesteps:
                    ax.axvline(
                        x=cp, color="r", linestyle="--", linewidth=1.0, alpha=0.6
                    )
                break

    # If we haven't added change points yet, try using change_points from results
    if "true" in results["change_points"]:
        true_cp = results["change_points"]["true"]

        pos_cols = [
            col
            for col in true_cp.columns
            if any(
                term in str(col).lower()
                for term in ["position", "time", "step", "timestep", "loc", "cp"]
            )
        ]

        if pos_cols:
            pos_col = pos_cols[0]
            for cp in true_cp[pos_col]:
                if not pd.isna(cp):  # Skip NaN values
                    ax.axvline(
                        x=cp, color="r", linestyle="--", linewidth=1.0, alpha=0.6
                    )

    # Add threshold line if available with improved styling
    for combination, config in results["configs"].items():
        if "detection" in config and "threshold" in config["detection"]:
            threshold = float(config["detection"]["threshold"])
            ax.axhline(
                y=threshold,
                color="k",
                linestyle="-.",
                linewidth=1.0,
                alpha=0.7,
                label=f"Threshold = {threshold}",
            )
            break

    # Add labels with optimal size
    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylabel("Martingale Value", fontweight="bold")
    ax.set_title("Martingale Values Comparison", fontsize=10, pad=6)

    # Add grid with minimal styling
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_axisbelow(True)

    # Create a compact, multi-column legend
    n_entries = len(ax.get_legend_handles_labels()[0])
    ncol = min(3, n_entries)  # Use at most 3 columns
    legend = ax.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=6,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(0.5)

    # Tight layout for efficient use of space
    fig.tight_layout()

    # Save the plot in both PNG and PDF formats for publication
    output_path_png = os.path.join(output_dir, "martingale_comparison.png")
    output_path_pdf = os.path.join(output_dir, "martingale_comparison.pdf")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    logger.info(
        f"Saved compact martingale comparison plots to {output_path_png} and {output_path_pdf}"
    )

    # Also save a larger version for presentations/posters if needed
    plt.figure(figsize=(7, 4))
    plt.savefig(
        os.path.join(output_dir, "martingale_comparison_large.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    # Save raw data as CSV for reference
    martingale_df = pd.DataFrame()
    for combination, df in results["martingales"].items():
        # Try to pick the right martingale column
        if "traditional_sum_martingales" in df.columns:
            martingale_df[combination] = df["traditional_sum_martingales"]
        elif "horizon_sum_martingales" in df.columns:
            martingale_df[combination] = df["horizon_sum_martingales"]
        elif "traditional_avg_martingales" in df.columns:
            martingale_df[combination] = df["traditional_avg_martingales"]

    if not martingale_df.empty:
        max_rows = max(len(df) for df in results["martingales"].values())
        martingale_df.index = range(max_rows)[: len(martingale_df)]
        martingale_df.index.name = "Timestep"

        csv_path = os.path.join(output_dir, "martingale_values.csv")
        martingale_df.to_csv(csv_path)

    plt.close()


def create_detection_frequency_plot(results, output_dir):
    """
    Create a compact, publication-ready plot of detection frequencies.

    Args:
        results: Dictionary of results data frames
        output_dir: Directory to save the plot
    """
    if not results["detection_frequencies"]:
        logger.warning("No detection frequency data available for plotting")
        return

    logger.info("Creating compact detection frequency plot")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-quality plot style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    # Get true change points if available
    true_positions = []
    if "true" in results["change_points"]:
        true_cp = results["change_points"]["true"]
        if "position" in true_cp.columns:
            true_positions = true_cp["position"].dropna().values
        elif "timestep" in true_cp.columns:
            true_positions = true_cp["timestep"].dropna().values
        elif true_cp.columns[0]:  # Assume first column contains positions
            true_positions = true_cp[true_cp.columns[0]].dropna().values

    # Prepare data for plotting
    all_positions = []
    all_frequencies = []
    all_combinations = []

    for combination, df in results["detection_frequencies"].items():
        betting_func, distance = combination.split("_")

        # Determine the column names based on the available columns
        pos_col = None
        freq_col = None

        if "position" in df.columns and "frequency" in df.columns:
            pos_col, freq_col = "position", "frequency"
        elif "timestep" in df.columns and "frequency" in df.columns:
            pos_col, freq_col = "timestep", "frequency"
        elif "position" in df.columns and "count" in df.columns:
            pos_col, freq_col = "position", "count"
        elif (
            df.columns[0] and df.columns[1]
        ):  # Assume first is position, second is frequency
            pos_col, freq_col = df.columns[0], df.columns[1]

        if not pos_col or not freq_col:
            logger.warning(
                f"Could not determine position and frequency columns for {combination}"
            )
            continue

        # Extract positions and frequencies
        positions = df[pos_col].values
        frequencies = df[freq_col].values

        for pos, freq in zip(positions, frequencies):
            all_positions.append(pos)
            all_frequencies.append(freq)
            all_combinations.append(f"{betting_func} + {distance}")

    # Create DataFrame for plotting and sort by position
    plot_df = pd.DataFrame(
        {
            "Position": all_positions,
            "Frequency": all_frequencies,
            "Combination": all_combinations,
        }
    ).sort_values("Position")

    # Create compact figure for publication
    fig, ax = plt.subplots(figsize=(7, 3))

    # Plot detection frequencies with a better visual separation
    sns.barplot(
        x="Position",
        y="Frequency",
        hue="Combination",
        data=plot_df,
        ax=ax,
        palette="viridis",
        saturation=0.8,
        alpha=0.8,
    )

    # Add vertical lines for true change points with improved styling
    for cp in true_positions:
        ax.axvline(x=cp, color="r", linestyle="--", linewidth=1.0, alpha=0.7)

    # Improve readability of the plot
    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylabel("Detection Frequency", fontweight="bold")
    ax.set_title("Detection Frequencies by Combination", fontsize=10, pad=6)

    # Make x-axis labels more readable
    plt.xticks(rotation=45, ha="right")

    # Create a compact, better-placed legend
    n_entries = len(ax.get_legend_handles_labels()[0])
    ncol = min(3, n_entries)  # Use at most 3 columns
    legend = ax.legend(
        title="Combination",
        fontsize=6,
        ncol=ncol,
        loc="upper right",
        framealpha=0.9,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.5)

    # Use tight layout for better space usage
    fig.tight_layout()

    # Save the plot in both PNG and PDF formats
    output_path_png = os.path.join(output_dir, "detection_frequency.png")
    output_path_pdf = os.path.join(output_dir, "detection_frequency.pdf")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    logger.info(
        f"Saved compact detection frequency plot to {output_path_png} and {output_path_pdf}"
    )

    plt.close()


def create_faceted_martingale_plot(results, output_dir):
    """
    Create publication-ready faceted martingale plots optimized for space efficiency.

    Args:
        results: Dictionary of results data frames
        output_dir: Directory to save the plot
    """
    if not results["martingales"]:
        logger.warning("No martingale data available for faceted plotting")
        return

    logger.info("Creating compact faceted martingale plots")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-quality plot style optimized for compact presentation
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    # Extract unique betting functions and distance measures
    betting_functions = sorted(
        set(
            bf
            for bf, _ in (combo.split("_") for combo in results["martingales"].keys())
        )
    )
    distance_measures = sorted(
        set(
            dm
            for _, dm in (combo.split("_") for combo in results["martingales"].keys())
        )
    )

    logger.info(
        f"Found {len(betting_functions)} betting functions and {len(distance_measures)} distance measures"
    )

    # Extract threshold value if available
    threshold = None
    for combination, config in results["configs"].items():
        if "detection" in config and "threshold" in config["detection"]:
            threshold = float(config["detection"]["threshold"])
            break

    # Extract true change points if available
    true_change_points = []
    if "true" in results["change_points"]:
        true_cp = results["change_points"]["true"]
        pos_cols = [
            col
            for col in true_cp.columns
            if any(
                term in str(col).lower()
                for term in ["position", "time", "step", "timestep", "loc", "cp"]
            )
        ]
        if pos_cols:
            pos_col = pos_cols[0]
            for cp in true_cp[pos_col]:
                if not pd.isna(cp):
                    true_change_points.append(cp)
        elif true_cp.columns[0]:  # Assume first column contains positions
            for cp in true_cp[true_cp.columns[0]]:
                if not pd.isna(cp):
                    true_change_points.append(cp)

    # Determine the column to use for martingale values for consistency
    martingale_col_name = None
    for combination, df in results["martingales"].items():
        if "traditional_sum_martingales" in df.columns:
            martingale_col_name = "traditional_sum_martingales"
            break
        elif "horizon_sum_martingales" in df.columns:
            martingale_col_name = "horizon_sum_martingales"
            break

    # Find max y-value across all datasets for consistent scaling
    max_y_value = 0
    for combination, df in results["martingales"].items():
        if martingale_col_name and martingale_col_name in df.columns:
            current_max = df[martingale_col_name].max()
        else:
            # Try to find any martingale column
            martingale_cols = [
                col for col in df.columns if "martingale" in str(col).lower()
            ]
            if martingale_cols:
                current_max = df[martingale_cols[0]].max()
            else:
                continue

        if current_max > max_y_value:
            max_y_value = current_max

    # 1. Create a horizontal layout with betting functions (1 row, multiple columns)
    n_bf = len(betting_functions)

    # Calculate figure size for a row layout - wider but less tall
    fig_width = 7.0  # Full journal page width (inches)
    row_height = 2.5  # Height for a single row

    fig1, axes1 = plt.subplots(
        1, n_bf, figsize=(fig_width, row_height), sharex=True, sharey=True
    )

    # Ensure axes1 is always an array even with a single subplot
    if n_bf == 1:
        axes1_flat = [axes1]
    else:
        axes1_flat = axes1

    # Colors for different distance measures - use a perceptually uniform colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(distance_measures)))
    color_map = {distance: colors[i] for i, distance in enumerate(distance_measures)}

    # Plot each betting function in its subplot
    for i, betting_func in enumerate(betting_functions):
        if i < len(axes1_flat):  # Ensure we don't go out of bounds
            ax = axes1_flat[i]

            for distance in distance_measures:
                combination = f"{betting_func}_{distance}"
                if combination in results["martingales"]:
                    df = results["martingales"][combination]

                    # Find the right columns to plot
                    if martingale_col_name and martingale_col_name in df.columns:
                        x_col = (
                            "timestep" if "timestep" in df.columns else df.columns[0]
                        )
                        y_col = martingale_col_name
                    else:
                        martingale_cols = [
                            col
                            for col in df.columns
                            if "martingale" in str(col).lower()
                        ]
                        if martingale_cols:
                            x_col = (
                                "timestep"
                                if "timestep" in df.columns
                                else df.columns[0]
                            )
                            y_col = martingale_cols[0]
                        else:
                            logger.warning(
                                f"No martingale column found for {combination}"
                            )
                            continue

                    # Plot with refined styling
                    ax.plot(
                        df[x_col],
                        df[y_col],
                        label=f"{distance}",
                        color=color_map[distance],
                        linewidth=1.0,
                        alpha=0.8,
                    )

                    # Add arrow and annotation when martingale crosses threshold
                    if threshold is not None:
                        values = df[y_col].values
                        # Find first crossing point
                        for j in range(1, len(values)):
                            if values[j - 1] <= threshold < values[j]:
                                # Get exact timestep and value
                                timestep = df[x_col].iloc[j]
                                value = values[j]
                                # Determine if annotation should be on left or right side
                                distance_index = list(distance_measures).index(distance)
                                if (
                                    distance_index % 2 == 0
                                ):  # Even index - place on right side
                                    x_offset = 7
                                else:  # Odd index - place on left side
                                    x_offset = -7

                                # Calculate a safe y position with vertical offset that stays in bounds
                                max_val = max(values)
                                vertical_factor = 0.15 + 0.08 * list(
                                    distance_measures
                                ).index(distance)
                                # Cap at 90% of max height to stay within bounds
                                y_pos = min(
                                    value + max_val * vertical_factor, max_val * 0.9
                                )

                                ax.annotate(
                                    f"t={timestep}",
                                    xy=(timestep, value),
                                    xytext=(
                                        timestep + x_offset,
                                        y_pos,
                                    ),  # Use bounded y position
                                    arrowprops=dict(
                                        facecolor=color_map[distance],
                                        shrink=0.05,
                                        width=0.5,  # Much thinner arrow
                                        headwidth=2,  # Smaller arrowhead width
                                        headlength=2,  # Shorter arrowhead
                                        alpha=0.8,
                                        edgecolor="none",  # Remove outline
                                    ),
                                    fontsize=6,
                                    bbox=dict(
                                        boxstyle="round,pad=0.4",
                                        fc="white",
                                        ec=color_map[distance],
                                        alpha=0.8,
                                    ),
                                    horizontalalignment=(
                                        "center" if i % 2 == 0 else "right"
                                    ),  # Adjust text alignment
                                )
                                break

            # Add title and styling to each subplot
            ax.set_title(f"betting: {betting_func}", fontweight="bold", fontsize=8)
            ax.grid(True, alpha=0.2, linestyle=":")

            # Add true change points and threshold
            for cp in true_change_points:
                ax.axvline(x=cp, color="r", linestyle="--", linewidth=0.8, alpha=0.6)

            if threshold is not None:
                ax.axhline(
                    y=threshold, color="k", linestyle="-.", linewidth=0.8, alpha=0.7
                )

            # Only add y-label to leftmost subplot
            if i == 0:
                ax.set_ylabel("Martingale Value", fontweight="bold")
                leg = ax.legend(
                    title="Distance",
                    ncol=1,
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
                leg.get_frame().set_linewidth(0.5)

    # Set y limits consistently across all subplots
    for ax in axes1_flat:
        ax.set_ylim(0, max_y_value * 1.1)

    # Hide any unused subplots
    for j in range(n_bf, len(axes1_flat)):
        axes1_flat[j].set_visible(False)

    # # Add overall title
    # fig1.suptitle(
    #     "Martingale Values by Betting Function", fontsize=10, fontweight="bold", y=0.99
    # )

    # Adjust spacing for compact horizontal layout
    fig1.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.2)

    # Save the plot
    output_path_png = os.path.join(output_dir, "martingale_by_betting_function.png")
    output_path_pdf = os.path.join(output_dir, "martingale_by_betting_function.pdf")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    logger.info(
        f"Saved compact betting function plot to {output_path_png} and {output_path_pdf}"
    )

    plt.close()

    # 2. Create a space-efficient 2x2 grid layout with distance measures
    n_dm = len(distance_measures)
    n_cols = min(2, n_dm)  # Use at most 2 columns
    n_rows = (
        n_dm + n_cols - 1
    ) // n_cols  # Ceiling division to get number of rows needed

    # Calculate figure size for a wider layout (full journal page width)
    fig_width = 7.0  # Full journal page width is usually around 7-7.5 inches

    fig2, axes2 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, row_height * n_rows),
        sharex=True,
        sharey=True,
    )

    # Flatten axes array for easier indexing if it's multi-dimensional
    if n_dm > 1:
        if n_cols > 1:
            axes2_flat = axes2.flatten()
        else:
            axes2_flat = axes2
    else:
        axes2_flat = [axes2]

    # Colors for different betting functions - use a perceptually uniform colormap
    colors = plt.cm.plasma(np.linspace(0, 1, len(betting_functions)))
    color_map = {
        betting_func: colors[i] for i, betting_func in enumerate(betting_functions)
    }

    # Plot each distance measure in its subplot
    for i, distance in enumerate(distance_measures):
        if i < len(axes2_flat):  # Ensure we don't go out of bounds
            ax = axes2_flat[i]

            for betting_func in betting_functions:
                combination = f"{betting_func}_{distance}"
                if combination in results["martingales"]:
                    df = results["martingales"][combination]

                    # Find the right columns to plot
                    if martingale_col_name and martingale_col_name in df.columns:
                        x_col = (
                            "timestep" if "timestep" in df.columns else df.columns[0]
                        )
                        y_col = martingale_col_name
                    else:
                        martingale_cols = [
                            col
                            for col in df.columns
                            if "martingale" in str(col).lower()
                        ]
                        if martingale_cols:
                            x_col = (
                                "timestep"
                                if "timestep" in df.columns
                                else df.columns[0]
                            )
                            y_col = martingale_cols[0]
                        else:
                            logger.warning(
                                f"No martingale column found for {combination}"
                            )
                            continue

                    # Plot with refined styling
                    ax.plot(
                        df[x_col],
                        df[y_col],
                        label=f"{betting_func}",
                        color=color_map[betting_func],
                        linewidth=1.0,
                        alpha=0.8,
                    )

                    # Add arrow and annotation when martingale crosses threshold
                    if threshold is not None:
                        values = df[y_col].values
                        # Find first crossing point
                        for j in range(1, len(values)):
                            if values[j - 1] <= threshold < values[j]:
                                # Get exact timestep and value
                                timestep = df[x_col].iloc[j]
                                value = values[j]
                                # Determine if annotation should be on left or right side
                                bf_index = list(betting_functions).index(betting_func)
                                if (
                                    bf_index % 2 == 0
                                ):  # Even index - place on right side
                                    x_offset = 7
                                else:  # Odd index - place on left side
                                    x_offset = -7

                                # Calculate a safe y position with vertical offset that stays in bounds
                                max_val = max(values)
                                vertical_factor = 0.15 + 0.08 * list(
                                    betting_functions
                                ).index(betting_func)
                                # Cap at 90% of max height to stay within bounds
                                y_pos = min(
                                    value + max_val * vertical_factor, max_val * 0.9
                                )

                                ax.annotate(
                                    f"t={timestep}",
                                    xy=(timestep, value),
                                    xytext=(
                                        timestep + x_offset,
                                        y_pos,
                                    ),  # Use bounded y position
                                    arrowprops=dict(
                                        facecolor=color_map[betting_func],
                                        shrink=0.05,
                                        width=0.5,  # Much thinner arrow
                                        headwidth=2,  # Smaller arrowhead width
                                        headlength=2,  # Shorter arrowhead
                                        alpha=0.8,
                                        edgecolor="none",  # Remove outline
                                    ),
                                    fontsize=6,
                                    bbox=dict(
                                        boxstyle="round,pad=0.4",
                                        fc="white",
                                        ec=color_map[betting_func],
                                        alpha=0.8,
                                    ),
                                    horizontalalignment=(
                                        "center" if i % 2 == 0 else "right"
                                    ),  # Adjust text alignment
                                )
                                break

            # Add title and styling to each subplot
            ax.set_title(f"distance: {distance}", fontweight="bold", fontsize=8)
            ax.grid(True, alpha=0.2, linestyle=":")

            # Add true change points and threshold
            for cp in true_change_points:
                ax.axvline(x=cp, color="r", linestyle="--", linewidth=0.8, alpha=0.6)

            if threshold is not None:
                ax.axhline(
                    y=threshold, color="k", linestyle="-.", linewidth=0.8, alpha=0.7
                )

            # Only add y-label to leftmost subplots
            if i % n_cols == 0:
                ax.set_ylabel("Martingale Value", fontweight="bold")

            # Only add x-label to bottom subplots
            if i >= n_dm - n_cols:
                ax.set_xlabel("Timestep", fontweight="bold")

            # Add a compact legend to the first subplot only
            if i == 0:
                leg = ax.legend(
                    title="Betting Function",
                    ncol=1,
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
                leg.get_frame().set_linewidth(0.5)

    # Set y limits consistently
    for ax in axes2_flat:
        ax.set_ylim(0, max_y_value * 1.1)

    # Hide any unused subplots
    for j in range(n_dm, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    # # Add overall title
    # fig2.suptitle(
    #     "Martingale Values by Distance Measure", fontsize=10, fontweight="bold", y=0.99
    # )

    # Adjust spacing for compact layout
    fig2.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.15)

    # Save the plot
    output_path_png = os.path.join(output_dir, "martingale_by_distance.png")
    output_path_pdf = os.path.join(output_dir, "martingale_by_distance.pdf")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    logger.info(
        f"Saved compact distance measure plot to {output_path_png} and {output_path_pdf}"
    )

    plt.close()


def run_comparative_analysis(combinations_dir, output_dir=None):
    """
    Run the analysis and create publication-ready plots.

    Args:
        combinations_dir: Directory containing combination results
        output_dir: Directory to save output plots
    """
    if output_dir is None:
        output_dir = os.path.join(combinations_dir, "comparative_plots")

    # Load all results
    results = load_combination_results(combinations_dir)

    if not results:
        logger.error("Failed to load results, cannot generate plots")
        return

    # Create compact, publication-ready plots
    create_martingale_comparison_plot(results, output_dir)
    create_detection_frequency_plot(results, output_dir)
    create_faceted_martingale_plot(results, output_dir)

    logger.info(
        f"Comparative analysis complete. Publication-ready plots saved to {output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create comparative plots from algorithm results"
    )
    parser.add_argument(
        "combinations_dir", help="Directory containing all combination results"
    )
    parser.add_argument("--output-dir", help="Directory to save output plots")

    args = parser.parse_args()

    run_comparative_analysis(args.combinations_dir, args.output_dir)
