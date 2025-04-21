# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results with research-quality plot generation."""

from typing import Dict, List, Any

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .plotting_config import (
    FIGURE_DIMENSIONS as FD,
    TYPOGRAPHY as TYPO,
    LINE_STYLE as LS,
    COLORS,
    LEGEND_STYLE as LEGEND,
    GRID_STYLE as GRID,
    EXPORT_SETTINGS as EXPORT,
    get_matplotlib_rc_params,
    get_grayscale_style,
)

# Constants for MartingaleVisualizer
MARTINGALE_CONSTANTS = {
    # Detection result keys
    "KEYS": {
        "TRADITIONAL_SUM_MARTINGALES": "traditional_sum_martingales",
        "TRADITIONAL_AVG_MARTINGALES": "traditional_avg_martingales",
        "TRADITIONAL_MARTINGALES": "traditional_martingales",
        "HORIZON_SUM_MARTINGALES": "horizon_sum_martingales",
        "HORIZON_AVG_MARTINGALES": "horizon_avg_martingales",
        "HORIZON_MARTINGALES": "horizon_martingales",
        "INDIVIDUAL_TRADITIONAL_MARTINGALES": "individual_traditional_martingales",
        "INDIVIDUAL_HORIZON_MARTINGALES": "individual_horizon_martingales",
        "TRADITIONAL_CHANGE_POINTS": "traditional_change_points",
        "HORIZON_CHANGE_POINTS": "horizon_change_points",
    },
    # Betting function keys
    "BETTING": {
        "FUNCTION": "function",
        "PARAMS": "params",
        "PARAM_STR": "param_str",
        "FUNCTIONS": {
            "POWER": "power",
            "EXPONENTIAL": "exponential",
            "MIXTURE": "mixture",
            "BETA": "beta",
            "KERNEL": "kernel",
            "CONSTANT": "constant",
        },
        "FUNCTION_PARAMS": {
            "POWER": {
                "EPSILON": "epsilon",
                "DEFAULT_EPSILON": 0.7,
            },
            "EXPONENTIAL": {
                "LAMBDA": "lambda",
                "DEFAULT_LAMBDA": 1.0,
            },
            "MIXTURE": {
                "EPSILONS": "epsilons",
                "DEFAULT_EPSILONS": [0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "BETA": {
                "ALPHA": "alpha",
                "BETA": "beta",
                "DEFAULT_ALPHA": 0.5,
                "DEFAULT_BETA": 1.5,
            },
            "KERNEL": {
                "BANDWIDTH": "bandwidth",
                "DEFAULT_BANDWIDTH": 0.1,
            },
        },
    },
    # Method types
    "METHODS": {
        "MULTIVIEW": "multiview",
        "SINGLE_VIEW": "single_view",
    },
    # Feature names
    "FEATURES": [
        "mean_degree",
        "density",
        "mean_clustering",
        "mean_betweenness",
        "mean_eigenvector",
        "mean_closeness",
        "max_singular_value",
        "min_nonzero_laplacian",
    ],
    # Martingale types
    "MARTINGALE_TYPES": {
        "TRADITIONAL": "traditional",
        "HORIZON": "horizon",
    },
    # Plot titles and labels
    "PLOT_TEXT": {
        "TRADITIONAL": "Traditional",
        "HORIZON": "Horizon",
        "TIME": "Time",
        "VALUE": "Value",
        "MARTINGALE_VALUE": "Martingale Value",
        "SHAP_VALUE": "SHAP Value",
        "TRADITIONAL_SHAP": "Traditional SHAP Values",
        "PREDICTION_SHAP": "Prediction SHAP Values",
        "NO_PREDICTION_SHAP": "No prediction SHAP values available",
        "TRADITIONAL_MARTINGALES": "Traditional Martingales",
        "PREDICTION_MARTINGALES": "Prediction Martingales",
        "COMBINED_SUM": "Combined (Sum)",
        "MARTINGALE": "Martingale",
        "THRESHOLD": "Threshold",
        "DETECTION": "Detection",
        "FEATURE_MARTINGALES": "Feature Martingales",
        "NETWORK_VIEW": "Network View",
    },
    # Result keys for the processed data
    "RESULT_KEYS": {
        "COMBINED": "combined",
        "MARTINGALES": "martingales",
        "MARTINGALES_SUM": "martingales_sum",
        "MARTINGALES_AVG": "martingales_avg",
        "PREDICTION_MARTINGALE_SUM": "prediction_martingale_sum",
        "PREDICTION_MARTINGALE_AVG": "prediction_martingale_avg",
        "PREDICTION_MARTINGALES": "prediction_martingales",
    },
    # Default values
    "DEFAULTS": {
        "SEQUENCE_LENGTH": 200,
        "TICK_INTERVAL": 50,
        "FIGURE_SIZE": (12, 8),
        "SUBFIGURE_HEIGHT": 5,
        "MARKER_SIZE": 10,
        "ANNOTATION_OFFSET": (10, 10),
        "Y_MARGIN_FACTOR": 0.1,
        "MIN_Y_MARGIN": 0.1,
        "Y_TICKS_COUNT": 4,
    },
}


class MartingaleVisualizer:
    """Visualization class for martingale analysis results with research-quality output."""

    def __init__(
        self,
        martingales: Dict[str, Any],
        change_points: List[int],
        threshold: float,
        betting_config: Dict[str, Any],
        output_dir: str = "results",
        prefix: str = "",
        skip_shap: bool = False,
        method: str = MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"],
    ):
        """Initialize the visualizer with research-quality plot settings."""
        self.change_points = change_points
        self.threshold = threshold
        self.betting_config = betting_config
        self.output_dir = output_dir
        self.prefix = prefix
        self.skip_shap = skip_shap
        self.method = method

        # Get betting function parameters for visualization
        self.betting_params = self._get_betting_params()

        # Process martingales into a format suitable for plotting
        self.martingales = self._process_martingales(martingales)

        # Initialize SHAP values
        self.shap_values = None
        self.feature_names = None
        self.prediction_shap_values = None

        # Compute SHAP values if needed
        if not skip_shap and method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]:
            try:
                # Get sequence length from combined martingales
                sequence_length = len(
                    self.martingales[MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]][
                        MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES_SUM"]
                    ]
                )

                # Get feature names
                self.feature_names = MARTINGALE_CONSTANTS["FEATURES"]

                # Prepare data for SHAP computation
                traditional_data = []
                prediction_data = []

                for feature in self.feature_names:
                    if feature in self.martingales:
                        trad_vals = self.martingales[feature][
                            MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"]
                        ]
                        pred_vals = self.martingales[feature].get(
                            MARTINGALE_CONSTANTS["RESULT_KEYS"][
                                "PREDICTION_MARTINGALES"
                            ],
                            [],
                        )

                        traditional_data.append(trad_vals)
                        if len(pred_vals) > 0:
                            prediction_data.append(pred_vals)

                # Convert to numpy arrays and transpose to get (time, features) shape
                if traditional_data:
                    self.shap_values = np.array(traditional_data).T
                if prediction_data:
                    self.prediction_shap_values = np.array(prediction_data).T

            except (ValueError, KeyError) as e:
                print(f"Warning: Could not compute SHAP values: {str(e)}")
                self.shap_values = None
                self.feature_names = None
                self.prediction_shap_values = None

        # Set publication-quality parameters
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(get_matplotlib_rc_params())

        # Store color schemes
        self.colors = COLORS
        self.grayscale_colors = get_grayscale_style()

        # Set default figure settings
        self.fig_settings = {
            "dpi": EXPORT["DPI"],
            "bbox_inches": EXPORT["BBOX_INCHES"],
            "pad_inches": EXPORT["PAD_INCHES"],
            "format": EXPORT["FORMAT"],
        }

    def _get_betting_params(self) -> Dict[str, Any]:
        """Extract relevant betting parameters for visualization."""
        function = self.betting_config[MARTINGALE_CONSTANTS["BETTING"]["FUNCTION"]]
        params = self.betting_config[MARTINGALE_CONSTANTS["BETTING"]["PARAMS"]].get(
            function, {}
        )

        # Create a formatted string of parameters for visualization
        param_str = ""
        if function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["POWER"]:
            epsilon = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["POWER"]["EPSILON"],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["POWER"][
                    "DEFAULT_EPSILON"
                ],
            )
            param_str = f"ε={epsilon}"
        elif function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["EXPONENTIAL"]:
            lambd = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["EXPONENTIAL"][
                    "LAMBDA"
                ],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["EXPONENTIAL"][
                    "DEFAULT_LAMBDA"
                ],
            )
            param_str = f"λ={lambd}"
        elif function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["MIXTURE"]:
            epsilons = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["MIXTURE"][
                    "EPSILONS"
                ],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["MIXTURE"][
                    "DEFAULT_EPSILONS"
                ],
            )
            param_str = f"ε={min(epsilons)}-{max(epsilons)}"
        elif function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["BETA"]:
            alpha = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["BETA"]["ALPHA"],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["BETA"][
                    "DEFAULT_ALPHA"
                ],
            )
            beta = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["BETA"]["BETA"],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["BETA"][
                    "DEFAULT_BETA"
                ],
            )
            param_str = f"α={alpha}, β={beta}"
        elif function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["KERNEL"]:
            bandwidth = params.get(
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["KERNEL"][
                    "BANDWIDTH"
                ],
                MARTINGALE_CONSTANTS["BETTING"]["FUNCTION_PARAMS"]["KERNEL"][
                    "DEFAULT_BANDWIDTH"
                ],
            )
            param_str = f"bw={bandwidth}"
        elif function == MARTINGALE_CONSTANTS["BETTING"]["FUNCTIONS"]["CONSTANT"]:
            param_str = "fixed"

        return {
            MARTINGALE_CONSTANTS["BETTING"]["FUNCTION"]: function,
            MARTINGALE_CONSTANTS["BETTING"]["PARAMS"]: params,
            MARTINGALE_CONSTANTS["BETTING"]["PARAM_STR"]: param_str,
        }

    def _process_martingales(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process detector results into a format suitable for plotting."""
        processed = {MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]: {}}

        # Get the sequence length from the combined martingales
        if self.method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]:
            seq_len = len(
                detection_results.get(
                    MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_SUM_MARTINGALES"], []
                )
            )
        else:
            seq_len = len(
                detection_results.get(
                    MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_MARTINGALES"], []
                )
            )

        if self.method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]:
            # Process multiview results
            processed[MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]].update(
                {
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "MARTINGALES_SUM"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_SUM_MARTINGALES"], []
                    )[
                        :seq_len
                    ],
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "MARTINGALES_AVG"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_AVG_MARTINGALES"], []
                    )[
                        :seq_len
                    ],
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "PREDICTION_MARTINGALE_SUM"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["HORIZON_SUM_MARTINGALES"], []
                    ),
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "PREDICTION_MARTINGALE_AVG"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["HORIZON_AVG_MARTINGALES"], []
                    ),
                }
            )

            # Process individual feature martingales
            individual_traditional = detection_results.get(
                MARTINGALE_CONSTANTS["KEYS"]["INDIVIDUAL_TRADITIONAL_MARTINGALES"], []
            )
            individual_horizon = detection_results.get(
                MARTINGALE_CONSTANTS["KEYS"]["INDIVIDUAL_HORIZON_MARTINGALES"], []
            )

            features = MARTINGALE_CONSTANTS["FEATURES"]

            for i, feature in enumerate(features):
                if i < len(individual_traditional):
                    # Use actual sequence length
                    traditional_values = individual_traditional[i][:seq_len]
                    processed[feature] = {
                        MARTINGALE_CONSTANTS["RESULT_KEYS"][
                            "MARTINGALES"
                        ]: traditional_values,
                        MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"]: (
                            individual_horizon[i] if individual_horizon else None
                        ),
                    }

        else:  # single_view
            # Process single-view results
            processed[MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]].update(
                {
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "MARTINGALES"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_MARTINGALES"], []
                    )[
                        :seq_len
                    ],
                    MARTINGALE_CONSTANTS["RESULT_KEYS"][
                        "PREDICTION_MARTINGALES"
                    ]: detection_results.get(
                        MARTINGALE_CONSTANTS["KEYS"]["HORIZON_MARTINGALES"], []
                    ),
                }
            )

        return processed

    def create_visualization(self) -> None:
        """Create all visualizations."""
        # Create detection analysis plots
        self._plot_detection_analysis()

        # Only create feature plots for multiview results with individual martingales
        if (
            self.method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]
            and len(self.martingales) > 1
        ):
            self._plot_feature_martingales()
            self._plot_overlaid_martingales()

            # Plot SHAP values if available and not explicitly skipped
            if (
                not self.skip_shap
                and hasattr(self, "shap_values")
                and self.shap_values is not None
            ):
                self._plot_shap_values()

    def _create_research_figure(self, width="single", height_scale=1.0) -> plt.Figure:
        """Create a figure with research-quality dimensions."""
        width_inches = (
            FD["SINGLE_COLUMN_WIDTH"]
            if width == "single"
            else FD["DOUBLE_COLUMN_WIDTH"]
        )
        height_inches = FD["STANDARD_HEIGHT"] * height_scale
        return plt.figure(figsize=(width_inches, height_inches))

    def _plot_detection_analysis(self) -> None:
        """Create publication-quality detection analysis visualization."""
        fig = self._create_research_figure(width="double", height_scale=1.0)
        gs = GridSpec(1, 2, figure=fig, wspace=0.25)

        # Create subplots with enhanced styling
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        self._plot_detection_subplot(
            ax1,
            MARTINGALE_CONSTANTS["MARTINGALE_TYPES"]["TRADITIONAL"],
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["TRADITIONAL"],
        )
        self._plot_detection_subplot(
            ax2,
            MARTINGALE_CONSTANTS["MARTINGALE_TYPES"]["HORIZON"],
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["HORIZON"],
        )

        # Ensure consistent y-axis limits across subplots
        self._synchronize_axis_limits([ax1, ax2])

        # Save with publication settings
        self._save_research_figure(fig, "detection_analysis.png")

    def _plot_feature_martingales(self) -> None:
        """Create publication-quality feature martingale plots."""
        fig = self._create_research_figure(width="double", height_scale=2.0)

        # Create grid with optimal spacing
        gs = GridSpec(
            4,
            2,
            figure=fig,
            hspace=FD["GRID_SPACING"] * 1.5,
            wspace=FD["GRID_SPACING"] * 2,
        )

        # Plot features with enhanced styling
        self._plot_feature_grid(fig, gs)

        # Add betting function info to figure title
        title = self._get_enhanced_title()
        fig.suptitle(title, fontsize=TYPO["TITLE_SIZE"], y=0.95)

        # Save with publication settings
        self._save_research_figure(fig, "feature_martingales.png")

    def _save_research_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure with publication-quality settings."""
        output_path = os.path.join(self.output_dir, f"{self.prefix}{filename}")

        # Save only PNG format with high quality settings
        fig.savefig(
            output_path,
            dpi=EXPORT["DPI"],
            bbox_inches=EXPORT["BBOX_INCHES"],
            pad_inches=EXPORT["PAD_INCHES"],
            format="png",
            transparent=EXPORT["TRANSPARENT"],
        )

        plt.close(fig)

    def _optimize_axis_settings(self, ax: plt.Axes) -> None:
        """Apply publication-quality axis settings."""
        # Set proper axis limits
        ax.set_xlim(0, MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"])
        ax.set_xticks(
            np.arange(
                0,
                MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"] + 1,
                MARTINGALE_CONSTANTS["DEFAULTS"]["TICK_INTERVAL"],
            )
        )

        # Optimize y-axis range
        y_data = [line.get_ydata() for line in ax.get_lines()]
        if y_data:
            y_data = np.concatenate(y_data)
            y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)
            margin = (y_max - y_min) * MARTINGALE_CONSTANTS["DEFAULTS"][
                "Y_MARGIN_FACTOR"
            ]
            ax.set_ylim(max(0, y_min - margin), y_max + margin)

        # Enhanced grid
        ax.grid(
            True,
            which="major",
            linestyle=GRID["LINE_STYLE"],
            alpha=GRID["MAJOR_ALPHA"],
            linewidth=GRID["MAJOR_LINE_WIDTH"],
        )

        # Only set y-axis label, x-axis label is handled separately
        ax.set_ylabel(
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["VALUE"],
            fontsize=TYPO["LABEL_SIZE"],
            labelpad=TYPO["LABEL_PAD"],
        )

        # Enhanced ticks
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TYPO["TICK_SIZE"],
            length=GRID["MAJOR_TICK_LENGTH"],
        )

    def _add_optimized_legend(self, ax: plt.Axes) -> None:
        """Add publication-quality legend."""
        ax.legend(
            ncol=LEGEND["COLUMNS"],
            loc=LEGEND["LOCATION"],
            fontsize=TYPO["LEGEND_SIZE"],
            frameon=True,
            edgecolor=COLORS["text"],
            facecolor=COLORS["background"],
            framealpha=LEGEND["FRAME_ALPHA"],
            borderaxespad=LEGEND["BORDER_PAD"],
            handlelength=LEGEND["HANDLE_LENGTH"],
            columnspacing=LEGEND["COLUMN_SPACING"],
        )

    def _plot_detection_subplot(self, ax, mart_type: str, title: str) -> None:
        """Helper method to plot detection analysis for a specific martingale type."""
        if mart_type == MARTINGALE_CONSTANTS["MARTINGALE_TYPES"]["TRADITIONAL"]:
            changes = self.martingales[
                MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
            ].get(MARTINGALE_CONSTANTS["KEYS"]["TRADITIONAL_CHANGE_POINTS"], [])
            if self.method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]:
                values = self.martingales[
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
                ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES_SUM"], [])
            else:
                values = self.martingales[
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
                ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"], [])
        else:  # horizon
            changes = self.martingales[
                MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
            ].get(MARTINGALE_CONSTANTS["KEYS"]["HORIZON_CHANGE_POINTS"], [])
            if self.method == MARTINGALE_CONSTANTS["METHODS"]["MULTIVIEW"]:
                values = self.martingales[
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
                ].get(
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALE_SUM"], []
                )
            else:
                values = self.martingales[
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
                ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"], [])

        # Plot martingale values
        if values is not None and len(values) > 0:
            ax.plot(
                values,
                color=self.colors["actual"],
                label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["MARTINGALE"],
            )

            # Mark detection points
            for idx, cp in enumerate(changes):
                # Make sure cp is a scalar index
                if isinstance(cp, (np.ndarray, list)):
                    continue  # Skip if cp is an array

                # Safely access the value at the change point
                if 0 <= cp < len(values):
                    value_at_cp = values[cp]

                    # Create a label only for the first change point
                    is_first = idx == 0

                    ax.plot(
                        cp,
                        value_at_cp,
                        "ro",
                        markersize=MARTINGALE_CONSTANTS["DEFAULTS"]["MARKER_SIZE"],
                        label=(
                            MARTINGALE_CONSTANTS["PLOT_TEXT"]["DETECTION"]
                            if is_first
                            else ""
                        ),
                    )
                    ax.annotate(
                        f"t={cp}",
                        (cp, value_at_cp),
                        xytext=MARTINGALE_CONSTANTS["DEFAULTS"]["ANNOTATION_OFFSET"],
                        textcoords="offset points",
                    )

        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color=self.colors["threshold"],
            linestyle="--",
            label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["THRESHOLD"],
        )

        ax.set_title(title)
        ax.set_xlabel(MARTINGALE_CONSTANTS["PLOT_TEXT"]["TIME"])
        ax.set_ylabel(MARTINGALE_CONSTANTS["PLOT_TEXT"]["VALUE"])
        ax.legend()
        ax.grid(True)

    def _plot_overlaid_martingales(self) -> None:
        """Create plot overlaying individual feature martingales with combined martingale."""
        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 2))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        features = MARTINGALE_CONSTANTS["FEATURES"]

        # Use a different color for each feature
        colors = plt.cm.tab10(np.linspace(0, 1, len(features)))

        # Plot traditional martingales (top subplot)
        ax1 = fig.add_subplot(gs[0])

        # Plot individual feature martingales
        for i, feature in enumerate(features):
            if feature in self.martingales:
                martingales = self.martingales[feature].get(
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"], []
                )
                if isinstance(martingales, (list, np.ndarray)) and len(martingales) > 0:
                    ax1.plot(
                        martingales,
                        color=colors[i],
                        label=feature.replace("_", " ").title(),
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the combined sum martingale
        combined_sum = self.martingales[
            MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
        ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES_SUM"], [])
        if isinstance(combined_sum, (list, np.ndarray)) and len(combined_sum) > 0:
            ax1.plot(
                combined_sum,
                color="blue",
                linestyle="--",
                label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["COMBINED_SUM"],
                linewidth=1.5,
                alpha=0.8,
                zorder=10,
            )

        # Add threshold and change points to top subplot
        self._add_threshold_and_changes(ax1)
        ax1.set_title(
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["TRADITIONAL_MARTINGALES"],
            fontsize=TYPO["TITLE_SIZE"],
        )

        # Plot prediction martingales (bottom subplot)
        ax2 = fig.add_subplot(gs[1])

        # Plot individual feature prediction martingales
        for i, feature in enumerate(features):
            if feature in self.martingales:
                pred_martingales = self.martingales[feature].get(
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"], []
                )
                if (
                    isinstance(pred_martingales, (list, np.ndarray))
                    and len(pred_martingales) > 0
                ):
                    # Get traditional martingales to calculate history size
                    martingales = self.martingales[feature].get(
                        MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"], []
                    )
                    history_size = len(martingales) - len(pred_martingales)
                    time_points = np.arange(
                        history_size, history_size + len(pred_martingales)
                    )
                    ax2.plot(
                        time_points,
                        pred_martingales,
                        color=colors[i],
                        label=feature.replace("_", " ").title(),
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the combined prediction sum martingale
        combined_pred_sum = self.martingales[
            MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
        ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALE_SUM"], [])
        if (
            isinstance(combined_pred_sum, (list, np.ndarray))
            and len(combined_pred_sum) > 0
        ):
            # Get traditional sum martingale to calculate history size
            combined_sum = self.martingales[
                MARTINGALE_CONSTANTS["RESULT_KEYS"]["COMBINED"]
            ].get(MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES_SUM"], [])
            history_size = len(combined_sum) - len(combined_pred_sum)
            time_points = np.arange(history_size, history_size + len(combined_pred_sum))
            ax2.plot(
                time_points,
                combined_pred_sum,
                color="orange",
                linestyle="--",
                label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["COMBINED_SUM"],
                linewidth=1.5,
                alpha=0.8,
                zorder=10,
            )

        # Add threshold and change points to bottom subplot
        self._add_threshold_and_changes(ax2)
        ax2.set_title(
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["PREDICTION_MARTINGALES"],
            fontsize=TYPO["TITLE_SIZE"],
        )

        # Set common parameters for both subplots
        for ax in [ax1, ax2]:
            ax.grid(True, which="major", linestyle=":", alpha=GRID["MAJOR_ALPHA"])
            ax.grid(True, which="minor", linestyle=":", alpha=GRID["MAJOR_ALPHA"] / 2)
            ax.set_xticks(
                np.arange(
                    0,
                    MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"] + 1,
                    MARTINGALE_CONSTANTS["DEFAULTS"]["TICK_INTERVAL"],
                )
            )
            ax.set_xlim(0, MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"])

            # Create consistent legend positioning for both subplots
            ax.legend(
                ncol=2,
                loc="upper right",
                fontsize=TYPO["LEGEND_SIZE"] * 0.8,
                frameon=True,
                handlelength=LEGEND["HANDLE_LENGTH"],
                handletextpad=0.5,
                columnspacing=LEGEND["COLUMN_SPACING"],
                borderaxespad=0.1,  # Consistent padding
            )

            ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])

        # Set labels only for bottom subplot
        ax2.set_xlabel(
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["TIME"],
            fontsize=TYPO["LABEL_SIZE"],
            labelpad=4,
        )
        for ax in [ax1, ax2]:
            ax.set_ylabel(
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["MARTINGALE_VALUE"],
                fontsize=TYPO["LABEL_SIZE"],
                labelpad=4,
            )

        plt.tight_layout(pad=0.3)
        self._save_figure("overlaid_martingales.png")

    def _add_threshold_and_changes(self, ax) -> None:
        """Add threshold line and change point markers to an axis."""
        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.3,
            linewidth=GRID["MAJOR_LINE_WIDTH"],
        )

        # Add change points
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.3,
                linewidth=GRID["MAJOR_LINE_WIDTH"],
            )

        # Set y-axis limits based on data
        lines = ax.get_lines()
        if lines:
            y_data = []
            for line in lines:
                data = line.get_ydata()
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    y_data.append(data)

            if y_data:
                y_data = np.concatenate(y_data)
                y_data = y_data[~np.isnan(y_data)]  # Remove NaN values
                if len(y_data) > 0:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    margin = (y_max - y_min) * MARTINGALE_CONSTANTS["DEFAULTS"][
                        "Y_MARGIN_FACTOR"
                    ]
                    ax.set_ylim(y_min - margin, y_max + margin)

    def _plot_shap_subplot(
        self, ax, shap_values, feature_names, colors, title, is_prediction=False
    ):
        """Plot SHAP values for a single subplot."""
        # Get history size for prediction offset
        history_size = 0
        if is_prediction:
            # Calculate history size from martingales
            feature = feature_names[0]  # Use first feature to get history size
            if feature in self.martingales:
                martingales = self.martingales[feature].get(
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"], []
                )
                pred_martingales = self.martingales[feature].get(
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"], []
                )
                if len(martingales) > 0 and len(pred_martingales) > 0:
                    history_size = len(martingales) - len(pred_martingales)

        # Plot each feature's SHAP values
        for i, feature in enumerate(feature_names):
            if i < len(colors):
                time_points = np.arange(len(shap_values))
                if is_prediction:
                    time_points = np.arange(
                        history_size, history_size + len(shap_values)
                    )

                ax.plot(
                    time_points,
                    shap_values[:, i],
                    label=feature.replace("_", " ").title(),
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.0,
                )

        # Add threshold line
        ax.axhline(
            y=0,  # SHAP values are centered around 0
            color=self.colors["threshold"],
            linestyle="--",
            alpha=0.2,  # Reduced alpha
            linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner line
        )

        # Dynamic y-axis range with small margin
        if isinstance(shap_values, (list, np.ndarray)) and len(shap_values) > 0:
            y_min, y_max = np.nanmin(shap_values), np.nanmax(shap_values)
            if not (np.isnan(y_min) or np.isnan(y_max)):
                y_margin = max(
                    (y_max - y_min)
                    * MARTINGALE_CONSTANTS["DEFAULTS"]["Y_MARGIN_FACTOR"],
                    MARTINGALE_CONSTANTS["DEFAULTS"]["MIN_Y_MARGIN"],
                )  # At least 0.1 margin
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Legend configuration
        if len(ax.get_lines()) > 0:  # Only add legend if we have plotted lines
            ax.legend(
                ncol=2,
                loc="upper right",
                fontsize=TYPO["LEGEND_SIZE"] * 0.8,
                frameon=True,
                handlelength=LEGEND["HANDLE_LENGTH"],
                handletextpad=0.2,
                columnspacing=LEGEND["COLUMN_SPACING"],
                borderaxespad=0.1,
            )

        ax.set_title(title, fontsize=TYPO["TITLE_SIZE"])

    def _plot_shap_values(self) -> None:
        """Create plot for SHAP values with two subplots: traditional and prediction."""
        if self.shap_values is None or self.feature_names is None:
            return

        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 1.6))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        # Create color map for available features
        n_features = len(self.feature_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_features))

        # Plot traditional SHAP values (top subplot)
        ax1 = fig.add_subplot(gs[0])
        self._plot_shap_subplot(
            ax1,
            self.shap_values,
            self.feature_names,
            colors,
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["TRADITIONAL_SHAP"],
            is_prediction=False,
        )

        # Plot prediction SHAP values if available (bottom subplot)
        ax2 = fig.add_subplot(gs[1])
        if (
            hasattr(self, "prediction_shap_values")
            and self.prediction_shap_values is not None
        ):
            self._plot_shap_subplot(
                ax2,
                self.prediction_shap_values,
                self.feature_names,
                colors,
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["PREDICTION_SHAP"],
                is_prediction=True,
            )
        else:
            ax2.text(
                0.5,
                0.5,
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["NO_PREDICTION_SHAP"],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=TYPO["LABEL_SIZE"],
            )
            ax2.set_title(
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["PREDICTION_SHAP"],
                fontsize=TYPO["TITLE_SIZE"],
            )

        # Set common parameters
        for ax in [ax1, ax2]:
            ax.set_xticks(
                np.arange(
                    0,
                    MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"] + 1,
                    MARTINGALE_CONSTANTS["DEFAULTS"]["TICK_INTERVAL"],
                )
            )
            ax.set_xlim(0, MARTINGALE_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"])
            ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])
            ax.grid(True, linestyle=":", alpha=GRID["MAJOR_ALPHA"] * 0.7)

            # Add threshold and change points
            self._add_threshold_and_changes(ax)

        # Set labels
        ax2.set_xlabel(
            MARTINGALE_CONSTANTS["PLOT_TEXT"]["TIME"],
            fontsize=TYPO["LABEL_SIZE"],
            labelpad=1,
        )
        for ax in [ax1, ax2]:
            ax.set_ylabel(
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["SHAP_VALUE"],
                fontsize=TYPO["LABEL_SIZE"],
                labelpad=1,
            )

        plt.tight_layout(pad=0.1)  # Tighter layout
        self._save_figure("shap_values.png")

    def _save_figure(self, filename: str) -> None:
        """Save figure with publication-quality settings."""
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        final_filename = f"{self.prefix}{filename}" if self.prefix else filename
        plt.savefig(
            os.path.join(self.output_dir, final_filename),
            dpi=EXPORT["DPI"],
            bbox_inches=EXPORT["BBOX_INCHES"],
            pad_inches=EXPORT["PAD_INCHES"],
            format=EXPORT["FORMAT"],
            transparent=EXPORT["TRANSPARENT"],
        )

    def _get_enhanced_title(self) -> str:
        """Generate publication-quality title with mathematical notation."""
        function_name = self.betting_config[
            MARTINGALE_CONSTANTS["BETTING"]["FUNCTION"]
        ].title()
        title = f"{MARTINGALE_CONSTANTS['PLOT_TEXT']['FEATURE_MARTINGALES']} ({function_name}"
        if self.betting_params[MARTINGALE_CONSTANTS["BETTING"]["PARAM_STR"]]:
            # Convert to proper mathematical notation
            param_str = self.betting_params[
                MARTINGALE_CONSTANTS["BETTING"]["PARAM_STR"]
            ].replace("ε", "$\\varepsilon$")
            param_str = param_str.replace("λ", "$\\lambda$")
            param_str = param_str.replace("α", "$\\alpha$")
            param_str = param_str.replace("β", "$\\beta$")
            title += f", {param_str}"
        title += ")"
        return title

    def _add_threshold_line(self, ax: plt.Axes) -> None:
        """Add threshold line with enhanced visibility."""
        ax.axhline(
            y=self.threshold,
            color=self.colors["threshold"],
            linestyle=LS["THRESHOLD_LINE_STYLE"],
            alpha=0.5,
            linewidth=LS["LINE_WIDTH"],
            label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["THRESHOLD"],
            zorder=5,
        )

    def _add_change_point_lines(self, ax: plt.Axes) -> None:
        """Add change point lines with enhanced visibility."""
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=self.colors["change_point"],
                linestyle=LS["CHANGE_POINT_LINE_STYLE"],
                alpha=0.3,
                linewidth=LS["LINE_WIDTH"] * 0.8,
                zorder=4,
            )

    def _find_global_maximum(self, features: List[str]) -> float:
        """Find global maximum value across all features."""
        max_mart = 0
        for feature in features:
            if feature in self.martingales:
                results = self.martingales[feature]
                for key in [
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"],
                    MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"],
                ]:
                    values = results.get(key, [])
                    if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                        max_mart = max(max_mart, np.nanmax(values))
        return max_mart

    def _synchronize_axis_limits(self, axes: List[plt.Axes]) -> None:
        """Ensure consistent axis limits across multiple subplots."""
        y_min, y_max = float("inf"), float("-inf")

        # Find global min and max
        for ax in axes:
            for line in ax.get_lines():
                data = line.get_ydata()
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    y_min = min(y_min, np.nanmin(data))
                    y_max = max(y_max, np.nanmax(data))

        # Add margin
        margin = (y_max - y_min) * 0.1
        y_min = max(0, y_min - margin)  # Don't go below 0
        y_max = y_max + margin

        # Set consistent limits
        for ax in axes:
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, 200)
            ax.set_xticks(np.arange(0, 201, 50))

    def _plot_feature_grid(self, fig: plt.Figure, gs: GridSpec) -> None:
        """Plot feature martingales in a grid layout."""
        features = MARTINGALE_CONSTANTS["FEATURES"]

        # Find global maximum for consistent y-axis
        max_mart_global = self._find_global_maximum(features)

        # Plot each feature
        for idx, feature in enumerate(features):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[row, col])

            if feature in self.martingales:
                self._plot_single_feature(
                    ax,
                    feature,
                    max_mart_global,
                    show_xlabel=(row == 3),  # Only bottom row shows x-label
                    show_ylabel=(col == 0),  # Left column shows y-label
                    show_legend=(idx == 0),  # First plot shows legend
                )

    def _plot_single_feature(
        self,
        ax: plt.Axes,
        feature: str,
        max_mart_global: float,
        show_xlabel: bool = False,
        show_ylabel: bool = False,
        show_legend: bool = False,
    ) -> None:
        """Plot a single feature with enhanced styling."""
        results = self.martingales[feature]

        # Plot traditional martingale
        martingales = results.get(
            MARTINGALE_CONSTANTS["RESULT_KEYS"]["MARTINGALES"], []
        )
        if isinstance(martingales, (list, np.ndarray)) and len(martingales) > 0:
            ax.plot(
                martingales,
                color=self.colors["actual"],
                label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["TRADITIONAL"],
                linewidth=LS["LINE_WIDTH"],
                alpha=LS["LINE_ALPHA"],
                zorder=10,
            )

        # Plot prediction martingale
        pred_martingales = results.get(
            MARTINGALE_CONSTANTS["RESULT_KEYS"]["PREDICTION_MARTINGALES"], []
        )
        if (
            isinstance(pred_martingales, (list, np.ndarray))
            and len(pred_martingales) > 0
        ):
            history_size = len(martingales) - len(pred_martingales)
            time_points = np.arange(history_size, history_size + len(pred_martingales))
            ax.plot(
                time_points,
                pred_martingales,
                color=self.colors["predicted"],
                label=MARTINGALE_CONSTANTS["PLOT_TEXT"]["HORIZON"],  # Updated label
                linewidth=LS["LINE_WIDTH"],
                linestyle=LS["PREDICTION_LINE_STYLE"],
                alpha=LS["LINE_ALPHA"],
                zorder=9,
            )

        # Add threshold and change points
        self._add_threshold_line(ax)
        self._add_change_point_lines(ax)

        # Set consistent y-axis limits
        y_max = max_mart_global * 1.05
        ax.set_ylim(0, y_max)
        ax.set_yticks(
            np.linspace(0, y_max, MARTINGALE_CONSTANTS["DEFAULTS"]["Y_TICKS_COUNT"])
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

        # Enhanced title and labels
        ax.set_title(
            feature.replace("mean_", "").replace("_", " ").title(),
            fontsize=TYPO["TITLE_SIZE"],
            pad=TYPO["TITLE_PAD"] * 2,  # Increased padding for title
        )

        # Show x-label only on bottom row
        if show_xlabel:
            ax.set_xlabel(
                MARTINGALE_CONSTANTS["PLOT_TEXT"]["TIME"],
                fontsize=TYPO["LABEL_SIZE"],
                labelpad=TYPO["LABEL_PAD"],
            )
        else:
            ax.set_xlabel("")  # Remove x-label for other subplots
            ax.set_xticklabels([])  # Also remove x-tick labels for non-bottom rows

        # Handle y-axis label
        if not show_ylabel:
            ax.set_ylabel("")  # Remove y-label if not on left column

        # Legend only in first plot
        if show_legend:
            self._add_optimized_legend(ax)

        # Enhanced grid and ticks
        self._optimize_axis_settings(ax)
