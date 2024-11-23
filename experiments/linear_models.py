# experiments/linear_models.py

"""Linear model analysis module for change point detection.

Implements martingale-based change detection using linear equations
and SHAP value analysis for feature importance interpretation.
"""

import logging
import numpy as np
import shap  # type: ignore
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Callable
from config.config import load_config, Config

logger = logging.getLogger(__name__)


def generate_martingales(
    instants: np.ndarray, equations: List[Tuple[float, float]], change_point: int
) -> np.ndarray:
    """Generate martingale values from linear equations.

    For each equation y = mx + b, computes:
    - M(t) = 1 for t <= change_point
    - M(t) = mx + b - (m * change_point + b) for t > change_point

    Args:
        instants (np.ndarray): Time points t.
        equations (List[Tuple[float, float]]): List of (slope m, intercept b) pairs.
        change_point (int): Time instant where change occurs.

    Returns:
        np.ndarray: Matrix [n_instants x n_equations] of martingale values.

    Raises:
        ValueError: If equations list is empty or change_point is negative.
        RuntimeError: If martingale generation fails.
    """
    if not equations:
        logger.error("Empty equation list provided")
        raise ValueError("Empty equation list")
    if change_point < 0:
        logger.error(f"Invalid change point: {change_point}")
        raise ValueError("Change point must be non-negative")

    try:
        logger.info(f"Generating martingales for {len(equations)} equations")
        logger.debug(f"Change point at t={change_point}")

        martingales = []
        for slope, intercept in equations:
            logger.debug(f"Processing equation: y = {slope}x + {intercept}")
            martingale = np.array(
                [
                    (
                        1
                        if i <= change_point
                        else slope * i + intercept - (slope * change_point + intercept)
                    )
                    for i in instants
                ]
            )
            martingales.append(martingale)

        result = np.vstack(martingales).T
        logger.debug(f"Generated martingale matrix of shape {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Martingale generation failed: {str(e)}")
        raise RuntimeError(f"Martingale generation failed: {str(e)}")


def define_models(
    threshold_sum: float, threshold_avg: float
) -> Tuple[Callable, Callable]:
    """Define threshold-based change detection models.

    Creates two models:
    1. Sum model: y = 1[sum_i x_i > threshold_sum]
    2. Average model: y = 1[mean(x) > threshold_avg]

    Args:
        threshold_sum (float): Sum threshold.
        threshold_avg (float): Average threshold.

    Returns:
        Tuple[Callable, Callable]: (sum_model, avg_model) functions.
    """
    logger.debug(
        f"Defining models with thresholds - sum: {threshold_sum}, avg: {threshold_avg}"
    )

    def model_sum(X: np.ndarray) -> np.ndarray:
        return (np.sum(X, axis=1) > threshold_sum).astype(int)

    def model_avg(X: np.ndarray) -> np.ndarray:
        return (np.mean(X, axis=1) > threshold_avg).astype(int)

    return model_sum, model_avg


def compute_shap_values(
    model: Callable, background: np.ndarray, X_explain: np.ndarray
) -> np.ndarray:
    """Compute SHAP values for model interpretability.
    Uses KernelExplainer to compute Shapley values:
    phi_i = sum_S (1 - |S| / n) * (f(x_S + i) - f(x_S))
    where x_S is the subset of features in S.
    """
    try:
        logger.info("Computing SHAP values")
        logger.debug(
            f"Data shapes - background: {background.shape}, explain: {X_explain.shape}"
        )

        explainer = shap.KernelExplainer(model, background)
        shap_values = explainer.shap_values(X_explain)

        logger.debug(f"Generated SHAP values of shape {np.array(shap_values).shape}")
        return shap_values

    except Exception as e:
        logger.error(f"SHAP computation failed: {str(e)}")
        raise RuntimeError(f"SHAP computation failed: {str(e)}")


def plot_martingales_subplot(
    ax: plt.Axes,
    instants: np.ndarray,
    X: np.ndarray,
    sum_x: np.ndarray,
    avg_x: np.ndarray,
    equations: List[Tuple[float, float]],
    colors: List[str],
    title: str = "Martingales and Their Sum/Average",
    save_plots: bool = False,
    output_dir: str = "",
) -> None:
    """Plot martingale sequences and their aggregations."""
    try:
        logger.debug(f"Plotting martingales for {title}")

        ax.plot(instants, sum_x, color="cyan", label="Martingale Sum", linewidth=1)
        ax.plot(instants, avg_x, color="teal", label="Martingale Average", linewidth=1)

        for idx, (slope, intercept) in enumerate(equations):
            label = f"y={slope}x+{intercept}"
            ax.plot(
                instants,
                X[:, idx],
                color=colors[idx],
                label=label,
                linewidth=1,
            )
            logger.debug(f"Added line for equation: {label}")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Martingale Values")
        ax.legend(fontsize="small")

        if save_plots:
            save_path = Path(output_dir) / f"{title.replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved martingale plot to {save_path}")

    except Exception as e:
        logger.error(f"Martingale plotting failed: {str(e)}")
        raise RuntimeError(f"Martingale plotting failed: {str(e)}")


def plot_shap_values_subplot(
    ax: plt.Axes,
    instants_explain: np.ndarray,
    shap_values: np.ndarray,
    equations: List[Tuple[float, float]],
    colors: List[str],
    model_name: str = "Model",
    save_plots: bool = False,
    output_dir: str = "",
) -> None:
    """Plot SHAP values over time."""
    try:
        logger.debug(f"Plotting SHAP values for {model_name}")

        for idx, (slope, intercept) in enumerate(equations):
            label = f"SHAP y={slope}x+{intercept}"
            ax.plot(
                instants_explain,
                shap_values[:, idx],
                color=colors[idx],
                label=label,
            )
            logger.debug(f"Added SHAP values for equation: {label}")

        ax.set_title(f"SHAP Values over Time for {model_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("SHAP Value")
        ax.legend(fontsize="small")

        if save_plots:
            save_path = Path(output_dir) / f"{model_name.replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP plot to {save_path}")

    except Exception as e:
        logger.error(f"SHAP plotting failed: {str(e)}")
        raise RuntimeError(f"SHAP plotting failed: {str(e)}")


def run_scenario_dashboard(
    axes: List[plt.Axes],
    config: Config,
    scenario: Config,
    save_plots: bool,
    output_dir: str = "",
) -> None:
    """Generate and visualize scenario analysis results."""
    try:
        logger.info(f"Running scenario: {scenario.name}")

        # Get equations and generate data
        equations = getattr(config.equations, scenario.equations)
        equations = [(slope, intercept) for slope, intercept in equations]
        instants = np.arange(config.time.total_instants)

        logger.debug(
            f"Using {len(equations)} equations over {len(instants)} time instants"
        )

        # Generate martingales
        X = generate_martingales(instants, equations, scenario.change_point)
        sum_x = np.sum(X, axis=1)
        avg_x = np.mean(X, axis=1)
        logger.debug(f"Generated martingale matrix of shape {X.shape}")

        # Prepare SHAP analysis
        logger.info("Preparing SHAP analysis")
        background_full = X[: scenario.change_point + 1, :]
        background = shap.sample(background_full, config.shap.background_samples)
        logger.debug(f"Using {len(background)} background samples")

        X_explain = (
            X if config.shap.compute_full_range else X[scenario.change_point + 1 :, :]
        )
        instants_explain = (
            instants
            if config.shap.compute_full_range
            else instants[scenario.change_point + 1 :]
        )
        logger.debug(f"Explanation range: {len(X_explain)} samples")

        # Define and analyze models
        logger.info("Running model analysis")
        model_sum, model_avg = define_models(
            scenario.thresholds.sum, scenario.thresholds.avg
        )
        shap_values_sum = compute_shap_values(model_sum, background, X_explain)
        shap_values_avg = compute_shap_values(model_avg, background, X_explain)

        # Create visualizations
        logger.info("Creating visualizations")
        plot_martingales_subplot(
            axes[0],
            instants,
            X,
            sum_x,
            avg_x,
            equations,
            colors=["orange", "lightgreen", "red", "purple"],
            title=scenario.name,
            save_plots=save_plots,
            output_dir=output_dir,
        )
        plot_shap_values_subplot(
            axes[1],
            instants_explain,
            shap_values_sum,
            equations,
            colors=["orange", "lightgreen", "red", "purple"],
            model_name="Model Sum",
            save_plots=save_plots,
            output_dir=output_dir,
        )
        plot_shap_values_subplot(
            axes[2],
            instants_explain,
            shap_values_avg,
            equations,
            colors=["orange", "lightgreen", "red", "purple"],
            model_name="Model Average",
            save_plots=save_plots,
            output_dir=output_dir,
        )

        logger.info("Scenario analysis complete")

    except Exception as e:
        logger.error(f"Scenario analysis failed: {str(e)}")
        raise RuntimeError(f"Scenario analysis failed: {str(e)}")


def main(config_path: str) -> None:
    """Execute linear model analysis pipeline."""
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        save_plots = config.visualization.save_plots
        output_dir = config.paths.output.dir

        # Configure visualization
        plt.rcParams.update({"font.size": 8})

        # Get number of scenarios
        scenarios = [
            s
            for s in config.scenarios.__dict__.values()
            if not s.__str__().startswith("_")
        ]
        num_scenarios = len(scenarios)
        logger.info(f"Found {num_scenarios} scenarios to analyze")

        # Create figure
        fig, axes = plt.subplots(
            nrows=num_scenarios,
            ncols=3,
            figsize=(18, 4 * num_scenarios),
        )

        # Run scenarios
        for idx, scenario in enumerate(scenarios):
            logger.info(f"Processing scenario {idx+1}/{num_scenarios}: {scenario.name}")
            scenario_axes = axes[idx] if num_scenarios > 1 else axes
            run_scenario_dashboard(
                scenario_axes, config, scenario, save_plots, output_dir
            )

        plt.tight_layout()
        logger.debug("Displaying results")
        plt.show()

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


# if __name__ == "__main__":
#     main("config/linear_models.yaml")
