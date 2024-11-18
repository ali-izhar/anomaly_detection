# tests/run_synthetic.py

"""Synthetic data analysis module for change point detection.

Implements pipeline for generating and analyzing synthetic graph sequences:
1. Graph generation with controlled structural changes
2. Feature extraction and martingale-based detection
3. SHAP-based interpretability analysis
4. Comprehensive visualization
"""

import numpy as np
import shap  # type: ignore
from pathlib import Path
from typing import Dict, List, Any

from sklearn.model_selection import train_test_split

from config.config import load_config
from src.changepoint.detector import ChangePointDetector
from src.graph.graph_generator import GraphGenerator
from src.models.models import CustomThresholdModel
from src.utils import SyntheticDataVisualizer
from src.utils.log_handling import get_logger

logger = get_logger(__name__)


def run_synthetic_pipeline(config_path: str) -> None:
    """Execute anomaly detection pipeline for synthetic graph sequences."""
    try:
        # Load configuration
        if not Path(config_path).exists():
            logger.critical(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Initialize components
        logger.debug("Initializing pipeline components")
        graph_generator = GraphGenerator()

        # Generate graphs for each model type
        generated_graphs = {}

        # Generate BA graphs
        logger.info("Generating Barabasi-Albert graphs")
        try:
            ba_params = _extract_ba_params(config)
            generated_graphs["barabasi_albert"] = graph_generator.barabasi_albert(
                **ba_params
            )
            logger.debug(f"Generated BA graphs with parameters: {ba_params}")
        except Exception as e:
            logger.error(f"Failed to generate BA graphs: {str(e)}")

        # Generate ER graphs
        logger.info("Generating Erdos-Renyi graphs")
        try:
            er_params = _extract_er_params(config)
            generated_graphs["erdos_renyi"] = graph_generator.erdos_renyi(**er_params)
            logger.debug(f"Generated ER graphs with parameters: {er_params}")
        except Exception as e:
            logger.error(f"Failed to generate ER graphs: {str(e)}")

        # Generate NW graphs
        logger.info("Generating Newman-Watts graphs")
        try:
            nw_params = _extract_nw_params(config)
            generated_graphs["newman_watts"] = graph_generator.newman_watts(**nw_params)
            logger.debug(f"Generated NW graphs with parameters: {nw_params}")
        except Exception as e:
            logger.error(f"Failed to generate NW graphs: {str(e)}")

        # Initialize other components
        visualizer = SyntheticDataVisualizer()
        detector = ChangePointDetector()

        # Process each graph type
        for graph_type, graphs in generated_graphs.items():
            if graphs:  # Only process if generation was successful
                logger.info(f"Processing {graph_type} graphs")
                try:
                    analyze_graph_sequence(
                        graphs=graphs,
                        graph_type=graph_type,
                        detector=detector,
                        visualizer=visualizer,
                        config=config,
                    )
                except Exception as e:
                    logger.error(f"Failed to analyze {graph_type} graphs: {str(e)}")
                    continue

        logger.info("Synthetic data analysis completed successfully")

    except Exception as e:
        logger.critical(f"Synthetic analysis failed: {str(e)}")
        raise


def analyze_graph_sequence(
    graphs: List[np.ndarray],
    graph_type: str,
    detector: ChangePointDetector,
    visualizer: SyntheticDataVisualizer,
    config: Any,
) -> None:
    """Analyze single graph sequence for structural changes."""
    try:
        logger.info(f"Starting analysis of {graph_type} graphs")

        # Extract features
        logger.debug(f"Extracting features for {graph_type}")
        detector.initialize(graphs)
        centralities = detector.extract_features()
        logger.debug(f"Extracted {len(centralities)} features")

        # Compute martingales with detection
        logger.debug("Computing martingales with detection")
        martingales_detect = {
            cent: detector.martingale_test(
                centralities[cent],
                threshold=config.analysis.parameters.threshold,
                detect=True,
            )
            for cent in config.analysis.centrality_metrics
        }
        logger.debug(
            f"Found {sum(len(m['change_detected_instant']) for m in martingales_detect.values())} change points"
        )

        # Compute martingales without detection
        logger.debug("Computing martingales without detection")
        martingales_no_detect = {
            cent: detector.martingale_test(
                centralities[cent],
                threshold=config.analysis.parameters.threshold,
                detect=False,
            )
            for cent in config.analysis.centrality_metrics
        }

        # Prepare data for SHAP analysis
        logger.debug("Preparing SHAP analysis")
        X = np.vstack([m["martingales"] for m in martingales_no_detect.values()]).T
        y = create_labels(
            length=len(X),
            change_point=config.analysis.parameters.change_point,
            high=config.analysis.labels.high,
            low=config.analysis.labels.low,
        )

        # Split data and train model
        logger.debug("Training model for SHAP analysis")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.analysis.parameters.test_size,
            random_state=config.analysis.parameters.random_state,
        )

        model = CustomThresholdModel(
            threshold=config.analysis.parameters.model_threshold
        )
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        logger.info(f"Model accuracy for {graph_type}: {score}")
        if score < 0.7:  # Example threshold
            logger.warning(f"Low model accuracy ({score}) for {graph_type}")

        # Compute SHAP values
        logger.debug("Computing SHAP values")
        shap_values = compute_shap_values(model, X_train, X)

        # Create visualization
        logger.debug("Creating visualization dashboard")
        create_dashboard(
            graphs=graphs,
            graph_type=graph_type,
            martingales_detect=martingales_detect,
            martingales_no_detect=martingales_no_detect,
            shap_values=shap_values,
            centralities=centralities,
            config=config,
            visualizer=visualizer,
        )

        logger.info(f"Completed analysis for {graph_type}")

    except Exception as e:
        logger.error(f"Graph sequence analysis failed for {graph_type}: {str(e)}")
        raise


def create_labels(length: int, change_point: int, high: int, low: int) -> List[int]:
    """Create binary labels for change point detection.

    Labels:
    - y[t] = high for t >= change_point
    - y[t] = low for t < change_point

    Args:
        length: Sequence length
        change_point: Change time index
        high: Label for change
        low: Label for no change

    Returns:
        Binary label sequence
    """
    logger.debug(f"Creating labels with change point at {change_point}")
    return [high if i >= change_point else low for i in range(length)]


def compute_shap_values(
    model: CustomThresholdModel, background: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Compute SHAP values for model interpretability."""
    try:
        logger.debug(
            f"Computing SHAP values with shapes - background: {background.shape}, X: {X.shape}"
        )
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)
        logger.debug(f"Generated SHAP values of shape {np.array(shap_values).shape}")
        return shap_values
    except Exception as e:
        logger.error(f"SHAP computation failed: {str(e)}")
        raise RuntimeError(f"SHAP computation failed: {str(e)}")


def create_dashboard(
    graphs: List[np.ndarray],
    graph_type: str,
    martingales_detect: Dict[str, Dict[str, Any]],
    martingales_no_detect: Dict[str, Dict[str, Any]],
    shap_values: np.ndarray,
    centralities: Dict[str, List[List[float]]],
    config: Any,
    visualizer: SyntheticDataVisualizer,
) -> None:
    """Create comprehensive visualization dashboard."""
    try:
        logger.info(f"Creating dashboard for {graph_type}")

        # Select sample graphs
        num_samples = min(config.visualization.num_graph_samples, len(graphs))
        sample_indices = np.linspace(
            0, len(graphs) - 1, num_samples, dtype=int
        ).tolist()
        sample_graphs = [graphs[i] for i in sample_indices]
        logger.debug(f"Selected {num_samples} sample graphs")

        # Create description
        description = (
            f"{graph_type.replace('_', ' ').title()} graphs generated with "
            f"specified parameters."
        )

        # Generate dashboard
        logger.debug("Generating comprehensive dashboard")
        visualizer.plot_comprehensive_dashboard(
            sample_graphs=sample_graphs,
            sample_indices=sample_indices,
            graph_type=graph_type.replace("_", " ").title(),
            description=description,
            martingales_detect=martingales_detect,
            martingales_no_detect=martingales_no_detect,
            shap_values=shap_values,
            centralities=centralities,
            config=config,
        )
        logger.info("Dashboard creation complete")

    except Exception as e:
        logger.error(f"Dashboard creation failed: {str(e)}")
        raise RuntimeError(f"Dashboard creation failed: {str(e)}")


def main(config_path: str) -> None:
    """Main entry point for synthetic data experiments."""
    try:
        logger.info(f"Starting synthetic data analysis with config: {config_path}")
        run_synthetic_pipeline(config_path)
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Synthetic analysis failed: {str(e)}")
        raise


def _extract_ba_params(config: Any) -> Dict[str, Any]:
    """Extract Barabási-Albert parameters from config."""
    logger.debug("Extracting BA parameters")
    ba_config = config.graph.barabasi_albert
    params = {
        "n": ba_config.n,
        "m1": ba_config.edges.initial,
        "m2": ba_config.edges.changed,
        "set1": ba_config.set1,
        "set2": ba_config.set2,
    }
    logger.debug(f"BA parameters: {params}")
    return params


def _extract_er_params(config: Any) -> Dict[str, Any]:
    """Extract Erdos-Renyi parameters from config."""
    logger.debug("Extracting ER parameters")
    er_config = config.graph.erdos_renyi
    params = {
        "n": er_config.n,
        "p1": er_config.probabilities.initial,
        "p2": er_config.probabilities.changed,
        "set1": er_config.set1,
        "set2": er_config.set2,
    }
    logger.debug(f"ER parameters: {params}")
    return params


def _extract_nw_params(config: Any) -> Dict[str, Any]:
    """Extract Newman-Watts parameters from config."""
    logger.debug("Extracting NW parameters")
    nw_config = config.graph.newman_watts
    params = {
        "n": nw_config.n,
        "p1": nw_config.rewiring_prob.initial,
        "p2": nw_config.rewiring_prob.changed,
        "k1": nw_config.neighbors.initial,
        "k2": nw_config.neighbors.changed,
        "set1": nw_config.set1,
        "set2": nw_config.set2,
    }
    logger.debug(f"NW parameters: {params}")
    return params


# if __name__ == "__main__":
#     main("config/synthetic_data_config.yaml")
