#!/usr/bin/env python

"""Entry point for running the graph change point detection pipeline."""

import argparse
import logging
import sys

from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection

logger = logging.getLogger(__name__)


def main(
    config_path,
    prediction=None,
    visualize=None,
    save_csv=None,
):
    """
    Run the detection pipeline with the given configuration.

    Args:
        config_path: Path to YAML configuration file
        prediction: Whether to generate and use predictions for detection.
                   If None, uses the value from the config file.
        visualize: Whether to create visualizations of the results.
                   If None, uses the value from the config file.
        save_csv: Whether to save results to CSV files.
                 If None, uses the value from the config file.

    Returns:
        Dictionary containing all results
    """
    try:
        pipeline = GraphChangeDetection(config_path)

        # Ensure martingale data is saved for analysis
        if "output" not in pipeline.config:
            pipeline.config["output"] = {}
        pipeline.config["output"]["save_martingales"] = True

        logger.info("Starting pipeline execution")
        if prediction is not None:
            logger.info(f"Prediction override: {prediction}")

        results = pipeline.run(
            prediction=prediction, visualize=visualize, save_csv=save_csv
        )

        # Log summary information about the results
        if results and "features" in results:
            logger.info(
                f"Final data dimensions: {results['features'].shape} (timesteps × features)"
            )

        if (
            results
            and "predicted_features" in results
            and results["predicted_features"] is not None
        ):
            pred_features = results["predicted_features"]
            logger.info(f"Final prediction dimensions: {pred_features.shape}")
            logger.info(
                f"Predictions were available for {len(pred_features)} timesteps"
            )

        # Print out martingale streams for analysis
        if results:
            # Look for martingale data directly in results (from prepare_result_data)
            # Print traditional martingale values
            if "traditional_sum_martingales" in results:
                trad_mart = results["traditional_sum_martingales"]
                logger.info(f"Traditional Martingale Stream (showing every 5th value):")
                for i in range(0, len(trad_mart), 5):
                    values = [
                        f"{trad_mart[j]:.4f}"
                        for j in range(i, min(i + 5, len(trad_mart)))
                    ]
                    logger.info(
                        f"  t=[{i}-{min(i+4, len(trad_mart)-1)}]: {', '.join(values)}"
                    )

            # Print horizon martingale values if available
            if "horizon_sum_martingales" in results:
                horizon_mart = results["horizon_sum_martingales"]
                prediction_start = results.get("prediction_start_time", 0)

                logger.info(
                    f"Horizon Martingale Stream (showing every 5th value, starting at t={prediction_start}):"
                )
                # Start printing from prediction_start
                for i in range(prediction_start, len(horizon_mart), 5):
                    values = [
                        f"{horizon_mart[j]:.4f}"
                        for j in range(i, min(i + 5, len(horizon_mart)))
                    ]
                    logger.info(
                        f"  t=[{i}-{min(i+4, len(horizon_mart)-1)}]: {', '.join(values)}"
                    )

                # Print high values in horizon martingale to find significant points
                threshold = 10.0  # Show values over this threshold
                high_values = [
                    (i, val) for i, val in enumerate(horizon_mart) if val > threshold
                ]
                if high_values:
                    logger.info(f"High horizon martingale values (>{threshold}):")
                    for i, val in high_values:
                        logger.info(f"  t={i}: {val:.4f}")

                # If prediction_start is provided, make it clear
                if "prediction_start_time" in results:
                    logger.info(
                        f"Note: Horizon martingale values before t={prediction_start} are not meaningful (no predictions available yet)"
                    )

            # Print true change points if available
            if "true_change_points" in results:
                logger.info(f"True change points: {results['true_change_points']}")

            # Print detected change points if available
            if "traditional_change_points" in results:
                logger.info(
                    f"Traditional change points detected: {results['traditional_change_points']}"
                )
            if "horizon_change_points" in results:
                logger.info(
                    f"Horizon change points detected: {results['horizon_change_points']}"
                )

        logger.info("Pipeline execution completed successfully")
        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Run graph change detection pipeline")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument(
        "--no-prediction", action="store_true", help="Disable prediction in detection"
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Disable result visualization"
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Disable CSV export of results"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Convert command-line flags to parameter values
    prediction = not args.no_prediction if args.no_prediction else None
    visualize = not args.no_visualization if args.no_visualization else None
    save_csv = not args.no_csv if args.no_csv else None

    # Run the pipeline
    try:
        main(
            args.config_path,
            prediction=prediction,
            visualize=visualize,
            save_csv=save_csv,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        sys.exit(1)
