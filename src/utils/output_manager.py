# src/utils/output_manager.py

"""Output manager for exporting change detection results to CSV."""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutputManager:
    """A minimal output manager that exports change detection results to CSV."""

    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the output manager.

        Args:
            output_dir: Directory to save output files
            config: Configuration parameters from the algorithm
        """
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(output_dir, exist_ok=True)

    def export_to_csv(
        self,
        detection_results: Dict[str, Any],
        true_change_points: List[int],
        individual_trials: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Export detection results to a CSV file with multiple sheets.

        Args:
            detection_results: Aggregated detection results (not used, kept for compatibility)
            true_change_points: List of actual change points
            individual_trials: List of results from individual trials, each will get its own sheet
        """
        try:
            # Determine the number of timesteps from configuration
            n_timesteps = self._get_timestep_count()
            logger.debug(f"Creating Excel file with {n_timesteps} timesteps")

            # Skip export if no individual trials are provided
            if not individual_trials or not any(individual_trials):
                logger.warning("No individual trial results to export")
                return

            # Create Excel writer with pandas
            excel_path = os.path.join(self.output_dir, "detection_results.xlsx")
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

                # Create individual trial sheets
                for i, trial_result in enumerate(individual_trials):
                    if trial_result:  # Check if trial result is not None
                        sheet_name = f"Trial{i+1}"
                        df_trial = self._create_results_dataframe(
                            trial_result, true_change_points, n_timesteps
                        )
                        df_trial.to_excel(writer, sheet_name=sheet_name, index=False)

                # Also add a summary sheet with detection information
                if individual_trials:
                    self._create_detection_summary(
                        individual_trials, true_change_points, writer
                    )

                    # Add expanded detection details sheet
                    self._create_expanded_detection_details(
                        individual_trials, true_change_points, writer
                    )

            logger.info(f"Results saved to {excel_path}")

        except Exception as e:
            logger.error(f"Error exporting results to CSV: {str(e)}")
            raise  # Re-raise the exception for better debugging

    def _create_detection_summary(
        self,
        individual_trials: List[Dict[str, Any]],
        true_change_points: List[int],
        writer: pd.ExcelWriter,
    ) -> None:
        """Create a summary sheet showing detection info across trials.

        Args:
            individual_trials: List of results from individual trials
            true_change_points: List of actual change points
            writer: Excel writer object to append the sheet to
        """
        # Create summary dataframe - simplify to just count detections
        summary_data = {
            "True CP Index": true_change_points,
        }

        # Count detections per trial
        for i, trial in enumerate(individual_trials):
            # Traditional martingale detections
            if "traditional_change_points" in trial:
                # Use detections directly without adjusting points
                detections = trial["traditional_change_points"]
                summary_data[f"Trial {i+1} Traditional Detection Count"] = [
                    self._count_detections_near_cp(detections, cp)
                    for cp in true_change_points
                ]

                # Record detection latency (distance from true CP to first detection)
                summary_data[f"Trial {i+1} Traditional Latency"] = [
                    self._calc_detection_latency(detections, cp)
                    for cp in true_change_points
                ]

            # Horizon martingale detections
            if "horizon_change_points" in trial:
                # Use detections directly without adjusting points
                horizon_detections = trial["horizon_change_points"]
                summary_data[f"Trial {i+1} Horizon Detection Count"] = [
                    self._count_detections_near_cp(horizon_detections, cp)
                    for cp in true_change_points
                ]

                # Record detection latency for horizon detections
                summary_data[f"Trial {i+1} Horizon Latency"] = [
                    self._calc_detection_latency(horizon_detections, cp)
                    for cp in true_change_points
                ]

        # Create DataFrame
        df_summary = pd.DataFrame(summary_data)

        # Add total and average rows
        if len(individual_trials) > 1:
            trial_cols = [col for col in df_summary.columns if "Detection Count" in col]
            if trial_cols:
                df_summary.loc["Total"] = ["Total"] + [
                    df_summary[col].sum() for col in df_summary.columns[1:]
                ]
                df_summary.loc["Average"] = ["Average"] + [
                    df_summary[col].mean() if "Count" in col else np.nan
                    for col in df_summary.columns[1:]
                ]

        df_summary.to_excel(writer, sheet_name="Detection Summary", index=False)

    def _create_expanded_detection_details(
        self,
        individual_trials: List[Dict[str, Any]],
        true_change_points: List[int],
        writer: pd.ExcelWriter,
    ) -> None:
        """Create a detailed sheet showing all detections for each trial.

        Args:
            individual_trials: List of results from individual trials
            true_change_points: List of actual change points
            writer: Excel writer object to append the sheet to
        """
        details_rows = []

        # For each trial, list all detections
        for i, trial in enumerate(individual_trials):
            trial_num = i + 1

            # Process traditional martingale detections
            if "traditional_change_points" in trial:
                detections = trial["traditional_change_points"]

                # Process each detection point
                for j, detection_point in enumerate(detections):
                    # Find the nearest true change point
                    nearest_cp, distance = self._find_nearest_cp(
                        detection_point, true_change_points
                    )

                    details_rows.append(
                        {
                            "Trial": trial_num,
                            "Type": "Traditional",
                            "Detection #": j + 1,
                            "Detection Index": detection_point,
                            "Nearest True CP": nearest_cp,
                            "Distance to CP": distance,
                            "Is Within 10 Steps": abs(distance) <= 10,
                        }
                    )

            # Process horizon martingale detections
            if "horizon_change_points" in trial:
                horizon_detections = trial["horizon_change_points"]

                # Process each detection point
                for j, detection_point in enumerate(horizon_detections):
                    # Find the nearest true change point
                    nearest_cp, distance = self._find_nearest_cp(
                        detection_point, true_change_points
                    )

                    details_rows.append(
                        {
                            "Trial": trial_num,
                            "Type": "Horizon",
                            "Detection #": j + 1,
                            "Detection Index": detection_point,
                            "Nearest True CP": nearest_cp,
                            "Distance to CP": distance,
                            "Is Within 10 Steps": abs(distance) <= 10,
                        }
                    )

        # Create detailed DataFrame
        if details_rows:
            df_details = pd.DataFrame(details_rows)
            df_details.to_excel(writer, sheet_name="Detection Details", index=False)
        else:
            # Create empty dataframe with headers if no detections
            df_details = pd.DataFrame(
                columns=[
                    "Trial",
                    "Type",
                    "Detection #",
                    "Detection Index",
                    "Nearest True CP",
                    "Distance to CP",
                    "Is Within 10 Steps",
                ]
            )
            df_details.to_excel(writer, sheet_name="Detection Details", index=False)

    def _count_detections_near_cp(
        self, detections: List[int], change_point: int, window: int = 10
    ) -> int:
        """Count detections that occur within a window of a change point.

        Args:
            detections: List of detection points
            change_point: True change point to check against
            window: Window size to consider a detection associated with the change point

        Returns:
            Number of detections within the window of the change point
        """
        return sum(1 for d in detections if abs(d - change_point) <= window)

    def _calc_detection_latency(
        self, detections: List[int], change_point: int, window: int = 20
    ) -> int:
        """Calculate detection latency (time from change point to first detection).

        Args:
            detections: List of detection points
            change_point: True change point to check against
            window: Maximum window to consider for latency calculation

        Returns:
            Detection latency or np.nan if no detection within window
        """
        # Find detections that occur after the change point and within window
        valid_detections = [
            d for d in detections if d >= change_point and d - change_point <= window
        ]

        if valid_detections:
            # Return the minimum distance (first detection)
            return min(valid_detections) - change_point
        return np.nan

    def _find_nearest_cp(
        self, detection: int, change_points: List[int]
    ) -> Tuple[int, int]:
        """Find the nearest true change point to a detection.

        Args:
            detection: Detection point
            change_points: List of true change points

        Returns:
            Tuple of (nearest CP, distance to CP)
        """
        if not change_points:
            return (None, np.inf)

        nearest_cp = min(change_points, key=lambda cp: abs(cp - detection))
        distance = (
            detection - nearest_cp
        )  # Positive means detection after CP, negative means before

        return (nearest_cp, distance)

    def _create_results_dataframe(
        self,
        detection_results: Dict[str, Any],
        true_change_points: List[int],
        n_timesteps: int,
    ) -> pd.DataFrame:
        """Create a dataframe with detection results.

        Args:
            detection_results: Detection results to include in the dataframe
            true_change_points: List of actual change points
            n_timesteps: Number of timesteps

        Returns:
            DataFrame with detection results
        """
        # Create a base dataframe with timesteps
        df_data = {"timestep": list(range(n_timesteps))}

        # Add change point indicators
        df_data["true_change_point"] = [
            1 if t in true_change_points else 0 for t in range(n_timesteps)
        ]

        # Get threshold from config
        threshold = 60.0  # Default value
        if (
            self.config
            and "detection" in self.config
            and "threshold" in self.config["detection"]
        ):
            threshold = self.config["detection"]["threshold"]

        # Add martingale values to the dataframe (both traditional and horizon)
        # Define the martingale keys we want to include
        traditional_martingale_keys = [
            "traditional_martingales",
            "traditional_sum_martingales",
            "traditional_avg_martingales",
        ]

        horizon_martingale_keys = [
            "horizon_martingales",
            "horizon_sum_martingales",
            "horizon_avg_martingales",
        ]

        # Add individual feature martingales if they exist
        individual_martingale_keys = []
        for key in detection_results.keys():
            if key.startswith("individual_") and key.endswith("_martingales"):
                individual_martingale_keys.append(key)

        # Add individual feature martingales
        for key in individual_martingale_keys:
            if key in detection_results:
                individual_martingales = detection_results[key]
                # Each feature has its own array of martingales
                for i, feature_martingales in enumerate(individual_martingales):
                    if len(feature_martingales) > 0:
                        feature_key = f"{key}_feature{i}"
                        max_idx = min(len(feature_martingales), n_timesteps)
                        df_data[feature_key] = list(feature_martingales[:max_idx]) + [
                            None
                        ] * (n_timesteps - max_idx)

        # Add traditional martingale values
        for martingale_key in traditional_martingale_keys:
            if martingale_key in detection_results:
                martingale_values = detection_results[martingale_key]
                if len(martingale_values) > 0:
                    max_idx = min(len(martingale_values), n_timesteps)
                    df_data[martingale_key] = list(martingale_values[:max_idx]) + [
                        None
                    ] * (n_timesteps - max_idx)

                    # Immediately after adding a martingale column, add its detection column if it's a sum martingale
                    if martingale_key == "traditional_sum_martingales":
                        # Initialize all detection flags to 0
                        df_data["traditional_detected"] = [0] * n_timesteps

                        # Set flag to 1 at indices where martingale exceeds threshold
                        for i in range(max_idx):
                            if (
                                df_data[martingale_key][i] is not None
                                and df_data[martingale_key][i] > threshold
                            ):
                                df_data["traditional_detected"][i] = 1

        # Add horizon martingale values
        for martingale_key in horizon_martingale_keys:
            if martingale_key in detection_results:
                martingale_values = detection_results[martingale_key]
                if len(martingale_values) > 0:
                    max_idx = min(len(martingale_values), n_timesteps)
                    df_data[martingale_key] = list(martingale_values[:max_idx]) + [
                        None
                    ] * (n_timesteps - max_idx)

                    # Add detection column for horizon sum martingales
                    if martingale_key == "horizon_sum_martingales":
                        # Initialize all detection flags to 0
                        df_data["horizon_detected"] = [0] * n_timesteps

                        # Set flag to 1 at indices where martingale exceeds threshold
                        for i in range(max_idx):
                            if (
                                df_data[martingale_key][i] is not None
                                and df_data[martingale_key][i] > threshold
                            ):
                                df_data["horizon_detected"][i] = 1

        # Store actual detection points for internal reference
        if "traditional_change_points" in detection_results:
            self._traditional_detection_points = detection_results[
                "traditional_change_points"
            ]

        # Store horizon detection points for internal reference
        if "horizon_change_points" in detection_results:
            self._horizon_detection_points = detection_results["horizon_change_points"]

        # Create the dataframe with a specific column order
        columns = ["timestep", "true_change_point"]

        # Add individual feature columns
        individual_feature_columns = [
            col for col in df_data.keys() if col.startswith("individual_")
        ]
        columns.extend(sorted(individual_feature_columns))

        # Add traditional columns
        if "traditional_martingales" in df_data:
            columns.append("traditional_martingales")
        if "traditional_sum_martingales" in df_data:
            columns.append("traditional_sum_martingales")
            columns.append(
                "traditional_detected"
            )  # Put detection flag immediately after sum
        if "traditional_avg_martingales" in df_data:
            columns.append("traditional_avg_martingales")

        # Add horizon columns
        if "horizon_martingales" in df_data:
            columns.append("horizon_martingales")
        if "horizon_sum_martingales" in df_data:
            columns.append("horizon_sum_martingales")
            columns.append(
                "horizon_detected"
            )  # Put detection flag immediately after sum
        if "horizon_avg_martingales" in df_data:
            columns.append("horizon_avg_martingales")

        # Create DataFrame with specific column order
        return pd.DataFrame({col: df_data[col] for col in columns if col in df_data})

    def _get_timestep_count(self) -> int:
        """Determine the number of timesteps from configuration.

        Returns:
            Number of timesteps (defaults to 200 if not found)
        """
        try:
            # Try to get sequence length from model parameters
            if "model" in self.config and "params" in self.config:
                params = self.config["params"]
                if hasattr(params, "seq_len"):
                    return params.seq_len

            # Try to access params as dictionary
            if "params" in self.config and isinstance(self.config["params"], dict):
                params = self.config["params"]
                if "seq_len" in params:
                    return params["seq_len"]

            # If that fails, look for sequence configuration in detection settings
            if "detection" in self.config and "sequence" in self.config["detection"]:
                sequence = self.config["detection"]["sequence"]
                if "length" in sequence:
                    return sequence["length"]

            # Default fallback value
            return 200

        except Exception as e:
            logger.warning(f"Could not determine timestep count from config: {str(e)}")
            return 200  # Default fallback
