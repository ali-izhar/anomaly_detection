# src/utils/output_manager.py

"""Output manager for exporting change detection results to CSV."""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants for OutputManager
OUTPUT_CONSTANTS = {
    # File paths and output names
    "FILES": {
        "EXCEL_FILENAME": "detection_results.xlsx",
    },
    # Sheet names
    "SHEETS": {
        "TRIAL_PREFIX": "Trial",
        "DETECTION_SUMMARY": "Detection Summary",
        "DETECTION_DETAILS": "Detection Details",
    },
    # Detection result keys
    "DETECTION_KEYS": {
        "TRADITIONAL_CHANGE_POINTS": "traditional_change_points",
        "HORIZON_CHANGE_POINTS": "horizon_change_points",
        "TRADITIONAL_MARTINGALES": "traditional_martingales",
        "TRADITIONAL_SUM_MARTINGALES": "traditional_sum_martingales",
        "TRADITIONAL_AVG_MARTINGALES": "traditional_avg_martingales",
        "HORIZON_MARTINGALES": "horizon_martingales",
        "HORIZON_SUM_MARTINGALES": "horizon_sum_martingales",
        "HORIZON_AVG_MARTINGALES": "horizon_avg_martingales",
        "INDIVIDUAL_TRADITIONAL_MARTINGALES": "individual_traditional_martingales",
        "INDIVIDUAL_HORIZON_MARTINGALES": "individual_horizon_martingales",
    },
    # Column names and labels
    "COLUMNS": {
        # Base columns
        "TIMESTEP": "timestep",
        "TRUE_CHANGE_POINT": "true_change_point",
        "TRADITIONAL_DETECTED": "traditional_detected",
        "HORIZON_DETECTED": "horizon_detected",
        # Summary sheet columns
        "TRUE_CP_INDEX": "True CP Index",
        "TRIAL_TRADITIONAL_PREFIX": "Trial {} Traditional Detection Count",
        "TRIAL_TRADITIONAL_LATENCY": "Trial {} Traditional Latency",
        "TRIAL_HORIZON_PREFIX": "Trial {} Horizon Detection Count",
        "TRIAL_HORIZON_LATENCY": "Trial {} Horizon Latency",
        "TOTAL": "Total",
        "AVERAGE": "Average",
        # Detail sheet columns
        "TRIAL": "Trial",
        "TYPE": "Type",
        "DETECTION_NUM": "Detection #",
        "RAW_DETECTION": "Raw Detection Index",
        "ADJUSTED_DETECTION": "Adjusted Detection Index",
        "NEAREST_CP": "Nearest True CP",
        "DISTANCE_TO_CP": "Distance to CP",
        "IS_WITHIN_RANGE": "Is Within 10 Steps",
    },
    # Detection types
    "TYPES": {
        "TRADITIONAL": "Traditional",
        "HORIZON": "Horizon",
    },
    # Config path keys
    "CONFIG_KEYS": {
        "DETECTION": "detection",
        "THRESHOLD": "threshold",
        "MODEL": "model",
        "PARAMS": "params",
        "SEQ_LEN": "seq_len",
        "SEQUENCE": "sequence",
        "LENGTH": "length",
    },
    # Default values
    "DEFAULTS": {
        "THRESHOLD": 60.0,
        "PROXIMITY_WINDOW": 10,
        "LATENCY_WINDOW": 20,
        "SEQUENCE_LENGTH": 200,
    },
}


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
            excel_path = os.path.join(
                self.output_dir, OUTPUT_CONSTANTS["FILES"]["EXCEL_FILENAME"]
            )
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

                # Create individual trial sheets
                for i, trial_result in enumerate(individual_trials):
                    if trial_result:  # Check if trial result is not None
                        sheet_name = (
                            f"{OUTPUT_CONSTANTS['SHEETS']['TRIAL_PREFIX']}{i+1}"
                        )
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
            OUTPUT_CONSTANTS["COLUMNS"]["TRUE_CP_INDEX"]: true_change_points,
        }

        # Count detections per trial
        for i, trial in enumerate(individual_trials):
            # Traditional martingale detections
            if OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"] in trial:
                # Adjust detection points
                adjusted_detections = [
                    max(0, point - 1)
                    for point in trial[
                        OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"]
                    ]
                ]
                summary_data[
                    OUTPUT_CONSTANTS["COLUMNS"]["TRIAL_TRADITIONAL_PREFIX"].format(
                        i + 1
                    )
                ] = [
                    self._count_detections_near_cp(adjusted_detections, cp)
                    for cp in true_change_points
                ]

                # Record detection latency (distance from true CP to first detection)
                summary_data[
                    OUTPUT_CONSTANTS["COLUMNS"]["TRIAL_TRADITIONAL_LATENCY"].format(
                        i + 1
                    )
                ] = [
                    self._calc_detection_latency(adjusted_detections, cp)
                    for cp in true_change_points
                ]

            # Horizon martingale detections
            if OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"] in trial:
                # Adjust detection points
                adjusted_horizon_detections = [
                    max(0, point - 1)
                    for point in trial[
                        OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"]
                    ]
                ]
                summary_data[
                    OUTPUT_CONSTANTS["COLUMNS"]["TRIAL_HORIZON_PREFIX"].format(i + 1)
                ] = [
                    self._count_detections_near_cp(adjusted_horizon_detections, cp)
                    for cp in true_change_points
                ]

                # Record detection latency for horizon detections
                summary_data[
                    OUTPUT_CONSTANTS["COLUMNS"]["TRIAL_HORIZON_LATENCY"].format(i + 1)
                ] = [
                    self._calc_detection_latency(adjusted_horizon_detections, cp)
                    for cp in true_change_points
                ]

        # Create DataFrame
        df_summary = pd.DataFrame(summary_data)

        # Add total and average rows
        if len(individual_trials) > 1:
            trial_cols = [col for col in df_summary.columns if "Detection Count" in col]
            if trial_cols:
                df_summary.loc[OUTPUT_CONSTANTS["COLUMNS"]["TOTAL"]] = [
                    OUTPUT_CONSTANTS["COLUMNS"]["TOTAL"]
                ] + [df_summary[col].sum() for col in df_summary.columns[1:]]
                df_summary.loc[OUTPUT_CONSTANTS["COLUMNS"]["AVERAGE"]] = [
                    OUTPUT_CONSTANTS["COLUMNS"]["AVERAGE"]
                ] + [
                    df_summary[col].mean() if "Count" in col else np.nan
                    for col in df_summary.columns[1:]
                ]

        df_summary.to_excel(
            writer,
            sheet_name=OUTPUT_CONSTANTS["SHEETS"]["DETECTION_SUMMARY"],
            index=False,
        )

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
            if OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"] in trial:
                raw_detections = trial[
                    OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"]
                ]
                adjusted_detections = [max(0, point - 1) for point in raw_detections]

                # Process each detection point
                for j, (raw_point, adj_point) in enumerate(
                    zip(raw_detections, adjusted_detections)
                ):
                    # Find the nearest true change point
                    nearest_cp, distance = self._find_nearest_cp(
                        adj_point, true_change_points
                    )

                    details_rows.append(
                        {
                            OUTPUT_CONSTANTS["COLUMNS"]["TRIAL"]: trial_num,
                            OUTPUT_CONSTANTS["COLUMNS"]["TYPE"]: OUTPUT_CONSTANTS[
                                "TYPES"
                            ]["TRADITIONAL"],
                            OUTPUT_CONSTANTS["COLUMNS"]["DETECTION_NUM"]: j + 1,
                            OUTPUT_CONSTANTS["COLUMNS"]["RAW_DETECTION"]: raw_point,
                            OUTPUT_CONSTANTS["COLUMNS"][
                                "ADJUSTED_DETECTION"
                            ]: adj_point,
                            OUTPUT_CONSTANTS["COLUMNS"]["NEAREST_CP"]: nearest_cp,
                            OUTPUT_CONSTANTS["COLUMNS"]["DISTANCE_TO_CP"]: distance,
                            OUTPUT_CONSTANTS["COLUMNS"]["IS_WITHIN_RANGE"]: abs(
                                distance
                            )
                            <= OUTPUT_CONSTANTS["DEFAULTS"]["PROXIMITY_WINDOW"],
                        }
                    )

            # Process horizon martingale detections
            if OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"] in trial:
                raw_horizon_detections = trial[
                    OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"]
                ]
                adjusted_horizon_detections = [
                    max(0, point - 1) for point in raw_horizon_detections
                ]

                # Process each detection point
                for j, (raw_point, adj_point) in enumerate(
                    zip(raw_horizon_detections, adjusted_horizon_detections)
                ):
                    # Find the nearest true change point
                    nearest_cp, distance = self._find_nearest_cp(
                        adj_point, true_change_points
                    )

                    details_rows.append(
                        {
                            OUTPUT_CONSTANTS["COLUMNS"]["TRIAL"]: trial_num,
                            OUTPUT_CONSTANTS["COLUMNS"]["TYPE"]: OUTPUT_CONSTANTS[
                                "TYPES"
                            ]["HORIZON"],
                            OUTPUT_CONSTANTS["COLUMNS"]["DETECTION_NUM"]: j + 1,
                            OUTPUT_CONSTANTS["COLUMNS"]["RAW_DETECTION"]: raw_point,
                            OUTPUT_CONSTANTS["COLUMNS"][
                                "ADJUSTED_DETECTION"
                            ]: adj_point,
                            OUTPUT_CONSTANTS["COLUMNS"]["NEAREST_CP"]: nearest_cp,
                            OUTPUT_CONSTANTS["COLUMNS"]["DISTANCE_TO_CP"]: distance,
                            OUTPUT_CONSTANTS["COLUMNS"]["IS_WITHIN_RANGE"]: abs(
                                distance
                            )
                            <= OUTPUT_CONSTANTS["DEFAULTS"]["PROXIMITY_WINDOW"],
                        }
                    )

        # Create detailed DataFrame
        if details_rows:
            df_details = pd.DataFrame(details_rows)
            df_details.to_excel(
                writer,
                sheet_name=OUTPUT_CONSTANTS["SHEETS"]["DETECTION_DETAILS"],
                index=False,
            )
        else:
            # Create empty dataframe with headers if no detections
            df_details = pd.DataFrame(
                columns=[
                    OUTPUT_CONSTANTS["COLUMNS"]["TRIAL"],
                    OUTPUT_CONSTANTS["COLUMNS"]["TYPE"],
                    OUTPUT_CONSTANTS["COLUMNS"]["DETECTION_NUM"],
                    OUTPUT_CONSTANTS["COLUMNS"]["RAW_DETECTION"],
                    OUTPUT_CONSTANTS["COLUMNS"]["ADJUSTED_DETECTION"],
                    OUTPUT_CONSTANTS["COLUMNS"]["NEAREST_CP"],
                    OUTPUT_CONSTANTS["COLUMNS"]["DISTANCE_TO_CP"],
                    OUTPUT_CONSTANTS["COLUMNS"]["IS_WITHIN_RANGE"],
                ]
            )
            df_details.to_excel(
                writer,
                sheet_name=OUTPUT_CONSTANTS["SHEETS"]["DETECTION_DETAILS"],
                index=False,
            )

    def _count_detections_near_cp(
        self, detections: List[int], change_point: int, window: int = None
    ) -> int:
        """Count detections that occur within a window of a change point.

        Args:
            detections: List of detection points
            change_point: True change point to check against
            window: Window size to consider a detection associated with the change point

        Returns:
            Number of detections within the window of the change point
        """
        if window is None:
            window = OUTPUT_CONSTANTS["DEFAULTS"]["PROXIMITY_WINDOW"]
        return sum(1 for d in detections if abs(d - change_point) <= window)

    def _calc_detection_latency(
        self, detections: List[int], change_point: int, window: int = None
    ) -> int:
        """Calculate detection latency (time from change point to first detection).

        Args:
            detections: List of detection points
            change_point: True change point to check against
            window: Maximum window to consider for latency calculation

        Returns:
            Detection latency or np.nan if no detection within window
        """
        if window is None:
            window = OUTPUT_CONSTANTS["DEFAULTS"]["LATENCY_WINDOW"]

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

        Note: There is a known 1-index offset in the martingale detection framework where
        detections are logged at index t when the martingale value actually exceeds
        the threshold at index t-1. The 'traditional_detected' column accurately shows
        where threshold exceedance occurs, while internal detection tracking is offset by 1.

        Args:
            detection_results: Detection results to include in the dataframe
            true_change_points: List of actual change points
            n_timesteps: Number of timesteps

        Returns:
            DataFrame with detection results
        """
        # Create a base dataframe with timesteps
        df_data = {OUTPUT_CONSTANTS["COLUMNS"]["TIMESTEP"]: list(range(n_timesteps))}

        # Add change point indicators
        df_data[OUTPUT_CONSTANTS["COLUMNS"]["TRUE_CHANGE_POINT"]] = [
            1 if t in true_change_points else 0 for t in range(n_timesteps)
        ]

        # Get threshold from config
        threshold = OUTPUT_CONSTANTS["DEFAULTS"]["THRESHOLD"]  # Default value
        if (
            self.config
            and OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"] in self.config
            and OUTPUT_CONSTANTS["CONFIG_KEYS"]["THRESHOLD"]
            in self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"]]
        ):
            threshold = self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"]][
                OUTPUT_CONSTANTS["CONFIG_KEYS"]["THRESHOLD"]
            ]

        # Add martingale values to the dataframe (both traditional and horizon)
        # Define the martingale keys we want to include
        traditional_martingale_keys = [
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_MARTINGALES"],
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_SUM_MARTINGALES"],
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_AVG_MARTINGALES"],
        ]

        horizon_martingale_keys = [
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_MARTINGALES"],
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_SUM_MARTINGALES"],
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_AVG_MARTINGALES"],
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
                    if (
                        martingale_key
                        == OUTPUT_CONSTANTS["DETECTION_KEYS"][
                            "TRADITIONAL_SUM_MARTINGALES"
                        ]
                    ):
                        # Initialize all detection flags to 0
                        df_data[OUTPUT_CONSTANTS["COLUMNS"]["TRADITIONAL_DETECTED"]] = [
                            0
                        ] * n_timesteps

                        # Set flag to 1 at indices where martingale exceeds threshold
                        for i in range(max_idx):
                            if (
                                df_data[martingale_key][i] is not None
                                and df_data[martingale_key][i] > threshold
                            ):
                                df_data[
                                    OUTPUT_CONSTANTS["COLUMNS"]["TRADITIONAL_DETECTED"]
                                ][i] = 1

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
                    if (
                        martingale_key
                        == OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_SUM_MARTINGALES"]
                    ):
                        # Initialize all detection flags to 0
                        df_data[OUTPUT_CONSTANTS["COLUMNS"]["HORIZON_DETECTED"]] = [
                            0
                        ] * n_timesteps

                        # Set flag to 1 at indices where martingale exceeds threshold
                        for i in range(max_idx):
                            if (
                                df_data[martingale_key][i] is not None
                                and df_data[martingale_key][i] > threshold
                            ):
                                df_data[
                                    OUTPUT_CONSTANTS["COLUMNS"]["HORIZON_DETECTED"]
                                ][i] = 1

        # Calculate traditional detection values for internal analysis
        if (
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"]
            in detection_results
        ):
            detected_points = detection_results[
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_CHANGE_POINTS"]
            ]
            # Store for internal analysis
            self._raw_detection_points = detected_points
            self._adjusted_detection_points = [
                max(0, point - 1) for point in detected_points
            ]

        # Calculate horizon detection values for internal analysis
        if (
            OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"]
            in detection_results
        ):
            horizon_detected_points = detection_results[
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_CHANGE_POINTS"]
            ]
            # Store for internal analysis
            self._raw_horizon_detection_points = horizon_detected_points
            self._adjusted_horizon_detection_points = [
                max(0, point - 1) for point in horizon_detected_points
            ]

        # Create the dataframe with a specific column order
        columns = [
            OUTPUT_CONSTANTS["COLUMNS"]["TIMESTEP"],
            OUTPUT_CONSTANTS["COLUMNS"]["TRUE_CHANGE_POINT"],
        ]

        # Add individual feature columns
        individual_feature_columns = [
            col for col in df_data.keys() if col.startswith("individual_")
        ]
        columns.extend(sorted(individual_feature_columns))

        # Add traditional columns
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_MARTINGALES"] in df_data:
            columns.append(
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_MARTINGALES"]
            )
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_SUM_MARTINGALES"] in df_data:
            columns.append(
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_SUM_MARTINGALES"]
            )
            columns.append(
                OUTPUT_CONSTANTS["COLUMNS"]["TRADITIONAL_DETECTED"]
            )  # Put detection flag immediately after sum
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_AVG_MARTINGALES"] in df_data:
            columns.append(
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["TRADITIONAL_AVG_MARTINGALES"]
            )

        # Add horizon columns
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_MARTINGALES"] in df_data:
            columns.append(OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_MARTINGALES"])
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_SUM_MARTINGALES"] in df_data:
            columns.append(
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_SUM_MARTINGALES"]
            )
            columns.append(
                OUTPUT_CONSTANTS["COLUMNS"]["HORIZON_DETECTED"]
            )  # Put detection flag immediately after sum
        if OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_AVG_MARTINGALES"] in df_data:
            columns.append(
                OUTPUT_CONSTANTS["DETECTION_KEYS"]["HORIZON_AVG_MARTINGALES"]
            )

        # Create DataFrame with specific column order
        return pd.DataFrame({col: df_data[col] for col in columns if col in df_data})

    def _get_timestep_count(self) -> int:
        """Determine the number of timesteps from configuration.

        Returns:
            Number of timesteps (defaults to 200 if not found)
        """
        try:
            # Try to get sequence length from model parameters
            if (
                OUTPUT_CONSTANTS["CONFIG_KEYS"]["MODEL"] in self.config
                and OUTPUT_CONSTANTS["CONFIG_KEYS"]["PARAMS"] in self.config
            ):
                params = self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["PARAMS"]]
                if hasattr(params, OUTPUT_CONSTANTS["CONFIG_KEYS"]["SEQ_LEN"]):
                    return params.seq_len

            # Try to access params as dictionary
            if OUTPUT_CONSTANTS["CONFIG_KEYS"]["PARAMS"] in self.config and isinstance(
                self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["PARAMS"]], dict
            ):
                params = self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["PARAMS"]]
                if OUTPUT_CONSTANTS["CONFIG_KEYS"]["SEQ_LEN"] in params:
                    return params[OUTPUT_CONSTANTS["CONFIG_KEYS"]["SEQ_LEN"]]

            # If that fails, look for sequence configuration in detection settings
            if (
                OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"] in self.config
                and OUTPUT_CONSTANTS["CONFIG_KEYS"]["SEQUENCE"]
                in self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"]]
            ):
                sequence = self.config[OUTPUT_CONSTANTS["CONFIG_KEYS"]["DETECTION"]][
                    OUTPUT_CONSTANTS["CONFIG_KEYS"]["SEQUENCE"]
                ]
                if OUTPUT_CONSTANTS["CONFIG_KEYS"]["LENGTH"] in sequence:
                    return sequence[OUTPUT_CONSTANTS["CONFIG_KEYS"]["LENGTH"]]

            # Default fallback value
            return OUTPUT_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"]

        except Exception as e:
            logger.warning(f"Could not determine timestep count from config: {str(e)}")
            return OUTPUT_CONSTANTS["DEFAULTS"]["SEQUENCE_LENGTH"]  # Default fallback
