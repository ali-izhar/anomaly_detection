# src/setup/config.py

"""
Configuration module for network experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Configuration for network experiments.
    Contains all parameters needed for running network prediction and analysis experiments.
    """

    # Graph generation parameters
    model: str  # Type of network model (e.g., 'ba', 'ws', 'er', 'sbm')
    params: Dict[str, Any]  # Model-specific parameters

    # Prediction parameters
    min_history: int  # Minimum history length required for prediction
    prediction_window: int  # Number of steps to predict ahead

    # Change point detection parameters
    martingale_threshold: float = 60.0  # Threshold for martingale-based detection
    martingale_epsilon: float = 0.85  # Epsilon parameter for martingale calculation
    shap_threshold: float = 30.0  # Threshold for SHAP value significance
    shap_window_size: int = 5  # Window size for SHAP value calculation

    # Experiment run parameters
    n_runs: int = 1  # Number of experiment runs
    save_individual: bool = False  # Whether to save individual run results
    visualize_individual: bool = False  # Whether to visualize individual run results

    # Feature weights for martingale analysis
    feature_weights: Optional[Dict[str, float]] = (
        None  # Weights for different network features
    )

    # Predictor configuration
    predictor_type: str = (
        "weighted"  # Type of predictor to use ('weighted' or 'hybrid')
    )

    def __post_init__(self):
        """Initialize default feature weights if not provided."""
        if self.feature_weights is None:
            self.feature_weights = {
                "betweenness": 1.0,  # Most reliable
                "clustering": 0.85,  # Most consistent
                "closeness": 0.7,  # Moderate reliability
                "degree": 0.5,  # Least reliable
            }


@dataclass
class OutputConfig:
    """Configuration for output handling and visualization."""

    output_dir: Optional[Path] = None  # Base output directory
    timestamp_format: str = "%Y%m%d_%H%M%S"  # Format for timestamp in directory names
    results_dir: str = "results"  # Name of results directory
    dpi: int = 600  # DPI for saved figures
    figure_format: str = "png"  # Format for saved figures
    figure_metadata: Dict[str, str] = None  # Metadata for saved figures

    def __post_init__(self):
        """Initialize default figure metadata if not provided."""
        if self.figure_metadata is None:
            self.figure_metadata = {"Creator": "Matplotlib"}


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Figure sizes (in inches)
    single_column_width: float = 3.3
    double_column_width: float = 7.0
    standard_height: float = 2.5
    grid_height: float = 3.3

    # Grid parameters
    grid_spacing: float = 0.4  # Space between subplots

    # Font sizes
    title_size: int = 8
    label_size: int = 8
    tick_size: int = 6
    legend_size: int = 6
    annotation_size: int = 6

    # Line parameters
    line_width: float = 0.8
    line_alpha: float = 0.8
    grid_alpha: float = 0.3
    grid_width: float = 0.5

    # Colors
    colors: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default colors if not provided."""
        if self.colors is None:
            self.colors = {
                "actual": "blue",
                "predicted": "orange",
                "average": "#2ecc71",
                "pred_avg": "#9b59b6",
                "threshold": "#FF7F7F",
                "change_point": "red",
            }


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing and feature extraction."""

    # Feature weights for martingale analysis
    feature_weights: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default feature weights if not provided."""
        if self.feature_weights is None:
            self.feature_weights = {
                "betweenness": 1.0,  # Most reliable
                "clustering": 0.85,  # Most consistent
                "closeness": 0.7,  # Moderate reliability
                "degree": 0.5,  # Least reliable
            }


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
