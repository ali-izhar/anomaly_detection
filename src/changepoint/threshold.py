# src/changepoint/threshold.py

"""Threshold-based classifier for change point detection in graph sequences.

This module implements a threshold-based classifier for detecting change points in
graph sequences using a fixed decision rule. The classifier is designed to work with
both single-view and multiview martingale sequences and provides SHAP-based
explanations for detected changes.

Mathematical Framework:
---------------------
1. Decision Rule:
   For a feature vector x ∈ ℝᵈ, the classifier predicts:
   y = 1[∑ᵢ xᵢ > τ]
   where τ is the decision threshold.

2. Probability Estimation:
   P(change) = ∑ᵢ xᵢ / (∑ᵢ xᵢ + τ)
   This provides a continuous score in [0,1] for SHAP analysis.

3. SHAP Analysis:
   - Feature-level: Which graph properties contribute to changes
   - Martingale-level: How different views influence detection

Properties:
----------
1. Interpretable decision boundary
2. Fast computation (linear in feature dimension)
3. No parameter estimation required
4. Compatible with scikit-learn API
5. Supports both hard and soft predictions

Example Usage:
------------
```python
# Basic change point detection
model = CustomThresholdModel(threshold=50.0)
predictions = model.predict(X)

# SHAP analysis of a detected change point
df = pd.read_excel("martingale_data.xlsx", sheet_name="Aggregate")
feature_cols = [col for col in df.columns 
                if col.startswith("individual_traditional_martingales_feature") 
                and col.endswith("_mean")]
change_points = [40, 120]  # From metadata

# Comprehensive change point analysis with multiple visualizations
contributions = model.analyze_change_points(
    df=df,
    feature_cols=feature_cols,
    sum_col="traditional_sum_martingales_mean",
    change_points=change_points,
    threshold=50.0,
    output_dir="results/change_point_analysis"
)

# Individual SHAP over time visualization
model.visualize_shap_over_time(
    df=df,
    feature_cols=feature_cols,
    sum_col="traditional_sum_martingales_mean",
    change_points=change_points,
    output_path="results/shap_time_analysis.png"
)
```

References:
----------
[1] Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model
    Predictions." Advances in Neural Information Processing Systems 30.
[2] Ribeiro, M. T., et al. (2016). "Why Should I Trust You?: Explaining the
    Predictions of Any Classifier." KDD '16.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

logger = logging.getLogger(__name__)

# Define a fixed feature order for consistent visualization and analysis in SHAP plots.
# Updated to match the feature names used in the project
FEATURE_ORDER = [
    "degree",  # Feature 0
    "density",  # Feature 1
    "clustering",  # Feature 2
    "betweenness",  # Feature 3
    "eigenvector",  # Feature 4
    "closeness",  # Feature 5
    "spectral",  # Feature 6 (renamed from singular_value for consistency)
    "laplacian",  # Feature 7
]

# Feature mapping to use in visualization
FEATURE_NAMES = {
    "degree": "Degree",
    "density": "Density",
    "clustering": "Clustering",
    "betweenness": "Betweenness",
    "eigenvector": "Eigenvector",
    "closeness": "Closeness",
    "spectral": "Spectral",
    "laplacian": "Laplacian",
}

# Feature ID mapping (for compatibility with martingale data formats)
FEATURE_ID_MAP = {
    "0": "degree",
    "1": "density",
    "2": "clustering",
    "3": "betweenness",
    "4": "eigenvector",
    "5": "closeness",
    "6": "spectral",
    "7": "laplacian",
}


@dataclass(frozen=True)
class ShapConfig:
    """Configuration for SHAP analysis.

    Attributes:
        window_size: Size of window around change points for labeling.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        use_probabilities: Whether to analyze probability outputs (via predict_proba).
        positive_class: Whether to analyze positive class predictions.
    """

    window_size: int = 5
    test_size: float = 0.2
    random_state: int = 42
    use_probabilities: bool = (
        True  # Changed default to True for better probability estimates
    )
    positive_class: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0,1), got {self.test_size}")


class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    """Threshold-based classifier for change point detection.

    This classifier implements a simple fixed-threshold rule:
        y = 1[∑ᵢ xᵢ > τ]
    where x are the input features and τ is the decision threshold.

    The model is designed for:
      1. Binary classification of change points.
      2. Probability estimation for uncertain predictions.
      3. SHAP-based feature importance analysis.
      4. Martingale contribution analysis in multiview detection.

    Mathematical Properties:
    -------------------------
      - Linear decision boundary in feature space.
      - Monotonic in feature values.
      - Scale-invariant threshold.
      - Interpretable decision rule.
    """

    def __init__(self, threshold: float = 60.0) -> None:
        """Initialize the model with a decision threshold.

        Args:
            threshold: Decision boundary τ. Predicts change if ∑ᵢ xᵢ > τ.

        Raises:
            ValueError: If threshold is not positive.
        """
        # Validate threshold: must be positive.
        if threshold <= 0:
            logger.error(f"Invalid threshold value: {threshold}")
            raise ValueError("Threshold must be positive")
        self.threshold = threshold
        logger.debug(f"Initialized model with threshold={threshold}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "CustomThresholdModel":
        """Fit the model (stores dimensions for validation).

        No parameter estimation is needed as the decision rule is fixed.
        This method only validates and stores input dimensions.

        Args:
            X: Feature matrix [n_samples × n_features].
            y: Binary labels (0: no change, 1: change).
            sample_weight: Optional sample weights (not used).

        Returns:
            self: The fitted model.
        """
        # Validate X and y using scikit-learn's utility.
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[
            1
        ]  # Store number of features for later validation.
        self.classes_ = np.unique(y)  # Save unique classes.
        logger.info(f"Model fitted with {self.n_features_in_} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change points using the threshold rule.

        Implements the decision rule:
            y = 1[∑ᵢ xᵢ > τ]

        Args:
            X: Feature matrix [n_samples × n_features].

        Returns:
            Binary predictions (0: no change, 1: change).

        Raises:
            ValueError: If X has wrong dimensions.
        """
        check_is_fitted(self)  # Ensure model has been fitted.
        X = check_array(X, accept_sparse=False)

        # Validate that the input feature dimension matches that seen during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Sum features and apply threshold to determine change.
        predictions = (np.sum(X, axis=1) > self.threshold).astype(int)
        n_changes = np.sum(predictions)
        logger.debug(f"Predicted {n_changes}/{len(predictions)} changes")
        return predictions

    def predict_proba(self, X: np.ndarray, positive_class: bool = True) -> np.ndarray:
        """Compute change point probabilities.

        Uses the formula:
            P(change) = ∑ᵢ xᵢ / (∑ᵢ xᵢ + τ)

        Args:
            X: Feature matrix [n_samples × n_features].
            positive_class: If True, return P(change).

        Returns:
            Probability array [n_samples].
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Validate input dimensions.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        sums = np.sum(X, axis=1)  # Sum over features for each sample.
        # Calculate probability using the given formula.
        probs_change = sums / (sums + self.threshold)
        return probs_change if positive_class else 1 - probs_change

    def compute_shap_values(
        self,
        X: np.ndarray,
        change_points: List[int],
        sequence_length: int,
        config: Optional[ShapConfig] = None,
    ) -> np.ndarray:
        """Compute SHAP values for feature importance analysis.

        Creates binary labels around change points and uses SHAP's
        KernelExplainer to compute feature contributions.

        Args:
            X: Feature matrix [n_samples × n_features].
            change_points: Indices of detected changes.
            sequence_length: Total sequence length.
            config: SHAP analysis configuration.

        Returns:
            SHAP values [n_samples × n_features].
        """
        config = config or ShapConfig()

        # Create binary labels where a window around each change point is marked as 1.
        y = np.zeros(sequence_length)
        for cp in change_points:
            # Label a window around each change point.
            start_idx = max(0, cp - config.window_size)
            end_idx = min(len(y), cp + config.window_size)
            y[start_idx:end_idx] = 1

        # Split the dataset for training and testing.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        # Fit the model on the training split.
        self.fit(X_train, y_train)

        try:
            # Choose the appropriate prediction function based on configuration.
            predict_fn = (
                self.predict_proba if config.use_probabilities else self.predict
            )

            # Use SHAP's KernelExplainer to compute SHAP values.
            explainer = shap.KernelExplainer(predict_fn, X_train)
            shap_values = explainer.shap_values(X)

            # If SHAP returns a list (one per class), select the desired class.
            if isinstance(shap_values, list):
                shap_values = (
                    shap_values[1] if config.positive_class else shap_values[0]
                )
            return shap_values

        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            # In case of failure, return a zero matrix.
            return np.zeros((len(X), X.shape[1]))

    def compute_martingale_shap_values(
        self,
        martingales: Dict[str, Dict[str, Any]],
        change_points: List[int],
        sequence_length: int,
        config: Optional[ShapConfig] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute SHAP values for martingale contributions.

        Analyzes how different features' martingales influence
        change point detection in multiview settings.

        Args:
            martingales: Martingale values per feature.
            change_points: True change points.
            sequence_length: Sequence length.
            config: SHAP analysis configuration.

        Returns:
            Tuple of (SHAP values, feature names).
        """
        # Convert the martingales dictionary into a feature matrix.
        feature_matrix = []
        feature_names = []

        for feature in FEATURE_ORDER:
            # Skip combined features and only process individual features.
            if feature in martingales and feature != "combined":
                martingales_array = np.array(
                    [
                        x.item() if isinstance(x, np.ndarray) else x
                        for x in martingales[feature]["martingales"]
                    ]
                )
                feature_matrix.append(martingales_array)
                feature_names.append(feature)

        if not feature_matrix:
            raise ValueError("No valid features in martingales dictionary")

        # Stack features column-wise to form a 2D matrix.
        X = np.vstack(feature_matrix).T

        # Compute SHAP values using the previously defined method.
        shap_values = self.compute_shap_values(
            X=X,
            change_points=change_points,
            sequence_length=sequence_length,
            config=config,
        )

        return shap_values, feature_names

    def compute_shap_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        change_points: Optional[List[int]] = None,
        timesteps: Optional[np.ndarray] = None,
        detection_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute SHAP values directly from a pandas DataFrame containing martingale data.

        Args:
            df: DataFrame with martingale values
            feature_cols: List of column names for individual feature martingales
            change_points: Optional indices of true change points
            timesteps: Optional array of timesteps (if not provided, uses range(len(df)))
            detection_index: Optional index where detection occurred

        Returns:
            Tuple of (SHAP values, feature names)
        """
        # If timesteps not provided, use range
        if timesteps is None:
            timesteps = np.arange(len(df))

        # If change_points not provided but detection_index is, use that
        if change_points is None and detection_index is not None:
            change_points = [timesteps[detection_index]]
        elif change_points is None:
            # Try to find change points from true_change_point column
            if "true_change_point" in df.columns:
                change_points = timesteps[df["true_change_point"] == 1].tolist()
            else:
                change_points = []

        # Get feature matrix
        X = df[feature_cols].values

        # Set up config
        config = ShapConfig(window_size=5, test_size=0.2, use_probabilities=True)

        # Create binary labels for SHAP analysis
        sequence_length = len(df)

        # Compute SHAP values
        shap_values = self.compute_shap_values(
            X=X,
            change_points=change_points,
            sequence_length=sequence_length,
            config=config,
        )

        return shap_values, feature_cols

    def visualize_shap_values(
        self,
        shap_values: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: str = "SHAP Values Analysis",
    ) -> None:
        """Create visualizations for SHAP values analysis.

        Args:
            shap_values: SHAP values from compute_shap_values
            X: Feature data used for SHAP analysis
            feature_names: Optional list of feature names
            output_path: Optional file path to save visualization
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))

        # If X is a DataFrame, convert to numpy array and extract feature names
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        # Create a summary plot
        plt.subplot(2, 1, 1)
        shap.summary_plot(
            shap_values, X, feature_names=feature_names, show=False, plot_size=(8, 6)
        )
        plt.title(f"{title} - Feature Importance")

        # Create a decision plot
        plt.subplot(2, 1, 2)
        try:
            shap.decision_plot(
                self.threshold, shap_values, feature_names=feature_names, show=False
            )
            plt.title(f"{title} - Decision Contributions")
        except Exception as e:
            logger.warning(f"Could not create decision plot: {e}")
            # Fall back to bar plot
            plt.barh(
                range(len(feature_names) if feature_names else shap_values.shape[1]),
                np.mean(np.abs(shap_values), axis=0),
                color="skyblue",
            )
            plt.yticks(
                range(len(feature_names) if feature_names else shap_values.shape[1]),
                (
                    feature_names
                    if feature_names
                    else [f"Feature {i}" for i in range(shap_values.shape[1])]
                ),
            )
            plt.title(f"{title} - Average SHAP Values")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP visualization to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_shap_over_time(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sum_col: str,
        change_points: Optional[List[int]] = None,
        detection_indices: Optional[List[int]] = None,
        timesteps: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        threshold: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create a time-based visualization of SHAP values and feature contributions.

        Args:
            df: DataFrame containing martingale data
            feature_cols: List of column names for individual feature martingales
            sum_col: Column name for the sum martingale
            change_points: Optional list of true change points
            detection_indices: Optional list of detection point indices
            timesteps: Optional array of timesteps (defaults to range(len(df)))
            output_path: Optional file path to save visualization
            threshold: Optional threshold value (defaults to self.threshold)
            feature_names: Optional list of display names for features (defaults to feature_cols)

        Returns:
            DataFrame containing feature contributions at detection points
        """
        if threshold is None:
            threshold = self.threshold

        if timesteps is None:
            timesteps = np.arange(len(df))

        # If feature_names not provided, use column names
        if feature_names is None:
            feature_names = feature_cols

        if len(feature_names) != len(feature_cols):
            logger.warning(
                f"Length mismatch: {len(feature_names)} feature names for {len(feature_cols)} columns. Using column names."
            )
            feature_names = feature_cols

        # Create a DataFrame for feature values
        X = df[feature_cols].copy()

        # Compute SHAP values
        model = LinearRegression(fit_intercept=False)
        model.fit(X, df[sum_col])

        # Verify the model accuracy
        predictions = model.predict(X)
        r2 = sklearn.metrics.r2_score(df[sum_col], predictions)
        logger.info(f"R² score of linear model: {r2:.6f}")

        # Compute SHAP values
        try:
            # Sample background data for the explainer
            background_indices = np.random.choice(
                len(X), size=min(100, len(X)), replace=False
            )
            background = X.iloc[background_indices]

            # Create explainer and compute SHAP values
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            logger.warning(f"Error computing SHAP values with KernelExplainer: {e}")
            logger.info("Using feature values * coefficients as SHAP approximation")
            # Approximate SHAP values for linear model
            shap_values = np.zeros(X.shape)
            for i, col in enumerate(X.columns):
                shap_values[:, i] = X[col].values * model.coef_[i]

        # If no detection indices provided but threshold is, find crossing points
        if detection_indices is None and threshold is not None:
            detection_indices = []
            for i in range(1, len(df)):
                if (
                    df[sum_col].iloc[i - 1] <= threshold
                    and df[sum_col].iloc[i] > threshold
                ):
                    detection_indices.append(i)

            # If still no detection points, use peaks near change points
            if not detection_indices and change_points:
                for cp in change_points:
                    # Find the closest index to the change point
                    cp_idx = np.argmin(np.abs(timesteps - cp))

                    # Define a window around the change point
                    window_start = cp_idx
                    window_end = min(len(df), cp_idx + 10)

                    # Find the index of maximum sum martingale in this window
                    max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                    detection_indices.append(max_idx)

        # Compute threshold-based classifier SHAP values
        classifier_shap_values = np.zeros(X.shape)

        # Process all detection points if available
        if detection_indices:
            for detection_index in detection_indices:
                # Calculate contribution at detection point
                feature_values = X.iloc[detection_index].values
                total = sum(feature_values)

                if total > 0:
                    # Contributions as percentages
                    for j, val in enumerate(feature_values):
                        classifier_shap_values[detection_index, j] = val / total

                    # Add decaying contributions around detection point
                    window = 2
                    for i in range(
                        max(0, detection_index - window),
                        min(len(df), detection_index + window + 1),
                    ):
                        if i != detection_index:
                            decay = 0.2 ** abs(i - detection_index)
                            vals = X.iloc[i].values
                            val_sum = sum(vals)
                            if val_sum > 0:
                                for j, val in enumerate(vals):
                                    classifier_shap_values[i, j] = (
                                        val / val_sum
                                    ) * decay

        # Setup plot style
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "figure.figsize": (8, 10),
                "figure.dpi": 300,
            }
        )

        # Create color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

        # Create figure with three panels
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(8, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.15},
        )

        # Panel 1: Martingale Values
        for i, col in enumerate(feature_cols):
            axs[0].plot(
                timesteps,
                df[col],
                label=feature_names[i],
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )

        # Add sum martingale
        axs[0].plot(
            timesteps,
            df[sum_col],
            label="Sum Martingale",
            color="black",
            linewidth=2,
        )

        # Add threshold
        axs[0].axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold})",
        )

        # Mark change points if provided
        if change_points:
            for cp in change_points:
                cp_idx = np.where(timesteps == cp)[0]
                if len(cp_idx) > 0:
                    axs[0].axvline(
                        x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                    )

        # Mark detection points if provided
        if detection_indices:
            for idx in detection_indices:
                if 0 <= idx < len(timesteps):
                    dp = timesteps[idx]
                    axs[0].axvline(
                        x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
                    )

        axs[0].set_title("Martingale Values Over Time", fontsize=11)
        axs[0].set_ylabel("Martingale Value", fontsize=10)
        axs[0].legend(loc="upper right", fontsize=8)
        axs[0].grid(True, alpha=0.3)

        # Panel 2: SHAP Values
        # Calculate feature-specific R² values between feature values and SHAP values
        feature_r2_values = []
        for i in range(len(feature_cols)):
            # Calculate correlation between feature and its SHAP values
            feature_vals = X.iloc[:, i].values
            feature_shap = shap_values[:, i]
            # Only calculate if there's variance in the feature
            if np.std(feature_vals) > 0 and np.std(feature_shap) > 0:
                corr = np.corrcoef(feature_vals, feature_shap)[0, 1]
                r2 = corr**2
            else:
                r2 = 0
            feature_r2_values.append(r2)

        # Plot with R² values in legend
        for i, col in enumerate(feature_cols):
            axs[1].plot(
                timesteps,
                shap_values[:, i],
                label=f"{feature_names[i]} (R²={feature_r2_values[i]:.3f})",
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )

        # Mark change points if provided
        if change_points:
            for cp in change_points:
                cp_idx = np.where(timesteps == cp)[0]
                if len(cp_idx) > 0:
                    axs[1].axvline(
                        x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                    )

        # Mark detection points if provided
        if detection_indices:
            for idx in detection_indices:
                if 0 <= idx < len(timesteps):
                    dp = timesteps[idx]
                    axs[1].axvline(
                        x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
                    )

        # Add note about R² score for the overall model
        if r2 > 0.99:
            model_note = f"Model R²={r2:.4f} (perfect additive)"
        else:
            model_note = f"Model R²={r2:.4f}"
        axs[1].text(
            0.02,
            0.97,
            model_note,
            transform=axs[1].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
        )

        axs[1].set_title("SHAP Values Over Time", fontsize=11)
        axs[1].set_ylabel("SHAP Value", fontsize=10)
        axs[1].legend(loc="upper right", fontsize=8)
        axs[1].grid(True, alpha=0.3)

        # Panel 3: Classifier SHAP Values
        # Calculate top feature contributions for legend if detection indices available
        if detection_indices:
            # Create more compact labels with percentages from both detection points
            compact_labels = []

            # Check if we have multiple detection points
            if len(detection_indices) >= 2:
                # Get the first two detection indices
                first_idx = detection_indices[0]
                second_idx = detection_indices[1]

                # Calculate percentages for both detection points
                first_percentages = classifier_shap_values[first_idx] * 100
                second_percentages = classifier_shap_values[second_idx] * 100

                # Create labels with both percentages
                for i, col in enumerate(feature_cols):
                    compact_labels.append(
                        f"{feature_names[i]} ({first_percentages[i]:.2f}, {second_percentages[i]:.2f})"
                    )
            else:
                # Only one detection point available
                detection_index = detection_indices[0]
                percentages = classifier_shap_values[detection_index] * 100

                # Create labels with single percentage
                for i, col in enumerate(feature_cols):
                    compact_labels.append(f"{feature_names[i]} ({percentages[i]:.2f})")

            # Plot with the compact labels
            for i, col in enumerate(feature_cols):
                axs[2].plot(
                    timesteps,
                    classifier_shap_values[:, i],
                    label=compact_labels[i],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.2,
                )
        else:
            # If no detection indices, use regular labels
            for i, col in enumerate(feature_cols):
                axs[2].plot(
                    timesteps,
                    classifier_shap_values[:, i],
                    label=feature_names[i],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.2,
                )

        # Mark change points if provided
        if change_points:
            for cp in change_points:
                cp_idx = np.where(timesteps == cp)[0]
                if len(cp_idx) > 0:
                    axs[2].axvline(
                        x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                    )

        # Mark detection points with annotations if provided
        panel_title = "Feature Contributions at Peak Martingales"
        if detection_indices:
            for idx in detection_indices:
                if 0 <= idx < len(timesteps):
                    dp = timesteps[idx]
                    axs[2].axvline(
                        x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
                    )

                    # Add text annotations for significant contributors
                    text_y = 0.8  # Starting y position
                    for i, col in enumerate(feature_cols):
                        if i % 2 == 0:  # Split annotations on either side
                            text_x = dp + 5
                            ha = "left"
                        else:
                            text_x = dp - 5
                            ha = "right"

                        percentage = classifier_shap_values[idx, i] * 100
                        if percentage > 3.0:  # Only show significant contributors
                            axs[2].annotate(
                                f"{feature_names[i]}: {percentage:.2f}",
                                xy=(text_x, text_y - i * 0.05),
                                color=colors[i],
                                fontsize=8,
                                fontweight="bold",
                                ha=ha,
                                va="center",
                            )

            # Set title based on threshold crossing
            if any(df[sum_col].iloc[idx] > threshold for idx in detection_indices):
                panel_title = "Threshold-Based Classifier SHAP Values"
            else:
                panel_title = "Feature Contributions at Peak Martingales"
        else:
            panel_title = "Feature Contributions (No Detection Points)"

        axs[2].set_title(panel_title, fontsize=11)
        axs[2].set_xlabel("Timestep", fontsize=10)
        axs[2].set_ylabel("Feature Contribution (0 to 1)", fontsize=10)
        axs[2].legend(loc="upper right", fontsize=8)
        axs[2].grid(True, alpha=0.3)

        # Save the figure
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP over time plot to {output_path}")
        else:
            plt.savefig("shap_over_time.png", dpi=300, bbox_inches="tight")
            logger.info("Saved SHAP over time plot to shap_over_time.png")
        plt.close()

        # Generate feature contribution report for all detection points
        all_contributions = []
        if detection_indices:
            for idx, detection_index in enumerate(detection_indices):
                if 0 <= detection_index < len(timesteps):
                    detection_time = timesteps[detection_index]

                    # Get contributions for this detection point
                    contrib_df = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Martingale Value": X.iloc[detection_index].values,
                            "Contribution %": classifier_shap_values[detection_index]
                            * 100,
                            "Detection Point": detection_time,
                        }
                    )

                    # Sort by contribution percentage
                    contrib_df = contrib_df.sort_values(
                        "Contribution %", ascending=False
                    )
                    all_contributions.append(contrib_df)

            # Combine all detection point analyses
            if all_contributions:
                combined_df = pd.concat(all_contributions)
                return combined_df

        # Return empty DataFrame if no contributions found
        return pd.DataFrame()

    def plot_feature_contributions_comparison(
        self,
        all_contributions: List[pd.DataFrame],
        feature_names: List[str],
        output_path: Optional[str] = None,
    ) -> None:
        """Create a bar chart comparing feature contributions across multiple detection points.

        Args:
            all_contributions: List of DataFrames with feature contributions
            feature_names: List of feature names
            output_path: Optional file path to save visualization
        """
        # Create a pivot table with features as rows and detection points as columns
        pivot_df = pd.concat(all_contributions)
        pivot_df = pivot_df.pivot(
            index="Feature", columns="Detection Point", values="Contribution %"
        )

        # Sort features by average contribution
        pivot_df["Average"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("Average", ascending=False)
        pivot_df = pivot_df.drop("Average", axis=1)

        # Create bar chart
        plt.figure(figsize=(10, 6))

        # Set bar width based on number of detection points
        bar_width = 0.8 / len(pivot_df.columns)

        # Plot bars for each detection point
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
        for i, (detection_point, contributions) in enumerate(pivot_df.items()):
            # Calculate position for this group of bars
            positions = np.arange(len(pivot_df.index)) + i * bar_width

            # Plot with feature-specific colors
            for j, (feature, value) in enumerate(contributions.items()):
                color_idx = (
                    feature_names.index(feature) if feature in feature_names else j
                )
                plt.bar(
                    positions[j],
                    value,
                    bar_width,
                    label=f"CP {int(detection_point)}" if j == 0 else None,
                    color=colors[color_idx % len(colors)],
                    alpha=0.7,
                )

        # Add labels and title
        plt.xlabel("Feature")
        plt.ylabel("Contribution %")
        plt.title("Feature Contributions Across Detection Points")
        plt.xticks(
            np.arange(len(pivot_df.index))
            + bar_width * (len(pivot_df.columns) - 1) / 2,
            pivot_df.index,
            rotation=45,
            ha="right",
        )
        plt.legend(title="Detection Point")
        plt.tight_layout()

        # Save the figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature contribution comparison to {output_path}")
        else:
            plt.savefig("contribution_comparison.png", dpi=300, bbox_inches="tight")
            logger.info(
                "Saved feature contribution comparison to contribution_comparison.png"
            )
        plt.close()

    def get_feature_importances(self) -> np.ndarray:
        """Get uniform feature importances for SHAP analysis."""
        check_is_fitted(self)
        # Return equal importance for all features (since no training is done).
        return np.ones(self.n_features_in_) / self.n_features_in_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Accuracy score in [0,1].
        """
        return np.mean(self.predict(X) == y)

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters for scikit-learn compatibility."""
        return {"threshold": self.threshold}

    def set_params(self, **params: dict) -> "CustomThresholdModel":
        """Set model parameters for scikit-learn compatibility."""
        for param, value in params.items():
            logger.debug(f"Setting parameter {param}={value}")
            setattr(self, param, value)
        return self

    def analyze_detection_point(
        self,
        X: np.ndarray,
        detection_index: int,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Analyze feature contributions at a specific detection point.

        Args:
            X: Feature matrix
            detection_index: Index of detection point
            feature_names: Optional list of feature names

        Returns:
            DataFrame with feature contributions
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Get feature values at detection point
        detection_values = X[detection_index]
        total = np.sum(detection_values)

        # Calculate percentage contributions
        percentages = (
            100 * detection_values / total
            if total > 0
            else np.zeros_like(detection_values)
        )

        # Create DataFrame with results
        results = pd.DataFrame(
            {
                "Feature": feature_names,
                "Martingale Value": detection_values,
                "Contribution %": percentages,
            }
        )

        # Sort by contribution percentage in descending order
        results = results.sort_values("Contribution %", ascending=False)

        return results

    def analyze_change_points(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sum_col: str = "traditional_sum_martingales_mean",
        change_points: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        timesteps: Optional[np.ndarray] = None,
        output_dir: str = "results/shap_analysis",
    ) -> pd.DataFrame:
        """Perform a comprehensive SHAP analysis of change points in a time series.

        This method combines multiple visualization techniques to provide a complete
        explanation of feature contributions to change point detection.

        Args:
            df: DataFrame containing martingale data
            feature_cols: List of column names for individual feature martingales
            sum_col: Column name for the sum martingale
            change_points: Optional list of true change points
            threshold: Optional threshold value (defaults to self.threshold)
            timesteps: Optional array of timesteps (defaults to range(len(df)))
            output_dir: Directory to save output visualizations

        Returns:
            DataFrame containing feature contributions at detection points
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        if threshold is None:
            threshold = self.threshold

        if timesteps is None:
            timesteps = np.arange(len(df))

        logger.info(f"Starting change point analysis with threshold={threshold}")

        # Find detection points
        detection_indices = []
        for i in range(1, len(df)):
            if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
                detection_indices.append(i)

        # If no threshold crossing, use peaks near change points
        if not detection_indices and change_points:
            logger.info(
                "No threshold crossing found. Using peak values near change points."
            )
            for cp in change_points:
                # Find the closest index to the change point
                cp_idx = np.argmin(np.abs(timesteps - cp))

                # Define a window around the change point
                window_start = cp_idx
                window_end = min(len(df), cp_idx + 10)

                # Find the index of maximum sum martingale in this window
                max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                detection_indices.append(max_idx)

        if detection_indices:
            logger.info(
                f"Found {len(detection_indices)} detection points at timesteps: {[timesteps[i] for i in detection_indices]}"
            )
        else:
            logger.warning("No detection points found. Analysis will be limited.")

        # Create time-based SHAP visualization
        contributions_df = self.visualize_shap_over_time(
            df=df,
            feature_cols=feature_cols,
            sum_col=sum_col,
            change_points=change_points,
            detection_indices=detection_indices,
            timesteps=timesteps,
            output_path=os.path.join(output_dir, "shap_over_time.png"),
            threshold=threshold,
            feature_names=feature_cols,
        )

        # Create traditional SHAP visualization for key detection points
        if detection_indices:
            # Take first detection index for traditional SHAP
            detection_idx = detection_indices[0]
            X_detection = df[feature_cols].iloc[detection_idx : detection_idx + 1]

            # Compute SHAP values for this single point
            shap_values = self.compute_shap_from_dataframe(
                df=df,
                feature_cols=feature_cols,
                change_points=change_points,
                timesteps=timesteps,
                detection_index=detection_idx,
            )

            # Visualize
            self.visualize_shap_values(
                shap_values=shap_values[0],
                X=X_detection,
                feature_names=feature_cols,
                output_path=os.path.join(output_dir, "shap_summary.png"),
                title=f"SHAP Values at Detection Point (t={timesteps[detection_idx]})",
            )

            # Also create detection point contribution analysis
            point_df = self.analyze_detection_point(
                X=df[feature_cols].values,
                detection_index=detection_idx,
                feature_names=feature_cols,
            )

            # Save point analysis
            point_df.to_csv(
                os.path.join(output_dir, "detection_point_analysis.csv"), index=False
            )

            # Feature comparison across multiple detection points
            if len(detection_indices) > 1:
                # Extract contributions for each detection point
                all_contribs = []
                for idx in detection_indices:
                    contrib = pd.DataFrame(
                        {
                            "Feature": feature_cols,
                            "Martingale Value": df[feature_cols].iloc[idx].values,
                            "Contribution %": df[feature_cols].iloc[idx].values
                            / df[feature_cols].iloc[idx].sum()
                            * 100,
                            "Detection Point": timesteps[idx],
                        }
                    )
                    all_contribs.append(contrib)

                # Create comparison plot
                self.plot_feature_contributions_comparison(
                    all_contributions=all_contribs,
                    feature_names=feature_cols,
                    output_path=os.path.join(output_dir, "feature_comparison.png"),
                )

        # Save contributions to CSV
        if not contributions_df.empty:
            contributions_df.to_csv(
                os.path.join(output_dir, "feature_contributions.csv"), index=False
            )

        return contributions_df
