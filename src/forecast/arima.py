"""ARIMA-based graph state forecaster."""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Dict, Optional, Tuple
import logging
import warnings
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import copy

logger = logging.getLogger(__name__)


class ARIMAGraphForecaster:
    """ARIMA-based forecaster for graph states.

    This forecaster uses ARIMA models to predict future graph states by:
    1. Extracting features from historical graph states
    2. Fitting ARIMA models to each feature
    3. Combining predictions back into graph states
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),  # Using simpler default model
        seasonal_order: Optional[tuple] = None,
        enforce_connectivity: bool = True,
        threshold: float = 0.5,
        max_iter: int = 100,
        method: str = "powell",
        force_posparams: bool = True,
        auto_order: bool = True,
        enable_preprocessing: bool = True,
    ):
        """Initialize the ARIMA forecaster.

        Args:
            order: The (p,d,q) order of the ARIMA model
            seasonal_order: The (P,D,Q,s) order of the seasonal component
            enforce_connectivity: Whether to ensure predicted graphs are connected
            threshold: Threshold for binarizing predicted adjacency matrices
            max_iter: Maximum number of iterations for ARIMA fitting
            method: Optimization method ('lbfgs', 'newton', 'nm', 'bfgs', 'cg', 'ncg', 'powell')
            force_posparams: Whether to enforce positivity constraints on parameters
            auto_order: Whether to automatically determine ARIMA order
            enable_preprocessing: Whether to apply preprocessing to time series
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_connectivity = enforce_connectivity
        self.threshold = threshold
        self.max_iter = max_iter
        self.method = method
        self.force_posparams = force_posparams
        self.auto_order = auto_order
        self.enable_preprocessing = enable_preprocessing
        self.models = {}  # Will store ARIMA models for each feature
        self.feature_transformers = {}  # Will store preprocessing transformers
        self.feature_orders = {}  # Will store best ARIMA orders for each feature

    def _extract_features(self, history: List[Dict]) -> np.ndarray:
        """Extract features from historical graph states.

        Args:
            history: List of dictionaries containing past adjacency matrices
                    [{"adjacency": adj_matrix}, ...]

        Returns:
            numpy array of shape (n_timesteps, n_features)
        """
        features = []
        for state in history:
            adj = state["adjacency"]
            if not isinstance(adj, np.ndarray):
                adj = np.array(adj)

            # Extract basic graph features
            n = adj.shape[0]
            density = np.sum(adj) / (n * (n - 1))
            degrees = np.sum(adj, axis=1)
            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)

            # Combine features
            feature_vector = [
                density,
                mean_degree,
                std_degree,
                np.max(degrees),
                np.min(degrees),
            ]
            features.append(feature_vector)

        return np.array(features)

    def _force_positive_callback(self, params):
        """Callback to force parameters to be positive."""
        params[params < 0] = 0
        return params

    def _preprocess_timeseries(self, ts_data, feature_idx):
        """Apply preprocessing transformations to make time series more ARIMA-friendly.

        Args:
            ts_data: Raw time series data
            feature_idx: Feature index for storing transformer

        Returns:
            Preprocessed time series and transformation info
        """
        # Initialize with original data
        processed = pd.Series(copy.deepcopy(ts_data))
        transformations = []

        # Skip if too few data points
        if len(processed) < 8:
            return processed, {"transformations": []}

        # Check for zeros and near-zeros
        if np.min(np.abs(processed)) < 1e-8:
            # Add small constant to avoid zeros
            processed = processed + 1e-5
            transformations.append(("add_constant", 1e-5))

        # Check for extreme values/outliers
        z_scores = stats.zscore(processed)
        if np.max(np.abs(z_scores)) > 3:
            # Cap extreme values (windsorize)
            cap_low, cap_high = np.percentile(processed, [5, 95])
            processed = processed.clip(lower=cap_low, upper=cap_high)
            transformations.append(("clip", (cap_low, cap_high)))

        # Check for skewness - apply log transform if highly skewed
        if processed.min() > 0 and stats.skew(processed) > 1.0:
            processed = np.log1p(processed)  # log(1+x) to handle small values
            transformations.append(("log1p", None))

        # Store transformation info
        transform_info = {
            "transformations": transformations,
            "original_mean": np.mean(ts_data),
            "original_std": np.std(ts_data),
        }

        # Store for inverse transform during prediction
        self.feature_transformers[feature_idx] = transform_info

        return processed, transform_info

    def _inverse_transform(self, predictions, feature_idx):
        """Apply inverse transforms to predictions.

        Args:
            predictions: Transformed predictions
            feature_idx: Feature index for retrieving transform info

        Returns:
            Inverse-transformed predictions
        """
        if feature_idx not in self.feature_transformers:
            return predictions

        transform_info = self.feature_transformers[feature_idx]
        result = copy.deepcopy(predictions)

        # Apply inverse transformations in reverse order
        for transform_type, params in reversed(transform_info["transformations"]):
            if transform_type == "log1p":
                result = np.expm1(result)  # inverse of log1p
            elif transform_type == "clip":
                # No inverse needed for clipping
                pass
            elif transform_type == "add_constant":
                result = result - params

        return result

    def _check_stationarity(self, timeseries):
        """Check if a time series is stationary using ADF test.

        Args:
            timeseries: Time series to check

        Returns:
            Tuple of (is_stationary, recommended_d)
        """
        # Need sufficient data for testing
        if len(timeseries) < 20:
            return False, 1

        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(timeseries, autolag="AIC")
            p_value = adf_result[1]

            # If p-value > 0.05, time series is non-stationary
            if p_value > 0.05:
                # Check if first differencing helps
                if len(timeseries) > 3:
                    diff1 = np.diff(timeseries)
                    adf_diff_result = adfuller(diff1, autolag="AIC")
                    if adf_diff_result[1] <= 0.05:
                        return False, 1
                    # Check second differencing
                    if len(diff1) > 3:
                        diff2 = np.diff(diff1)
                        adf_diff2_result = adfuller(diff2, autolag="AIC")
                        if adf_diff2_result[1] <= 0.05:
                            return False, 2
                return False, 1
            else:
                return True, 0

        except Exception as e:
            logger.debug(f"Error in stationarity check: {str(e)}")
            return False, 1

    def _auto_arima_order(self, timeseries, max_p=2, max_q=2):
        """Simplified auto.arima-like functionality to find best ARIMA order.

        Args:
            timeseries: Time series data
            max_p: Maximum p value to try
            max_q: Maximum q value to try

        Returns:
            Best ARIMA order as (p,d,q)
        """
        # Check stationarity and determine 'd'
        is_stationary, recommended_d = self._check_stationarity(timeseries)
        d = 0 if is_stationary else recommended_d

        # For very short series, use simpler models
        if len(timeseries) < 30:
            return (1, d, 0)  # Simple AR(1) model with appropriate differencing

        # Set up candidate models
        best_aic = float("inf")
        best_order = (1, d, 0)  # Default to AR(1)

        try:
            # Try different combinations of p and q
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    # Skip if p=0 and q=0
                    if p == 0 and q == 0:
                        continue

                    try:
                        # Fit the model with current parameters
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            model = ARIMA(timeseries, order=(p, d, q))
                            result = model.fit(method=self.method, maxiter=50, disp=0)
                            aic = result.aic

                            # Update best model if AIC is lower
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Error in auto ARIMA: {str(e)}")

        # Use the best order found
        return best_order

    def _get_start_params(self, ts_data, order):
        """Get better start parameters for ARIMA fitting.

        This helps convergence by using AR coefficients from OLS regression
        and reasonable starting MA parameters.
        """
        n_params = sum(order) - order[1]  # p + q

        # Start with small values rather than zeros
        start_params = np.ones(n_params) * 0.1

        # Try to estimate AR coefficients if possible
        if order[0] > 0 and len(ts_data) > order[0] + 2:
            # Simple AR coefficient estimates using lagged values
            ar_coefs = []
            y = ts_data[order[0] :]

            for i in range(1, order[0] + 1):
                x = ts_data[order[0] - i : -i]
                # Add small constant to avoid division by zero
                coef = np.cov(y, x)[0, 1] / (np.var(x) + 1e-8)
                # Ensure stability by keeping AR coefficients below 1
                coef = np.clip(coef, -0.9, 0.9)
                ar_coefs.append(coef)

            # Set AR coefficients
            start_params[: order[0]] = ar_coefs

        return start_params

    def _fit_arima_with_fallback(
        self,
        timeseries: np.ndarray,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple],
        feature_idx: int = 0,
    ) -> Tuple[object, np.ndarray]:
        """Fit ARIMA model with fallback to simpler models if convergence fails.

        Args:
            timeseries: Time series data to fit
            order: ARIMA order (p,d,q)
            seasonal_order: Seasonal order (P,D,Q,s)
            feature_idx: Index of the feature being modeled (for logging)

        Returns:
            Tuple of (fitted model, forecast array)
        """
        # Skip if too few observations
        if len(timeseries) < 4:
            logger.warning(
                f"Too few observations ({len(timeseries)}) for ARIMA modeling"
            )
            return None, np.repeat(np.mean(timeseries), 5)

        # Handle constant/near constant series
        if np.std(timeseries) < 1e-8:
            logger.info(
                f"Feature {feature_idx} is nearly constant, using simple mean forecast"
            )
            return None, np.repeat(np.mean(timeseries), 5)

        # Check for NaN values
        if np.isnan(timeseries).any():
            logger.warning(f"Feature {feature_idx} contains NaN values, using fallback")
            valid_vals = timeseries[~np.isnan(timeseries)]
            return None, np.repeat(np.mean(valid_vals) if len(valid_vals) > 0 else 0, 5)

        # Use pandas Series for more consistent results
        ts_series = pd.Series(timeseries)

        # Apply preprocessing if enabled
        if self.enable_preprocessing:
            ts_series, _ = self._preprocess_timeseries(ts_series, feature_idx)

        # Determine ARIMA order if auto_order is enabled
        if self.auto_order:
            best_order = self._auto_arima_order(ts_series)
            self.feature_orders[feature_idx] = best_order
            logger.debug(f"Feature {feature_idx}: Auto-selected ARIMA{best_order}")
            use_order = best_order
        else:
            use_order = order

        # Try original parameters
        p, d, q = use_order

        # Define methods to try
        methods = [self.method, "powell", "lbfgs", "cg", "ncg", "bfgs"]

        # Define fallback orders from most to least complex
        fallback_orders = [
            use_order,  # Auto-selected or user-specified order
            (min(p, 2), d, min(q, 2)),  # Reduced AR and MA
            (1, d, 1),  # Simple ARIMA(1,d,1)
            (1, 0, 0),  # Simple AR(1)
            (0, 1, 0),  # Simple differencing
            (0, 0, 0),  # White noise
        ]

        # Generate better start parameters if needed
        start_params = self._get_start_params(ts_series, use_order)

        # Try each method and order combination
        forecast = None
        model_fit = None

        for method in methods:
            for try_order in fallback_orders:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=RuntimeWarning)

                        model = ARIMA(
                            ts_series, order=try_order, seasonal_order=seasonal_order
                        )

                        # Use callback for positivity constraints if requested
                        fit_kwargs = {
                            "method": method,
                            "maxiter": self.max_iter,
                            "disp": 0,
                        }

                        # Add callback for positive parameters if requested
                        if self.force_posparams:
                            fit_kwargs["callback"] = self._force_positive_callback

                        # Use start_params for the original order
                        if (
                            try_order == use_order
                            and len(start_params) == sum(try_order) - try_order[1]
                        ):
                            fit_kwargs["start_params"] = start_params

                        model_fit = model.fit(**fit_kwargs)
                        forecast = model_fit.forecast(
                            steps=5
                        )  # Default forecast length

                        # If we get here without exceptions, break the loop
                        logger.debug(
                            f"Feature {feature_idx}: ARIMA converged with order={try_order}, method={method}"
                        )

                        # Get forecast values, apply inverse transform if needed
                        forecast_values = (
                            forecast.values if hasattr(forecast, "values") else forecast
                        )

                        if self.enable_preprocessing:
                            forecast_values = self._inverse_transform(
                                forecast_values, feature_idx
                            )

                        return model_fit, forecast_values

                except Exception as e:
                    logger.debug(
                        f"Feature {feature_idx}: ARIMA fitting failed with order={try_order}, method={method}: {str(e)}"
                    )
                    continue

        # If all attempts failed, use exponential smoothing as fallback
        logger.info(f"Feature {feature_idx}: Using exponential smoothing fallback")

        # Get original time series if preprocessing was applied
        orig_ts = timeseries

        # Apply simple exponential smoothing with alpha=0.3
        alpha = 0.3
        smoothed = np.zeros(len(orig_ts))
        smoothed[0] = orig_ts[0]

        for i in range(1, len(orig_ts)):
            smoothed[i] = alpha * orig_ts[i] + (1 - alpha) * smoothed[i - 1]

        # Forecast using the last smoothed value
        last_value = smoothed[-1]
        forecast = np.array([last_value] * 5)

        return None, forecast

    def _reconstruct_graph(self, features: np.ndarray, n_nodes: int) -> np.ndarray:
        """Reconstruct a graph from predicted features.

        Args:
            features: Predicted feature vector
            n_nodes: Number of nodes in the graph

        Returns:
            Predicted adjacency matrix
        """
        density, mean_degree, std_degree, max_degree, min_degree = features

        # Ensure valid feature values
        density = np.clip(density, 0.01, 0.99)  # Prevent extreme densities
        mean_degree = np.clip(mean_degree, 1, n_nodes - 1)  # Reasonable mean degree
        std_degree = max(0.1, std_degree)  # Prevent zero std
        max_degree = min(
            n_nodes - 1, max(mean_degree, max_degree)
        )  # Keep max degree reasonable
        min_degree = max(1, min(mean_degree, min_degree))  # Keep min degree reasonable

        # Create initial random graph with target density
        adj = np.random.random((n_nodes, n_nodes))
        adj = (adj + adj.T) / 2  # Make symmetric
        np.fill_diagonal(adj, 0)  # Zero diagonal

        # Threshold to achieve target density
        threshold = np.percentile(adj, (1 - density) * 100)
        adj = (adj > threshold).astype(int)

        # Adjust degrees to match predicted statistics
        current_degrees = np.sum(adj, axis=1)
        target_degrees = np.random.normal(mean_degree, std_degree, n_nodes)
        target_degrees = np.clip(target_degrees, min_degree, max_degree)

        # Adjust edges to match target degrees
        for i in range(n_nodes):
            diff = int(target_degrees[i] - current_degrees[i])
            if diff > 0:
                # Add edges
                non_edges = np.where(adj[i] == 0)[0]
                non_edges = non_edges[non_edges != i]  # Exclude self-loops
                if len(non_edges) > 0:
                    to_add = np.random.choice(non_edges, min(diff, len(non_edges)))
                    adj[i, to_add] = 1
                    adj[to_add, i] = 1
            elif diff < 0:
                # Remove edges
                edges = np.where(adj[i] == 1)[0]
                if len(edges) > 0:
                    to_remove = np.random.choice(edges, min(-diff, len(edges)))
                    adj[i, to_remove] = 0
                    adj[to_remove, i] = 0

        # Ensure the graph is connected if required
        if self.enforce_connectivity and n_nodes > 1:
            # Simple check for connectedness - ensure minimum degree is at least 1
            for i in range(n_nodes):
                if np.sum(adj[i]) == 0:  # Isolated node
                    # Connect to a random node
                    j = np.random.choice([j for j in range(n_nodes) if j != i])
                    adj[i, j] = 1
                    adj[j, i] = 1

        return adj

    def predict(self, history: List[Dict], horizon: int = 5) -> List[np.ndarray]:
        """Predict future graph states.

        Args:
            history: List of dictionaries containing past adjacency matrices
                    [{"adjacency": adj_matrix}, ...]
            horizon: Number of future time steps to predict

        Returns:
            List of predicted adjacency matrices
        """
        if len(history) < 2:
            raise ValueError("Need at least 2 historical states for prediction")

        # Extract features from history
        features = self._extract_features(history)
        n_features = features.shape[1]
        n_nodes = history[0]["adjacency"].shape[0]

        # Reset feature transformers and orders
        self.feature_transformers = {}
        self.feature_orders = {}

        # Fit ARIMA models for each feature
        predicted_features = []
        for i in range(n_features):
            feature_ts = features[:, i]

            # Use robust fitting with fallback, passing feature index for better logging
            _, forecast = self._fit_arima_with_fallback(
                feature_ts, self.order, self.seasonal_order, feature_idx=i
            )

            # Ensure forecast has the right length
            if len(forecast) > horizon:
                forecast = forecast[:horizon]
            elif len(forecast) < horizon:
                # Extend forecast if needed
                last_val = forecast[-1] if len(forecast) > 0 else feature_ts[-1]
                extension = np.repeat(last_val, horizon - len(forecast))
                forecast = np.concatenate([forecast, extension])

            predicted_features.append(forecast)

        # Combine predictions
        predicted_features = np.array(predicted_features).T

        # Reconstruct graphs from predicted features
        predicted_graphs = []
        for features in predicted_features:
            adj = self._reconstruct_graph(features, n_nodes)
            predicted_graphs.append(adj)

        # Log feature transformation and ARIMA order information if any
        if self.feature_transformers and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Applied {len(self.feature_transformers)} feature transformations"
            )

        if self.feature_orders and logger.isEnabledFor(logging.DEBUG):
            orders_str = ", ".join(
                [f"F{i}: {o}" for i, o in self.feature_orders.items()]
            )
            logger.debug(f"Used ARIMA orders: {orders_str}")

        return predicted_graphs
