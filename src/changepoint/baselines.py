"""Baseline change point detectors: CUSUM and EWMA."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class CUSUMConfig:
    """Configuration for CUSUM detector."""
    threshold: float = 15.0
    k: float = 0.5  # Sensitivity parameter
    startup_period: int = 20
    reset: bool = True
    adaptive: bool = True


@dataclass
class EWMAConfig:
    """Configuration for EWMA detector."""
    threshold: float = 15.0
    lambda_param: float = 0.1  # Smoothing parameter
    L: float = 3.0  # Control limit width
    startup_period: int = 20
    reset: bool = True


class CUSUMDetector:
    """CUSUM (Cumulative Sum) change point detector."""

    def __init__(self, config: Optional[CUSUMConfig] = None):
        self.config = config or CUSUMConfig()

    def run(self, data: np.ndarray) -> Dict[str, Any]:
        """Run CUSUM detection.

        Args:
            data: 1D or 2D array of observations

        Returns:
            Dict with change_points and cusum values
        """
        data = np.asarray(data)
        if data.ndim == 1:
            return self._run_univariate(data)
        return self._run_multivariate(data)

    def _run_univariate(self, data: np.ndarray) -> Dict[str, Any]:
        n = len(data)
        cfg = self.config

        # Initialize
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        change_points = []

        # Baseline statistics from startup period
        startup_end = min(cfg.startup_period, n)
        mean = np.mean(data[:startup_end])
        std = max(np.std(data[:startup_end]), 1e-8)
        last_change = 0

        for i in range(startup_end, n):
            z = (data[i] - mean) / std

            cusum_pos[i] = max(0, cusum_pos[i - 1] + z - cfg.k)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - z - cfg.k)

            if cusum_pos[i] > cfg.threshold or cusum_neg[i] > cfg.threshold:
                change_points.append(i)

                if cfg.reset:
                    cusum_pos[i] = cusum_neg[i] = 0
                    if cfg.adaptive and i > last_change:
                        mean = np.mean(data[last_change:i])
                        std = max(np.std(data[last_change:i]), 1e-8)
                    last_change = i

        combined = np.maximum(cusum_pos, cusum_neg)
        return {
            "change_points": change_points,
            "traditional_change_points": change_points,
            "cusum_values": combined,
            "traditional_martingales": combined,
        }

    def _run_multivariate(self, data: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = data.shape
        cfg = self.config

        cusum_all = np.zeros((n_features, n_samples))
        all_change_points = set()

        startup_end = min(cfg.startup_period, n_samples)
        means = np.mean(data[:startup_end], axis=0)
        stds = np.maximum(np.std(data[:startup_end], axis=0), 1e-8)

        for f in range(n_features):
            cusum_pos = np.zeros(n_samples)
            cusum_neg = np.zeros(n_samples)
            last_change = 0

            for i in range(startup_end, n_samples):
                z = (data[i, f] - means[f]) / stds[f]

                cusum_pos[i] = max(0, cusum_pos[i - 1] + z - cfg.k)
                cusum_neg[i] = max(0, cusum_neg[i - 1] - z - cfg.k)
                cusum_all[f, i] = max(cusum_pos[i], cusum_neg[i])

                if cusum_all[f, i] > cfg.threshold:
                    all_change_points.add(i)

                    if cfg.reset:
                        cusum_pos[i] = cusum_neg[i] = 0
                        if cfg.adaptive and i > last_change:
                            means[f] = np.mean(data[last_change:i, f])
                            stds[f] = max(np.std(data[last_change:i, f]), 1e-8)
                        last_change = i

        change_points = sorted(all_change_points)
        sum_cusum = np.sum(cusum_all, axis=0)
        avg_cusum = np.mean(cusum_all, axis=0)

        return {
            "change_points": change_points,
            "traditional_change_points": change_points,
            "cusum_values": cusum_all,
            "traditional_sum_martingales": sum_cusum,
            "traditional_avg_martingales": avg_cusum,
            "individual_traditional_martingales": [cusum_all[f] for f in range(n_features)],
        }


class EWMADetector:
    """EWMA (Exponentially Weighted Moving Average) change point detector."""

    def __init__(self, config: Optional[EWMAConfig] = None):
        self.config = config or EWMAConfig()

    def run(self, data: np.ndarray) -> Dict[str, Any]:
        """Run EWMA detection.

        Args:
            data: 1D or 2D array of observations

        Returns:
            Dict with change_points and EWMA values
        """
        data = np.asarray(data)
        if data.ndim == 1:
            return self._run_univariate(data)
        return self._run_multivariate(data)

    def _run_univariate(self, data: np.ndarray) -> Dict[str, Any]:
        n = len(data)
        cfg = self.config
        lam = cfg.lambda_param

        ewma = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        change_points = []

        startup_end = min(cfg.startup_period, n)
        mean = np.mean(data[:startup_end])
        std = max(np.std(data[:startup_end]), 1e-8)
        ewma[:startup_end] = mean

        for i in range(startup_end, n):
            ewma[i] = lam * data[i] + (1 - lam) * ewma[i - 1]

            var_factor = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i - startup_end + 1)))
            se = std * np.sqrt(var_factor)

            upper[i] = mean + cfg.L * se
            lower[i] = mean - cfg.L * se

            if ewma[i] > upper[i] or ewma[i] < lower[i]:
                change_points.append(i)

                if cfg.reset:
                    window_start = max(0, i - cfg.startup_period)
                    mean = np.mean(data[window_start:i])
                    std = max(np.std(data[window_start:i]), 1e-8)
                    ewma[i] = mean

        ewma_stat = np.abs((ewma - mean) / std)
        return {
            "change_points": change_points,
            "traditional_change_points": change_points,
            "ewma_values": ewma,
            "traditional_martingales": ewma_stat,
            "upper_limits": upper,
            "lower_limits": lower,
        }

    def _run_multivariate(self, data: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = data.shape
        cfg = self.config
        lam = cfg.lambda_param

        ewma_all = np.zeros((n_features, n_samples))
        ewma_stat = np.zeros((n_features, n_samples))
        all_change_points = set()

        startup_end = min(cfg.startup_period, n_samples)
        means = np.mean(data[:startup_end], axis=0)
        stds = np.maximum(np.std(data[:startup_end], axis=0), 1e-8)

        for f in range(n_features):
            ewma_all[f, :startup_end] = means[f]
            mean_f, std_f = means[f], stds[f]

            for i in range(startup_end, n_samples):
                ewma_all[f, i] = lam * data[i, f] + (1 - lam) * ewma_all[f, i - 1]

                var_factor = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i - startup_end + 1)))
                se = std_f * np.sqrt(var_factor)
                ucl = mean_f + cfg.L * se
                lcl = mean_f - cfg.L * se

                ewma_stat[f, i] = abs((ewma_all[f, i] - mean_f) / std_f)

                if ewma_all[f, i] > ucl or ewma_all[f, i] < lcl:
                    all_change_points.add(i)

                    if cfg.reset:
                        window_start = max(0, i - cfg.startup_period)
                        mean_f = np.mean(data[window_start:i, f])
                        std_f = max(np.std(data[window_start:i, f]), 1e-8)
                        ewma_all[f, i] = mean_f

        change_points = sorted(all_change_points)
        sum_stat = np.sum(ewma_stat, axis=0)
        avg_stat = np.mean(ewma_stat, axis=0)

        return {
            "change_points": change_points,
            "traditional_change_points": change_points,
            "ewma_values": ewma_all,
            "traditional_sum_martingales": sum_stat,
            "traditional_avg_martingales": avg_stat,
            "individual_traditional_martingales": [ewma_stat[f] for f in range(n_features)],
        }
