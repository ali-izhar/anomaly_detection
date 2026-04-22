"""Feature-space forecaster — EWMA (Eq 21) behind a pluggable Protocol.

- **Protocol, not ABC.** Any class with matching `predict` / `predict_multi`
  signatures plugs in; no inheritance. Swap EWMA for ARIMA/GNN in one line.
- **Feature-space, not graph-space.** Eq 21 predicts each of the K scalar
  coordinates independently. Graph-space forecasting (Eqs 22-25) is the
  N×N constrained optimisation the paper proposes but does not evaluate.
- **EWMA, not Holt's double-exponential.** Holt's trend term `b_t` compounds
  noise across h-step forecasts under H₀ and inflates predictive-p-value
  variance (breaks Def 8 calibration → breaks Thm 4 false-alarm control).
  EWMA is the conservative calibrated choice the paper uses.
- **h-step = 1-step for stationary EWMA.** No trend term ⇒ E[X_{t+h}] = level_t.
  The paper's optimal h ≈ 5; beyond that the forecast is stale but bounded.

Eq 21 form:  X̂_{t+h} = Σ_{j=1}^{w} w_j · X_{t-j+1},  w_j = α^j / Σ_l α^l.
The j-th most-recent obs carries weight ∝ α^j, so α → 0 sharpens focus on
history[-1]. Paper uses α = 0.5 as the default; we match.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Predictor", "EWMA", "HoltForecaster", "default"]


@runtime_checkable
class Predictor(Protocol):
    """Forecaster protocol.

    Implementations must be stateless w.r.t. the detector loop: the caller
    passes the history window in each call. Both ``predict`` (scalar) and
    ``predict_multi`` (vector) are required — the detector uses the vector
    form for multi-feature efficiency.
    """

    def predict(self, history: np.ndarray, horizon: int) -> float:
        """Scalar h-step-ahead forecast from a 1-D history of shape (w,)."""
        ...

    def predict_multi(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Vector h-step-ahead forecast from a 2-D history of shape (w, K).

        Returns a length-K array. Required by HorizonDetector's inner loop.
        """
        ...


class EWMA:
    """§IV-B(a), Eq 21 — exponentially weighted moving average forecaster.

        X̂_{t+h} = Σ_{j=1}^{w} w_j · X_{t-j+1},   w_j = α^j / Σ_{l=1}^{w} α^l

    With history ordered history[-1] = X_t (most recent), history[0] = X_{t-w+1}
    (oldest), the weight on history[-j] (the j-th most recent observation,
    1-indexed) is α^j / Σ α^l. So history[-1] gets weight α / Σα^l, history[-2]
    gets α²/Σα^l, etc. — more recent observations get higher weight iff α < 1.

    Parameters
    ----------
    alpha : float in (0, 1)
        Decay. Smaller α → sharper focus on the latest observation. The paper
        uses α = 0.5 as the default stationary-regime setting.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"EWMA requires α ∈ (0, 1); got {alpha!r}")
        self.alpha = float(alpha)

    # Why two methods, not one with broadcasting:
    # - predict(): 1-D scalar forecast, matches the Protocol.
    # - predict_multi(): (w, K) -> (K,), avoids a Python for-loop over features.
    def predict(self, history: np.ndarray, horizon: int) -> float:
        """Scalar EWMA h-step forecast.

        Parameters
        ----------
        history : np.ndarray, shape (w,)
        horizon : int, ≥ 1 (unused under stationary EWMA — see module Why#4).

        Returns
        -------
        float
        """
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1; got {horizon}")
        h = np.asarray(history, dtype=np.float64).ravel()
        w = h.shape[0]
        if w == 0:
            return 0.0  # no history → trivial forecast
        # j-th most recent element (1-indexed) is h[-j]; j = 1..w.
        j = np.arange(1, w + 1, dtype=np.float64)
        wj_unnorm = np.power(self.alpha, j)
        wj = wj_unnorm / wj_unnorm.sum()
        # h[::-1][j-1] == h[-j]
        return float(np.dot(wj, h[::-1]))

    def predict_multi(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Vectorized EWMA forecast across K features.

        Parameters
        ----------
        history : np.ndarray, shape (w, K)
        horizon : int, ≥ 1

        Returns
        -------
        np.ndarray of shape (K,).
        """
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1; got {horizon}")
        H = np.asarray(history, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError(f"history must be 2-D (w, K); got shape {H.shape}")
        w, K = H.shape
        if w == 0:
            return np.zeros(K, dtype=np.float64)
        j = np.arange(1, w + 1, dtype=np.float64)
        wj_unnorm = np.power(self.alpha, j)
        wj = wj_unnorm / wj_unnorm.sum()  # (w,)
        # H[::-1] reverses rows so row 0 is most recent → weight wj[0].
        return (wj[:, None] * H[::-1, :]).sum(axis=0)


class HoltForecaster:
    """Holt's double-exponential smoothing forecaster (experimental ablation).

    Recurrence:
        level_t  = α · x_t + (1 − α) · (level_{t−1} + trend_{t−1})
        trend_t  = β · (level_t − level_{t−1}) + (1 − β) · trend_{t−1}
        forecast_{t+h} = level_t + h · trend_t

    Fit to a contiguous window of history each call (stateless w.r.t. detector),
    then extrapolate h steps ahead.

    NOTE: unlike EWMA, Holt's trend term `b_t` integrates noise across
    observations. Under H₀ (stationary regime) this inflates the variance of
    h-step forecasts, which can degrade predictive-p-value calibration (Def 8)
    for large h. For small h (≤5) the effect is modest in practice.

    Parameters
    ----------
    alpha : float in (0, 1)
        Level smoothing. Default 0.3 per CLAUDE.md FeaturePredictor notes.
    beta : float in (0, 1)
        Trend smoothing. Default 0.1 per CLAUDE.md FeaturePredictor notes.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Holt requires α ∈ (0, 1); got {alpha!r}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"Holt requires β ∈ (0, 1); got {beta!r}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _fit_and_forecast_1d(self, h: np.ndarray, horizon: int) -> float:
        """Fit Holt on 1-D history, return h-step forecast."""
        w = h.shape[0]
        if w == 0:
            return 0.0
        if w == 1:
            return float(h[0])
        # Init: level = x_0, trend = x_1 - x_0  (standard Holt initialization).
        level = float(h[0])
        trend = float(h[1] - h[0])
        for t in range(1, w):
            prev_level = level
            level = self.alpha * float(h[t]) + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        return level + horizon * trend

    def predict(self, history: np.ndarray, horizon: int) -> float:
        """Scalar Holt h-step forecast from 1-D history (w,)."""
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1; got {horizon}")
        h = np.asarray(history, dtype=np.float64).ravel()
        return float(self._fit_and_forecast_1d(h, horizon))

    def predict_multi(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Vectorized Holt h-step forecast across K features; (w, K) -> (K,)."""
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1; got {horizon}")
        H = np.asarray(history, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError(f"history must be 2-D (w, K); got shape {H.shape}")
        w, K = H.shape
        if w == 0:
            return np.zeros(K, dtype=np.float64)
        if w == 1:
            return H[0].copy()
        # Vectorized recurrence across K features.
        level = H[0].copy()
        trend = H[1] - H[0]
        for t in range(1, w):
            prev_level = level.copy()
            level = self.alpha * H[t] + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        return level + horizon * trend


def default() -> Predictor:
    """EWMA(α=0.5) — paper's Sec IV-B(a) default."""
    return EWMA(alpha=0.5)
