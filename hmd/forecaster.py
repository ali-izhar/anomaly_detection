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

import networkx as nx
import numpy as np

__all__ = [
    "Predictor",
    "GraphPredictor",
    "EWMA",
    "HoltForecaster",
    "GraphSpaceForecaster",
    "default",
]


@runtime_checkable
class Predictor(Protocol):
    """Feature-space forecaster protocol (§IV-B(a)).

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


@runtime_checkable
class GraphPredictor(Protocol):
    """Graph-space forecaster protocol (§IV-B(b), Eqs 22-25).

    Forecasts the next *graph* rather than next feature vector. The detector
    then extracts features from the forecasted graph using the same
    `hmd.features.extract` call applied to observed graphs, so the horizon
    stream sees feature values in the same coordinate system as traditional.
    """

    def predict_graph(self, history: list["nx.Graph"], horizon: int) -> "nx.Graph":
        """Forecast a graph h steps ahead from a window of recent graphs."""
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


# ----------------------------------------------------------------------------
# §IV-B(b) Graph-space forecaster with structural constraints
# ----------------------------------------------------------------------------


class GraphSpaceForecaster:
    """§IV-B(b), Eqs 22-25 — constrained adjacency-matrix forecaster.

    Forecasts the next graph by blending a temporal EWMA on adjacency matrices
    with a structural prior that projects onto the space of graphs with
    preserved degree/community/sparsity characteristics.

    Forecast rule (Eq 22):
        Â_{t+h} = β_t · Ã^(T)_{t+h}  +  (1 − β_t) · Â^(S)_{t+h}

    where
      - Ã^(T) is the per-step EWMA of recent adjacency matrices (continuous
        weights in [0, 1])
      - Â^(S) is the sparsity-projected version of Ã^(T) onto A(n, κ_t)
        where κ_t is the average density over the window
      - β_t ∈ (0, 1) is the adaptive blend weight (Eq 22 text)

    The paper's Â^(S) is the argmin over A(n, κ_t) of ‖Ã^(T) − A‖_F + γ·R(A)
    with R(A) = ω_1 R_1(A) + ω_2 R_2(A) + ω_3 R_3(A) — degree preservation
    (Eq 23), community structure (Eq 24), sparsity (Eq 25). We implement
    Â^(S) as the top-κ_t·n(n-1)/2 edge-weight selection from Ã^(T); this
    satisfies R_3 exactly and is the Frobenius-projection onto the sparsity
    constraint set, which reduces to the paper's optimization when ω_3 → 1.
    R_1 and R_2 are computed as diagnostics (stored on the forecaster) but
    not iteratively minimized — full R(A) minimization requires a discrete
    search (combinatorial over edge subsets) that's out of scope here.

    The adaptive β_t tracks `δ_t = ‖Ã^(T)_{t-1} − A_t‖_F − ‖Â^(S)_{t-1} − A_t‖_F`
    internally — the forecaster maintains one scalar of state across calls.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be ≥ 0; got {gamma}")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (3,) or (w < 0).any() or not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must be 3 non-negative numbers summing to 1; got {weights}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.weights = tuple(w)
        # Running state for adaptive β_t (exponentially-smoothed δ_t).
        self._delta_ema: float = 0.0

    # -- Core graph-space forecast ------------------------------------------

    def predict_graph(self, history: list[nx.Graph], horizon: int) -> nx.Graph:
        """Forecast Â_{t+horizon} from a window of recent graphs.

        Parameters
        ----------
        history : list[nx.Graph]
            Window of recent snapshots, ordered oldest→newest. All graphs
            must share a common node set (same size, same labels).
        horizon : int
            Steps ahead (≥ 1). Unused under stationary EWMA (all horizons
            yield the same temporal forecast; structural prior is h-invariant).

        Returns
        -------
        nx.Graph : forecast graph with the same node set.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1; got {horizon}")
        if not history:
            raise ValueError("history must be non-empty")

        node_list = sorted(history[-1].nodes())
        n = len(node_list)
        w = len(history)

        # --- 1. Temporal EWMA on adjacency matrices (Eq 22 top) ------------
        # w_j = α^j / Σα^l, the j-th most recent snapshot carries weight w_j.
        js = np.arange(1, w + 1, dtype=np.float64)
        weights_j = np.power(self.alpha, js)
        weights_j /= weights_j.sum()
        A_tilde = np.zeros((n, n), dtype=np.float64)
        for j in range(w):
            A_j = nx.to_numpy_array(history[-(j + 1)], nodelist=node_list, dtype=np.float64)
            A_tilde += weights_j[j] * A_j
        # Ensure symmetry (numerical safety).
        A_tilde = 0.5 * (A_tilde + A_tilde.T)
        np.fill_diagonal(A_tilde, 0.0)

        # --- 2. Target sparsity κ_t (avg density over window) --------------
        # Eq 24 text: κ_t = (1/w) Σ_s ‖A_s‖_0 / (n(n-1)).
        # NOTE: paper's formula is edges-in-asymmetric-matrix / possible pairs.
        # For undirected graphs we use 2|E| / (n(n-1)) (i.e., fraction of
        # undirected pairs that are edges), equivalent up to a factor of 2.
        n_pairs = n * (n - 1) // 2
        densities = np.asarray(
            [g.number_of_edges() / n_pairs if n_pairs > 0 else 0.0 for g in history]
        )
        kappa_t = float(densities.mean())
        n_edges_target = int(round(kappa_t * n_pairs))

        # --- 3. Structural prior Â^(S): project Ã^(T) onto A(n, κ_t) --------
        # Top-k edge selection on the upper triangle of Ã^(T) satisfies the
        # sparsity constraint R_3 = 0 exactly and minimizes the Frobenius
        # distance to Ã^(T) subject to that constraint.
        iu = np.triu_indices(n, k=1)
        weights_flat = A_tilde[iu]
        A_hat_S = np.zeros((n, n), dtype=np.float64)
        if 0 < n_edges_target <= len(weights_flat):
            top_k_idx = np.argpartition(weights_flat, -n_edges_target)[-n_edges_target:]
            A_hat_S[iu[0][top_k_idx], iu[1][top_k_idx]] = 1.0
            A_hat_S = A_hat_S + A_hat_S.T

        # --- 4. Adaptive blend β_t (Eq 22 text) -----------------------------
        # β_t = (1 + exp(δ_t))^{-1} where δ_t tracks recent prediction errors.
        # We maintain an EMA of δ as state across calls.
        beta_t = 1.0 / (1.0 + np.exp(self._delta_ema))

        # --- 5. Blend and binarize ------------------------------------------
        A_hat_cont = beta_t * A_tilde + (1.0 - beta_t) * A_hat_S
        # Final binarize to sparsity κ_t (keep top-k edges).
        flat = A_hat_cont[iu]
        binary = np.zeros((n, n), dtype=np.uint8)
        if 0 < n_edges_target <= len(flat):
            top = np.argpartition(flat, -n_edges_target)[-n_edges_target:]
            binary[iu[0][top], iu[1][top]] = 1
            binary = binary + binary.T

        # --- 6. Update δ state using last observed graph as "ground truth"
        # for the previous step's forecast (lag-1 online self-calibration).
        A_last = nx.to_numpy_array(history[-1], nodelist=node_list, dtype=np.float64)
        err_T = float(np.linalg.norm(A_tilde - A_last, ord="fro"))
        err_S = float(np.linalg.norm(A_hat_S - A_last, ord="fro"))
        # δ_t > 0 means structural prior fits better ⇒ β_t decreases (more weight on S).
        delta_t = err_T - err_S
        # Exponential smoothing of δ with the same α as temporal EWMA.
        self._delta_ema = self.alpha * delta_t + (1.0 - self.alpha) * self._delta_ema

        # --- 7. Compute diagnostic regularizer values (stored for inspection)
        self.last_R1 = self._degree_preservation(binary, history, node_list)
        self.last_R2 = self._community_structure(binary, history[-1], node_list)
        self.last_R3 = abs(
            (binary.sum() / 2) / n_pairs - kappa_t if n_pairs > 0 else 0.0
        )

        return nx.from_numpy_array(binary.astype(np.int8))

    # -- Regularizer diagnostics --------------------------------------------

    @staticmethod
    def _degree_preservation(A: np.ndarray, history: list[nx.Graph], nodes: list) -> float:
        """Eq 23: R_1(A) = (1/|V|) Σ_v ((d_A(v) − μ_t(d(v))) / σ_t(d(v)))^2."""
        n = A.shape[0]
        degrees_hist = np.zeros((len(history), n), dtype=np.float64)
        for i, g in enumerate(history):
            for j, v in enumerate(nodes):
                degrees_hist[i, j] = g.degree(v)
        mu = degrees_hist.mean(axis=0)
        sigma = degrees_hist.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)  # avoid /0 on constant degree
        d_A = A.sum(axis=1)
        z2 = ((d_A - mu) / sigma) ** 2
        return float(z2.mean())

    @staticmethod
    def _community_structure(A: np.ndarray, G_last: nx.Graph, nodes: list) -> float:
        """Eq 24: R_2(A) = Σ_{C ∈ C_t} |e_C(A)/|C|² − e_C(A_hist)/|C|²|.

        Communities C_t are detected by greedy modularity on G_last; A_hist
        uses G_last's edges to compute reference per-community edge counts.
        """
        try:
            from networkx.algorithms import community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(G_last))
        except Exception:
            return 0.0
        node_idx = {v: i for i, v in enumerate(nodes)}
        A_hist = nx.to_numpy_array(G_last, nodelist=nodes, dtype=np.float64)
        total = 0.0
        for C in communities:
            idx = np.asarray([node_idx[v] for v in C if v in node_idx], dtype=np.int64)
            size2 = max(len(idx) ** 2, 1)
            e_C_A = 0.5 * A[np.ix_(idx, idx)].sum()
            e_C_H = 0.5 * A_hist[np.ix_(idx, idx)].sum()
            total += abs(e_C_A / size2 - e_C_H / size2)
        return float(total)


def default() -> Predictor:
    """EWMA(α=0.5) — paper's Sec IV-B(a) default."""
    return EWMA(alpha=0.5)
