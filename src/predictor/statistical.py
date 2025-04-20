# src/predictor/statistical.py

############################################
#### EXPERIMENTAL STATISTICAL PREDICTOR ####
############################################

from typing import List, Dict, Any
from collections import deque

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

from .base import BasePredictor


class StatisticalAdaptivePredictor(BasePredictor):
    """
    A purely statistical, model-based predictor for evolving random graphs.

    Overview:
    1) Identifies which random-graph model best fits the most recent adjacency.
    2) Estimates or updates the parameter(s) for that model (e.g., 'p' for ER, 'm' for BA, etc.).
    3) Tracks model fitness over time and detects distribution shifts (changepoints)
       if there's a sudden drop in log-likelihood or a switch in best-fitting model.
    4) Forecasts the next adjacency by sampling from or using the expected adjacency
       of the chosen model with the updated parameter(s).
    """

    def __init__(
        self,
        n_history: int = 5,
        alpha: float = 0.8,  # Smoothing factor for parameter updates
        change_threshold: float = 5,  # Threshold for log-likelihood drops to trigger changes
        min_phase_length: int = 40,
        history_size: int = 40,  # We'll store up to 40 past adjacencies
    ):
        """
        Parameters
        ----------
        n_history : int
            Number of historical states to keep.
        alpha : float
            Exponential smoothing factor for parameter updates (0 < alpha < 1).
        change_threshold : float

            Threshold for detecting a large drop in model fit => indicates a changepoint.
        min_phase_length : int
            Minimum length of a "phase" before we consider another changepoint.
        history_size : int
            Number of past adjacencies to store. Also used for advanced calculations.
        """
        self.n_history = n_history
        self._alpha = alpha
        self._change_threshold = change_threshold
        self._min_phase_length = min_phase_length

        # Internal trackers
        self._t = 0
        self._last_cp = 0  # last changepoint time
        self._best_model = None
        self._model_params = {}  # e.g., {"er": {"p": 0.05}, "ba": {"m": 2}, ...}
        self._current_params = {}  # for the currently chosen model
        self._last_loglik = None
        self._adj_history = deque(maxlen=history_size)

        # Pre-initialize known models
        self._known_models = ["er", "ba", "ws", "sbm"]
        self._init_model_params()

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """
        Predict 'horizon' future adjacency matrices by sampling (or taking the expectation)
        from the currently chosen model with its latest parameter estimates.

        We do not attempt to detect changes in "unseen" timesteps; we simply roll forward
        with the current best model and parameters.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical states, each with key 'adjacency'.
        horizon : int
            Number of steps to predict.

        Returns
        -------
        List[np.ndarray]
            List of adjacency matrices (or expected adjacency) for t+1 ... t+horizon
        """
        if not history:
            # Fallback: return empty or identity if no history
            return [np.zeros((0, 0)) for _ in range(horizon)]

        if len(history) < self.n_history:
            raise ValueError(f"Need at least {self.n_history} historical states")

        # Identify the best model and parameters from the last update_state (or do quick check)
        current_model = self._best_model
        current_params = self._current_params.copy()

        predictions = []
        for _ in range(horizon):
            # Sample or compute expected adjacency from the chosen model
            if current_model == "er":
                A_pred = self._sample_er(current_params)
            elif current_model == "ba":
                A_pred = self._sample_ba(current_params)
            elif current_model == "ws":
                A_pred = self._sample_ws(current_params)
            elif current_model == "sbm":
                A_pred = self._sample_sbm(current_params)
            else:
                # default fallback => ER with some p
                A_pred = self._sample_er({"n": 50, "p": 0.1})

            predictions.append(A_pred)

        return predictions

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """
        Observe the new adjacency, estimate parameters for each known model,
        select best model, check for changepoint, update internal states.
        """
        A_t = actual_state["adjacency"]
        self._adj_history.append(A_t)

        # 1) Fit each known model => get log-likelihood & parameter estimates
        logliks = {}
        fitted_params = {}
        for model_name in self._known_models:
            ll, params_hat = self._fit_model(model_name, A_t)
            logliks[model_name] = ll
            fitted_params[model_name] = params_hat

        # 2) Pick best model by highest log-likelihood
        best_model = max(logliks, key=logliks.get)
        best_ll = logliks[best_model]

        if self._best_model is None:
            # Initialize
            self._best_model = best_model
            self._current_params = fitted_params[best_model]
            self._last_loglik = best_ll
        else:
            # 3) Check for large drop in log-likelihood or model switch
            # If best model has changed or log-likelihood plummets => suspect changepoint
            if (best_model != self._best_model) or (
                self._last_loglik is not None
                and (self._last_loglik - best_ll > self._change_threshold)
            ):
                # Check if we have enough time since last CP
                if (self._t - self._last_cp) >= self._min_phase_length:
                    # Register changepoint
                    self._last_cp = self._t
                    self._best_model = best_model
                    self._current_params = fitted_params[best_model]
                    self._last_loglik = best_ll
                else:
                    # Not enough time passed; keep old model but do partial update
                    self._update_parameters(self._best_model, fitted_params[best_model])
                    self._last_loglik = best_ll
            else:
                # 4) If same model or no big drop, do a smooth update of parameters
                if best_model == self._best_model:
                    self._update_parameters(best_model, fitted_params[best_model])
                    self._last_loglik = best_ll
                else:
                    # The new model is slightly better but no huge jump => we can switch or blend
                    self._best_model = best_model
                    self._current_params = fitted_params[best_model]
                    self._last_loglik = best_ll

        self._t += 1

    def get_state(self) -> Dict[str, Any]:
        """Return current predictor state."""
        return {
            "time_index": self._t,
            "last_changepoint": self._last_cp,
            "best_model": self._best_model,
            "current_params": self._current_params,
            "last_loglik": self._last_loglik,
            "history_length": len(self._adj_history),
        }

    def reset(self) -> None:
        """Reset the internal state."""
        self._t = 0
        self._last_cp = 0
        self._best_model = None
        self._current_params = {}
        self._last_loglik = None
        self._adj_history.clear()
        self._init_model_params()

    # -------------------------------------------------------------------------
    # Internal: Model fitting and parameter updates
    # -------------------------------------------------------------------------
    def _fit_model(self, model_name: str, A: np.ndarray):
        """
        Fit the parameters of the given model to adjacency A,
        return (log_likelihood, fitted_params).
        """
        if model_name == "er":
            return self._fit_er(A)
        elif model_name == "ba":
            return self._fit_ba(A)
        elif model_name == "ws":
            return self._fit_ws(A)
        elif model_name == "sbm":
            return self._fit_sbm(A)
        else:
            return -np.inf, {}

    def _fit_er(self, A: np.ndarray):
        """
        MLE for Erdos-Renyi => p_hat = (number of edges) / [n*(n-1)/2].
        Log-likelihood under Bernoulli(p).
        """
        n = A.shape[0]
        edges = np.sum(A[np.triu_indices(n, k=1)])
        total_possible = n * (n - 1) / 2
        p_hat = edges / total_possible

        # Likelihood = product of p^(#edges) * (1-p)^(#non-edges)
        # log-likelihood = #edges*log(p) + (#non-edges)*log(1-p)
        if p_hat <= 0 or p_hat >= 1:
            # edge-case => clamp
            p_hat = np.clip(p_hat, 1e-4, 1 - 1e-4)

        ll = edges * np.log(p_hat) + (total_possible - edges) * np.log(1 - p_hat)
        return ll, {"n": n, "p": p_hat}

    def _fit_ba(self, A: np.ndarray):
        """
        Barabasi-Albert MLE is trickier. As a proxy, we estimate 'm' by average degree/2
        (because total edges ~ m*(n - m)).
        Then compute approximate log-likelihood using power-law distribution assumptions.
        """
        n = A.shape[0]
        degrees = A.sum(axis=0)
        avg_deg = np.mean(degrees)
        # Estimate m
        m_hat = avg_deg / 2.0
        m_hat = max(1.0, min(m_hat, n - 1))

        # Approx. log-likelihood using power-law exponent ~ 2 + m/...
        # We'll do a basic approach: if the degree distribution is ~ power-law,
        # we compute the log-likelihood of a power-law fit with exponent alpha.
        # alpha ~ 2 + (1 / (avg_deg - 1)) ??? (very rough)
        alpha_hat = 2.5 if avg_deg < 2 else (2 + 1.0 / (avg_deg - 1 + 1e-6))
        # log-likelihood of degrees ~ x^(-alpha_hat)
        # ignoring normalization constants for demonstration
        degrees_pos = degrees[degrees > 0]
        ll = -alpha_hat * np.sum(np.log(degrees_pos))

        return ll, {"n": n, "m": m_hat, "alpha": alpha_hat}

    def _fit_ws(self, A: np.ndarray):
        """
        Watts-Strogatz MLE can be complicated.
        We'll estimate 'k' from average degree, and 'beta' (rewire_prob) from clustering or short paths.
        For demonstration, approximate 'k' = avg_deg, and rewire_prob from average clustering.
        """
        n = A.shape[0]
        G = nx.from_numpy_array(A)
        avg_deg = np.mean([deg for _, deg in G.degree()])
        k_hat = int(np.round(avg_deg))
        # bounding
        k_hat = max(2, min(k_hat, n - 2))

        # approximate rewire_prob from the average clustering
        c = nx.average_clustering(G)
        # For WS: c ~ (1 - beta)^3 => beta ~ 1 - c^(1/3) (rough approximation)
        beta_hat = max(0.0, 1.0 - c ** (1 / 3))

        # Simple log-likelihood proxy:
        # we compare the observed clustering with the WS expected clustering
        # for given (k, beta).
        # or do a small negative MSE.
        c_ws = (
            (3 * (k_hat - 2)) / (4 * (k_hat - 1)) * (1 - beta_hat) ** 3
            if k_hat > 2
            else 0.0
        )
        ll = -abs(c - c_ws) * 100.0  # negative scaled difference => smaller is better

        return ll, {"n": n, "k": k_hat, "beta": beta_hat}

    def _fit_sbm(self, A: np.ndarray):
        """
        For a 2-block SBM, estimate intra- and inter-block probabilities via spectral or
        a rough guess from clustering. Then compute approximate log-likelihood.
        """
        n = A.shape[0]
        # We try a 2-cluster spectral partition
        clustering = SpectralClustering(
            n_clusters=2, affinity="precomputed", random_state=42
        )
        labels = clustering.fit_predict(A)
        block0 = labels == 0
        block1 = labels == 1
        # Probability estimates
        A_00 = A[block0][:, block0]
        A_11 = A[block1][:, block1]
        A_01 = A[block0][:, block1]

        n0 = block0.sum()
        n1 = block1.sum()

        # compute #edges and #possible
        e00 = A_00[np.triu_indices(n0, k=1)].sum()
        e11 = A_11[np.triu_indices(n1, k=1)].sum()
        e01 = A_01.sum()

        poss00 = n0 * (n0 - 1) / 2
        poss11 = n1 * (n1 - 1) / 2
        poss01 = n0 * n1

        if poss00 > 0:
            p_intra0 = e00 / poss00
        else:
            p_intra0 = 0.0
        if poss11 > 0:
            p_intra1 = e11 / poss11
        else:
            p_intra1 = 0.0
        p_intra = (e00 + e11) / (poss00 + poss11 + 1e-9)
        p_inter = e01 / (poss01 + 1e-9)

        p_intra = np.clip(p_intra, 1e-4, 1 - 1e-4)
        p_inter = np.clip(p_inter, 1e-4, 1 - 1e-4)

        # approximate log-likelihood
        ll_intra = (e00 + e11) * np.log(p_intra) + (
            (poss00 + poss11) - (e00 + e11)
        ) * np.log(1 - p_intra)
        ll_inter = e01 * np.log(p_inter) + (poss01 - e01) * np.log(1 - p_inter)
        ll = ll_intra + ll_inter

        return ll, {
            "n": n,
            "num_blocks": 2,
            "labels": labels,
            "p_intra": p_intra,
            "p_inter": p_inter,
        }

    def _update_parameters(self, model_name: str, new_params: Dict[str, float]):
        """
        Smoothly update self._current_params for the chosen model
        using exponential smoothing with factor alpha.
        """
        for key, val in new_params.items():
            if key not in self._current_params:
                self._current_params[key] = val
            else:
                old_val = self._current_params[key]
                self._current_params[key] = (
                    self._alpha * old_val + (1 - self._alpha) * val
                )

    def _init_model_params(self):
        """
        Initialize typical default parameter sets for each known model
        (used if we haven't fit anything yet).
        """
        self._model_params = {
            "er": {"n": 50, "p": 0.05},
            "ba": {"n": 50, "m": 2, "alpha": 2.5},
            "ws": {"n": 50, "k": 4, "beta": 0.1},
            "sbm": {"n": 50, "num_blocks": 2, "p_intra": 0.9, "p_inter": 0.1},
        }

    # -------------------------------------------------------------------------
    # Internal: Sampling / Prediction from model
    # -------------------------------------------------------------------------
    def _sample_er(self, params: Dict[str, float]) -> np.ndarray:
        n = int(params.get("n", 50))
        p = params.get("p", 0.05)
        A = (np.random.rand(n, n) < p).astype(float)
        np.fill_diagonal(A, 0)
        A = np.triu(A, 1)
        A = A + A.T
        return A

    def _sample_ba(self, params: Dict[str, float]) -> np.ndarray:
        """
        Sample from BA(n, m). We'll do a simple incremental approach:
        1) Start with a small random connected graph of m+1 nodes fully connected
        2) Add nodes one by one, each new node attaches to m existing nodes with probability
           proportional to degree.
        """
        n = int(params.get("n", 50))
        m = int(round(params.get("m", 2)))
        m = max(1, min(m, n - 1))

        # Start adjacency
        A = np.zeros((n, n))
        # small complete subgraph of size m+1
        A[: m + 1, : m + 1] = 1
        np.fill_diagonal(A, 0)

        degrees = A.sum(axis=0)
        # incremental addition
        for node in range(m + 1, n):
            # attach 'node' to m existing nodes with probability ~ deg / sum_deg
            p = degrees / degrees.sum()
            chosen = np.random.choice(
                np.arange(node), size=m, replace=False, p=p[:node]
            )
            for c in chosen:
                A[node, c] = 1
                A[c, node] = 1
            degrees[node] = m
            for c in chosen:
                degrees[c] += 1

        return A

    def _sample_ws(self, params: Dict[str, float]) -> np.ndarray:
        """
        Generate a Watts-Strogatz graph with n, k, beta.
        1) Start with ring of n nodes each connected to k nearest neighbors
        2) Rewire each edge with probability beta
        """
        n = int(params.get("n", 50))
        k = int(params.get("k", 4))
        beta = params.get("beta", 0.1)
        # Must keep k even for ring, or approximate
        k = k if k % 2 == 0 else (k + 1)
        k = min(k, n - 2)

        # ring adjacency
        A = np.zeros((n, n))
        half_k = k // 2
        for i in range(n):
            for j in range(1, half_k + 1):
                r = (i + j) % n
                l = (i - j) % n
                A[i, r] = 1
                A[r, i] = 1
                A[i, l] = 1
                A[l, i] = 1

        # rewire
        for i in range(n):
            for j in range(1, half_k + 1):
                neighbor = (i + j) % n
                if np.random.rand() < beta:
                    # rewire edge (i->neighbor)
                    A[i, neighbor] = 0
                    A[neighbor, i] = 0

                    # pick a random new target not i or neighbor, and not a duplicate
                    new_target = np.random.choice(
                        [x for x in range(n) if x != i and A[i, x] == 0]
                    )
                    A[i, new_target] = 1
                    A[new_target, i] = 1

        return A

    def _sample_sbm(self, params: Dict[str, float]) -> np.ndarray:
        """
        Sample from a 2-block SBM with p_intra, p_inter.
        """
        n = params.get("n", 50)
        n_blocks = int(params.get("num_blocks", 2))
        if n_blocks != 2:
            # fallback or extension
            n_blocks = 2
        p_intra = params.get("p_intra", 0.9)
        p_inter = params.get("p_inter", 0.1)

        # assume half nodes in block0, half in block1
        block_size = n // 2
        A = np.zeros((n, n))

        # Fill block0
        for i in range(block_size):
            for j in range(i + 1, block_size):
                if np.random.rand() < p_intra:
                    A[i, j] = 1
                    A[j, i] = 1

        # Fill block1
        for i in range(block_size, n):
            for j in range(i + 1, n):
                if np.random.rand() < p_intra:
                    A[i, j] = 1
                    A[j, i] = 1

        # Fill inter-block
        for i in range(block_size):
            for j in range(block_size, n):
                if np.random.rand() < p_inter:
                    A[i, j] = 1
                    A[j, i] = 1

        return A
