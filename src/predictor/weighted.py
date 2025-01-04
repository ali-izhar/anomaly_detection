# src/predictor/weighted.py

"""Weighted average network predictor."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional

from .base import BasePredictor


class WeightedPredictor(BasePredictor):
    """Predict future adjacency matrices (or probability matrices) via a weighted average
    of recent states, with optional adaptive weighting and connectivity enforcement.

    Let A_{t0}, A_{t1}, ..., A_{t(n_history-1)} be the n_history most recent adjacency
    matrices. Each entry is either 0/1 for unweighted adjacency. We define weights
    w_0, w_1, ..., w_{n_history-1}, which sum to 1.

    Step 1: Probability matrix
        P = w_0 * A_{t} + w_1 * A_{t-1} + ... + w_{n_history-1} * A_{t-(n_history-1)}

        We then rescale P into [0,1] by subtracting min(P) and dividing by (max(P)-min(P)).

    Step 2 (if binary=True):
      - We determine a 'target_avg_degree' from the last observed adjacency.
      - We convert P to a binary adjacency by sorting edges by probability (descending)
        and picking the top E edges, where E = floor(target_avg_degree * n / 2).
      - If enforce_connectivity=True, we add bridging edges for any disconnected components,
        using the highest-probability edges across those components.

    Step 3 (if adaptive=True):
      - After we predict the new adjacency (or probability matrix), we measure the L1
        difference of that new adjacency from each of the n_history old ones.
      - We re-weight the old adjacency that is "closest" to the new adjacency with a higher
        weight, and re-normalize. This modifies self.weights for the next iteration.

    Parameters
    ----------
    n_history : int, optional
        Number of recent states to use. Default = 3.
    weights : np.ndarray, optional
        Array of shape (n_history,). Will be normalized to sum=1.
        Defaults to [0.5, 0.3, 0.2] if None.
        The order is assumed to be newest -> oldest.
    adaptive : bool, optional
        If True, adapt the weights after each horizon step based on closeness
        of the newly predicted adjacency to each historical adjacency.
        Default = True.
    enforce_connectivity : bool, optional
        If False (default), and binary=True, ensure the resulting adjacency is
        (at least) a single connected component by bridging disconnected parts
        with the highest-probability edges.
    binary : bool, optional
        If True (default), produce a 0/1 adjacency. If False, produce a matrix
        of probabilities and skip connectivity enforcement.

    Notes
    -----
    - The adaptive weighting approach here is naive.
    - If binary=False, the returned matrix is never thresholded to top edges,
      so the 'target_avg_degree' is ignored.
    """

    def __init__(
        self,
        n_history: int = 3,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        enforce_connectivity: bool = False,
        binary: bool = True,
    ):
        self.n_history = n_history

        if weights is None:
            weights = np.array([0.5, 0.3, 0.2])

        # Normalize weights
        weights = np.array(weights, dtype=float)
        self.weights = weights / weights.sum()

        self.adaptive = adaptive
        self.enforce_connectivity = enforce_connectivity
        self.binary = binary

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states for 'horizon' steps.

        Algorithm:
        1. For each horizon step, gather the last n_history adjacency matrices.
        2. Compute the weighted average probability matrix P.
        3. If binary=True, pick top edges to match average degree from the most
           recent adjacency, and (optionally) enforce connectivity.
        4. If binary=False, just return the probability matrix P.
        5. If adaptive=True, re-weight self.weights based on closeness to the
           newly predicted adjacency (or probability matrix).
        6. Append the new adjacency/probability to history and repeat.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical records, each must have:
              - 'adjacency': np.ndarray (the adjacency matrix)
              - 'graph': nx.Graph object (optional, used here to find degrees)
              - 'params': any metadata (not strictly required)
        horizon : int, optional
            How many future steps to predict, default=1.

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency (or probability) matrices, one per horizon step.

        Raises
        ------
        ValueError
            If len(history) < self.n_history.
        """
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        predictions = []
        current_history = list(history)  # local copy

        for _ in range(horizon):
            # 1. Gather last n_history adjacency
            last_states = current_history[-self.n_history :]

            # 2. Weighted average => probability matrix
            avg_adj = self._compute_weighted_average(
                [st["adjacency"] for st in last_states]
            )

            # 3. If binary => pick top edges to match last graph's average degree
            if self.binary:
                latest_graph = last_states[-1].get("graph")
                if latest_graph is not None:
                    degrees = [deg for _, deg in latest_graph.degree()]
                    target_avg_degree = float(np.mean(degrees))
                else:
                    # fallback if no graph in the dictionary
                    target_avg_degree = float(np.mean(avg_adj.sum(axis=1)))

                predicted = self._prob_to_binary(
                    avg_adj, target_avg_degree, enforce=self.enforce_connectivity
                )
            else:
                # Non-binary => just keep prob matrix
                predicted = avg_adj

            predictions.append(predicted)

            # 4. Update history with this predicted adjacency
            #    So the next step can incorporate it if horizon>1
            current_history.append(
                {
                    "adjacency": predicted,
                    "graph": nx.from_numpy_array(predicted) if self.binary else None,
                }
            )

            # 5. Possibly adapt weights
            if self.adaptive:
                self._adapt_weights([st["adjacency"] for st in last_states], predicted)

        return predictions

    def _compute_weighted_average(self, adj_mats: List[np.ndarray]) -> np.ndarray:
        """Compute weighted average of adjacency matrices, shape (n,n).

        P = w_0 * adj_mats[0] + ... + w_(n-1) * adj_mats[n-1],
        and then scale P into [0,1].
        """
        avg_adj = np.zeros_like(adj_mats[0], dtype=float)

        for w, A in zip(self.weights, adj_mats):
            avg_adj += w * A

        # scale into [0,1]
        mn, mx = avg_adj.min(), avg_adj.max()
        if abs(mx - mn) < 1e-12:
            return np.zeros_like(avg_adj)  # all zero if no variance
        avg_adj = (avg_adj - mn) / (mx - mn)
        return avg_adj

    def _prob_to_binary(
        self,
        prob_matrix: np.ndarray,
        target_avg_degree: float,
        enforce: bool = False,
    ) -> np.ndarray:
        """Convert probability matrix => binary adjacency by:
        - Sorting edges in descending probability
        - Taking top E edges, where E = floor( (target_avg_degree * n)/2 )
        - Optionally bridging disconnected components with highest-prob edges
        """
        n = prob_matrix.shape[0]
        target_edges = int(np.floor((target_avg_degree * n) / 2))

        # gather all edges from upper triangular
        triu = np.triu_indices(n, k=1)
        probs = prob_matrix[triu]
        edges = list(zip(probs, triu[0], triu[1]))
        # sort descending by prob
        edges.sort(key=lambda x: x[0], reverse=True)

        predicted = np.zeros_like(prob_matrix, dtype=float)

        # pick top edges
        for _, i, j in edges[:target_edges]:
            predicted[i, j] = 1.0
            predicted[j, i] = 1.0

        # enforce connectivity if desired
        if enforce:
            graph_temp = nx.from_numpy_array(predicted)
            comps = list(nx.connected_components(graph_temp))
            if len(comps) > 1:
                # identify largest comp
                main_comp = max(comps, key=len)
                others = [c for c in comps if c != main_comp]

                # for each other comp, connect it to main_comp
                for cset in others:
                    best_edge = None
                    best_p = -1
                    for c1 in main_comp:
                        for c2 in cset:
                            p = prob_matrix[c1, c2]
                            if p > best_p:
                                best_p = p
                                best_edge = (c1, c2)

                    if best_edge is not None:
                        i2, j2 = best_edge
                        predicted[i2, j2] = 1.0
                        predicted[j2, i2] = 1.0

        return predicted

    def _adapt_weights(
        self, old_adjs: List[np.ndarray], new_adj: np.ndarray, alpha: float = 1.0
    ):
        """Adapt self.weights based on closeness of old_adjs to new_adj.

        We measure the difference (L1 norm) of each old adjacency to new_adj:
          diff_i = || old_adjs[i] - new_adj ||_1
        Then we define a raw weight as w'_i = 1 / (1 + diff_i).

        Next:
          w_i = w'_i / sum_j w'_j

        The parameter alpha can be used to amplify or dampen differences.
        For example, alpha=2.0 squares the differences, alpha=0.5 takes
        sqrt. Here we do a simple approach with alpha=1.0.

        This modifies self.weights in place, so next iteration uses them.

        Parameters
        ----------
        old_adjs : List[np.ndarray]
            The n_history adjacency matrices we just used.
        new_adj : np.ndarray
            The predicted adjacency (binary or probability).
        alpha : float
            Exponent to apply to differences, default=1.0 means direct usage.
        """
        # measure difference
        diffs = []
        for A in old_adjs:
            # L1 difference
            d = np.sum(np.abs(A - new_adj))
            diffs.append(d)

        # convert difference => weight = 1/(1 + diff^alpha)
        raw_ws = []
        for d in diffs:
            val = d**alpha
            w_i = 1.0 / (1.0 + val)
            raw_ws.append(w_i)

        # normalize
        raw_ws = np.array(raw_ws, dtype=float)
        w_sum = raw_ws.sum()
        if w_sum < 1e-12:
            # fallback, uniform
            new_weights = np.ones_like(raw_ws) / len(raw_ws)
        else:
            new_weights = raw_ws / w_sum

        # update self.weights (newest -> oldest)
        # we assume old_adjs are in the same order as self.weights: newest->oldest
        if len(new_weights) == len(self.weights):
            self.weights = new_weights
        else:
            # fallback if mismatch
            pass
