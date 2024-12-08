# src/changepoint/detector.py

import logging
from typing import List, Dict, Any
import numpy as np

from .martingale import compute_martingale, multiview_martingale_test
from ..graph.features import extract_centralities, compute_embeddings

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detects structural changes in temporal graph sequences using martingale-based methods.
    Implements both univariate and multivariate martingale tests on graph features (centralities, embeddings).
    Uses power martingales: M_n = product(i=1 to n) (epsilon * |p_i|^(epsilon - 1)), where p_i are p-values and epsilon in (0,1).
    """

    def __init__(self) -> None:
        """Initialize detector with empty graph sequence and centrality storage."""
        self._graphs: List[np.ndarray] = []
        self._centralities: Dict[str, List[List[float]]] = {}
        logger.debug("Initialized empty ChangePointDetector")

    @property
    def graphs(self) -> List[np.ndarray]:
        """Get current graph sequence."""
        return self._graphs

    @property
    def centralities(self) -> Dict[str, List[List[float]]]:
        """Get computed centrality measures."""
        return self._centralities

    def initialize(self, graphs: List[np.ndarray]) -> None:
        """Set graph sequence for analysis.

        Args:
            graphs: List of adjacency matrices [n x n] representing temporal graph sequence.

        Raises:
            ValueError: If graphs list is empty or matrices have invalid dimensions.
        """
        if not graphs:
            logger.error("Attempted to initialize with empty graph sequence")
            raise ValueError("Graph sequence cannot be empty")

        if not all(g.shape == graphs[0].shape for g in graphs):
            logger.error(
                f"Inconsistent graph dimensions detected: {[g.shape for g in graphs]}"
            )
            raise ValueError("All adjacency matrices must have same dimensions")

        self._graphs = graphs
        logger.info(f"Initialized detector with {len(graphs)} graphs.")
        logger.debug(
            f"Memory usage for graph sequence: {sum(g.nbytes for g in graphs) / 1e6:.2f} MB"
        )

    def extract_features(self) -> Dict[str, List[List[float]]]:
        if not self._graphs:
            logger.error("Feature extraction attempted without initialized graphs")
            raise ValueError("No graphs initialized. Call initialize() first")

        logger.info("Starting feature extraction process")
        logger.debug(f"Extracting features from {len(self._graphs)} graphs")

        self._centralities = extract_centralities(self._graphs)
        logger.debug(
            f"Extracted centrality measures: {list(self._centralities.keys())}"
        )

        # Compute embeddings
        logger.info("Computing graph embeddings")
        svd_list = compute_embeddings(
            self._graphs, method="svd"
        )  # Each element should be (n_nodes, embedding_dim) or (n_nodes,)
        lsvd_list = compute_embeddings(self._graphs, method="lsvd")  # Same as above

        # Ensure that both svd_list and lsvd_list arrays are 2D
        # If an array is 1D (n_nodes,), reshape it to (n_nodes, 1)
        def ensure_2d(arr_list):
            new_list = []
            for arr in arr_list:
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]  # Make it (n_nodes,1)
                new_list.append(arr)
            return new_list

        svd_list = ensure_2d(svd_list)
        lsvd_list = ensure_2d(lsvd_list)

        # Now we can safely average over axis=1
        svd_node_level = [arr.mean(axis=1) for arr in svd_list]  # seq_len x n_nodes
        lsvd_node_level = [arr.mean(axis=1) for arr in lsvd_list]  # seq_len x n_nodes

        embeddings = {
            "svd": svd_node_level,
            "lsvd": lsvd_node_level,
        }

        self._centralities.update(embeddings)

        logger.info(
            f"Feature extraction complete. Total features: {len(self._centralities)}"
        )
        return self._centralities

    def martingale_test(
        self,
        data: List[float],
        threshold: float,
        epsilon: float = 0.8,
        reset: bool = True,
    ) -> Dict[str, Any]:
        """Perform univariate martingale-based change detection.
        Implements sequential martingale test: M_n exceeding threshold indicates change.
        M_n is reset after detection if reset is True for finding multiple changes.

        Args:
            data: Univariate time series of feature values
            threshold: Detection threshold for martingale value
            epsilon: Sensitivity parameter epsilon in (0,1), smaller = more sensitive
            reset: Whether to reset martingale after detection

        Returns:
            Dictionary containing:
            - change_detected_instant: List of detection times
            - pvalues: Sequence of p-values
            - strangeness: Sequence of strangeness values
            - martingales: Sequence of martingale values
        """
        logger.debug(
            f"Running martingale test with threshold={threshold}, epsilon={epsilon}"
        )
        results = compute_martingale(data, threshold, epsilon, reset)

        if results["change_detected_instant"]:
            logger.info(
                f"Change points detected at: {results['change_detected_instant']}"
            )
        else:
            logger.debug("No change points detected")

        return results

    def multiview_martingale_test(
        self, data: List[List[float]], threshold: float = 20, epsilon: float = 0.8
    ) -> Dict[str, Any]:
        """Perform multivariate martingale-based change detection.
        Combines evidence from multiple features using sum of individual martingales:
        M_total = sum(j=1 to d) M_n^j, where d is number of features.

        Args:
            data: List of feature sequences [samples x features]
            threshold: Detection threshold for combined martingale
            epsilon: Sensitivity parameter epsilon in (0,1)

        Returns:
            Dictionary containing:
            - change_detected_instant: List of detection times
            - pvalues: Sequence of p-values
            - strangeness: Sequence of strangeness values
            - martingales: Sequence of martingale values
        """
        logger.info(f"Starting multiview martingale test with {len(data[0])} features")
        logger.debug(f"Parameters: threshold={threshold}, epsilon={epsilon}")

        results = multiview_martingale_test(data, threshold, epsilon)

        if results["change_detected_instant"]:
            logger.info(
                f"Multiple change points detected at: {results['change_detected_instant']}"
            )
            logger.debug(f"Maximum martingale sum: {max(results['martingales'])}")
        else:
            logger.debug("No change points detected in multiview analysis")

        return results
