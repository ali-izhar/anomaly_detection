# src/utils/rm_evaluator.py

"""Evaluator class for the MIT Reality Mining dataset."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class RealityMiningEvaluator:
    """Evaluates change point detection performance using multiple metrics.

    Computes standard binary classification metrics and detection delays
    for assessing algorithm performance against ground truth.

    A detection delay is the time between the ground truth change point and
    the detection instant and could be defined as:
    d = min{t: Mt > tau} - t* where t* is the ground truth change point.
    """

    def __init__(self, trials: int, ground_truth: int, thresholds: List[float]) -> None:
        """Initialize evaluator with experiment parameters.

        Args:
            trials: Number of repeated experiments
            ground_truth: True change point location
            thresholds: Detection thresholds to evaluate

        Raises:
            ValueError: If parameters are invalid
        """
        if trials <= 0 or ground_truth < 0:
            logger.error(
                f"Invalid parameters: trials={trials}, ground_truth={ground_truth}"
            )
            raise ValueError("Trials and ground truth must be positive")
        if not thresholds:
            logger.error("Empty threshold list provided")
            raise ValueError("Must provide at least one threshold")

        self.trials = trials
        self.ground_truth = ground_truth
        self.thresholds = thresholds
        logger.info(
            f"Initialized evaluator with {trials} trials, ground truth at t={ground_truth}"
        )
        logger.debug(f"Evaluation thresholds: {thresholds}")

    def run_test(
        self,
        detector: Any,
        threshold: float,
        adjacency_matrices: List[np.ndarray],
        feature_vectors: List[float],
        title: str,
    ) -> Dict[int, pd.DataFrame]:
        """Run detection experiment with given parameters.

        For each trial, computes:
        - Martingale sequence M_n
        - Binary detection results y_n = 1[M_n > threshold]

        Args:
            detector: Change point detection algorithm
            threshold: Detection threshold tau
            adjacency_matrices: Graph sequence [G1, ..., Gn]
            feature_vectors: Feature sequence [x1, ..., xn]
            title: Experiment identifier

        Returns:
            Dictionary mapping trial index to results DataFrame
        """
        logger.info(f"Running detection test '{title}' with threshold={threshold}")
        logger.debug(
            f"Input: {len(adjacency_matrices)} graphs, {len(feature_vectors)} feature vectors"
        )

        results: Dict[int, pd.DataFrame] = {}

        for trial in range(1, self.trials + 1):
            logger.debug(f"Starting trial {trial}/{self.trials}")

            # Run martingale test
            detection = detector.martingale_test(feature_vectors, threshold)

            # Extract results
            martingales = detection["martingales"]
            change_points = detection["change_detected_instant"]

            logger.debug(f"Trial {trial}: detected {len(change_points)} change points")

            # Create results DataFrame
            results[trial] = pd.DataFrame(
                {
                    "Martingales": martingales,
                    "Is Anomaly": [
                        i in change_points for i in range(len(adjacency_matrices))
                    ],
                }
            )

        logger.info(f"Completed {self.trials} trials")
        return results

    def true_false_positives(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Calculate true and false positives from detection results."""
        tp, fp, delay_time = [], [], []
        anomaly_indices = df[df["Is Anomaly"]].index

        for i in anomaly_indices:
            if i > self.ground_truth and not tp:
                tp.append(i)
                delay = i - self.ground_truth
                delay_time.append(delay)
                logger.debug(f"True positive at t={i} (delay={delay})")
            else:
                fp.append(i)
                logger.debug(f"False positive at t={i}")

        return {"True Positives": tp, "False Positives": fp, "Delay Time": delay_time}

    def true_false_negatives(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Calculate true and false negatives from detection results."""
        fn = df[df["Is Anomaly"]].index.tolist()
        tn = df[~df["Is Anomaly"]].index.tolist()
        logger.debug(f"Found {len(fn)} false negatives, {len(tn)} true negatives")
        return {"False Negatives": fn, "True Negatives": tn}

    def precision_calculation(
        self, data_dict: Dict[int, pd.DataFrame]
    ) -> Dict[str, List[int]]:
        """Calculate precision-related metrics across all trials."""
        logger.debug("Computing precision metrics")
        tp, fp, delay_time = [], [], []

        for trial, df in data_dict.items():
            result = self.true_false_positives(df)
            tp.extend(result["True Positives"])
            fp.extend(result["False Positives"])
            delay_time.extend(result["Delay Time"])

        logger.debug(f"Total: {len(tp)} true positives, {len(fp)} false positives")
        return {"True Positives": tp, "False Positives": fp, "Delay Time": delay_time}

    def positive_rates_calculation(
        self, data_dict: Dict[int, pd.DataFrame]
    ) -> Dict[str, List[int]]:
        """Calculate rates for false/true negatives across all trials."""
        logger.debug("Computing negative rate metrics")
        fn, tn = [], []

        for trial, df in data_dict.items():
            result = self.true_false_negatives(df)
            fn.extend(result["False Negatives"])
            tn.extend(result["True Negatives"])

        logger.debug(f"Total: {len(fn)} false negatives, {len(tn)} true negatives")
        return {"False Negatives": fn, "True Negatives": tn}

    def evaluate(
        self,
        obj_p: Any,
        obj_n: Any,
        graphs_p: List[np.ndarray],
        graphs_n: List[np.ndarray],
        feature: str = "lsvd",
    ) -> pd.DataFrame:
        """Evaluate detection performance across all thresholds."""
        logger.info(f"Starting evaluation using feature: {feature}")
        results = []

        # Extract features
        lsvd_p = obj_p.extract_features()[feature]
        lsvd_n = obj_n.extract_features()[feature]
        logger.debug(
            f"Extracted features: {len(lsvd_p)} positive, {len(lsvd_n)} negative"
        )

        for th in self.thresholds:
            logger.info(f"Evaluating threshold {th}")

            # Run detection tests
            df_dict_cent_p = self.run_test(
                detector=obj_p,
                threshold=th,
                adjacency_matrices=graphs_p,
                feature_vectors=lsvd_p,
                title=feature,
            )
            df_dict_cent_n = self.run_test(
                detector=obj_n,
                threshold=th,
                adjacency_matrices=graphs_n,
                feature_vectors=lsvd_n,
                title=feature,
            )

            # Calculate metrics
            tp_fp = self.precision_calculation(df_dict_cent_p)
            fn_tn = self.positive_rates_calculation(df_dict_cent_n)

            tp, fp = len(tp_fp["True Positives"]), len(tp_fp["False Positives"])
            fn, tn = len(fn_tn["False Negatives"]), len(fn_tn["True Negatives"])

            # Compute performance metrics
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            fpr = fp / (fp + tn) if fp + tn > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )
            miss_detection = self.trials - tp
            mean_delay = np.mean(tp_fp["Delay Time"]) if tp_fp["Delay Time"] else 0

            logger.info(
                f"Threshold {th} results - Precision: {precision}, Recall: {recall}, F1: {f1_score}"
            )
            logger.debug(
                f"Detailed metrics - FPR: {fpr}, Miss Rate: {miss_detection}/{self.trials}, Mean Delay: {mean_delay}"
            )

            results.append(
                {
                    "Precision": precision,
                    "FPR": fpr,
                    "TPR": recall,
                    "F1 score": f1_score,
                    "missdetection": [miss_detection],
                    "meanDelay": mean_delay,
                    "Threshold": th,
                }
            )

        logger.info("Evaluation complete")
        return pd.DataFrame(results)
