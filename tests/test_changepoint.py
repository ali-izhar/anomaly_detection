# tests/test_changepoint.py

"""
Test the `src/changepoint` module.

1. Single-view martingale detection
2. Multi-view martingale detection
3. Memory efficiency
4. Early stopping
5. Edge cases and error handling
"""

import unittest
import sys
import os
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.changepoint.detector import ChangePointDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMartingaleDetection(unittest.TestCase):
    """Test suite for martingale-based change point detection."""

    def setUp(self):
        """Initialize detector and test data."""
        self.detector = ChangePointDetector()

        # Generate synthetic data with known change points
        np.random.seed(42)
        self.seq_len = 1000
        self.change_points = [250, 500, 750]

        # Create sequence with mean shifts - reshape to 2D array
        self.data = []
        current_mean = 0
        for i in range(self.seq_len):
            if i in self.change_points:
                current_mean += 1
            self.data.append(
                [np.random.normal(current_mean, 0.1)]
            )  # Make each point 2D
        self.data = np.array(self.data)

    def test_single_view_detection(self):
        """Test single-view martingale detection."""
        results = self.detector.detect_changes(
            data=self.data, threshold=20, epsilon=0.8
        )

        # Verify detection
        self.assertTrue(len(results["change_points"]) > 0)

        # Verify detection near true change points
        detected = results["change_points"]
        for true_cp in self.change_points:
            # Allow for detection delay up to 50 time steps
            self.assertTrue(
                any(abs(d - true_cp) <= 50 for d in detected),
                f"Failed to detect change point near {true_cp}",
            )

        # Verify martingale properties
        martingales = results["martingale_values"]
        self.assertTrue(all(m >= 0 for m in martingales))
        self.assertEqual(len(martingales), len(self.data))

    def test_multiview_detection(self):
        """Test multi-view martingale detection."""
        # Create multi-feature data
        features = []
        for shift in [0, 0.5, 1.0]:  # Different mean shifts
            feature_data = []
            current_mean = shift
            for i in range(self.seq_len):
                if i in self.change_points:
                    current_mean += 1
                feature_data.append([np.random.normal(current_mean, 0.1)])
            features.append(np.array(feature_data))

        results = self.detector.detect_changes_multiview(
            data=features,
            threshold=50,  # Higher threshold for combined martingales
            epsilon=0.8,
        )

        # Verify detection
        self.assertTrue(len(results["change_points"]) > 0)

        # Verify detection near true change points
        detected = results["change_points"]
        for true_cp in self.change_points:
            self.assertTrue(
                any(abs(d - true_cp) <= 50 for d in detected),
                f"Failed to detect change point near {true_cp}",
            )

        # Verify martingale properties
        martingale_values = results["martingale_values"]
        self.assertTrue(all(m >= 0 for m in martingale_values))

    def test_memory_efficiency(self):
        """Test memory efficiency with window size limits."""
        window_size = 100
        results = self.detector.detect_changes(
            data=self.data, threshold=20, epsilon=0.8, max_window=window_size
        )

        # Verify window size constraint
        self.assertTrue(
            all(len(str(m)) < 1000 for m in results["martingale_values"]),
            "Martingale values growing too large",
        )

    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_threshold = 1000
        features = [self.data] * 3  # Multiple copies of the same feature

        results = self.detector.detect_changes_multiview(
            data=features, threshold=20, epsilon=0.8, max_martingale=early_threshold
        )

        # Verify early stopping
        if len(results["martingale_values"]) < self.seq_len:
            max_martingale = max(results["martingale_values"])
            self.assertGreaterEqual(
                max_martingale,
                early_threshold,
                "Early stopping occurred before threshold",
            )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test invalid epsilon
        with self.assertRaises(ValueError):
            self.detector.detect_changes(data=self.data, threshold=20, epsilon=1.5)

        # Test invalid threshold
        with self.assertRaises(ValueError):
            self.detector.detect_changes(data=self.data, threshold=-1, epsilon=0.8)

        # Test empty data
        with self.assertRaises(ValueError):
            self.detector.detect_changes(data=np.array([]), threshold=20, epsilon=0.8)

    def test_parameter_sensitivity(self):
        """Test sensitivity to epsilon and threshold parameters."""
        # Test different epsilon values
        epsilons = [0.2, 0.5, 0.8]
        for eps in epsilons:
            results = self.detector.detect_changes(
                data=self.data, threshold=20, epsilon=eps
            )
            self.assertTrue(
                isinstance(results["change_points"], list), f"Failed with epsilon={eps}"
            )

        # Test different thresholds
        thresholds = [10, 20, 50]
        for thresh in thresholds:
            results = self.detector.detect_changes(
                data=self.data, threshold=thresh, epsilon=0.8
            )
            self.assertTrue(
                isinstance(results["change_points"], list),
                f"Failed with threshold={thresh}",
            )


if __name__ == "__main__":
    unittest.main()
