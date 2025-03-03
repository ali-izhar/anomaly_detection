# tests/test_changepoint/test_visualizer.py

"""Tests for visualization utilities in the changepoint detection module."""

import os
import shutil
import sys
import pytest
import numpy as np
from pathlib import Path

# Set matplotlib backend to 'Agg' for testing (must be done before importing matplotlib)
import matplotlib

matplotlib.use("Agg")

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.plot.plot_changepoint import MartingaleVisualizer


# Test data fixtures
@pytest.fixture
def sample_martingales():
    """Create sample martingale data for testing."""
    return {
        "traditional_sum_martingales": np.random.rand(200),
        "traditional_avg_martingales": np.random.rand(200),
        "horizon_sum_martingales": np.random.rand(50),
        "horizon_avg_martingales": np.random.rand(50),
        "individual_traditional_martingales": [np.random.rand(200) for _ in range(8)],
        "individual_horizon_martingales": [np.random.rand(50) for _ in range(8)],
    }


@pytest.fixture
def sample_change_points():
    """Create sample change points for testing."""
    return [50, 100, 150]


@pytest.fixture
def sample_betting_config():
    """Create sample betting configuration for testing."""
    return {"function": "power", "params": {"power": {"epsilon": 0.7}}}


@pytest.fixture
def test_output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def visualizer(
    sample_martingales, sample_change_points, sample_betting_config, test_output_dir
):
    """Create a MartingaleVisualizer instance for testing."""
    return MartingaleVisualizer(
        martingales=sample_martingales,
        change_points=sample_change_points,
        threshold=2.0,
        betting_config=sample_betting_config,
        output_dir=test_output_dir,
    )


class TestMartingaleVisualizer:
    """Test suite for MartingaleVisualizer class."""

    def test_initialization(self, visualizer, sample_change_points, test_output_dir):
        """Test proper initialization of MartingaleVisualizer."""
        assert visualizer.change_points == sample_change_points
        assert visualizer.threshold == 2.0
        assert visualizer.output_dir == test_output_dir
        assert visualizer.prefix == ""
        assert visualizer.skip_shap is False
        assert visualizer.method == "multiview"

    def test_betting_params_power(
        self, sample_martingales, sample_change_points, test_output_dir
    ):
        """Test betting parameters processing for power function."""
        betting_config = {"function": "power", "params": {"power": {"epsilon": 0.7}}}
        visualizer = MartingaleVisualizer(
            martingales=sample_martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config=betting_config,
            output_dir=test_output_dir,
        )
        assert visualizer.betting_params["function"] == "power"
        assert visualizer.betting_params["param_str"] == "ε=0.7"

    def test_betting_params_exponential(
        self, sample_martingales, sample_change_points, test_output_dir
    ):
        """Test betting parameters processing for exponential function."""
        betting_config = {
            "function": "exponential",
            "params": {"exponential": {"lambda": 1.0}},
        }
        visualizer = MartingaleVisualizer(
            martingales=sample_martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config=betting_config,
            output_dir=test_output_dir,
        )
        assert visualizer.betting_params["function"] == "exponential"
        assert visualizer.betting_params["param_str"] == "λ=1.0"

    def test_process_martingales_multiview(self, visualizer):
        """Test processing of multiview martingales."""
        processed = visualizer.martingales
        assert "combined" in processed
        assert "martingales_sum" in processed["combined"]
        assert "martingales_avg" in processed["combined"]
        assert "prediction_martingale_sum" in processed["combined"]
        assert "prediction_martingale_avg" in processed["combined"]

    def test_create_visualization(self, visualizer):
        """Test creation of all visualizations."""
        visualizer.create_visualization()

        # Check if files were created
        expected_files = [
            "combined_martingales.png",
            "detection_analysis.png",
            "feature_martingales.png",
            "overlaid_martingales.png",
            "shap_values.png",
        ]

        for file in expected_files:
            assert os.path.exists(os.path.join(visualizer.output_dir, file))

    def test_single_view_mode(self, sample_change_points, test_output_dir):
        """Test visualizer in single-view mode."""
        martingales = {
            "traditional_martingales": np.random.rand(200),
            "horizon_martingales": np.random.rand(50),
        }

        visualizer = MartingaleVisualizer(
            martingales=martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config={"function": "constant", "params": {}},
            output_dir=test_output_dir,
            method="single_view",
        )

        processed = visualizer.martingales
        assert "combined" in processed
        assert "martingales" in processed["combined"]
        assert "prediction_martingales" in processed["combined"]

    def test_custom_prefix(
        self, sample_martingales, sample_change_points, test_output_dir
    ):
        """Test visualization with custom prefix."""
        prefix = "test_prefix_"
        visualizer = MartingaleVisualizer(
            martingales=sample_martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config={"function": "constant", "params": {}},
            output_dir=test_output_dir,
            prefix=prefix,
        )

        visualizer.create_visualization()
        assert os.path.exists(
            os.path.join(test_output_dir, f"{prefix}combined_martingales.png")
        )

    def test_skip_shap(self, sample_martingales, sample_change_points, test_output_dir):
        """Test visualization with SHAP values skipped."""
        visualizer = MartingaleVisualizer(
            martingales=sample_martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config={"function": "constant", "params": {}},
            output_dir=test_output_dir,
            skip_shap=True,
        )

        assert visualizer.shap_values is None
        assert visualizer.feature_names is None
        assert visualizer.prediction_shap_values is None

    @pytest.mark.parametrize(
        "betting_function,params,expected_str",
        [
            ("power", {"power": {"epsilon": 0.7}}, "ε=0.7"),
            ("exponential", {"exponential": {"lambda": 1.0}}, "λ=1.0"),
            ("mixture", {"mixture": {"epsilons": [0.5, 0.9]}}, "ε=0.5-0.9"),
            ("beta", {"beta": {"alpha": 0.5, "beta": 1.5}}, "α=0.5, β=1.5"),
            ("kernel", {"kernel": {"bandwidth": 0.1}}, "bw=0.1"),
            ("constant", {}, "fixed"),
        ],
    )
    def test_betting_params_all_functions(
        self,
        sample_martingales,
        sample_change_points,
        test_output_dir,
        betting_function,
        params,
        expected_str,
    ):
        """Test betting parameters processing for all betting functions."""
        betting_config = {"function": betting_function, "params": params}
        visualizer = MartingaleVisualizer(
            martingales=sample_martingales,
            change_points=sample_change_points,
            threshold=2.0,
            betting_config=betting_config,
            output_dir=test_output_dir,
        )
        assert visualizer.betting_params["function"] == betting_function
        assert visualizer.betting_params["param_str"] == expected_str

    def test_cleanup(self, test_output_dir):
        """Clean up test output directory after tests."""
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
