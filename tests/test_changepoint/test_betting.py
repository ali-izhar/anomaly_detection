# tests/test_changepoint/test_betting.py

"""Tests for betting functions in the changepoint detection module.

This module contains comprehensive tests for all betting functions and their
parameter validation, edge cases, and mathematical properties.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.betting import (
    PowerBetting,
    ExponentialBetting,
    MixtureBetting,
    ConstantBetting,
    BetaBetting,
    KernelBetting,
    get_betting_function,
    VALID_BETTING_FUNCTIONS,
    PowerParams,
    ExponentialParams,
    MixtureParams,
    BetaParams,
    KernelParams,
)


# Test parameter validation
def test_power_params_validation():
    """Test validation of PowerBetting parameters."""
    # Valid epsilon values
    PowerParams(epsilon=0.5)
    PowerParams(epsilon=0.1)
    PowerParams(epsilon=0.9)

    # Invalid epsilon values
    with pytest.raises(ValueError):
        PowerParams(epsilon=0)
    with pytest.raises(ValueError):
        PowerParams(epsilon=1)
    with pytest.raises(ValueError):
        PowerParams(epsilon=-0.1)
    with pytest.raises(ValueError):
        PowerParams(epsilon=1.1)


def test_exponential_params_validation():
    """Test validation of ExponentialBetting parameters."""
    # Valid lambda values
    ExponentialParams(lambd=1.0)
    ExponentialParams(lambd=0.5)
    ExponentialParams(lambd=5.0)

    # Invalid lambda values
    with pytest.raises(ValueError):
        ExponentialParams(lambd=0)
    with pytest.raises(ValueError):
        ExponentialParams(lambd=-1.0)


def test_mixture_params_validation():
    """Test validation of MixtureBetting parameters."""
    # Valid epsilon lists
    MixtureParams(epsilons=[0.3, 0.5, 0.7])
    MixtureParams()  # Test default values

    # Invalid epsilon values
    with pytest.raises(ValueError):
        MixtureParams(epsilons=[0, 0.5])
    with pytest.raises(ValueError):
        MixtureParams(epsilons=[0.5, 1.0])
    with pytest.raises(ValueError):
        MixtureParams(epsilons=[-0.1, 0.5])


def test_beta_params_validation():
    """Test validation of BetaBetting parameters."""
    # Valid a, b values
    BetaParams(a=0.5, b=1.5)
    BetaParams()  # Test default values

    # Invalid a, b values
    with pytest.raises(ValueError):
        BetaParams(a=0, b=1.5)
    with pytest.raises(ValueError):
        BetaParams(a=0.5, b=0)
    with pytest.raises(ValueError):
        BetaParams(a=-1, b=1.5)


def test_kernel_params_validation():
    """Test validation of KernelBetting parameters."""
    # Valid bandwidth values
    KernelParams(bandwidth=0.1)
    KernelParams()  # Test default values

    # Invalid bandwidth values
    with pytest.raises(ValueError):
        KernelParams(bandwidth=0)
    with pytest.raises(ValueError):
        KernelParams(bandwidth=-0.1)


# Test input validation for all betting functions
@pytest.mark.parametrize(
    "betting_class",
    [
        PowerBetting,
        ExponentialBetting,
        MixtureBetting,
        ConstantBetting,
        BetaBetting,
        KernelBetting,
    ],
)
def test_betting_function_input_validation(betting_class):
    """Test input validation for all betting functions."""
    # Initialize with default parameters for each betting class
    params = {}
    if betting_class == PowerBetting:
        params = {"epsilon": 0.5}
    betting_func = betting_class(params)

    # Valid inputs should work
    betting_func(1.0, 0.5)

    # Invalid previous martingale values
    with pytest.raises(ValueError):
        betting_func(-1.0, 0.5)

    # Invalid p-values
    with pytest.raises(ValueError):
        betting_func(1.0, -0.1)
    with pytest.raises(ValueError):
        betting_func(1.0, 1.1)


# Test specific betting function properties
def test_power_betting():
    """Test PowerBetting function properties."""
    betting = PowerBetting({"epsilon": 0.5})

    # Test basic functionality
    assert betting(1.0, 0.5) == 1.0 * 0.5 * (0.5**-0.5)

    # Test edge cases
    assert np.isinf(betting(1.0, 0))  # Should return infinity
    assert betting(1.0, 1) == 0.0  # Should return 0


def test_exponential_betting():
    """Test ExponentialBetting function properties."""
    betting = ExponentialBetting({"lambd": 1.0})

    # Test basic functionality
    expected = 1.0 * np.exp(-0.5) / ((1 - np.exp(-1)) / 1.0)
    np.testing.assert_almost_equal(betting(1.0, 0.5), expected)

    # Test edge cases
    assert betting(1.0, 0) > betting(1.0, 0.5)  # Larger for smaller p-values
    assert betting(1.0, 1) < betting(1.0, 0.5)  # Smaller for larger p-values


def test_mixture_betting():
    """Test MixtureBetting function properties."""
    betting = MixtureBetting({"epsilons": [0.3, 0.5, 0.7]})

    # Test basic functionality
    result = betting(1.0, 0.5)
    assert isinstance(result, float)
    assert result > 0

    # Test edge cases
    assert np.isinf(betting(1.0, 0))  # Should return infinity
    assert betting(1.0, 1) == 0.0  # Should return 0


def test_constant_betting():
    """Test ConstantBetting function properties."""
    betting = ConstantBetting()

    # Test step function behavior
    assert betting(1.0, 0.4) == 1.5  # p < 0.5
    assert betting(1.0, 0.6) == 0.5  # p >= 0.5
    assert betting(1.0, 0.5) == 0.5  # Exactly at 0.5


def test_beta_betting():
    """Test BetaBetting function properties."""
    betting = BetaBetting({"a": 0.5, "b": 1.5})

    # Test basic functionality
    result = betting(1.0, 0.5)
    assert isinstance(result, float)
    assert result > 0

    # Test edge cases with a < 1
    assert np.isinf(betting(1.0, 0))  # Should return infinity


def test_kernel_betting():
    """Test KernelBetting function properties."""
    past_pvalues = [0.1, 0.2, 0.3, 0.4, 0.5]
    betting = KernelBetting({"bandwidth": 0.1, "past_pvalues": past_pvalues})

    # Test basic functionality
    result = betting(1.0, 0.3)
    assert isinstance(result, float)
    assert result > 0

    # Test with no past values
    empty_betting = KernelBetting({"bandwidth": 0.1})
    assert empty_betting(1.0, 0.5) == 1.0  # Should return unchanged martingale


# Test factory function
def test_get_betting_function():
    """Test the betting function factory."""
    # Test all valid betting functions
    for name in VALID_BETTING_FUNCTIONS:
        config = {"name": name, "params": {"epsilon": 0.5} if name == "power" else {}}
        betting_func = get_betting_function(config)
        assert callable(betting_func)

        # Test basic functionality
        result = betting_func(1.0, 0.5)
        assert isinstance(result, float)
        assert result >= 0

    # Test invalid betting function name
    with pytest.raises(ValueError):
        get_betting_function({"name": "invalid_name"})


# Test martingale property
@pytest.mark.parametrize(
    "betting_class",
    [
        PowerBetting,
        ExponentialBetting,
        MixtureBetting,
        ConstantBetting,
        BetaBetting,
    ],
)
def test_martingale_property(betting_class):
    """Test that betting functions satisfy the martingale property.

    The martingale property states that the expected value of the betting
    function over uniform p-values should be approximately 1.
    """
    # Initialize with default parameters for each betting class
    params = {}
    if betting_class == PowerBetting:
        params = {"epsilon": 0.5}
    betting = betting_class(params)

    # Generate uniform p-values
    p_values = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 for numerical stability

    # Calculate average betting factor
    betting_factors = [betting(1.0, p) for p in p_values]
    avg_factor = np.mean(betting_factors)

    # Check if average is approximately 1 (allowing for numerical error)
    np.testing.assert_allclose(avg_factor, 1.0, rtol=0.1)


# Test sequential updates
def test_sequential_updates():
    """Test that betting functions work correctly in sequence."""
    betting = PowerBetting({"epsilon": 0.7})

    # Start with martingale value of 1.0
    m = 1.0

    # Apply sequence of p-values
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for p in p_values:
        m = betting(m, p)
        assert isinstance(m, float)
        assert m >= 0  # Martingale should remain non-negative
