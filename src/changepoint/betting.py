# src/changepoint/betting.py

"""Betting algorithms for martingale-based change point detection.

This module implements betting functions that transform a sequence of p-values into
an exchangeability martingale. Each function takes the previous martingale value
and the current p-value (and optionally other parameters) and returns the updated
martingale value.

The betting function g(p) must satisfy:
    ∫_0^1 g(p) dp = 1,
so that under the null hypothesis (p-values are uniformly distributed) the process
remains a martingale.
"""

from typing import List, Dict, Any, Callable, TypedDict

import numpy as np
from scipy.stats import beta, gaussian_kde


class BettingFunctionConfig(TypedDict, total=False):
    """Configuration for betting functions.

    The 'name' field is required, while 'params' is optional and depends on the specific betting function:
    - power: requires 'epsilon'
    - exponential: optional 'lambd' (default 1.0)
    - mixture: optional 'epsilons' list
    - constant: no additional params
    - beta: optional 'a' and 'b' (defaults 0.5, 1.5)
    - kernel: optional 'bandwidth' (default 0.1) and requires 'past_pvalues' list
    """

    name: str
    params: Dict[str, Any]


def power_martingale(prev_m: float, pvalue: float, epsilon: float) -> float:
    """Update function for the power martingale.

    Implements the update rule:
        M_n = M_(n-1) * epsilon * (pvalue)^(epsilon - 1)

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.
        epsilon: Sensitivity parameter in (0, 1); lower values lead to larger bets on small p-values.

    Returns:
        The updated martingale value.
    """
    return prev_m * epsilon * (pvalue ** (epsilon - 1))


def exponential_martingale(prev_m: float, pvalue: float, lambd: float = 1.0) -> float:
    """Update function for an exponential martingale.

    Uses the update rule:
        M_n = M_(n-1) * exp(-lambd * pvalue) / normalization
    where normalization = (1 - exp(-lambd)) / lambd ensures the betting function
    integrates to 1 over [0,1].

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.
        lambd: Rate parameter controlling sensitivity (default is 1.0).

    Returns:
        The updated martingale value.
    """
    normalization = (1 - np.exp(-lambd)) / lambd  # Integral of exp(-λp) from 0 to 1
    return prev_m * np.exp(-lambd * pvalue) / normalization


def mixture_martingale(
    prev_m: float, pvalue: float, epsilons: List[float] = None
) -> float:
    """Update function for a mixture martingale.

    For a list of sensitivity parameters (epsilons), computes a power martingale
    update for each epsilon and then averages the resulting betting factors:
        update_factor = average([epsilon * (pvalue)^(epsilon - 1) for epsilon in epsilons])
        M_n = M_(n-1) * update_factor

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.
        epsilons: List of sensitivity parameters (each in (0,1)). Defaults to [0.5, 0.6, 0.7, 0.8, 0.9].

    Returns:
        The updated martingale value.
    """
    if not epsilons:
        epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    updates = [eps * (pvalue ** (eps - 1)) for eps in epsilons]
    avg_update = sum(updates) / len(updates)
    return prev_m * avg_update


def constant_martingale(prev_m: float, pvalue: float) -> float:
    """Update function for a constant betting martingale.

    Uses a piecewise constant function defined as:
        g(p) = 1.5, if p ∈ [0, 0.5)
             = 0.5, if p ∈ [0.5, 1]

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.

    Returns:
        The updated martingale value.
    """
    factor = 1.5 if pvalue < 0.5 else 0.5
    return prev_m * factor


def beta_martingale(
    prev_m: float, pvalue: float, a: float = 0.5, b: float = 1.5
) -> float:
    """Update function for a beta martingale betting function.

    Uses the Beta probability density function (PDF) as the betting function:
        g(p) = Beta(p; a, b)

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.
        a: The alpha parameter of the Beta distribution (default 0.5).
        b: The beta parameter of the Beta distribution (default 1.5).

    Returns:
        The updated martingale value.
    """
    betting_factor = beta.pdf(pvalue, a, b)
    return prev_m * betting_factor


def kernel_density_martingale(
    prev_m: float, pvalue: float, past_pvalues: List[float], bandwidth: float = 0.1
) -> float:
    """Update function for a kernel density betting martingale.

    Estimates a density for p-values using a Gaussian kernel density estimator (KDE)
    on a list of previous p-values. The estimated density at the current p-value
    is used as the betting factor.

    Args:
        prev_m: The previous martingale value.
        pvalue: The current p-value.
        past_pvalues: List of past p-values for density estimation.
        bandwidth: The bandwidth for the Gaussian kernel (default 0.1).

    Returns:
        The updated martingale value.

    Note:
        If past_pvalues is empty, returns the previous martingale value unchanged.
    """
    if not past_pvalues:
        return prev_m

    # Convert the list to a numpy array
    past_array = np.array(past_pvalues)

    # Reflect points at boundaries to handle boundary effects
    reflected = np.concatenate(
        [
            -past_array,  # Reflect left of 0
            past_array,  # Original points
            2 - past_array,  # Reflect right of 1
        ]
    )

    # Create KDE with reflected points
    kde = gaussian_kde(reflected, bw_method=bandwidth)

    # Normalize the density to integrate to 1 over [0,1]
    x = np.linspace(0, 1, 1000)
    density_vals = kde.evaluate(x)
    normalization = np.trapz(density_vals, x)  # Numerical integration

    # Evaluate normalized density at pvalue
    density = kde.evaluate(pvalue)[0] / normalization

    return prev_m * density


# Mapping of betting function names to their implementations
BETTING_FUNCTIONS = {
    "power": power_martingale,
    "exponential": exponential_martingale,
    "mixture": mixture_martingale,
    "constant": constant_martingale,
    "beta": beta_martingale,
    "kernel": kernel_density_martingale,
}


def get_betting_function(config: BettingFunctionConfig) -> Callable:
    """Get the betting function and validate its configuration.

    Args:
        config: Betting function configuration with name and parameters.

    Returns:
        A callable that wraps the betting function with its configuration.

    Raises:
        ValueError: If the betting function name is unknown or required parameters are missing.
    """
    if config["name"] not in BETTING_FUNCTIONS:
        raise ValueError(
            f"Unknown betting function '{config['name']}'. "
            f"Available options are: {list(BETTING_FUNCTIONS.keys())}"
        )

    betting_func = BETTING_FUNCTIONS[config["name"]]
    params = config.get("params", {})

    # Validate required parameters
    if config["name"] == "power" and "epsilon" not in params:
        raise ValueError("Power betting function requires 'epsilon' parameter")
    elif config["name"] == "kernel" and "past_pvalues" not in params:
        raise ValueError("Kernel betting function requires 'past_pvalues' parameter")

    def wrapped_betting_func(prev_m: float, pvalue: float) -> float:
        return betting_func(prev_m, pvalue, **params)

    return wrapped_betting_func
