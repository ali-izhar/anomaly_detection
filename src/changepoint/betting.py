# src/changepoint/betting.py

"""Betting algorithms for martingale-based change point detection.

This module implements several betting functions (betting functions) that transform
a sequence of p-values into an exchangeability martingale. Each function takes the
previous martingale value and the current p-value (and optionally other parameters)
and returns the updated martingale value. The betting function g(p) must satisfy:

    ∫_0^1 g(p) dp = 1,
so that under the null hypothesis (p-values are uniformly distributed) the process
remains a martingale.
"""

from typing import List

import numpy as np
from scipy.stats import beta, gaussian_kde


def power_martingale(prev_m: float, pvalue: float, epsilon: float) -> float:
    """
    Update function for the power martingale.

    Implements the update rule:
        M_n = M_(n-1) * epsilon * (pvalue)^(epsilon - 1)

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.
    epsilon : float
        Sensitivity parameter in (0, 1); lower values lead to larger bets on small p-values.

    Returns
    -------
    float
        The updated martingale value.
    """
    return prev_m * epsilon * (pvalue ** (epsilon - 1))


def exponential_martingale(prev_m: float, pvalue: float, lambd: float = 1.0) -> float:
    """
    Update function for an exponential martingale.

    Uses the update rule:
        M_n = M_(n-1) * exp(-lambd * pvalue)

    This betting rule emphasizes rapid growth when the p-value is small.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.
    lambd : float, optional
        Rate parameter controlling sensitivity (default is 1.0).

    Returns
    -------
    float
        The updated martingale value.
    """
    return prev_m * np.exp(-lambd * pvalue)


def mixture_martingale(prev_m: float, pvalue: float, epsilons: list) -> float:
    """
    Update function for a mixture martingale.

    For a list of sensitivity parameters (epsilons), this function computes a power martingale
    update for each epsilon and then averages the resulting betting factors:
        update_factor = average( [epsilon * (pvalue)^(epsilon - 1) for epsilon in epsilons] )
        M_n = M_(n-1) * update_factor

    This hedges against the choice of a single epsilon value.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.
    epsilons : list
        A list of sensitivity parameters (each in (0,1)).

    Returns
    -------
    float
        The updated martingale value.

    Raises
    ------
    ValueError
        If the provided epsilons list is empty.
    """
    if not epsilons:
        # Use default epsilon values
        epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    updates = [eps * (pvalue ** (eps - 1)) for eps in epsilons]
    avg_update = sum(updates) / len(updates)
    return prev_m * avg_update


def constant_martingale(prev_m: float, pvalue: float) -> float:
    """
    Update function for a constant betting martingale.

    This betting function uses a piecewise constant function defined as:
        g(p) = 1.5, if p ∈ [0, 0.5)
             = 0.5, if p ∈ [0.5, 1]
    Note that the integral over [0, 1] is:
        0.5*1.5 + 0.5*0.5 = 0.75 + 0.25 = 1.
    Thus, under the null hypothesis of uniform p-values the martingale remains valid.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.

    Returns
    -------
    float
        The updated martingale value.
    """
    if pvalue < 0.5:
        factor = 1.5
    else:
        factor = 0.5
    return prev_m * factor


def beta_martingale(
    prev_m: float, pvalue: float, a: float = 0.5, b: float = 1.5
) -> float:
    """
    Update function for a beta martingale betting function.

    This function uses the Beta probability density function (PDF) as the betting function.
    The Beta PDF on [0, 1] is defined as:
        g(p) = Beta(p; a, b)
    which is normalized (its integral is 1). Choosing parameters such that a < 1 and b > 1
    emphasizes small p-values, which is desirable when detecting changes.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.
    a : float, optional
        The alpha parameter of the Beta distribution (default 0.5).
    b : float, optional
        The beta parameter of the Beta distribution (default 1.5).

    Returns
    -------
    float
        The updated martingale value.
    """
    # The beta.pdf function returns the density value at pvalue.
    betting_factor = beta.pdf(pvalue, a, b)
    return prev_m * betting_factor


def kernel_density_martingale(
    prev_m: float, pvalue: float, past_pvalues: List[float], bandwidth: float = 0.1
) -> float:
    """
    Update function for a kernel density betting martingale.

    This function estimates a density for p-values using a Gaussian kernel density estimator (KDE)
    on a list of previous p-values (typically from a sliding window). The estimated density at the
    current p-value is then used as the betting factor. The KDE is normalized so that its integral
    over [0, 1] is (approximately) 1. Note that this naive implementation does not correct for
    boundary effects at 0 and 1.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The current p-value.
    past_pvalues : List[float]
        A list (or window) of past p-values used for estimating the density.
    bandwidth : float, optional
        The bandwidth for the Gaussian kernel (default is 0.1).

    Returns
    -------
    float
        The updated martingale value.

    Notes
    -----
    If the past_pvalues list is empty, this function returns the previous martingale value
    unchanged (equivalent to betting with a uniform density of 1).
    """
    # If no past p-values are available, we default to a uniform betting factor of 1.
    if not past_pvalues:
        return prev_m

    # Convert the list to a numpy array for compatibility with gaussian_kde.
    past_array = np.array(past_pvalues)
    # Create the KDE estimator using the specified bandwidth.
    # Note: gaussian_kde uses 'scott' or 'silverman' rules by default; here we specify a fixed bandwidth.
    kde = gaussian_kde(past_array, bw_method=bandwidth)
    # Evaluate the estimated density at the current pvalue.
    density = kde.evaluate(pvalue)[0]
    # Update the martingale value by multiplying with the density.
    # Under the null hypothesis (uniform p-values), the density estimate should be close to 1.
    return prev_m * density
