# src/changepoint/bitting.py

"""Bitting algorithms for martingale-based change point detection."""

import numpy as np


def power_martingale(prev_m: float, pvalue: float, epsilon: float) -> float:
    """Update function for the power martingale.

    This implements the bitting function:
        M_n = M_(n-1) * epsilon * (p_n)^(epsilon - 1)
    which is widely used in conformal martingale-based change detection.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The conformal p-value for the current (or predicted) observation.
    epsilon : float
        The sensitivity parameter in (0,1).

    Returns
    -------
    float
        The updated martingale value.
    """
    return prev_m * epsilon * (pvalue ** (epsilon - 1))


def exponential_martingale(prev_m: float, pvalue: float, lambd: float = 1.0) -> float:
    """Update function for an exponential martingale.

    This update rule is:
        M_n = M_(n-1) * exp(-lambd * pvalue)
    It emphasizes rapid growth when the p-value is small. Note that under the null hypothesis,
    this update may require calibration (or normalization) to preserve an expected value of 1.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The conformal p-value for the current (or predicted) observation.
    lambd : float, optional
        A rate parameter controlling the sensitivity (default is 1.0).

    Returns
    -------
    float
        The updated martingale value.
    """
    return prev_m * np.exp(-lambd * pvalue)


def mixture_martingale(prev_m: float, pvalue: float, epsilons: list) -> float:
    """Update function for a mixture martingale.

    This update computes multiple power martingale updates for a set of epsilon values and returns
    their average. Formally, for a list of epsilons, it computes:
        update = average( [epsilon * (pvalue)^(epsilon - 1) for epsilon in epsilons] )
        M_n = M_(n-1) * update
    This can be useful to hedge against the choice of a single epsilon.

    Parameters
    ----------
    prev_m : float
        The previous martingale value.
    pvalue : float
        The conformal p-value for the current (or predicted) observation.
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
        raise ValueError("epsilons list must not be empty")
    # Compute the update factor for each epsilon value
    updates = [eps * (pvalue ** (eps - 1)) for eps in epsilons]
    avg_update = sum(updates) / len(updates)
    return prev_m * avg_update
