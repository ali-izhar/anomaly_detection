"""Betting-function calibration tests (Ali & Ho, ICDM 2025, Def 5).

WHAT is tested
--------------
Every valid betting function g: [0,1] -> [0, infty) must satisfy

    integral_0^1 g(p) dp = 1.                        (*)

This is the load-bearing condition that makes M_t = prod g(p_s) a
test martingale under H_0 (p_t | F_{t-1} ~ Unif(0,1)):

    E[M_t | F_{t-1}] = M_{t-1} * E[g(p_t)]
                     = M_{t-1} * integral_0^1 g(p) dp
                     = M_{t-1}.                      (martingale property)

WHY this matters
----------------
If (*) fails, Theorem 1 of the paper is false for that g and Ville's
inequality cannot be invoked to bound the false-alarm rate.

Two verification paths are used here:
  * ``hmd.betting.verify_calibration`` for power + mixture families. This
    helper uses a u = p^0.25 substitution which *only* tames the p -> 0
    singularity, so it is accurate for power / mixture but DOES NOT handle
    the p -> 1 singularity that appears for Beta(alpha, beta) with beta < 1.
    It also suffers a finite eps_clip bias bounded analytically by
    eps * eps_clip^eps (see betting.py docstring). At eps=0.2 that bias
    alone is ~0.002; on top of it the substitution + trapezoid adds
    additional quadrature error -> tolerance must be widened.
  * scipy.integrate.quad for Beta(alpha, beta): adaptive, handles both
    endpoint singularities.

Tests here:
  1. Power / mixture integrate to 1 within quadrature tolerance.
  2. Beta (via scipy.quad) integrates to 1 within 1e-6.
  3. Mixture weights must sum to 1 or else the constructor raises ValueError.
  4. Power(eps) at p=1 equals eps exactly (g(1) = eps * 1^(eps-1) = eps).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from hmd import betting as B


# ---------------------------------------------------------------------------
# (1) Calibration: integral_0^1 g(p) dp == 1 via verify_calibration helper.
# ---------------------------------------------------------------------------


def _cal_bound(eps_list: list[float], eps_clip: float = 1e-10) -> float:
    """Analytical + numerical tolerance for verify_calibration on power/mixture.

    The eps_clip floor excludes tail mass eps * eps_clip^eps. The substitution
    u = p^0.25 + trapezoid with 10_001 knots adds roughly 4x that magnitude
    in empirical quadrature error. We therefore allow 8 * eps * eps_clip^eps
    as the tolerance from each component, plus a 1e-4 baseline.
    """
    analytic = max(eps * eps_clip**eps for eps in eps_list)
    return 8.0 * analytic + 1e-4


@pytest.mark.parametrize("eps", [0.2, 0.5, 0.7, 0.9])
def test_power_betting_calibrated(eps):
    g = B.power(eps)
    I = B.verify_calibration(g)
    tol = _cal_bound([eps])
    assert abs(I - 1.0) <= tol, (
        f"power(eps={eps}) integrates to {I:.6f}; tol = {tol:.6f}"
    )


def test_default_mixture_calibrated():
    g = B.default()
    I = B.verify_calibration(g)
    # default() uses eps in {0.3, 0.5, 0.7}. Smallest eps dominates bias.
    tol = _cal_bound([0.3, 0.5, 0.7])
    assert abs(I - 1.0) <= tol, f"default mixture integrates to {I:.6f}; tol = {tol:.6f}"


@pytest.mark.parametrize(
    "weights,epsilons",
    [
        ([0.5, 0.5], [0.3, 0.7]),
        ([0.25, 0.25, 0.5], [0.2, 0.5, 0.8]),
        ([1.0], [0.6]),  # degenerate single-component mixture
    ],
)
def test_custom_mixture_calibrated(weights, epsilons):
    g = B.mixture(weights, epsilons)
    I = B.verify_calibration(g)
    tol = _cal_bound(list(epsilons))
    assert abs(I - 1.0) <= tol, (
        f"mixture({weights},{epsilons}) integrates to {I:.6f}; tol = {tol:.6f}"
    )


# ---------------------------------------------------------------------------
# (2) Calibration for Beta betting: verify_calibration does NOT handle the
#     p -> 1 singularity, so we use scipy.integrate.quad directly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,beta_",
    [
        (0.5, 0.5),  # U-shape -- singular at both endpoints
        (2.0, 5.0),
        (1.0, 1.0),  # uniform -- g(p) = 1
        (0.7, 3.0),
    ],
)
def test_beta_betting_calibrated(alpha, beta_):
    g = B.beta(alpha, beta_)
    val, err = quad(lambda p: float(g(np.asarray([p]))[0]), 0.0, 1.0, limit=200)
    assert abs(val - 1.0) <= 1e-6 + err, (
        f"beta({alpha},{beta_}) integrates to {val:.6f} +- {err:.2e}"
    )


# ---------------------------------------------------------------------------
# (2) Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.2, 0.5, 0.7, 0.9])
def test_power_at_one_equals_eps(eps):
    """g_eps(1) = eps * 1^(eps-1) = eps.  Elementary analytic identity."""
    g = B.power(eps)
    val = float(g(np.asarray([1.0]))[0])
    assert abs(val - eps) < 1e-12, f"power({eps})(1) = {val}, expected {eps}"


def test_mixture_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="weights must sum to 1"):
        B.mixture([0.3, 0.3], [0.3, 0.7])


def test_mixture_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        B.mixture([1.5, -0.5], [0.3, 0.7])


def test_mixture_rejects_out_of_range_eps():
    # eps = 1.0 is forbidden (endpoint excluded)
    with pytest.raises(ValueError, match="strictly in"):
        B.mixture([0.5, 0.5], [0.3, 1.0])
    with pytest.raises(ValueError, match="strictly in"):
        B.mixture([0.5, 0.5], [0.0, 0.5])


def test_power_rejects_boundary_eps():
    with pytest.raises(ValueError):
        B.power(0.0)
    with pytest.raises(ValueError):
        B.power(1.0)


def test_beta_rejects_non_positive():
    with pytest.raises(ValueError):
        B.beta(0.0, 0.5)
    with pytest.raises(ValueError):
        B.beta(0.5, -1.0)


def test_mixture_weights_shape_mismatch():
    with pytest.raises(ValueError, match="same length"):
        B.mixture([0.5, 0.5], [0.3])
