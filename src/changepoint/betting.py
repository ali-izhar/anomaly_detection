# src/changepoint/betting.py

"""Betting algorithms for martingale-based change point detection.

This module implements betting functions that transform a sequence of p-values into
an exchangeability martingale. Each betting function takes the previous martingale value
and the current p-value (and optionally other parameters) and returns the updated
martingale value.

Mathematical Framework:
---------------------
A betting function g(p) must satisfy the fundamental martingale property:
    ∫_0^1 g(p) dp = 1

This ensures that under the null hypothesis (when p-values are uniformly distributed),
the process remains a martingale:
    E[M_n | M_{n-1}] = M_{n-1}

References:
----------
[1] Vovk, V., & Wang, R. (2020). "Combining p-values via averaging."
    Biometrika, 107(4), 791-808.
[2] Volkhonskiy et al. (2017). "Inductive Conformal Martingales for Change-Point Detection."
    Proceedings of the Sixth Workshop on Conformal and Probabilistic Prediction, pp. 132-153.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    final,
    get_args,
)

import numpy as np
from scipy.stats import beta, gaussian_kde

# Define allowed betting function names as literal types
BettingFunctionName = Literal[
    "power", "exponential", "mixture", "constant", "beta", "kernel"
]
# Get a tuple of valid betting function names from the Literal type
VALID_BETTING_FUNCTIONS: tuple[str, ...] = get_args(BettingFunctionName)


class BettingFunctionConfig(TypedDict, total=False):
    """Configuration for betting functions.

    Attributes:
        name: The name of the betting function to use.
        params: Parameters specific to the chosen betting function.

    Example:
        >>> config = {
        ...     "name": "power",
        ...     "params": {"epsilon": 0.7}
        ... }
    """

    name: BettingFunctionName
    params: Dict[str, Union[float, List[float], List[Any]]]


@dataclass(frozen=True)
class BettingFunctionParams:
    """Base class for betting function parameters."""

    pass  # This serves as a base for specific parameter classes


@dataclass(frozen=True)
class PowerParams(BettingFunctionParams):
    """Parameters for power betting function.

    Attributes:
        epsilon: Sensitivity parameter in (0, 1). Lower values lead to larger
                 bets on small p-values.
    """

    epsilon: float

    def __post_init__(self):
        # Ensure that epsilon is within the open interval (0, 1)
        if not 0 < self.epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")


@dataclass(frozen=True)
class ExponentialParams(BettingFunctionParams):
    """Parameters for exponential betting function.

    Attributes:
        lambd: Rate parameter controlling sensitivity. Must be positive.
    """

    lambd: float = 1.0

    def __post_init__(self):
        # Check that the lambda parameter is positive
        if self.lambd <= 0:
            raise ValueError(f"lambd must be positive, got {self.lambd}")


@dataclass(frozen=True)
class MixtureParams(BettingFunctionParams):
    """Parameters for mixture betting function.

    Attributes:
        epsilons: List of sensitivity parameters, each in (0, 1).
    """

    epsilons: List[float] = None

    def __post_init__(self):
        # If epsilons is not provided, initialize with a default list
        if self.epsilons is None:
            object.__setattr__(self, "epsilons", [0.5, 0.6, 0.7, 0.8, 0.9])
        # Ensure that each epsilon value is in (0, 1)
        if not all(0 < eps < 1 for eps in self.epsilons):
            raise ValueError("All epsilons must be in (0, 1)")


@dataclass(frozen=True)
class BetaParams(BettingFunctionParams):
    """Parameters for beta betting function.

    Attributes:
        a: Alpha parameter of Beta distribution. Must be positive.
        b: Beta parameter of Beta distribution. Must be positive.
    """

    a: float = 0.5
    b: float = 1.5

    def __post_init__(self):
        # Validate that both a and b are positive
        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"a and b must be positive, got a={self.a}, b={self.b}")


@dataclass(frozen=True)
class KernelParams(BettingFunctionParams):
    """Parameters for kernel density betting function.

    Attributes:
        bandwidth: Bandwidth for Gaussian kernel. Must be positive.
        past_pvalues: List of past p-values for density estimation.
    """

    bandwidth: float = 0.1
    past_pvalues: List[float] = None

    def __post_init__(self):
        # Validate that bandwidth is a positive number
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")
        # Initialize past_pvalues as an empty list if not provided
        if self.past_pvalues is None:
            object.__setattr__(self, "past_pvalues", [])


class BettingFunction(ABC):
    """Abstract base class for betting functions.

    All betting functions must satisfy the fundamental martingale property:
        ∫_0^1 g(p) dp = 1

    This ensures that under the null hypothesis (when p-values are uniformly
    distributed), the process remains a martingale:
        E[M_n | M_{n-1}] = M_{n-1}
    """

    name: ClassVar[BettingFunctionName]
    params_class: ClassVar[type[BettingFunctionParams]]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # Initialize the betting function with specific parameters; use defaults if none provided
        self.params = self.params_class(**(params or {}))

    @abstractmethod
    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        pass

    @final
    def validate_inputs(self, prev_m: float, pvalue: float) -> None:
        """Validate input values.

        Args:
            prev_m: Previous martingale value.
            pvalue: Current p-value.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Ensure previous martingale value is not negative
        if prev_m < 0:
            raise ValueError(
                f"Previous martingale value must be non-negative, got {prev_m}"
            )
        # Ensure that pvalue is within the valid range [0, 1]
        if not 0 <= pvalue <= 1:
            raise ValueError(f"P-value must be in [0, 1], got {pvalue}")


class PowerBetting(BettingFunction):
    """Power betting function implementation.

    The power betting function uses the update rule:
        M_n = M_{n-1} * ε * p^(ε-1)

    where ε ∈ (0,1) is the sensitivity parameter. Lower values of ε lead to
    larger bets on small p-values, making the martingale more sensitive to
    potential changes.

    Mathematical Properties:
    ----------------------
    1. The betting function g(p) = ε * p^(ε-1) integrates to 1 over [0,1].
    2. For ε < 1, the function places larger bets on small p-values.
    3. As ε → 0, the betting becomes more aggressive on small p-values.
    4. As ε → 1, the betting approaches uniform (no betting).
    """

    name: ClassVar[BettingFunctionName] = "power"
    params_class: ClassVar[type[BettingFunctionParams]] = PowerParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using power betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate the inputs to ensure they are within expected bounds
        self.validate_inputs(prev_m, pvalue)

        # Handle numerical edge case: p-value is 0, which causes divergence
        if pvalue == 0:
            return float("inf")  # Betting function is undefined at p=0
        # Handle the limit case: p-value is 1, leading the betting factor to 0
        if pvalue == 1:
            return 0.0  # Limit as p→1

        # Calculate the updated martingale value using the power betting formula
        return prev_m * self.params.epsilon * (pvalue ** (self.params.epsilon - 1))


class ExponentialBetting(BettingFunction):
    """Exponential betting function implementation.

    The exponential betting function uses the update rule:
        M_n = M_{n-1} * exp(-λp) / ((1-exp(-λ))/λ)

    where λ > 0 is the rate parameter and the denominator normalizes
    the betting function to integrate to 1 over [0,1].

    Mathematical Properties:
    ----------------------
    1. The betting function g(p) = exp(-λp)/((1-exp(-λ))/λ) integrates to 1.
    2. Larger λ values lead to more aggressive betting on small p-values.
    3. As λ → 0, the betting approaches uniform (no betting).
    4. As λ → ∞, the betting concentrates all mass near p=0.
    """

    name: ClassVar[BettingFunctionName] = "exponential"
    params_class: ClassVar[type[BettingFunctionParams]] = ExponentialParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using exponential betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate the input values first
        self.validate_inputs(prev_m, pvalue)

        # Handle numerical edge cases when lambda is very large (for stability)
        if self.params.lambd > 100:  # Arbitrary threshold for numerical stability
            if pvalue == 0:
                return float("inf")
            if pvalue == 1:
                return 0.0

        # Compute normalization factor: the integral of exp(-λp) over [0,1]
        normalization = (1 - np.exp(-self.params.lambd)) / self.params.lambd

        # Update the martingale value using the exponential factor and normalization
        return prev_m * np.exp(-self.params.lambd * pvalue) / normalization


class MixtureBetting(BettingFunction):
    """Mixture betting function implementation.

    The mixture betting function averages the betting factors from multiple
    power martingales with different sensitivity parameters:
        M_n = M_{n-1} * (1/K) * Σ[ε_k * p^(ε_k-1)]

    where {ε_k} are K different sensitivity parameters in (0,1).

    Mathematical Properties:
    ----------------------
    1. The average of valid betting functions is also a valid betting function.
    2. Mixing different sensitivities provides robustness against different
       types of changes.
    3. The mixture is more stable than individual power martingales.
    """

    name: ClassVar[BettingFunctionName] = "mixture"
    params_class: ClassVar[type[BettingFunctionParams]] = MixtureParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using mixture betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate inputs for correctness
        self.validate_inputs(prev_m, pvalue)

        # Handle edge cases for extreme p-values
        if pvalue == 0:
            return float("inf")
        if pvalue == 1:
            return 0.0

        # Calculate the update factor for each epsilon parameter using the power betting rule
        updates = [eps * (pvalue ** (eps - 1)) for eps in self.params.epsilons]
        # Compute the average update factor from the list of updates
        avg_update = sum(updates) / len(updates)

        # Return the updated martingale value by applying the averaged update factor
        return prev_m * avg_update


class ConstantBetting(BettingFunction):
    """Constant betting function implementation.

    Uses a piecewise constant betting function defined as:
        g(p) = 1.5, if p ∈ [0, 0.5)
             = 0.5, if p ∈ [0.5, 1]

    Mathematical Properties:
    ----------------------
    1. The betting function integrates to 1 over [0,1]:
       ∫_0^1 g(p) dp = 1.5 * 0.5 + 0.5 * 0.5 = 1
    2. Places higher bets on the first half of the interval.
    3. Simplest possible non-uniform betting function.
    4. Discontinuous at p = 0.5.
    """

    name: ClassVar[BettingFunctionName] = "constant"
    # No additional parameters are required; use the base class
    params_class: ClassVar[type[BettingFunctionParams]] = BettingFunctionParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using constant betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate the inputs before computation
        self.validate_inputs(prev_m, pvalue)
        # Select the constant factor based on whether pvalue is less than 0.5 or not
        factor = 1.5 if pvalue < 0.5 else 0.5
        # Return the new martingale value after applying the constant betting factor
        return prev_m * factor


class BetaBetting(BettingFunction):
    """Beta betting function implementation.

    Uses the Beta probability density function (PDF) as the betting function:
        g(p) = Beta(p; α, β)

    where α, β > 0 are the shape parameters of the Beta distribution.

    Mathematical Properties:
    ----------------------
    1. The Beta PDF naturally integrates to 1 over [0,1].
    2. For α < 1 and β > 1, places higher bets on small p-values.
    3. For α > 1 and β < 1, places higher bets on large p-values.
    4. For α = β = 1, reduces to uniform betting (no betting).
    5. Smooth and continuous over the entire interval.
    """

    name: ClassVar[BettingFunctionName] = "beta"
    params_class: ClassVar[type[BettingFunctionParams]] = BetaParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using beta betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate input values for correctness
        self.validate_inputs(prev_m, pvalue)

        # Handle edge cases: when p-value is 0 or 1, under certain parameter settings the beta PDF can be infinite
        if pvalue == 0 and self.params.a < 1:
            return float("inf")
        if pvalue == 1 and self.params.b < 1:
            return float("inf")

        # Calculate the betting factor using the Beta probability density function
        betting_factor = beta.pdf(pvalue, self.params.a, self.params.b)
        # Return the updated martingale value after applying the beta-based betting factor
        return prev_m * betting_factor


class KernelBetting(BettingFunction):
    """Kernel density betting function implementation.

    Estimates a density for p-values using a Gaussian kernel density estimator (KDE)
    on a list of previous p-values:
        g(p) = KDE(p) / ∫_0^1 KDE(x) dx

    where KDE uses reflection at boundaries to handle edge effects.

    Mathematical Properties:
    ----------------------
    1. Adapts to the empirical distribution of p-values.
    2. Automatically normalized to integrate to 1 over [0,1].
    3. Smooth and continuous over the entire interval.
    4. Bandwidth parameter controls smoothness of the betting function.

    Notes:
    ------
    - Uses reflection at boundaries to handle edge effects:
      * Points < 0 are reflected about 0
      * Points > 1 are reflected about 1
    - This ensures proper density estimation near boundaries.
    """

    name: ClassVar[BettingFunctionName] = "kernel"
    params_class: ClassVar[type[BettingFunctionParams]] = KernelParams

    def __call__(self, prev_m: float, pvalue: float) -> float:
        """Update the martingale value using kernel density betting.

        Args:
            prev_m: Previous martingale value M_{n-1}.
            pvalue: Current p-value p_n.

        Returns:
            Updated martingale value M_n.
        """
        # Validate the input values
        self.validate_inputs(prev_m, pvalue)

        # If there are no past p-values provided, we cannot update the density estimate;
        # thus, return the current martingale value unchanged.
        if not self.params.past_pvalues:
            return prev_m

        # Convert the list of past p-values into a NumPy array for efficient computations
        past_array = np.array(self.params.past_pvalues)

        # Handle edge effects: reflect past values at the boundaries (0 and 1)
        reflected = np.concatenate(
            [
                -past_array,  # Reflect values less than 0 about 0
                past_array,  # Original p-values
                2 - past_array,  # Reflect values greater than 1 about 1
            ]
        )

        # Create a Gaussian Kernel Density Estimator with the specified bandwidth
        kde = gaussian_kde(reflected, bw_method=self.params.bandwidth)

        # Prepare a grid of values in [0, 1] for numerical integration
        x = np.linspace(0, 1, 1000)
        # Evaluate the KDE on the grid
        density_vals = kde.evaluate(x)
        # Compute the normalization constant using the trapezoidal rule to ensure the density integrates to 1
        normalization = np.trapz(density_vals, x)

        # Evaluate the KDE at the current p-value and normalize it
        density = kde.evaluate(pvalue)[0] / normalization

        # Return the updated martingale value using the estimated density
        return prev_m * density


# Mapping of betting function names to their respective class implementations
BETTING_FUNCTIONS: Dict[BettingFunctionName, type[BettingFunction]] = {
    "power": PowerBetting,
    "exponential": ExponentialBetting,
    "mixture": MixtureBetting,
    "constant": ConstantBetting,
    "beta": BetaBetting,
    "kernel": KernelBetting,
}


def get_betting_function(
    config: BettingFunctionConfig,
) -> Callable[[float, float], float]:
    """Get a configured betting function instance.

    Args:
        config: Betting function configuration with name and parameters.

    Returns:
        A callable that implements the betting function with the specified configuration.

    Raises:
        ValueError: If the betting function name is unknown or required parameters are missing.

    Example:
        >>> config = {"name": "power", "params": {"epsilon": 0.7}}
        >>> betting_func = get_betting_function(config)
        >>> new_value = betting_func(1.0, 0.05)  # Update martingale
    """
    # Check that the provided betting function name is valid
    if config["name"] not in BETTING_FUNCTIONS:
        raise ValueError(
            f"Unknown betting function '{config['name']}'. "
            f"Available options are: {list(BETTING_FUNCTIONS.keys())}"
        )

    # Retrieve the corresponding betting function class
    betting_class = BETTING_FUNCTIONS[config["name"]]
    # Instantiate the betting function using the provided parameters (or defaults if none provided)
    betting_instance = betting_class(config.get("params", {}))

    # Return the callable __call__ method of the betting function instance for updating martingale values
    return betting_instance.__call__
