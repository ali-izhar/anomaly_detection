"""Betting functions for martingale-based change point detection."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List
import numpy as np
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class BettingConfig:
    """Configuration for betting functions."""
    name: str = "power"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in BETTING_FUNCTIONS:
            raise ValueError(f"Unknown betting function: {self.name}")


def power_betting(epsilon: float = 0.7) -> Callable[[float, float], float]:
    """Power betting: g(p) = epsilon * p^(epsilon-1)."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0, 1)")

    def bet(prev_m: float, p: float) -> float:
        if p <= 0:
            return float("inf")
        if p >= 1:
            return 0.0
        return prev_m * epsilon * (p ** (epsilon - 1))
    return bet


def mixture_betting(epsilons: Optional[List[float]] = None) -> Callable[[float, float], float]:
    """Mixture of power betting functions."""
    epsilons = epsilons or [0.5, 0.6, 0.7, 0.8, 0.9]
    if not all(0 < e < 1 for e in epsilons):
        raise ValueError("All epsilons must be in (0, 1)")

    def bet(prev_m: float, p: float) -> float:
        if p <= 0:
            return float("inf")
        if p >= 1:
            return 0.0
        factors = [e * (p ** (e - 1)) for e in epsilons]
        return prev_m * np.mean(factors)
    return bet


def beta_betting(a: float = 0.5, b: float = 1.5) -> Callable[[float, float], float]:
    """Beta distribution betting: g(p) = Beta(p; a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")

    def bet(prev_m: float, p: float) -> float:
        if p <= 0 and a < 1:
            return float("inf")
        if p >= 1 and b < 1:
            return float("inf")
        return prev_m * beta_dist.pdf(p, a, b)
    return bet


def exponential_betting(lambd: float = 1.0) -> Callable[[float, float], float]:
    """Exponential betting: g(p) = exp(-lambda*p) / norm."""
    if lambd <= 0:
        raise ValueError("lambda must be positive")

    norm = (1 - np.exp(-lambd)) / lambd

    def bet(prev_m: float, p: float) -> float:
        return prev_m * np.exp(-lambd * p) / norm
    return bet


def constant_betting() -> Callable[[float, float], float]:
    """Constant betting: 1.5 if p < 0.5, else 0.5."""
    def bet(prev_m: float, p: float) -> float:
        return prev_m * (1.5 if p < 0.5 else 0.5)
    return bet


BETTING_FUNCTIONS = {
    "power": power_betting,
    "mixture": mixture_betting,
    "beta": beta_betting,
    "exponential": exponential_betting,
    "constant": constant_betting,
}


def create_betting_function(config: BettingConfig) -> Callable[[float, float], float]:
    """Create a betting function from configuration."""
    factory = BETTING_FUNCTIONS[config.name]
    return factory(**config.params)
