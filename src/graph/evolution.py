# src/graph/evolution.py

"""Controls the evolution of graph parameters."""

from typing import Dict, Any
import numpy as np


class EvolutionManager:
    """Manages parameter evolution rules for different graph models."""

    def __init__(self, rng: np.random.RandomState = None):
        """Initialize evolution manager.

        Args:
            rng: Random number generator to use
        """
        self.rng = rng or np.random.RandomState()

    def evolve_parameters(
        self, model: str, params: Dict[str, Any], use_gaussian: bool = False
    ) -> Dict[str, Any]:
        """Evolve parameters based on model-specific rules or Gaussian steps.

        Args:
            model: Model name ('ba', 'ws', 'er', 'sbm')
            params: Current parameters including bounds
            use_gaussian: If True, use Gaussian evolution with _std suffixes,
                        otherwise use uniform evolution with min/max bounds
        Returns:
            New parameter set
        """
        if use_gaussian:
            return self._evolve_gaussian(params)

        if hasattr(self, f"_evolve_{model}"):
            return getattr(self, f"_evolve_{model}")(params)
        return self._evolve_uniform(params)

    def _evolve_gaussian(self, params: Dict) -> Dict:
        """Evolve parameters by Gaussian steps for fields with _std suffix.

        Args:
            params: Current parameters
        Returns:
            Updated parameters
        """
        evolved = params.copy()

        for key, value in params.items():
            std_key = f"{key}_std"
            if std_key in params and params[std_key] is not None:
                std = params[std_key]
                new_val = self.rng.normal(value, std)

                if isinstance(value, int):
                    new_val = int(round(new_val))
                    if key not in ["min_changes", "max_changes"]:
                        new_val = max(1, new_val)
                elif isinstance(value, float):
                    if "prob" in key:
                        new_val = float(np.clip(new_val, 0.0, 1.0))
                    else:
                        new_val = max(0.0, new_val)

                evolved[key] = new_val

        return evolved

    def _evolve_uniform(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default evolution using uniform sampling between min/max bounds.

        Args:
            params: Current parameters including bounds
        Returns:
            New parameter set
        """
        new_params = params.copy()

        for key, value in params.items():
            min_key = f"min_{key}"
            max_key = f"max_{key}"
            if min_key in params and max_key in params:
                if isinstance(value, int):
                    new_params[key] = self.rng.randint(
                        params[min_key], params[max_key] + 1
                    )
                else:
                    new_params[key] = self.rng.uniform(params[min_key], params[max_key])

        return new_params

    def _evolve_sbm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve SBM parameters while preserving community structure.

        Args:
            params: Current parameters including bounds
        Returns:
            New parameter set
        """
        new_params = params.copy()

        # Generate new intra_prob with community preservation
        new_intra = self.rng.uniform(params["min_intra_prob"], params["max_intra_prob"])

        # Set inter_prob to maintain community structure
        min_ratio = 3.0  # Minimum ratio for community separation
        max_allowed_inter = new_intra / min_ratio
        new_inter = self.rng.uniform(
            params["min_inter_prob"], min(max_allowed_inter, params["max_inter_prob"])
        )

        new_params["intra_prob"] = new_intra
        new_params["inter_prob"] = new_inter

        # Keep structural parameters constant
        new_params["n"] = params["n"]
        new_params["num_blocks"] = params["num_blocks"]

        return new_params

    def _evolve_ba(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve BA parameters while maintaining scale-free properties.

        Args:
            params: Current parameters including bounds
        Returns:
            New parameter set
        """
        new_params = params.copy()

        # Evolve m while keeping it reasonable relative to n
        n = params["n"]
        max_m = min(params["max_m"], n - 1)
        new_m = self.rng.randint(params["min_m"], max_m + 1)

        new_params["m"] = new_m
        new_params["n"] = n  # Keep n constant

        return new_params

    def _evolve_ws(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve WS parameters while maintaining small-world properties.

        Args:
            params: Current parameters including bounds
        Returns:
            New parameter set
        """
        new_params = params.copy()
        n = params["n"]

        # More controlled evolution of k_nearest to maintain connectivity
        # Keep k in a reasonable range (not too sparse, not too dense)
        min_k = max(2, params.get("min_k", 2))  # At least 2 neighbors for small-world
        max_k = min(
            params.get("max_k", n // 4), n // 4
        )  # Cap at n/4 to avoid over-connection

        # Get current k or use a reasonable default
        current_k = params.get("k_nearest", min_k)

        # Allow k to change by at most 2 steps to avoid drastic changes
        new_k = self.rng.randint(
            max(min_k, current_k - 2), min(max_k, current_k + 2) + 1
        )

        # Evolve rewiring probability more smoothly
        # Keep p relatively low to maintain small-world property
        min_p = params.get("min_rewire_prob", 0.0)
        max_p = min(
            params.get("max_rewire_prob", 0.3), 0.3
        )  # Cap at 0.3 to preserve structure

        new_p = self.rng.uniform(min_p, max_p)

        new_params["k_nearest"] = new_k
        new_params["rewire_prob"] = new_p
        new_params["n"] = n  # Keep n constant

        return new_params

    def _evolve_er(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve ER parameters.

        Args:
            params: Current parameters including bounds
        Returns:
            New parameter set
        """
        new_params = params.copy()

        # Evolve probability while maintaining reasonable density
        new_p = self.rng.uniform(params["min_prob"], params["max_prob"])

        new_params["prob"] = new_p
        new_params["n"] = params["n"]  # Keep n constant

        return new_params
