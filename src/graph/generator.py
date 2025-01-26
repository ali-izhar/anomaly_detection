# src/graph/generator.py

from typing import Dict, List, Tuple

import logging
import numpy as np
import networkx as nx

from .utils import graph_to_adjacency

logger = logging.getLogger(__name__)


class GraphGenerator:
    """Generator for dynamic graph sequences."""

    # Generator functions
    _GENERATORS = {
        "ba": lambda n, m, **kwargs: nx.barabasi_albert_graph(
            n=n, m=m, seed=kwargs.get("seed")
        ),
        "ws": lambda n, k_nearest, rewire_prob, **kwargs: nx.watts_strogatz_graph(
            n=n, k=k_nearest, p=rewire_prob, seed=kwargs.get("seed")
        ),
        "er": lambda n, prob, **kwargs: nx.erdos_renyi_graph(
            n=n, p=prob, seed=kwargs.get("seed")
        ),
        "sbm": lambda n, num_blocks, intra_prob, inter_prob, **kwargs: nx.stochastic_block_model(
            sizes=[n // num_blocks] * (num_blocks - 1)
            + [n - (n // num_blocks) * (num_blocks - 1)],
            p=np.full((num_blocks, num_blocks), inter_prob)
            + np.diag([intra_prob - inter_prob] * num_blocks),
            seed=kwargs.get("seed"),
        ),
    }

    # Model name aliases
    MODEL_ALIASES = {
        "barabasi_albert": "ba",
        "watts_strogatz": "ws",
        "erdos_renyi": "er",
        "stochastic_block_model": "sbm",
    }

    # Parameters to exclude from model generation
    _EXCLUDED_PARAMS = {"seq_len", "min_segment", "min_changes", "max_changes", "seed"}

    def __init__(self, model: str):
        """Initialize generator for a specific model.

        Args:
            model: Model name to use ('ba', 'ws', 'er', 'sbm' or full names)
        """
        # Resolve alias if full name used
        model = self.MODEL_ALIASES.get(model, model)

        if model not in self._GENERATORS:
            raise ValueError(f"Unknown model: {model}")

        self.model = model
        self.generator = None  # Will be set when seed is known
        self.rng = None  # Will be set in generate_sequence
        logger.info(f"Initialized generator for {model} model")

    @property
    def is_initialized(self) -> bool:
        """Check if generator is properly initialized with random state."""
        return self.rng is not None and self.generator is not None

    def _setup_generator(self, seed: int = None) -> None:
        """Set up the generator with proper random state.

        Args:
            seed: Random seed to use
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            seed = np.random.randint(2**31 - 1)
            self.rng = np.random.RandomState(seed)

        # Set up generator with fixed random state
        self.generator = self._GENERATORS[self.model]

    def _validate_parameters(self, params: Dict) -> Dict:
        """Validate and adjust model parameters.

        Args:
            params: Model parameters
        Returns:
            Validated and adjusted parameters
        """
        current = params.copy()
        n = current.get("n", 0)
        if n < 1:
            raise ValueError(f"Number of nodes must be positive, got {n}")

        # Model-specific validations and adjustments
        if self.model == "ba":
            m = current.get("m", 0)
            if m >= n:
                current["m"] = max(1, n - 1)
            elif m < 1:
                raise ValueError(f"BA model requires m >= 1, got m={m}")
        elif self.model == "ws":
            k = current.get("k_nearest", 0)
            if k >= n:
                current["k_nearest"] = n - 1 if n > 1 else 1
        elif self.model == "sbm":
            num_blocks = current.get("num_blocks", 1)
            if num_blocks > n:
                current["num_blocks"] = n

        return current

    def generate_sequence(self, params: Dict) -> Dict:
        """Generate graph sequence with optional change points.

        Args:
            params: Model parameters from YAML config
        Returns:
            Dict with graphs, change points, parameters
        """
        params = params.copy()

        # Set up random state
        seed = params.get("seed")
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)  # Also set global numpy seed
        else:
            # If no seed provided, create a new random state with a random seed
            seed = np.random.randint(2**31 - 1)  # Max 32-bit signed int
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)  # Also set global numpy seed
            params["seed"] = seed

        # Set up generator with fixed random state
        if self.model == "ba":
            self.generator = lambda n, m, **kwargs: nx.barabasi_albert_graph(
                n=n, m=m, seed=self.rng
            )
        elif self.model == "ws":
            self.generator = (
                lambda n, k_nearest, rewire_prob, **kwargs: nx.watts_strogatz_graph(
                    n=n, k=k_nearest, p=rewire_prob, seed=self.rng
                )
            )
        elif self.model == "er":
            self.generator = lambda n, prob, **kwargs: nx.erdos_renyi_graph(
                n=n, p=prob, seed=self.rng
            )
        elif self.model == "sbm":

            def sbm_generator(n, num_blocks, intra_prob, inter_prob, **kwargs):
                sizes = [n // num_blocks] * (num_blocks - 1)
                sizes.append(n - sum(sizes))
                p = np.full((num_blocks, num_blocks), inter_prob)
                np.fill_diagonal(p, intra_prob)
                return nx.stochastic_block_model(sizes=sizes, p=p, seed=self.rng)

            self.generator = sbm_generator

        # Generate change points and parameter sets
        change_points, num_changes = self._generate_change_points(params)
        param_sets = self._generate_parameter_sets(params, num_changes)
        logger.info(f"Generated {num_changes} change points at: {change_points}")

        # Generate graphs for each segment
        all_graphs = []
        for i in range(len(change_points) + 1):
            start = change_points[i - 1] if i > 0 else 0
            end = change_points[i] if i < len(change_points) else params["seq_len"]
            segment = self._generate_graph_segment(param_sets[i], end - start)
            all_graphs.extend(segment)

        return {
            "graphs": all_graphs,
            "change_points": change_points,
            "parameters": param_sets,
            "model": self.model,
            "num_changes": num_changes,
            "n": params["n"],
            "sequence_length": params["seq_len"],
            "seed": seed,  # Return the actual seed used
        }

    def _generate_change_points(self, params: Dict) -> Tuple[List[int], int]:
        """Generate random change points for sequence.

        Args:
            params: Generation parameters from YAML
        Returns:
            (change points, number of changes)
        """
        seq_len = params["seq_len"]
        min_segment = params["min_segment"]
        min_changes = params["min_changes"]
        max_changes = params["max_changes"]

        max_possible = (seq_len - min_segment) // min_segment
        max_changes = min(max_changes, max_possible)
        min_changes = min(min_changes, max_changes)

        num_changes = self.rng.randint(min_changes, max_changes + 1)
        valid_positions = list(
            range(min_segment, seq_len - min_segment + 1, min_segment)
        )

        if len(valid_positions) < num_changes:
            return [], 0

        points = sorted(
            self.rng.choice(valid_positions, size=num_changes, replace=False)
        )
        return points, num_changes

    def _generate_parameter_sets(self, params: Dict, num_changes: int) -> List[Dict]:
        """Generate parameter sets for each segment.

        Args:
            params: Base parameters from YAML
            num_changes: Number of changes to generate
        Returns:
            List of parameter dictionaries
        """
        param_sets = [params]
        current = params.copy()

        for _ in range(num_changes):
            new_params = current.copy()

            # Mutate parameters with min/max bounds
            for key in current:
                min_key = f"min_{key}"
                max_key = f"max_{key}"
                if min_key in current and max_key in current:
                    if isinstance(current[key], int):
                        new_params[key] = self.rng.randint(
                            current[min_key], current[max_key] + 1
                        )
                    else:
                        new_params[key] = self.rng.uniform(
                            current[min_key], current[max_key]
                        )

            param_sets.append(new_params)
            current = new_params

        return param_sets

    def _generate_graph_segment(self, params: Dict, length: int) -> List[np.ndarray]:
        """Generate sequence of graphs with evolving parameters.

        Args:
            params: Model parameters from YAML
            length: Sequence length
        Returns:
            List of adjacency matrices
        """
        if self.rng is None:
            raise RuntimeError(
                "Random state not initialized. Call generate_sequence first."
            )

        graphs = []
        current = params.copy()

        # Filter out non-model parameters
        excluded_params = {
            "seq_len",
            "min_segment",
            "min_changes",
            "max_changes",
            "seed",
        }
        excluded_params.update(
            k
            for k in current
            if k.startswith("min_") or k.startswith("max_") or k.endswith("_std")
        )

        # Validate parameters
        n = current.get("n", 0)
        if n < 1:
            raise ValueError(f"Number of nodes must be positive, got {n}")

        # Model-specific validations and adjustments
        if self.model == "ba":
            m = current.get("m", 0)
            if m >= n:
                # For small graphs, adjust m to be valid
                current["m"] = max(1, n - 1)
            elif m < 1:
                raise ValueError(f"BA model requires m >= 1, got m={m}")
        elif self.model == "ws":
            k = current.get("k_nearest", 0)
            if k >= n:
                k = n - 1 if n > 1 else 1
                current["k_nearest"] = k
        elif self.model == "sbm":
            num_blocks = current.get("num_blocks", 1)
            if num_blocks > n:
                current["num_blocks"] = n

        for _ in range(length):
            # Only pass relevant parameters to generator
            model_params = {
                k: v for k, v in current.items() if k not in excluded_params
            }
            G = self.generator(**model_params)
            adj = graph_to_adjacency(G)
            graphs.append(adj)
            current = self._evolve_parameters(current)

        return graphs

    def _evolve_parameters(self, params: Dict) -> Dict:
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
