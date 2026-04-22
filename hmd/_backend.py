"""Array backend selector (numpy default; `HMD_BACKEND=cupy` for batched GPU runs).

N=50 graphs are tiny enough that NumPy CPU beats CuPy GPU on single-run
detection (kernel-launch overhead dominates). GPU wins only for batched
parameter sweeps — users flip the env var before importing.
"""

from __future__ import annotations

import os

_backend = os.environ.get("HMD_BACKEND", "numpy").lower()

if _backend == "cupy":
    import cupy as xp  # type: ignore[import-not-found]

    BACKEND_NAME = "cupy"
elif _backend == "numpy":
    import numpy as xp  # type: ignore[assignment]

    BACKEND_NAME = "numpy"
else:
    raise ValueError(f"HMD_BACKEND must be 'numpy' or 'cupy', got {_backend!r}")


