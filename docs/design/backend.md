# Backend — CPU vs GPU

`hmd/` defaults to NumPy on CPU.

## Why NumPy, not cupy?

N=50 nodes, T≤300 snapshots. A 50×50 eigendecomposition is ~10 µs on NumPy (MKL) vs ~50 µs on cupy (kernel-launch dominated). The full detector runs in ~3 seconds per sequence on CPU — GPU offers no single-run speedup and adds host↔device sync per step.

GPU wins only when we **batch** many runs (experiment sweeps over scenario × seed × config).

## Switching backends

```bash
HMD_BACKEND=cupy .venv/bin/python experiments/run_table3.py
```

Requires `pip install cupy-cuda13x`. Feature extraction (NetworkX centralities) stays CPU — graph algorithms have no dense-tensor equivalent at N=50.

## Where the backend actually matters

| Task | Backend | Why |
|---|---|---|
| Single detector run | NumPy CPU | Graphs too small to amortize GPU overhead |
| Table IV sweep (~90 runs) | NumPy CPU + `multiprocessing.Pool` | Embarrassingly parallel over seeds |
| Table III grid (~1350 runs) | cupy GPU (batched) | Kernel launches amortized |
| N > 1000 nodes (not in paper) | cupy GPU | Eigendecomp becomes bottleneck |

All four supported without source changes.
