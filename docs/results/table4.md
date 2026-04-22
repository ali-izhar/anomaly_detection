# Table IV — main comparison

Four detectors × nine scenarios × ten seeds. Metrics: TPR, FPR, ADD (Eqs 26-28) with Δ=20.

## Reproduce

```bash
.venv/bin/python experiments/run_table4.py --n-trials 10 --out results/table4.csv
```

Runs in ~28 s on 19 workers. Writes `results/table4.csv` (pivoted) and `results/table4_raw.csv`.

## Our numbers (paper-strict `per_feature`, EWMA, λ=50, startup=20)

| Detector | TPR | FPR | ADD |
|---|---:|---:|---:|
| Martingale | 0.994 | 0.017 | 6.98 |
| Horizon (EWMA, h=5) | 0.994 | 0.017 | 6.98 |
| CUSUM | 1.000 | 0.083 | 0.07 |
| EWMA chart | 1.000 | 0.086 | 0.02 |

## vs paper's claimed Table IV

| Metric | Ours (Martingale) | Paper (Martingale) | Status |
|---|---:|---:|---|
| TPR | 0.994 | ~1.00 | ≈ |
| FPR | 0.017 | ~0.002 | higher but within Ville (1/λ = 0.02) |
| ADD | 6.98 | 5-13 (varies) | ✓ |

## Horizon vs Martingale

With the paper's EWMA forecaster, Horizon gives identical ADD to Martingale on every scenario. See [theory/horizon](../theory/horizon.md) for the theoretical reason: EWMA cannot extrapolate, so Horizon's per-step signal ≤ Traditional's.
