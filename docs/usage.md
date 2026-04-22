# Usage — a new student's tour

Assumes you've cloned the repo and have a working Python 3.10+ environment with numpy, scipy, networkx, and matplotlib. Run everything from the repo root.

## 1. Quick detection on a synthetic sequence

```python
from hmd import HorizonDetector
from hmd.data.synthetic import sbm_community_merge

# 200-timestep SBM sequence with one community-merge at a random time in [40, 160].
seq = sbm_community_merge(seed=42)
print(f"True change point: {seq.change_points}")

# Detector with paper defaults.
det = HorizonDetector(threshold=50, horizon=5, history_size=20, startup_period=20)
result = det.run(seq.graphs)

print(f"Detected at: {result.change_points}")
print(f"Max traditional martingale: {result.M_traditional.max():.1f}")
print(f"Max horizon martingale:     {result.M_horizon.max():.1f}")
```

## 2. Attribution on a detected change

```python
if result.change_points:
    t_detect = result.change_points[0]
    shares = result.attribution_at(t_detect)  # {feature_name: percent}
    driver = max(shares, key=shares.get)
    print(f"Dominant driver at t={t_detect}: {driver} ({shares[driver]:.1f}%)")
    # Sorted breakdown
    for name, pct in sorted(shares.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}  {pct:5.1f}%")
```

## 3. Plotting the martingale traces

```python
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(len(seq.graphs))
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(t, result.M_traditional, label="Traditional")
ax.semilogy(t, result.M_horizon,     label="Horizon")
ax.axhline(50, color="red", ls="--", label="λ = 50")
for cp in seq.change_points:
    ax.axvline(cp, color="gray", ls=":")
for det in result.change_points:
    ax.plot(det, result.M_traditional[det], "D", markerfacecolor="none", markeredgecolor="k")
ax.set_xlabel("time"); ax.set_ylabel("M_t (log)")
ax.legend(); fig.tight_layout()
fig.savefig("my_first_figure.png", dpi=150)
```

## 4. Swapping a dependency

The detector has three pluggable pieces: features, betting function, forecaster.

=== "Custom betting"

    ```python
    from hmd import HorizonDetector
    from hmd.betting import power, mixture
    
    # More aggressive on small p-values
    det = HorizonDetector(betting=power(eps=0.3))
    
    # Custom mixture
    g = mixture(weights=[0.5, 0.5], epsilons=[0.2, 0.8])
    det = HorizonDetector(betting=g)
    ```

=== "Custom forecaster"

    Any class matching the `Predictor` protocol plugs in:
    
    ```python
    import numpy as np
    from hmd import HorizonDetector
    
    class ConstantForecaster:
        """Forecasts the last observed value (naive baseline)."""
        def predict(self, history, horizon):
            return float(history[-1])
        def predict_multi(self, history, horizon):
            return history[-1].copy()
    
    det = HorizonDetector(forecaster=ConstantForecaster())
    ```

=== "Custom feature set"

    ```python
    from hmd import HorizonDetector
    from hmd.features import FeatureSpec, default_set
    
    def mean_triangle_count(g):
        import networkx as nx
        return sum(nx.triangles(g).values()) / (3 * g.number_of_nodes())
    
    features = list(default_set()) + [FeatureSpec("mean_triangles", mean_triangle_count)]
    det = HorizonDetector(features=features)
    ```

## 5. Ablation: pure Horizon vs pure Martingale

```python
from hmd import HorizonDetector
from hmd.data.synthetic import er_density_increase

seq = er_density_increase(seed=0)

horizon_only = HorizonDetector(enable_traditional=False).run(seq.graphs)
trad_only    = HorizonDetector(enable_horizon=False).run(seq.graphs)

print("Horizon only:", horizon_only.change_points)
print("Trad only:   ", trad_only.change_points)
```

## 6. MIT Reality dataset

```python
from hmd import HorizonDetector
from hmd.data.mit_reality import load, MIT_EVENTS

seq, meta = load()   # bundled at hmd/data/mit_reality/Proximity.csv
det = HorizonDetector(threshold=20, startup_period=20, normalize_features=True)
result = det.run(seq.graphs)

for cp in result.change_points:
    matched_event = next((d for d in MIT_EVENTS if d <= cp <= d+20), None)
    if matched_event is not None:
        print(f"Day {cp}: matches '{MIT_EVENTS[matched_event]}' (true day {matched_event})")
```

## 7. Reproducing a paper figure

```bash
.venv/bin/python experiments/figure1_mit.py --out results/figure1.png
.venv/bin/python experiments/figure2.py    --out results/figure2.png
.venv/bin/python experiments/run_table4.py --n-trials 10 --out results/table4.csv
```

## 8. Running the tests

```bash
.venv/bin/pytest tests/ -v
```

This validates: betting calibration (∫g(p)dp ≈ 1), conformal p-value uniformity under H₀ (KS test), martingale property E[M_T]≈1, Ville's inequality P(sup M ≥ λ) ≤ 1/λ.
