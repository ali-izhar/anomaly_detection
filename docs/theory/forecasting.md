# §IV-B — Forecasting

Thm 4's validity depends on a *properly calibrated* forecaster (Def 8). Paper §IV-B describes two approaches; we implement the first (EWMA) and leave the second (graph-space constrained optimisation) as future work.

## Eq 21 — EWMA feature-space

![Paper §IV-B(a) Eq 21 — EWMA forecaster](../assets/eq-21-ewma-forecaster.png)

$$\hat{X}_{t+h}^{(k)} = \sum_{j=1}^{p_{\text{hist}}} w_j \cdot X_{t-j+1}^{(k)}, \quad w_j = \frac{\alpha^j}{\sum_l \alpha^l}$$

Weights sum to 1; recent observations weighted most heavily. Default `α=0.5`.

!!! question "Why EWMA, not Holt's?"
    Holt's adds a trend term. Under H₀ the trend integrates noise, inflating predictive-p-value variance — weakens Def 8 calibration, which Thm 4 relies on. EWMA is the conservative calibrated choice.

    Tradeoff: EWMA cannot extrapolate, so `X̂_{t+h}` is essentially the same for every h. Under H₁, Horizon's signal is weaker than Traditional's ⇒ Horizon ≤ Traditional empirically. See [horizon](horizon.md) for the theoretical analysis.

## Predictor protocol

```python
class Predictor(Protocol):
    def predict(self, history: np.ndarray, horizon: int) -> float: ...
    def predict_multi(self, history: np.ndarray, horizon: int) -> np.ndarray: ...
```

Swap forecaster in one line: `HorizonDetector(forecaster=MyPredictor())`. `hmd.forecaster.HoltForecaster` is available as a trend-aware alternative (deviates from EWMA's H₀-calibration; faster Horizon detection but higher FPR).

## §IV-B(b) graph-space — not implemented

The adjacency-matrix constrained optimisation (Eqs 22-25) is described in the paper but not evaluated in Table IV. We didn't implement it. Feature-space reproduces the paper's qualitative claims; graph-space is a forecaster swap, not a framework change.
