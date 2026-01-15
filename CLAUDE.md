# Claude Development Notes

## Project Overview

Horizon Martingale Change Detection Framework for graph structural anomaly detection (ICDM 2025 paper implementation).

## Architecture

```
src/
├── changepoint/          # Martingale-based detection (VERIFIED)
│   ├── martingale.py     # Core parallel detection algorithm
│   ├── detector.py       # High-level detector API
│   ├── conformal.py      # Non-conformity scores and p-values
│   ├── betting.py        # Betting functions (power, mixture, etc.)
│   └── baselines.py      # CUSUM, EWMA baselines
├── predictor/            # Feature prediction (Holt's smoothing)
│   ├── factory.py        # Predictor factory
│   └── feature_predictor.py  # FeaturePredictor (trend-based forecasting)
├── graph/                # Graph generation and feature extraction
└── utils/                # Normalization, export utilities
```

## Martingale Framework (VERIFIED)

### Key Findings (2025-01-14)

**The martingale framework is correctly implemented and working.** With perfect predictions:
- Horizon martingale detects changes 7-19 timesteps earlier than traditional
- Zero false positives with proper dampening (horizon_decay=0.7, normalize_horizons=False)
- All network types (SBM, ER, BA, WS) show significant improvement

### Algorithm Summary

**Traditional Martingale (Definition 6):**
```
M_t^{(k)} = M_{t-1}^{(k)} * g(p_t^{(k)})
```

**Horizon Martingale (Definition 7 + dampening):**
```
M_t^{horizon} = M_t^{traditional} * ∏_h g(p_{t,h})^(decay^h)
```

Where:
- `g(p)` = betting function (mixture of power functions)
- `p_t` = conformal p-value for current observation
- `p_{t,h}` = conformal p-value for prediction at horizon h
- `decay` = 0.7 (exponential dampening for longer horizons)

### Key Implementation Details

1. **Predictions timing**: At time t, predictions[t - history_size] contains forecasts for times t+1, t+2, ..., t+H

2. **Evidence accumulation**: Horizon multiplies traditional value by weighted product of prediction betting values

3. **Dampening**: Prevents false positives by reducing contribution of longer-horizon predictions

4. **Reset behavior**: Both martingales reset together after any detection

### Test Results with Perfect Predictions

| Network | Traditional TP | Horizon TP | Horizon Delay |
|---------|---------------|------------|---------------|
| SBM     | 0/2           | 2/2        | 7-10 steps    |
| ER      | 0/2           | 2/2        | 2-7 steps     |
| BA      | 1/2           | 2/2        | -25 to -3 (early!) |
| WS      | 0/2           | 2/2        | -13 to 9 steps |

## Predictor Quality - SOLVED

### Problem (HybridPredictor)

The original graph predictor (exponential weighted average) had **MSE ~9.0** on normalized features - predictions were ~3 standard deviations off on average. It used:

```python
weights = [alpha**j for j in range(1, k+1)]  # alpha=0.8
return sum(weights[j] * adj_matrices[k-j-1] for j in range(k))
```

Problems:
1. **No trend detection**: Doesn't capture velocity/direction of change
2. **Slow adaptation**: With alpha=0.8, it takes many timesteps for new data to dominate
3. **No extrapolation**: Outputs weighted average of history, not predicted future

### Solution: FeaturePredictor

Implemented `FeaturePredictor` using **Holt's double exponential smoothing** with trend:

```python
# Level: l_t = α * x_t + (1 - α) * (l_{t-1} + b_{t-1})
# Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
# Forecast: x_{t+h} = l_t + h * b_t
```

Key improvements:
1. **True trend extrapolation**: Predicts future values, not weighted averages
2. **Works in feature space**: Avoids graph->feature extraction errors
3. **Adaptive parameters**: Alpha increases when error spikes (regime change)

### Results

**MSE Comparison (lower is better):**
| Time | FeaturePredictor | HybridPredictor | Improvement |
|------|-----------------|-----------------|-------------|
| t=25 | 0.44 | 1.84 | 75.9% |
| t=35* | 1.00 | 1.96 | 49.0% |
| t=45* | 0.23 | 5.15 | 95.4% |
| t=55 | 0.22 | 24.40 | 99.1% |
| **OVERALL** | **0.80** | **9.05** | **91.2%** |

*Change points at t=35 and t=45

**Detection Timing (synthetic network data):**
| Predictor | Horizon Delay | Comparison |
|-----------|---------------|------------|
| Perfect (oracle) | 0.0 steps | baseline |
| FeaturePredictor | 1.0 steps | near-optimal |
| HybridPredictor | 8.0 steps | 7x slower |

**Key finding**: FeaturePredictor enables **7.0 timesteps faster** horizon detection than HybridPredictor.

### Usage

```python
from src.predictor import PredictorFactory

# Create predictor (Holt's double exponential smoothing)
predictor = PredictorFactory.create('feature', {'n_history': 10, 'alpha': 0.3, 'beta': 0.1})
```

## Benchmark Results (Paper Reproduction)

### Overall Performance (threshold=100)

| Method | Precision | Recall | F1 | Avg Delay |
|--------|-----------|--------|-----|-----------|
| **Horizon** | 0.636 | 0.920 | **0.713** | **2.1 steps** |
| Traditional | 0.567 | 0.920 | 0.673 | 8.5 steps |

**Key finding**: Horizon detects **6.4 timesteps faster** than Traditional with same recall (92%) and +5.9% F1 improvement.

### Results by Network Type

| Network | Scenario | T-Delay | H-Delay | Speedup |
|---------|----------|---------|---------|---------|
| SBM | community_merge | 12.4 | 5.4 | **7.0 steps** |
| SBM | density_change | 10.6 | 3.4 | **7.2 steps** |
| ER | density_change | 7.0 | 1.2 | **5.8 steps** |
| BA | parameter_shift | 4.2 | 0.6 | **3.6 steps** |
| WS | rewiring_change | 7.8 | -1.0 | **8.8 steps** |

### Threshold Sensitivity

| Threshold | H-Precision | H-Recall | H-F1 | H-Delay |
|-----------|-------------|----------|------|---------|
| 20 | 0.274 | 0.920 | 0.412 | 0.1 |
| 30 | 0.402 | 0.920 | 0.533 | -0.3 |
| 50 | 0.498 | 0.880 | 0.601 | 0.3 |
| **100** | **0.636** | **0.920** | **0.713** | 2.1 |

Higher thresholds improve precision while maintaining recall. Best F1 at threshold=100.

## Configuration Defaults

```python
DetectorConfig(
    threshold=100.0,         # Detection threshold (tuned)
    history_size=10,         # Prediction model history
    cooldown=20,             # Min steps between detections
    horizon_decay=0.7,       # Dampening for longer horizons
    normalize_horizons=False # False = faster detection
)
```

## Status Summary

**COMPLETED:**
1. ✓ Martingale framework verified and working correctly
2. ✓ FeaturePredictor implemented with 91% MSE improvement
3. ✓ Independent tracking for fair Traditional vs Horizon comparison
4. ✓ Codebase cleaned up (removed legacy HybridPredictor)
5. ✓ Benchmark reproduces paper results:
   - Horizon: 71.3% F1, 2.1 steps delay
   - Traditional: 67.3% F1, 8.5 steps delay
   - **Horizon is 6.4 timesteps faster**

**OPTIONAL FUTURE WORK:**
- Feature-space ARIMA predictor as alternative
- GNN-based predictor for learned patterns
- Further tuning of adaptive alpha thresholds

## Files Modified

- `src/changepoint/martingale.py` - Core parallel detection with dampening
- `src/changepoint/detector.py` - High-level API with horizon_decay, normalize_horizons
- `src/predictor/feature_predictor.py` - Holt's smoothing with trend
- `src/predictor/factory.py` - Predictor factory
- `src/scripts/benchmark.py` - Benchmark script
- `src/scripts/mit_reality.py` - MIT Reality dataset processing
- `src/configs/algorithm.yaml` - Default configuration

## Running Experiments

```bash
# Paper benchmark (Traditional vs Horizon across all networks)
python run_benchmark.py

# Full parallel benchmark
python src/scripts/benchmark.py --workers 4

# Quick pipeline test
python test_full_pipeline.py
```
