<i>This document outlines the intuition and design of the experiments located in the `tests` directory.</i>

# 1. Linear Models Experiment

The Linear Models Experiment is designed to detect and explain structural changes in time series data using linear equations combined with martingale frameworks.

## 1.1 Linear Equations Setup

From `config/linear_models.yaml`, the following equations serve as the baseline models before and after the change point, enabling the detection of structural shifts in the data.

```yaml
equations:
  default:              
    - [2, 3]            # y = 2x + 3
    - [3, 3]            # y = 3x + 3
    - [5, 8]            # y = 5x + 8
    - [7, 2]            # y = 7x + 2
```

## 1.2 Martingale Framework

From `tests/linear_models.py`:
```python
# Before change point (t ≤ change_point): M(t) = 1
# After change point: M(t) = mx + b - (m * change_point + b)
```

This design intentionally creates a stable baseline $(M(t) = 1)$ before the change point and allows the martingale values to follow the defined linear equations post-change. This clear distinction aids in visually and statistically identifying changes in the time series.

## 1.3 Experiment Dashboard

<!-- ![SHAP Dashboard](../assets/shap_dashboard.png) -->

- **Default Scenario (Row 1):** Martingale values remain constant at $1$ before the change point $(t=100)$ and follow the actual linear equations thereafter. Consequently, SHAP values exhibit a spike at the change point, indicating the detection moment.
- **Detection Thresholds (Row 2):** Adjusting detection thresholds impacts the sensitivity and stability of change detection:
  - **Low Thresholds:** Lower thresholds enable earlier detection with higher sensitivity but introduce more noise in SHAP values, resulting in jagged spikes and a less stable post-detection plateau.
  ```yaml
  low_threshold:
    thresholds:
      sum: 30
      avg: 7.5
  ```

  - **High Thresholds:** Higher thresholds delay detection but provide cleaner and more decisive SHAP spikes, smoother transitions, and a more stable plateau after detection.
  ```yaml
  high_threshold:
    thresholds:
      sum: 70
      avg: 17.5
  ```

  - **Visual Pattern Comparison:**
  ```text
    Low Threshold:  ___/\/\___  (Earlier, noisier)
    High Threshold: ____/‾\___  (Later, cleaner)
  ```

- **Modified Equations (Row 4):** This setup tests the robustness of change detection across different equation sets, ensuring the framework's adaptability to varied linear relationships.
  ```yaml
  modified:
    equations: "modified"
    - [1, 2]
    - [2, 4]
    - [4, 6]
    - [6, 1]
  ```

  - **Time Variation (Rows 5-6):** The experiment includes scenarios with early $(t=50)$ and late $(t=150)$ change points, demonstrating the framework's ability to consistently detect changes regardless of their timing within the series.


## 1.4 SHAP Value Interpretation

SHAP values are computed to interpret the model's detections, providing insights into feature contributions during change detection. From `linear_models.py`:

```python
def compute_shap_values(model, background, X_explain):
    """Compute SHAP values for model interpretability.
    Uses KernelExplainer to compute Shapley values:
    phi_i = sum_S (1 - |S| / n) * (f(x_S + i) - f(x_S))
    """
```

The SHAP value patterns observed include:

- **Pre-change Point:** SHAP values remain flat, indicating stable model behavior.
- **Change Point:** A spike in SHAP values signals the detection of a structural change.
- **Post-change Point:** SHAP values stabilize, reflecting the new normal state of the model.


## 1.5 Running the Experiment

```bash
python main.py linear -c config/linear_models.yaml
```
