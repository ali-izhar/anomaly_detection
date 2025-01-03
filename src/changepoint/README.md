# Change Point Detection

This module demonstrates **martingale-based change detection** in sequential data (including **multi-view** or multi-feature scenarios), grounded in **conformal prediction** principles. The approach combines **strangeness measures**, **p-values**, and a **power martingale** update rule to detect distributional shifts over time.

## What Is a Change Point?

Given a data stream $X_1, X_2, \ldots, X_n$, a **change point** at time $k$ indicates that the distribution of $X_{1:k}$ differs significantly from the distribution of $X_{k+1:n}$. In practice, we usually do **online** detection — we process observations one at a time and raise an alarm as soon as a suspected shift is detected.

## The Martingale Framework

### Background

A **martingale** $\{M_n\}_{n \ge 0}$ is a process that satisfies the property

$$
E[M_{n+1} \mid M_1, M_2, \ldots, M_n] = M_n
$$

Intuitively, the expected future value of a martingale equals its current value. This differs from:

- **Markov Processes**: Where the future state depends only on the current state.
- **Martingale Processes**: Where the *expected* future value is the current value (like fair games where your expected winnings remain constant).

In **change detection**, we construct a **test martingale** from sequentially computed p-values. Under a stable (unchanged) distribution, this martingale remains bounded or grows slowly; a sharp growth in the martingale suggests a change point.

### The Power Martingale Update

We use the **power martingale** update rule:

$$
M_n = M_{n-1} \times \epsilon \times (p_n)^{\,(\epsilon - 1)}
$$

where

- $p_n$ is the **p-value** at time $n$.
- $\epsilon \in (0,1)$ is a **sensitivity parameter**. Smaller $\epsilon$ places more emphasis on small p-values (anomalies), making the martingale spike faster.

**Key insight**:  
- If $p_n$ is high (meaning the new observation is not strange), $(p_n)^{(\epsilon - 1)}$ is not too large, and $M_n$ remains stable or decreases.  
- If $p_n$ is very low (the point is suspicious), $(p_n)^{(\epsilon - 1)}$ can be large, so $M_n$ increases significantly.

### Thresholding

We set a **detection threshold** $\tau$. Whenever $M_n$ exceeds $\tau$, we raise a **change point alarm**. Optionally, we can **reset** the martingale and/or the historical window upon detection to search for subsequent changes.

## P-Value via Strangeness

### Strangeness

Each new observation gets a **strangeness** value, $\alpha_n$, measuring how "unusual" it is relative to past observations. For instance, one might use:

- **Cluster-based distance**: Minimum distance to a KMeans cluster center.
- **Graph structural features**: Degree, betweenness, or subgraph patterns.
- **Embedding distance**: Distance in a learned embedding space (node2vec, GNN, etc.).

### Empirical P-value Calculation

Let $\{\alpha_1, \alpha_2, \ldots, \alpha_n\}$ be all strangeness values up to time $n$. We define:

$$
p_n = \frac{\text{\#}\{\alpha_i : \alpha_i > \alpha_n\} + \theta \,\text{\#}\{\alpha_i : \alpha_i = \alpha_n\}}{n},
$$

where $\theta \sim U(0,1)$ is a random tie-break. This is the **standard conformal prediction** p-value approach:

- **Low p-value** ($p_n \approx 0$) means $\alpha_n$ is larger than most previous strangeness scores (the new point is very unusual).
- **High p-value** ($p_n \approx 1$) means the new point is not unusual compared to history.

## Single-View vs. Multi-View Detection

1. **Single-View**: We have a single stream $\{X_t\}$. For each point, we compute strangeness, then p-value, then update one martingale $M_n$.
2. **Multi-View** (or multi-feature): Suppose we have $d$ different features or “views,” each producing its own martingale $M_j(n)$. Then we can **combine** them (often by summation) to get a global statistic:
   $$
     M_{\mathrm{total}}(n) = \sum_{j=1}^{d} M_j(n).
   $$
   If $M_{\mathrm{total}}(n)$ exceeds $\tau$, we declare a global change point.

## Step-by-Step Algorithm

1. **Compute Strangeness**: For each new observation (or network snapshot), compute a numeric measure of how unusual it is.
2. **Compute P-value**: Compare that strangeness to the empirical distribution of past strangeness, per conformal prediction.
3. **Update Martingale**: 
   $$
     M_n = M_{n-1} \times \epsilon \times \bigl(p_n\bigr)^{(\epsilon - 1)}
   $$
4. **Threshold**: If $M_n > \tau$, flag a change point and optionally reset.

**Advantages**:
- Nonparametric, distribution-free approach.
- Adaptable to complex data (graphs, embeddings, etc.) by customizing the strangeness measure.
- Online: can process data as it arrives.

**Challenges**:
- Choosing $\epsilon$ and $\tau$ can be domain-specific.
- Large dimensional data might need sophisticated strangeness measures (e.g., embeddings).

## Example Usage

1. **Single-View**:
   ```python
   from changepoint.detector import ChangePointDetector
   import numpy as np

   data = np.array([0.1, 0.2, 0.5, 1.2, 1.25, 0.3]).reshape(-1,1)
   detector = ChangePointDetector()
   result = detector.detect_changes(data, threshold=5.0, epsilon=0.6, max_window=3)
   print(result["change_points"])
   print(result["martingale_values"])
   ```

2. **Multi-View**:
    ```python
    feat1 = np.array([0.1, 0.2, 0.5, 1.2, 1.25, 0.3]).reshape(-1,1)
    feat2 = np.array([2.1, 2.2, 2.5, 1.9, 2.0, 2.1]).reshape(-1,1)

    detector = ChangePointDetector()
    result = detector.detect_changes_multiview(
        data=[feat1, feat2],
        threshold=7.0,
        epsilon=0.4,
        max_window=3
    )
    print(result["change_points"])
    print(result["martingale_values"])
    ```

## References

- Ho, S. S., & Wechsler, H. (2005). A martingale framework for detecting changes in data streams by testing exchangeability. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
- Doob, J. L. (1953). Stochastic Processes. John Wiley & Sons.
- Vovk, V. et al. (2005+). Conformal Prediction frameworks for nonparametric p-values in machine learning.
- Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. Journal of Machine Learning Research.
