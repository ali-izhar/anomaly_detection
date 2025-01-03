# archive/draw_predicts.py

import numpy as np
import matplotlib.pyplot as plt


def generate_feature_data(n_historical=100, n_future=50, n_features=6, seed=42):
    """Generate synthetic feature data with trends and anomalies"""
    np.random.seed(seed)

    # Generate historical data
    historical = np.zeros((n_historical, n_features))
    time = np.arange(n_historical)

    # Add different patterns for each feature
    for i in range(n_features):
        # Base pattern
        historical[:, i] = 0.5 + 0.1 * np.sin(time * (i + 1) * 0.1)
        # Add noise
        historical[:, i] += np.random.normal(0, 0.05, n_historical)

    # Generate future data with some anomalies
    future = np.zeros((n_future, n_features))
    future_time = np.arange(n_historical, n_historical + n_future)

    for i in range(n_features):
        # Continue the pattern
        future[:, i] = 0.5 + 0.1 * np.sin(future_time * (i + 1) * 0.1)
        # Add increased noise
        future[:, i] += np.random.normal(0, 0.08, n_future)

    # Add anomalies with different characteristics
    anomaly_points = [10, 30, 45]  # Points where anomalies occur

    for i, t in enumerate(anomaly_points):
        if i == 0:  # First anomaly - clearly visible
            future[t, :] += np.random.normal(0.3, 0.1, n_features)
        elif i == 1:  # Second anomaly - no actual change
            pass  # Skip to create false positive scenario
        else:  # Last anomaly - subtle change
            future[t, :] += np.random.normal(0.1, 0.05, n_features)

    return historical, future, anomaly_points


def plot_prediction_concept():
    """Visualize the concept of feature prediction and anomaly detection"""
    # Generate data
    n_historical, n_future = 100, 50
    historical, future, true_anomalies = generate_feature_data(n_historical, n_future)

    # Create figure with just 2 subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Create axes
    ax1 = fig.add_subplot(gs[0])  # Features plot
    ax2 = fig.add_subplot(gs[1])  # Martingale plot

    # Colors
    feature_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    martingale_color = "#4B6BF5"  # Blue
    anomaly_color = "#FF4B4B"  # Red
    success_color = "#45A247"  # Green
    warning_color = "#FFA500"  # Orange

    # Helper function for prediction arrows
    def add_prediction_arrow(ax, start_x, end_x, y, color, text, direction="up"):
        """Helper function to add prediction arrows"""
        if direction == "up":
            y_start = y
            y_end = y * 1.5
            y_text = y * 1.7
        else:
            y_start = y
            y_end = y * 0.6
            y_text = y * 0.4

        ax.annotate(
            "",
            xy=(end_x, y_end),
            xytext=(start_x, y_start),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=0.3"
            ),
        )
        ax.text(
            (start_x + end_x) / 2,
            y_text,
            text,
            color=color,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )

    # 1. Feature Plot with Predictions
    time = np.arange(n_historical + n_future)
    features = ["Degree", "Betweenness", "Eigenvector", "Closeness", "SVD", "LSVD"]

    # Plot historical and future data
    for i, (feature, color) in enumerate(zip(features, feature_colors)):
        # Historical data
        ax1.plot(time[:n_historical], historical[:, i], color=color, alpha=0.7)

        # Future predictions with uncertainty
        mean_pred = future[:, i]
        conf_interval = 0.1 + np.arange(n_future) * 0.002
        ax1.fill_between(
            time[n_historical:],
            mean_pred - conf_interval,
            mean_pred + conf_interval,
            color=color,
            alpha=0.2,
        )
        ax1.plot(time[n_historical:], mean_pred, color=color, linestyle="--", alpha=0.7)

    # Add present time and anomaly lines
    ax1.axvline(x=n_historical, color="black", linestyle=":", label="Present")
    for t in true_anomalies:
        ax1.axvline(x=n_historical + t, color=anomaly_color, alpha=0.3, linestyle="--")

    # 2. Martingale Detection Plot
    # Simulate martingale values with specific behaviors
    martingale = np.ones(n_historical + n_future)
    tau = 30  # threshold

    # 1. True Positive: Increase before first anomaly
    start_increase = n_historical + true_anomalies[0] - 8
    peak_at = n_historical + true_anomalies[0]
    martingale[start_increase:peak_at] = np.logspace(0, 4, peak_at - start_increase)

    # 2. False Positive: Increase but no anomaly
    false_alarm_start = n_historical + 20
    false_alarm_end = false_alarm_start + 5
    martingale[false_alarm_start:false_alarm_end] = np.logspace(0, 2, 5)
    martingale[false_alarm_end : false_alarm_end + 5] = np.logspace(
        2, 0, 5
    )  # Return to normal

    # 3. Missed Detection: Small increase but doesn't cross threshold
    missed_start = n_historical + true_anomalies[-1] - 5
    missed_end = n_historical + true_anomalies[-1]
    martingale[missed_start:missed_end] = np.logspace(
        0, np.log10(tau / 2), missed_end - missed_start
    )

    # Plot martingale with clear threshold
    ax2.semilogy(time, martingale, color=martingale_color, linewidth=2)
    ax2.axhline(y=tau, color=anomaly_color, linestyle="--", label="τ (Threshold)")
    ax2.axvline(x=n_historical, color="black", linestyle=":")

    # Add vertical lines for anomalies
    for t in true_anomalies:
        ax2.axvline(x=n_historical + t, color=anomaly_color, alpha=0.3, linestyle="--")

    # Add arrows and annotations
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)

    # 1. True Positive
    add_prediction_arrow(
        ax1,
        start_increase,
        peak_at,
        1.1,
        success_color,
        "Forecasting Anomaly\n(Confirmed)",
        "up",
    )
    add_prediction_arrow(
        ax2,
        start_increase,
        peak_at,
        10,
        success_color,
        "Early Warning\n(Correct)",
        "up",
    )

    # 2. False Positive
    add_prediction_arrow(
        ax1,
        false_alarm_start,
        false_alarm_end,
        0.3,
        warning_color,
        "False Alarm\n(No Actual Anomaly)",
        "down",
    )
    add_prediction_arrow(
        ax2,
        false_alarm_start,
        false_alarm_end,
        5,
        warning_color,
        "Unnecessary Warning",
        "down",
    )

    # 3. Missed Detection
    add_prediction_arrow(
        ax1, missed_start, missed_end, 0.2, anomaly_color, "Missed Anomaly", "down"
    )
    add_prediction_arrow(
        ax2, missed_start, missed_end, 3, anomaly_color, "Failed to Detect", "down"
    )

    # Add labels and titles
    ax1.set_title("Graph Feature Predictions", fontsize=12)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Feature Value")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Martingale Detection", fontsize=12)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Martingale Value (Mn)")
    ax2.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(
        "Early Anomaly Detection using Graph Features and Martingales",
        fontsize=14,
        y=0.95,
    )

    # Adjust y-axis limits
    ax1.set_ylim(-0.2, 1.4)
    ax2.set_ylim(1, 1e5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_prediction_concept()
