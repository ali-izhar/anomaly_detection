import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.changepoint import ChangePointDetector

def generate_change_point_data(n_points=200, change_point=100, seed=42):
    """Generate synthetic data with a change point"""
    np.random.seed(seed)
    
    # Generate two different normal distributions
    x1 = np.random.normal(0, 1, change_point)  # N(0,1) before change point
    x2 = np.random.normal(3, 1, n_points - change_point)  # N(3,1) after change point
    
    # Combine the sequences
    sequence = np.concatenate([x1, x2])
    time = np.arange(n_points)
    
    return time, sequence, change_point

def plot_change_point_distributions():
    """Create visualization of time series with change point and its distributions"""
    # Create figure with 2 subplots arranged vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Generate data
    time, sequence, change_point = generate_change_point_data()
    
    # Colors
    sequence_color = '#4B6BF5'  # Blue
    change_point_color = '#FF4B4B'  # Red
    
    # 1. Time Series Plot
    ax1.plot(time, sequence, color=sequence_color, label='Sequence')
    ax1.axvline(x=change_point, color=change_point_color, linestyle='--', label='Change Point')
    ax1.set_title('Time Series with Change Point')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.text(25, -2.5, r'$X_{1:k} \sim N(0,1)$', fontsize=10)
    ax1.text(change_point + 25, -2.5, r'$X_{k+1:n} \sim N(3,1)$', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Density Distribution Plot
    before_change = sequence[:change_point]
    after_change = sequence[change_point:]
    x_range = np.linspace(min(sequence), max(sequence), 200)
    kde_before = stats.gaussian_kde(before_change)
    kde_after = stats.gaussian_kde(after_change)
    
    ax2.plot(x_range, kde_before(x_range), color=sequence_color, label='Distribution before k')
    ax2.plot(x_range, kde_after(x_range), color=change_point_color, label='Distribution after k')
    ax2.set_title('Density Distributions')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_martingale_detection():
    """Create visualization of Martingale sequence detection with multiple epsilon values"""
    # Create figure with 3 rows (last row has 2 columns)
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], 
                         width_ratios=[1, 1],
                         hspace=0.3)
    
    # Generate data
    time, sequence, change_point = generate_change_point_data()
    
    # Colors for different epsilon values (Blue, Green, Red)
    colors = ['#4B6BF5', '#45A247', '#FF4B4B']
    epsilons = [0.7, 0.5, 0.3]  # Reversed order: less to more sensitive
    
    # Prepare data
    data = sequence.reshape(-1, 1)
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    detector = ChangePointDetector()
    
    # 1. Martingale Sequence Plot (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    all_pvalues = []
    for eps, color in zip(epsilons, colors):
        martingale_results = detector.martingale_test(
            data=normalized_data,
            threshold=30,
            epsilon=eps,
            reset=True
        )
        
        martingale_values = np.array(martingale_results['martingales'])
        martingale_values = np.clip(martingale_values, 1e-10, None)
        
        ax1.semilogy(time, martingale_values, color=color, 
                    label=f'Martingale (ε={eps}, {"more" if eps > 0.5 else "less" if eps < 0.5 else "medium"} sensitive)')
        all_pvalues.append(martingale_results['pvalues'])
    
    ax1.axvline(x=change_point, color='black', linestyle='--', label='Change Point')
    ax1.set_title('Martingale Sequence Detection\n' + r'$M_n = M_{n-1} \cdot \epsilon \cdot p_n^{(\epsilon-1)}$')
    ax1.set_xlabel('Time (n)')
    ax1.set_ylabel('Martingale Value (Mn) - Log Scale')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add text annotations for regions
    ax1.text(25, 1e-5, 'Normal Data Region\nHigh p-values\nMn stable/decreasing', 
             bbox=dict(facecolor='white', alpha=0.8))
    ax1.text(150, 1e-5, 'Unusual Data Region\nLow p-values\nMn increasing', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. P-values Plot (spans both columns)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Plot p-values for each epsilon with jittered y positions
    for i, (eps, color, pvals) in enumerate(zip(epsilons, colors, all_pvalues)):
        pvalues = np.array([p if isinstance(p, (int, float)) else p[0] if len(p) > 0 else np.nan 
                           for p in pvals])
        
        # Add small offset to separate the points
        jitter = np.random.normal(0, 0.01, len(pvalues))
        
        # Plot points with different markers for before/after change point
        before_change = slice(1, change_point)
        after_change = slice(change_point, None)
        
        ax2.scatter(time[before_change], pvalues[before_change] + jitter[before_change], 
                   color=color, alpha=0.5, label=f'ε={eps}')
        ax2.scatter(time[after_change], pvalues[after_change] + jitter[after_change],
                   color=color, alpha=0.5)
    
    ax2.axvline(x=change_point, color='black', linestyle='--', label='Change Point')
    ax2.axhline(y=0.05, color='red', linestyle=':', label='Significance Level')
    ax2.set_title('P-values Over Time')
    ax2.set_xlabel('Time (n)')
    ax2.set_ylabel('p-value')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Effect of epsilon on p-value sensitivity (left plot in bottom row)
    ax3 = fig.add_subplot(gs[2, 0])
    p_range = np.logspace(-3, 0, 100)
    
    for eps, color in zip(epsilons, colors):
        multiplier = eps * p_range**(eps-1)
        ax3.plot(p_range, multiplier, color=color, 
                label=f'ε={eps} ({"more" if eps > 0.5 else "less" if eps < 0.5 else "medium"} sensitive)')
    
    ax3.set_xscale('log')
    ax3.set_title('Effect of ε on p-value Sensitivity')
    ax3.set_xlabel('p-value')
    ax3.set_ylabel('Multiplier')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. P-value Distribution (right plot in bottom row)
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Plot distribution for medium sensitivity (ε=0.5)
    pvalues = np.array([p if isinstance(p, (int, float)) else p[0] if len(p) > 0 else np.nan 
                        for p in all_pvalues[1]])  # Using ε=0.5
    kde = stats.gaussian_kde(pvalues[~np.isnan(pvalues)])
    p_range = np.linspace(0, 1, 200)
    ax4.plot(p_range, kde(p_range), color=colors[1])
    ax4.fill_between(p_range, kde(p_range), alpha=0.3, color=colors[1])
    ax4.axvline(x=0.05, color='red', linestyle=':', label='Significance Level')
    ax4.set_title('P-value Distribution (ε=0.5)')
    ax4.set_xlabel('p-value')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_change_point_distributions()
    plot_martingale_detection()
