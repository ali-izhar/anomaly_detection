#!/usr/bin/env python3
# SHAP Analysis for Martingale-based Graph Change Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
import argparse
import os
import sys

def load_martingale_data(file_path):
    """
    Load martingale values from a CSV or Excel file.
    
    Expected format from user's data:
    - timestep: Timestep column
    - true_change_point: Column indicating true change points
    - individual_traditional_martingales_feature0, ...: Feature martingale columns
    - traditional_sum_martingales: Additive martingale column
    """
    print(f"Loading martingale data from {file_path}")
    
    # Determine file type and read accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}. Use .csv, .xlsx, or .xls")
    
    print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get timesteps
    if 'timestep' in df.columns:
        timesteps = df['timestep'].values
    else:
        timesteps = np.arange(len(df))
    
    # Get change point column
    cp_col = None
    if 'true_change_point' in df.columns:
        cp_col = 'true_change_point'
    
    # Extract feature martingale columns
    feature_cols = [col for col in df.columns if col.startswith('individual_traditional_martingales_feature')]
    
    if not feature_cols:
        # Try alternative naming
        feature_cols = [col for col in df.columns if col.startswith('feature') or 'martingale' in col.lower()]
        feature_cols = [col for col in feature_cols if 'traditional_sum' not in col and 'traditional_avg' not in col]
    
    if not feature_cols:
        raise ValueError("Could not identify feature martingale columns in the file.")
    
    print(f"Found {len(feature_cols)} feature martingales: {feature_cols}")
    
    # Rename columns for easier use
    column_mapping = {}
    for col in feature_cols:
        feature_num = col.split('feature')[-1]
        column_mapping[col] = f'M^{feature_num}'
    
    # Map additive martingale column
    if 'traditional_sum_martingales' in df.columns:
        column_mapping['traditional_sum_martingales'] = 'M^A'
    
    # Rename columns in dataframe
    df = df.rename(columns=column_mapping)
    
    # Update feature_cols list with new names
    feature_cols = [column_mapping[col] for col in feature_cols]
    
    return df, timesteps, feature_cols, cp_col

def compute_additive_martingale(df, feature_cols):
    """
    Compute the additive martingale if not already in the data.
    """
    if 'M^A' in df.columns:
        print("Using existing additive martingale column 'M^A'")
        return df
    
    print("Computing additive martingale as sum of feature martingales")
    df['M^A'] = df[feature_cols].sum(axis=1)
    return df

def plot_combined_analysis(df, timesteps, feature_cols, cp_col=None, threshold=None, detection_index=None):
    """
    Create a combined plot with martingale values on top and normalized SHAP values on bottom.
    
    Args:
        df: DataFrame containing martingale values
        timesteps: Array of timestep values
        feature_cols: List of column names for individual feature martingales
        cp_col: Name of change point column (optional)
        threshold: Detection threshold value (optional)
        detection_index: Index of detection point to analyze (optional)
    """
    print("Computing SHAP values...")
    
    # Prepare the data for SHAP analysis
    X = df[feature_cols]
    y = df['M^A']
    
    # Create a simple linear model (additive martingale is sum of features)
    model = LinearRegression(fit_intercept=False)  # No intercept since M^A is exact sum
    model.fit(X, y)
    
    # Verify the model is accurate (should be perfect for additive martingale)
    predictions = model.predict(X)
    r2 = sklearn.metrics.r2_score(y, predictions)
    print(f"R² score of linear model: {r2:.6f} (should be close to 1.0)")
    
    # Define prediction function for SHAP
    def predict_fn(x):
        return model.predict(x)
    
    # Compute SHAP values
    try:
        # Create a background dataset (sample of X)
        background_indices = np.random.choice(len(X), size=min(100, len(X)), replace=False)
        background = X.iloc[background_indices]
        
        # Create explainer and compute SHAP values
        print("Creating KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn, background)
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X)
        print(f"SHAP values shape: {np.array(shap_values).shape}")
    except Exception as e:
        print(f"Error using KernelExplainer: {e}")
        print("Computing SHAP values manually based on model coefficients...")
        # For linear models without intercept, SHAP values = feature_value * coefficient
        shap_values = np.zeros(X.shape)
        for i, col in enumerate(feature_cols):
            shap_values[:, i] = X[col].values * model.coef_[i]
    
    # Create a DataFrame with SHAP values over time
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{col}' for col in feature_cols])
    shap_df['timestep'] = timesteps
    
    # Normalize SHAP values by the max absolute SHAP value for any feature at any time
    shap_cols = [f'SHAP_{col}' for col in feature_cols]
    max_abs_shap = max(abs(shap_df[shap_cols].values.min()), 
                       abs(shap_df[shap_cols].values.max()))
    
    if max_abs_shap > 0:  # Avoid division by zero
        for col in shap_cols:
            shap_df[f'Norm_{col}'] = shap_df[col] / max_abs_shap
    
    # Create the combined 1 column, 2 row figure
    print("Creating combined plot...")
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'hspace': 0.3})
    
    # Top plot: Martingale values
    for col in feature_cols:
        axs[0].plot(timesteps, df[col], label=col, alpha=0.7, linewidth=1)
    
    # Plot additive martingale with thicker line
    axs[0].plot(timesteps, df['M^A'], label='Additive Martingale', 
             color='black', linewidth=2.5)
    
    # Plot threshold line if provided
    if threshold is not None:
        axs[0].axhline(y=threshold, color='r', linestyle='--', 
                    label=f'Threshold λ={threshold}')
    
    # Plot change points if available
    if cp_col is not None and cp_col in df.columns:
        change_points = timesteps[df[cp_col] == 1]
        if len(change_points) > 0:
            for cp in change_points:
                axs[0].axvline(x=cp, color='g', linestyle='--', alpha=0.7)
            axs[0].scatter(change_points, [0]*len(change_points), 
                       color='g', s=100, label='True Change Points')
    
    # Add detection point if detection_index is provided
    if detection_index is not None:
        dp = timesteps[detection_index]
        axs[0].axvline(x=dp, color='purple', linestyle=':', alpha=0.7)
        axs[0].scatter([dp], [0], color='purple', s=100, label='Detection Point')
    
    # Customize the top plot
    axs[0].set_title('Martingale Values Over Time', fontsize=16)
    axs[0].set_ylabel('Martingale Value', fontsize=14)
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)
    
    # Bottom plot: Normalized SHAP values
    for i, col in enumerate(feature_cols):
        axs[1].plot(timesteps, shap_df[f'Norm_SHAP_{col}'], label=f'{col}')
    
    # Add change points if available
    if cp_col is not None and cp_col in df.columns:
        change_points = timesteps[df[cp_col] == 1]
        if len(change_points) > 0:
            for cp in change_points:
                axs[1].axvline(x=cp, color='g', linestyle='--', alpha=0.7)
    
    # Add detection point if detection_index is provided
    if detection_index is not None:
        dp = timesteps[detection_index]
        axs[1].axvline(x=dp, color='purple', linestyle=':', alpha=0.7)
    
    # Customize the bottom plot
    axs[1].set_title('Normalized SHAP Values Over Time', fontsize=16)
    axs[1].set_xlabel('Timestep', fontsize=14)
    axs[1].set_ylabel('Normalized SHAP Value (-1 to 1)', fontsize=14)
    axs[1].legend(loc='best')
    axs[1].grid(True, alpha=0.3)
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig('combined_analysis.png', dpi=300)
    print("Saved combined plot to combined_analysis.png")
    plt.close()
    
    # If a detection index is provided, analyze feature contributions at that point
    if detection_index is not None:
        timestep_value = timesteps[detection_index]
        print(f"Analyzing detection point at timestep {timestep_value}...")
        
        # Get the data for the detection point
        detection_data = X.iloc[detection_index:detection_index+1]
        
        # Get SHAP values for detection point
        detection_shap = shap_values[detection_index] if detection_index < len(shap_values) else None
        
        if detection_shap is not None:
            # Compute probability of change as in CustomThresholdModel in threshold.py
            martingale_sum = df['M^A'].iloc[detection_index]
            change_prob = martingale_sum / (martingale_sum + threshold) if threshold else 0.5
            
            # Show the feature contributions at detection point
            contrib_df = pd.DataFrame({
                'Feature': feature_cols,
                'Martingale Value': detection_data.values[0],
                'SHAP Value': detection_shap,
                'Contribution %': 100 * detection_shap / sum(detection_shap) if sum(detection_shap) != 0 else 0
            })
            contrib_df = contrib_df.sort_values('Contribution %', ascending=False)
            print("\nFeature contributions at detection point:")
            print(contrib_df.to_string(index=False))
            print(f"\nProbability of change at detection point: {change_prob:.4f}")
            
            # Save the contributions to a CSV
            contrib_df.to_csv('detection_contributions.csv', index=False)
            print("Saved detection point contributions to detection_contributions.csv")
            
            return contrib_df

def find_detection_index(df, threshold):
    """Find the index where the additive martingale first exceeds the threshold."""
    for i in range(1, len(df)):
        if df['M^A'].iloc[i-1] < threshold and df['M^A'].iloc[i] >= threshold:
            return i
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze martingale data with SHAP.')
    parser.add_argument('file_path', help='Path to CSV or Excel file containing martingale values')
    parser.add_argument('--threshold', type=float, default=None, 
                       help='Detection threshold for additive martingale')
    args = parser.parse_args()
    
    # Load data
    df, timesteps, feature_cols, cp_col = load_martingale_data(args.file_path)
    
    # Compute additive martingale if needed
    df = compute_additive_martingale(df, feature_cols)
    
    # Find detection index if threshold is provided
    detection_index = None
    if args.threshold is not None:
        detection_index = find_detection_index(df, args.threshold)
        if detection_index is not None:
            print(f"Detection occurred at timestep {timesteps[detection_index]}")
        else:
            print("No detection found for the given threshold.")
    
    # Create the combined plot with martingale values and normalized SHAP values
    plot_combined_analysis(df, timesteps, feature_cols, cp_col, args.threshold, detection_index)

if __name__ == "__main__":
    main()
