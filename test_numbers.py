#!/usr/bin/env python3
"""
Calculate exact Shapley values for martingale-based change detection.

This script computes the Shapley values for each feature's martingale contribution
at timestep t=125 from the Erdős–Rényi graph example in our paper.
"""

import numpy as np
from itertools import combinations
import pandas as pd
import shap
from matplotlib import pyplot as plt

# Define the martingale values at t=125
martingale_values = {
    "mean_degree": 12.89,          # f0
    "density": 39.20,              # f1
    "mean_clustering": 0.001,      # f2
    "mean_betweenness": 0.63,      # f3
    "mean_eigenvector": 0.05,      # f4
    "mean_closeness": 5.06,        # f5
    "max_singular_value": 3.49,    # f6
    "min_laplacian": 1.13,         # f7
}

def calculate_exact_shapley_values(martingale_values):
    """
    Calculate exact Shapley values using the classic formula:
    
    φᵢ = Σ_S⊆N∖{i} |S|!(|N|-|S|-1)!/|N|! × [f(S∪{i}) - f(S)]
    
    where f(S) is the sum of martingale values for features in subset S.
    """
    features = list(martingale_values.keys())
    n = len(features)
    shapley_values = {}
    
    # For each feature
    for i, feature_i in enumerate(features):
        value = 0
        # Iterate through all possible subsets excluding feature_i
        for r in range(n):
            for subset in combinations([j for j in range(n) if j != i], r):
                # Calculate |S|!(|N|-|S|-1)!/|N|!
                weight = (factorial(r) * factorial(n - r - 1)) / factorial(n)
                
                # Calculate f(S)
                f_s = sum(martingale_values[features[j]] for j in subset)
                
                # Calculate f(S∪{i})
                f_s_i = f_s + martingale_values[feature_i]
                
                # Add weighted marginal contribution
                value += weight * (f_s_i - f_s)
        
        shapley_values[feature_i] = value
    
    return shapley_values

def factorial(n):
    """Calculate n! (factorial of n)"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def calculate_approximate_shapley_values_with_shap(martingale_values):
    """
    Calculate approximate Shapley values using SHAP's KernelExplainer
    """
    # Convert to format expected by SHAP
    X = pd.DataFrame([martingale_values])
    
    # Define a simple model that sums the inputs (like our change detection)
    def model(X):
        return np.sum(X, axis=1)
    
    # Create a background dataset (here we use the same point)
    background = X
    
    # Create explainer
    explainer = shap.KernelExplainer(model, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # Convert to dictionary
    result = {feature: value for feature, value in zip(martingale_values.keys(), shap_values[0])}
    return result

def main():
    """Main function to calculate and display Shapley values"""
    print("Martingale values at t=125:")
    for feature, value in martingale_values.items():
        print(f"  {feature}: {value}")
    print("\nTotal martingale sum:", sum(martingale_values.values()))
    
    print("\nCalculating exact Shapley values...")
    shapley_values = calculate_exact_shapley_values(martingale_values)
    
    # Sort by contribution
    sorted_shapley = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    
    print("\nExact Shapley values (sorted by contribution):")
    for feature, value in sorted_shapley:
        print(f"  {feature}: {value:.2f} ({100 * value / sum(shapley_values.values()):.1f}%)")
    
    print("\nSum of Shapley values:", sum(shapley_values.values()))
    
    print("\nCalculating approximate Shapley values using SHAP's KernelExplainer...")
    approx_shapley = calculate_approximate_shapley_values_with_shap(martingale_values)
    
    sorted_approx = sorted(approx_shapley.items(), key=lambda x: x[1], reverse=True)
    
    print("\nApproximate Shapley values (sorted by contribution):")
    for feature, value in sorted_approx:
        print(f"  {feature}: {value:.2f} ({100 * value / sum(approx_shapley.values()):.1f}%)")
    
    print("\nSum of approximate Shapley values:", sum(approx_shapley.values()))
    
    # Calculate normalized SHAP values (for P(change) explanation)
    threshold = 60.0
    martingale_sum = sum(martingale_values.values())
    p_change = martingale_sum / (martingale_sum + threshold)
    
    print(f"\nP(change) at t=125: {p_change:.2f}")
    
    print("\nNormalized SHAP values (for explaining P(change)):")
    for feature, value in sorted_shapley:
        norm_value = value * p_change / martingale_sum
        print(f"  {feature}: {norm_value:.4f} ({100 * norm_value / p_change:.1f}%)")

if __name__ == "__main__":
    main()
