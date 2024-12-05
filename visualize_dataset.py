"""
Dataset Visualization Script

This script reads the training data from the dataset directory and creates visualizations showing:
1. Timeline view of sequences with anomaly labels
2. Feature distributions and correlations
3. Training data statistics and balance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from pathlib import Path
import h5py
from src.changepoint import ChangePointDetector

class DatasetVisualizer:
    """Visualizer for graph anomaly detection dataset."""
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = Path(data_dir)
        self.feature_names = [
            'Degree', 'Betweenness', 'Eigenvector', 'Closeness', 
            'SVD', 'LSVD'
        ]
        self.colors = sns.color_palette("husl", n_colors=len(self.feature_names))
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Load training sequences, labels and change points from H5 files."""
        train_dir = self.data_dir / "train"
        
        with h5py.File(train_dir / "data.h5", "r") as hf:
            sequences = hf["sequences"][:]
            labels = hf["labels"][:]
            change_points = hf["change_points"][:]
            
        return sequences, labels, change_points
    
    def plot_sequence_timeline(
        self, 
        features: np.ndarray,
        labels: np.ndarray,
        change_points: List[List[int]],
        sequence_idx: int,
        save_path: str = None
    ) -> None:
        """Plot timeline view of a single sequence with features, martingales, and labels."""
        plt.figure(figsize=(15, 12))
        
        # Create three subplots
        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
        
        # Get change points for this sequence
        seq_change_points = change_points[sequence_idx]
        
        # Top subplot: Features
        ax1 = plt.subplot(gs[0])
        for i, (name, color) in enumerate(zip(self.feature_names, self.colors)):
            feature_values = features[sequence_idx, :, i]
            normalized_values = (feature_values - np.mean(feature_values)) / np.std(feature_values)
            ax1.plot(normalized_values, label=name, color=color, alpha=0.7)
        
        # Add change point vertical lines
        for cp in seq_change_points:
            ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        
        ax1.set_title(f'Sequence {sequence_idx}: Feature Values Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Feature Value (Normalized)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Middle subplot: Martingales
        ax2 = plt.subplot(gs[1])
        detector = ChangePointDetector()
        
        # Compute martingales for each feature
        martingales = []
        for i, (name, color) in enumerate(zip(self.feature_names, self.colors)):
            feature_values = features[sequence_idx, :, i]
            normalized_values = (feature_values - np.mean(feature_values)) / np.std(feature_values)
            feature_2d = normalized_values.reshape(-1, 1)
            
            result = detector.martingale_test(
                data=feature_2d,
                threshold=30.0,
                epsilon=0.8,
                reset=True
            )
            martingales.append(result['martingales'])
            ax2.plot(result['martingales'], label=name, color=color, alpha=0.7)
        
        # Plot combined martingales
        combined_martingales = np.maximum.reduce(martingales)
        ax2.plot(combined_martingales, label='Combined', color='black', linewidth=2, alpha=0.8)
        
        # Add threshold line and change points
        ax2.axhline(y=30.0, color='r', linestyle='--', label='Threshold')
        for cp in seq_change_points:
            ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        
        ax2.set_title('Martingale Values Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Martingale Value')
        ax2.set_yscale('log')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Bottom subplot: Labels
        ax3 = plt.subplot(gs[2])
        sequence_labels = labels[sequence_idx]
        ax3.fill_between(range(len(sequence_labels)), 0, sequence_labels, 
                        color='red', alpha=0.3, label='Anomaly')
        
        # Add change points
        for cp in seq_change_points:
            ax3.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_title('Anomaly Labels')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Label')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_feature_distributions(self, features: np.ndarray, save_path: str = None) -> None:
        """Plot distribution of each feature across all sequences."""
        plt.figure(figsize=(15, 10))
        
        for i, (name, color) in enumerate(zip(self.feature_names, self.colors)):
            plt.subplot(2, 3, i+1)
            feature_values = features[:, :, i].flatten()
            sns.histplot(feature_values, color=color, kde=True)
            plt.title(f'{name} Distribution')
            plt.xlabel('Value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_correlations(self, features: np.ndarray, save_path: str = None) -> None:
        """Plot correlation matrix between features."""
        flat_features = features.reshape(-1, features.shape[-1])
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = np.corrcoef(flat_features.T)
        sns.heatmap(correlation_matrix, 
                   xticklabels=self.feature_names,
                   yticklabels=self.feature_names,
                   annot=True, 
                   cmap='coolwarm',
                   center=0,
                   vmin=-1, 
                   vmax=1)
        plt.title('Feature Correlations')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_label_statistics(self, labels: np.ndarray, save_path: str = None) -> None:
        """Plot statistics about anomaly labels."""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Overall label distribution
        plt.subplot(1, 2, 1)
        total_samples = labels.size
        anomaly_samples = np.sum(labels)
        normal_samples = total_samples - anomaly_samples
        
        plt.pie([normal_samples, anomaly_samples],
                labels=['Normal', 'Anomaly'],
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        plt.title('Label Distribution')
        
        # Plot 2: Anomaly distribution across sequences
        plt.subplot(1, 2, 2)
        anomaly_per_seq = np.mean(labels, axis=1) * 100
        plt.hist(anomaly_per_seq, bins=20, color='lightcoral')
        plt.title('Anomaly Percentage per Sequence')
        plt.xlabel('Percentage of Anomalies')
        plt.ylabel('Number of Sequences')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def visualize_training_data(self, output_dir: str = "visualizations") -> None:
        """Create comprehensive visualizations of the training data."""
        print("Loading training data...")
        sequences, labels, change_points = self.load_training_data()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get shapes
        n_sequences, seq_length, n_features = sequences.shape
        print(f"\nTraining Data Shape:")
        print(f"Number of sequences: {n_sequences}")
        print(f"Sequence length: {seq_length}")
        print(f"Number of features: {n_features}")
        
        # 1. Plot example sequences with change points
        print("\nPlotting example sequences...")
        for i in range(min(5, n_sequences)):
            self.plot_sequence_timeline(
                sequences,
                labels,
                change_points,
                i,
                save_path=output_dir / f"sequence_{i}.png"
            )
        
        # 2. Plot feature distributions
        print("\nPlotting feature distributions...")
        self.plot_feature_distributions(
            sequences,
            save_path=output_dir / "feature_distributions.png"
        )
        
        # 3. Plot feature correlations
        print("\nPlotting feature correlations...")
        self.plot_feature_correlations(
            sequences,
            save_path=output_dir / "feature_correlations.png"
        )
        
        # 4. Plot label statistics
        print("\nPlotting label statistics...")
        self.plot_label_statistics(
            labels,
            save_path=output_dir / "label_statistics.png"
        )
        
        print(f"\nVisualizations saved to: {output_dir}/")

def main():
    """Main entry point for visualization."""
    print("Visualizing training data...")
    visualizer = DatasetVisualizer()
    visualizer.visualize_training_data()

if __name__ == "__main__":
    main()
