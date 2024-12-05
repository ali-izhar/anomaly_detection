"""
Dataset Generator for Graph Anomaly Detection

This module creates synthetic graph datasets with labeled anomalies based on 
martingale computations. Each graph in the sequence is represented by 6 features:
- Degree centrality
- Betweenness centrality  
- Eigenvector centrality
- Closeness centrality
- SVD embedding
- LSVD embedding (Laplacian SVD)

The anomaly labels are computed using martingale-based change point detection.
"""

import logging
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from synthetic_data.create_graphs import generate_ba_graphs
from src.changepoint import ChangePointDetector
from src.graph.features import extract_centralities, compute_embeddings

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Graph generation parameters
    n_nodes: int = 30  # Number of nodes in each graph
    edges: Dict[str, int] = None  # Edge parameters for BA graphs
    sequence_length: int = 200  # Total graphs per sequence
    num_sequences: int = 20  # Increased from 5 to 20 sequences
    change_points_per_sequence: int = 3  # Number of change points per sequence
    
    # Anomaly detection parameters
    threshold: float = 30.0  # Martingale threshold for anomaly detection
    epsilon: float = 0.8  # Sensitivity parameter for martingale computation
    window_size: int = 5  # Window size around change points to label as anomalous
    
    # Dataset split parameters
    split_ratio: Dict[str, float] = None  # Train/val/test split ratios
    output_dir: str = "dataset"  # Directory to save datasets
    
    def __post_init__(self):
        if self.edges is None:
            self.edges = {
                "initial": 3,
                "change_min": 2,
                "change_max": 8
            }
        if self.split_ratio is None:
            # Adjusted split ratios for small datasets
            self.split_ratio = {
                "train": 0.6,  # 60%
                "val": 0.2,    # 20%
                "test": 0.2    # 20%
            }

class GraphDataset:
    """Generator for graph anomaly detection datasets."""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize dataset generator with configuration."""
        self.config = config or DatasetConfig()
        
    def _calculate_change_points(self) -> List[int]:
        """Randomly calculate change points within a sequence."""
        sequence_length = self.config.sequence_length
        num_changes = self.config.change_points_per_sequence
        
        min_gap = sequence_length // (num_changes + 1)
        change_points = []
        for i in range(1, num_changes + 1):
            cp = i * min_gap + np.random.randint(-min_gap // 4, min_gap // 4)
            cp = max(1, min(sequence_length - 1, cp))
            change_points.append(cp)
        return sorted(change_points)
    
    def _compute_features(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Extract and combine all features for each graph."""
        try:
            # Get centrality features
            centralities = extract_centralities(graphs)
            
            # Get both SVD and LSVD embedding features
            svd_embeddings = compute_embeddings(graphs, method='svd', n_components=1)
            lsvd_embeddings = compute_embeddings(graphs, method='lsvd', n_components=1)
            
            # Combine all features
            features = []
            for i in range(len(graphs)):
                graph_features = []
                # Add centrality features
                for centrality_name in ['degree', 'betweenness', 'eigenvector', 'closeness']:
                    graph_features.append(np.mean(centralities[centrality_name][i]))
                # Add embedding features
                graph_features.append(float(svd_embeddings[i].mean()))
                graph_features.append(float(lsvd_embeddings[i].mean()))
                features.append(graph_features)
                
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature computation failed: {str(e)}")
            raise RuntimeError(f"Feature computation failed: {str(e)}")
    
    def _compute_anomaly_labels(
        self, 
        features: np.ndarray,
        change_points: List[int]
    ) -> np.ndarray:
        """Compute binary anomaly labels using martingales."""
        try:
            # Compute martingales for each feature dimension
            martingales = []
            
            for feature_idx in range(features.shape[1]):
                feature_values = features[:, feature_idx]
                normalized_values = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                
                detector = ChangePointDetector()
                feature_2d = normalized_values.reshape(-1, 1)
                
                result = detector.martingale_test(
                    data=feature_2d,
                    threshold=self.config.threshold,
                    epsilon=self.config.epsilon,
                    reset=True
                )
                martingales.append(result['martingales'])
            
            # Create labels: 1 for anomalies (within window of change points), 0 otherwise
            labels = np.zeros(len(features))
            w = self.config.window_size
            
            for cp in change_points:
                start_idx = max(0, cp - w)
                end_idx = min(len(features), cp + w + 1)
                labels[start_idx:end_idx] = 1
                
            return labels
            
        except Exception as e:
            logger.error(f"Label computation failed: {str(e)}")
            raise RuntimeError(f"Label computation failed: {str(e)}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to have zero mean and unit variance."""
        flat = features.reshape(-1, features.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-8  # Prevent division by zero
        return (features - mean) / std
    
    def _split_dataset(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray,
        change_points: List[List[int]]
    ) -> Dict:
        """Split dataset into train, validation, and test sets."""
        num_sequences = len(sequences)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)
        
        train_end = int(self.config.split_ratio["train"] * num_sequences)
        val_end = train_end + int(self.config.split_ratio["val"] * num_sequences)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            "train": {
                "sequences": sequences[train_idx], 
                "labels": labels[train_idx],
                "change_points": [change_points[i] for i in train_idx]
            },
            "val": {
                "sequences": sequences[val_idx], 
                "labels": labels[val_idx],
                "change_points": [change_points[i] for i in val_idx]
            },
            "test": {
                "sequences": sequences[test_idx], 
                "labels": labels[test_idx],
                "change_points": [change_points[i] for i in test_idx]
            }
        }
    
    def _save_to_hdf5(self, data_split: Dict):
        """Save dataset splits to HDF5 files."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        for split_name, data in data_split.items():
            split_dir = os.path.join(self.config.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Pad change points to max length
            max_changes = max(len(cp) for cp in data["change_points"])
            padded_change_points = np.array([
                cp + [-1] * (max_changes - len(cp)) 
                for cp in data["change_points"]
            ])
            
            with h5py.File(os.path.join(split_dir, "data.h5"), "w") as hf:
                hf.create_dataset("sequences", data=data["sequences"], compression="gzip")
                hf.create_dataset("labels", data=data["labels"], compression="gzip")
                hf.create_dataset("change_points", data=padded_change_points, compression="gzip")
                # Store the actual lengths
                hf.create_dataset("change_point_lengths", 
                                data=[len(cp) for cp in data["change_points"]], 
                                compression="gzip")
            
            logger.info(f"Saved {split_name} data: {data['sequences'].shape} sequences, "
                    f"{data['labels'].shape} labels, {len(data['change_points'])} change points")

    def generate(self, save_to_disk: bool = True) -> Dict:
        """Generate complete dataset with train/val/test splits."""
        try:
            all_sequences = []
            all_labels = []
            all_change_points = []
            
            for _ in range(self.config.num_sequences):
                # Generate single sequence with actual change points
                result = generate_ba_graphs()
                graphs = result['graphs']
                # Use the actual change points from BA graph generation
                change_points = result['change_points']  # These are the true structural changes
                
                # Extract features and compute labels
                features = self._compute_features(graphs)
                labels = self._compute_anomaly_labels(features, change_points)
                
                all_sequences.append(features)
                all_labels.append(labels)
                all_change_points.append(change_points)
            
            # Convert to arrays
            all_sequences = np.array(all_sequences)
            all_labels = np.array(all_labels)
            all_sequences = self._normalize_features(all_sequences)
            
            # Split dataset with change points
            data_split = self._split_dataset(
                all_sequences, 
                all_labels,
                all_change_points
            )
            
            if save_to_disk:
                self._save_to_hdf5(data_split)
            
            return data_split
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}")
            raise RuntimeError(f"Dataset generation failed: {str(e)}")

def create_dataset(
    config: Optional[DatasetConfig] = None,
    save_to_disk: bool = True
) -> Dict:
    """Convenience function to create a new dataset."""
    dataset = GraphDataset(config)
    return dataset.generate(save_to_disk=save_to_disk)

def main():
    """Main entry point for dataset creation."""
    print("\nGenerating Graph Anomaly Detection Dataset")
    print("----------------------------------------")
    
    # Create configuration
    config = DatasetConfig(
        n_nodes=30,
        sequence_length=200,
        num_sequences=20,
        change_points_per_sequence=3,
        threshold=30.0,
        epsilon=0.8,
        window_size=5,
        output_dir="dataset"
    )
    
    print(f"Configuration:")
    print(f"  - Nodes per graph: {config.n_nodes}")
    print(f"  - Sequence length: {config.sequence_length}")
    print(f"  - Number of sequences: {config.num_sequences}")
    print(f"  - Change points per sequence: {config.change_points_per_sequence}")
    print(f"  - Edge parameters: {config.edges}")
    print(f"  - Output directory: {config.output_dir}")
    
    # Create and save dataset
    data = create_dataset(config)
    
    # Print dataset statistics
    for split_name, split_data in data.items():
        sequences = split_data['sequences']
        labels = split_data['labels']
        print(f"\n{split_name.capitalize()} set:")
        print(f"  - Sequences shape: {sequences.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Anomaly ratio: {np.mean(labels):.2%}")

if __name__ == "__main__":
    main()
