# tests/test_dataset.py

"""Test script for verifying dataset loading and functionality."""

import os
import sys
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import GraphSequenceDataset

DATASET_PATH = "../src/dataset/link_prediction/link_prediction_dataset.h5"

# MAKE SURE THE DATASET EXISTS
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")


def test_basic_loading():
    """Test basic dataset loading and properties."""
    print("\n=== Testing Basic Dataset Loading ===")

    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of sequences: {dataset.num_sequences}")
    print(f"Sequence lengths: {dataset.sequence_lengths[:5]} ...")
    print(f"\nAvailable features:")
    for feat_name, dims in dataset.get_feature_dims().items():
        print(f"  {feat_name}: {dims}")


def test_feature_normalization():
    """Test feature normalization."""
    print("\n=== Testing Feature Normalization ===")

    # Load same data with and without normalization
    dataset_norm = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", normalize=True
    )

    dataset_raw = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", normalize=False
    )

    # Compare distributions
    idx = 0
    batch_norm = dataset_norm[idx]
    batch_raw = dataset_raw[idx]

    # Plot distributions for a few features
    features_to_plot = list(batch_norm["features"].keys())[:3]
    fig, axes = plt.subplots(
        len(features_to_plot), 2, figsize=(12, 4 * len(features_to_plot))
    )

    for i, feat_name in enumerate(features_to_plot):
        # Raw distribution
        sns.histplot(
            batch_raw["features"][feat_name].flatten().numpy(), ax=axes[i, 0], bins=50
        )
        axes[i, 0].set_title(f"{feat_name} (Raw)")

        # Normalized distribution
        sns.histplot(
            batch_norm["features"][feat_name].flatten().numpy(), ax=axes[i, 1], bins=50
        )
        axes[i, 1].set_title(f"{feat_name} (Normalized)")

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nFeature Statistics:")
    for feat_name in features_to_plot:
        norm_data = batch_norm["features"][feat_name].numpy()
        raw_data = batch_raw["features"][feat_name].numpy()
        print(f"\n{feat_name}:")
        print(f"  Raw - mean: {np.mean(raw_data):.3f}, std: {np.std(raw_data):.3f}")
        print(
            f"  Normalized - mean: {np.mean(norm_data):.3f}, std: {np.std(norm_data):.3f}"
        )


def test_sequence_windows():
    """Test sequence windowing functionality."""
    print("\n=== Testing Sequence Windows ===")

    # Test different window sizes and strides
    configs = [
        {"window_size": 5, "stride": 1},
        {"window_size": 10, "stride": 5},
        {"window_size": 20, "stride": 10},
    ]

    for config in configs:
        dataset = GraphSequenceDataset(data_path=DATASET_PATH, split="train", **config)

        print(f"\nWindow size: {config['window_size']}, Stride: {config['stride']}")
        print(f"Number of windows: {len(dataset)}")

        # Check a few windows
        for i in range(min(3, len(dataset))):
            batch = dataset[i]
            print(f"\nWindow {i}:")
            print(f"  Sequence: {batch['metadata']['sequence_idx']}")
            print(
                f"  Window: [{batch['metadata']['window_start']}, {batch['metadata']['window_end']})"
            )
            print(f"  Change points: {batch['metadata']['change_points'].tolist()}")
            print(f"  Adjacency shape: {batch['adjacency'].shape}")


def test_community_features():
    """Test loading and handling of community features."""
    print("\n=== Testing Community Features ===")

    # First check which features are available
    dataset_all = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    available_features = dataset_all.get_feature_dims().keys()
    community_features = [
        f
        for f in ["community_similarity", "block_membership"]
        if f in available_features
    ]

    if not community_features:
        print("No community features found in dataset")
        return

    # Load dataset with only available community features
    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH,
        split="train",
        window_size=5,
        stride=1,
        feature_set=community_features,
    )

    # Get a batch of data
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))

    # Check community features
    print("\nCommunity Features:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"  {feat_name}: {feat_tensor.shape}")

    # Check community labels in metadata
    if "community_labels" in batch["metadata"]:
        labels = batch["metadata"]["community_labels"]
        print(f"\nCommunity Labels Shape: {labels.shape}")
        print(f"Number of unique communities: {len(torch.unique(labels))}")
    else:
        print("\nNo community labels found in metadata")


def test_temporal_features():
    """Test loading and handling of temporal features."""
    print("\n=== Testing Temporal Features ===")

    # First check which features are available
    dataset_all = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    available_features = dataset_all.get_feature_dims().keys()
    temporal_features = [
        f
        for f in available_features
        if any(f.startswith(p) for p in ["temporal_", "link_", "cn_"])
    ]

    if not temporal_features:
        print("No temporal features found in dataset")
        return

    # Load dataset with only temporal features
    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH,
        split="train",
        window_size=5,
        stride=1,
        feature_set=temporal_features,
    )

    # Get a batch of data
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))

    # Check temporal features
    print("\nTemporal Features:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"  {feat_name}: {feat_tensor.shape}")
        # Verify that temporal features have the correct window size
        assert (
            feat_tensor.shape[1] == dataset.window_size
        ), f"Expected window size {dataset.window_size}, got {feat_tensor.shape[1]}"

    # Print feature statistics
    print("\nFeature Statistics:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"\n{feat_name}:")
        print(f"  Mean: {feat_tensor.mean().item():.3f}")
        print(f"  Std: {feat_tensor.std().item():.3f}")
        print(f"  Min: {feat_tensor.min().item():.3f}")
        print(f"  Max: {feat_tensor.max().item():.3f}")


def test_feature_dimensions():
    """Test handling of features with different dimensions."""
    print("\n=== Testing Feature Dimensions ===")

    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    # Get feature dimensions
    feature_dims = dataset.get_feature_dims()
    print("\nFeature Dimensions:")
    for feat_name, dims in feature_dims.items():
        print(f"  {feat_name}: {dims}")

    # Get a batch of data
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))

    # Verify feature shapes
    print("\nFeature Shapes in Batch:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"  {feat_name}: {feat_tensor.shape}")
        # First two dimensions should be batch_size and window_size
        assert (
            feat_tensor.shape[0] == 4
        ), f"Expected batch size 4, got {feat_tensor.shape[0]}"
        assert (
            feat_tensor.shape[1] == dataset.window_size
        ), f"Expected window size {dataset.window_size}, got {feat_tensor.shape[1]}"


def test_link_prediction_splits():
    """Test handling of link prediction data splits."""
    print("\n=== Testing Link Prediction Splits ===")

    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    # Get a batch of data
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))

    if "link_prediction" not in batch["metadata"]:
        print("No link prediction data found in dataset")
        return

    # Check link prediction data
    lp_data = batch["metadata"]["link_prediction"]
    print("\nLink Prediction Data:")
    for split in ["train", "val", "test"]:
        if split in lp_data:
            pos_edges = lp_data[split]["positive"]
            neg_edges = lp_data[split]["negative"]
            print(f"\n{split.upper()} Split:")
            print(f"  Positive edges shape: {pos_edges.shape}")
            print(f"  Negative edges shape: {neg_edges.shape}")

            # Verify edge indices are valid
            max_node_idx = dataset.num_nodes - 1
            valid_pos = (pos_edges >= -1) & (pos_edges <= max_node_idx)
            valid_neg = (neg_edges >= -1) & (neg_edges <= max_node_idx)
            assert valid_pos.all(), "Invalid node indices in positive edges"
            assert valid_neg.all(), "Invalid node indices in negative edges"


def test_link_prediction_data():
    """Test loading and handling of link prediction data."""
    print("\n=== Testing Link Prediction Data ===")

    # First check which features are available
    dataset_all = GraphSequenceDataset(
        data_path=DATASET_PATH, split="train", window_size=5, stride=1
    )

    available_features = dataset_all.get_feature_dims().keys()
    link_pred_features = [
        f
        for f in [
            "common_neighbors",
            "jaccard",
            "adamic_adar",
            "preferential_attachment",
            "resource_allocation",
            "degree_similarity",
            "link_frequency",
        ]
        if f in available_features
    ]

    if not link_pred_features:
        print("No link prediction features found in dataset")
        return

    # Load dataset with only available link prediction features
    dataset = GraphSequenceDataset(
        data_path=DATASET_PATH,
        split="train",
        window_size=5,
        stride=1,
        feature_set=link_pred_features,
    )

    # Get a batch of data
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))

    # Check link prediction features
    print("\nLink Prediction Features:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"  {feat_name}: {feat_tensor.shape}")

    # Print feature statistics
    print("\nFeature Statistics:")
    for feat_name, feat_tensor in batch["features"].items():
        print(f"\n{feat_name}:")
        print(f"  Mean: {feat_tensor.mean().item():.3f}")
        print(f"  Std: {feat_tensor.std().item():.3f}")
        print(f"  Min: {feat_tensor.min().item():.3f}")
        print(f"  Max: {feat_tensor.max().item():.3f}")


def main():
    """Run all tests."""
    test_basic_loading()
    test_feature_normalization()
    test_sequence_windows()
    test_community_features()
    test_temporal_features()
    test_feature_dimensions()
    test_link_prediction_splits()
    test_link_prediction_data()


if __name__ == "__main__":
    main()
