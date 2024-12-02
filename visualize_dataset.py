import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


def load_split(split_name, dataset_dir="dataset"):
    """Load sequences and labels from a specific split."""
    split_path = os.path.join(dataset_dir, split_name)
    with h5py.File(os.path.join(split_path, "sequences.h5"), "r") as hf:
        sequences = hf["sequences"][:]
    with h5py.File(os.path.join(split_path, "labels.h5"), "r") as hf:
        labels = hf["labels"][:]
    return sequences, labels


def visualize_split(split_name):
    sequences, labels = load_split(split_name)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{split_name} Split Analysis")

    # 1. Plot feature distributions for first sequence
    axes[0, 0].hist(sequences[0].flatten(), bins=50)
    axes[0, 0].set_title("Feature Distribution (First Sequence)")
    axes[0, 0].set_xlabel("Feature Values")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Plot label distribution over time for first sequence
    axes[0, 1].plot(labels[0], "r-", label="Labels")
    axes[0, 1].set_title("Labels Over Time (First Sequence)")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Label")
    axes[0, 1].grid(True)

    # 3. Plot average feature values over time for first sequence
    mean_features = sequences[0].mean(axis=(1))  # Average across nodes
    for i in range(mean_features.shape[1]):
        axes[1, 0].plot(mean_features[:, i], label=f"Feature {i}")
    axes[1, 0].set_title("Average Feature Values Over Time")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Feature Value")
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 4. Plot label distribution across all sequences
    axes[1, 1].hist(labels.flatten(), bins=2)
    axes[1, 1].set_title("Label Distribution (All Sequences)")
    axes[1, 1].set_xlabel("Label Value")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Print some basic statistics
    print(f"\n{split_name} Statistics:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {sequences.shape[1]}")
    print(f"Number of nodes: {sequences.shape[2]}")
    print(f"Feature dimension: {sequences.shape[3]}")
    print(f"Percentage of positive labels: {(labels == 1).mean() * 100:.2f}%")


# Visualize each split
for split in ["train", "val", "test"]:
    visualize_split(split)
