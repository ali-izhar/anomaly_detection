"""
Barabási-Albert (BA) Graph Sequence Generator with Data Preparation for Anomaly Prediction

=====================================================================
Overview
=====================================================================
This module is designed to generate multiple sequences of Barabási-Albert (BA) graphs 
with predefined parameter changes to simulate network evolution and structural 
anomalies. It extracts relevant features from each graph, labels anomalies based 
on martingale sums exceeding a specified threshold, and stores the processed data 
in HDF5 format for efficient consumption by machine learning models.

Key Features:
- **Dynamic Graph Generation:** Creates sequences of BA graphs with multiple change points.
- **Feature Extraction:** Computes centrality measures and dimensionality reduction embeddings.
- **Anomaly Labeling:** Labels time steps where martingale sums indicate anomalies.
- **GPU Acceleration:** Utilizes GPU for faster computation of embeddings and centrality measures.
- **Data Storage:** Saves processed data in an organized HDF5 structure for training, validation, and testing.

=====================================================================
Usage Instructions
=====================================================================
1. **Configuration:**
   - Update the `BA_CONFIG` dictionary with desired parameters such as the number of nodes, edge parameters, sequence lengths, etc.

2. **Dependencies:**
   - Ensure the following Python libraries are installed:
     - `networkx`
     - `numpy`
     - `h5py`
     - `scikit-learn`
     - `tqdm`
     - `cupy` (for GPU acceleration)
     - `cugraph` (for GPU-accelerated graph operations)
   - Install GPU-accelerated libraries using:
     ```bash
     pip install cupy-cuda11x cugraph
     ```
     Replace `cuda11x` with your specific CUDA version.

3. **Execution:**
   - Run the script directly using:
     ```bash
     python create_ba_graphs.py
     ```

=====================================================================
Section Breakdown
=====================================================================
1. **Imports and Setup:**
   - Imports necessary libraries and ensures reproducibility by setting random seeds.

2. **User Configuration (`BA_CONFIG`):**
   - Defines all configurable parameters for graph generation, feature extraction, anomaly labeling, and data storage.

3. **Graph Generation Class (`GraphGenerator`):**
   - Encapsulates the graph generation logic using GPU-accelerated libraries.

4. **Feature Extraction Functions:**
   - Extracts centrality measures and computes embeddings (SVD and LSVD) from adjacency matrices.

5. **Sequence Generation and Labeling:**
   - Generates individual graph sequences with multiple change points.
   - Computes martingale sums and creates binary anomaly labels based on a prediction horizon.

6. **Data Normalization and Splitting:**
   - Normalizes features across the entire dataset.
   - Splits the dataset into training, validation, and testing sets based on specified ratios.

7. **Data Storage (`save_to_hdf5`):**
   - Saves the processed sequences and labels into HDF5 files organized by dataset splits.

8. **Main Execution Flow (`main` Function):**
   - Orchestrates the entire data generation, feature extraction, labeling, normalization, splitting, and saving processes.

=====================================================================
Performance Enhancements with GPU
=====================================================================
To accelerate computation, especially for feature extraction tasks like centrality measures and embeddings, the script leverages GPU-accelerated libraries:
- **cuGraph:** Utilized for computing centrality measures on the GPU.
- **CuPy:** Used for GPU-accelerated numerical operations, particularly for embeddings.

**Note:** Ensure that your system has a compatible NVIDIA GPU and the appropriate CUDA drivers installed to utilize these features.

=====================================================================
Author: [Your Name]
Date: [Date]
"""

import os
import numpy as np
import h5py
import random
from typing import List, Dict
from tqdm import tqdm  # For progress bars

# GPU Libraries
try:
    import cupy as cp
    import cugraph

    GPU_AVAILABLE = True
except ImportError:
    print("CuPy or cuGraph not installed. Proceeding with CPU-based computations.")
    GPU_AVAILABLE = False

from sklearn.decomposition import TruncatedSVD

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
if GPU_AVAILABLE:
    cp.random.seed(42)

# -----------------------------------------------------------------------------#
#                              USER CONFIGURATION                              #
# -----------------------------------------------------------------------------#

BA_CONFIG = {
    "n": 50,  # Number of nodes in each graph
    "edges": {
        "initial": 3,  # Initial m parameter (edges per new node)
        "change_min": 2,  # Minimum change in edge count
        "change_max": 8,  # Maximum change in edge count
    },
    "sequence_length": 200,  # Total number of graphs per sequence
    # "num_sequences": 1500,         # Total sequences to generate
    "num_sequences": 10,  # Total sequences to generate
    "change_points_per_sequence": 3,  # Number of change points in each sequence
    "prediction_horizon": 5,  # Number of steps to predict ahead
    "threshold_tau": 30,  # Threshold for martingale sum to label anomalies
    "feature_dimension": 7,  # Number of features per node
    "output_dir": "dataset",  # Directory to save datasets
    "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
}

# -----------------------------------------------------------------------------#
#                           IMPLEMENTATION DETAILS                             #
# -----------------------------------------------------------------------------#


class GraphGenerator:
    """
    GraphGenerator utilizes cuGraph for GPU-accelerated graph operations.
    If GPU is unavailable, it falls back to NetworkX for CPU-based operations.
    """

    def __init__(self, use_gpu: bool = GPU_AVAILABLE):
        self.use_gpu = use_gpu
        if self.use_gpu:
            print("Using GPU-accelerated graph generation with cuGraph.")
        else:
            print("Using CPU-based graph generation with NetworkX.")
            import networkx as nx

            self.nx = nx

    def barabasi_albert(self, n: int, m: int) -> np.ndarray:
        """
        Generates a Barabási-Albert graph and returns its adjacency matrix.

        Parameters:
            n (int): Number of nodes.
            m (int): Number of edges to attach from a new node to existing nodes.

        Returns:
            adj_matrix (np.ndarray): Adjacency matrix of the generated graph.
        """
        if self.use_gpu:
            # cuGraph does not have a direct BA graph generator; using NetworkX on GPU via DataFrame
            # Create BA graph on CPU and transfer to GPU
            import networkx as nx

            G_cpu = nx.barabasi_albert_graph(n, m)
            df = cugraph.from_networkx(G_cpu).to_pandas_edgelist()
            G_gpu = cugraph.from_pandas_edgelist(
                df, source="source", destination="destination"
            )
            adj_matrix = cugraph.adj_matrix(G_gpu).toarray()
            return adj_matrix
        else:
            # CPU-based generation
            G = self.nx.barabasi_albert_graph(n, m)
            adj_matrix = self.nx.to_numpy_array(G)
            return adj_matrix


def extract_centralities(
    adj_matrix: np.ndarray, use_gpu: bool = GPU_AVAILABLE
) -> Dict[str, List[float]]:
    """
    Extracts centrality measures from an adjacency matrix using cuGraph or NetworkX.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        use_gpu (bool): Whether to use GPU-accelerated computation.

    Returns:
        centralities (Dict[str, List[float]]): Dictionary containing centrality measures.
    """
    if use_gpu:
        # Convert adjacency matrix to cuGraph graph
        G = cugraph.Graph()
        # Extract edges from adjacency matrix
        edges = np.transpose(np.nonzero(adj_matrix))
        edges_df = {"source": edges[:, 0], "destination": edges[:, 1]}
        import pandas as pd

        edges_pd = pd.DataFrame(edges_df)
        G.from_cudf_edgelist(
            cudf.DataFrame(edges_pd), source="source", destination="destination"
        )

        # Degree Centrality
        degree = G.degree["degree"].to_array().tolist()

        # Betweenness Centrality
        betweenness = (
            cugraph.betweenness_centrality(G)["betweenness_centrality"]
            .to_array()
            .tolist()
        )

        # Eigenvector Centrality
        eigenvector = (
            cugraph.eigenvector_centrality(G)["eigenvector_centrality"]
            .to_array()
            .tolist()
        )

        # Closeness Centrality
        closeness = (
            cugraph.closeness_centrality(G)["closeness_centrality"].to_array().tolist()
        )

    else:
        import networkx as nx

        G = nx.from_numpy_array(adj_matrix)
        degree = list(dict(G.degree()).values())
        betweenness = list(nx.betweenness_centrality(G).values())
        eigenvector = list(nx.eigenvector_centrality_numpy(G).values())
        closeness = list(nx.closeness_centrality(G).values())

    return {
        "degree": degree,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "closeness": closeness,
    }


def compute_embeddings(
    adj_matrix: np.ndarray, method: str = "svd", use_gpu: bool = GPU_AVAILABLE
) -> np.ndarray:
    """
    Computes dimensionality reduction embeddings (SVD or LSVD) from an adjacency matrix.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        method (str): 'svd' or 'lsvd' indicating the embedding method.
        use_gpu (bool): Whether to use GPU-accelerated computation.

    Returns:
        embeddings (np.ndarray): Flattened embedding vector.
    """
    if method not in ["svd", "lsvd"]:
        raise ValueError("Unsupported embedding method.")

    n_components = 2  # Example dimensionality

    if use_gpu:
        # Use CuPy for GPU-accelerated SVD
        adj_cp = cp.asarray(adj_matrix)
        if method == "svd":
            # Singular Value Decomposition
            U, S, Vt = cp.linalg.svd(adj_cp, full_matrices=False)
            embeddings = cp.asnumpy(U[:, :n_components] * S[:n_components])
        elif method == "lsvd":
            # Limited SVD (similar to Truncated SVD)
            U, S, Vt = cp.linalg.svd(adj_cp, full_matrices=False)
            embeddings = cp.asnumpy(U[:, :n_components] * S[:n_components])
    else:
        # Use scikit-learn's TruncatedSVD for CPU
        svd = TruncatedSVD(n_components=n_components)
        embeddings = svd.fit_transform(adj_matrix)

    return embeddings.flatten()  # Flatten to 1D array


def adjacency_to_graph(adj_matrix: np.ndarray, use_gpu: bool = GPU_AVAILABLE):
    """
    Converts an adjacency matrix to a NetworkX graph.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        use_gpu (bool): Whether to use GPU-accelerated computation.

    Returns:
        G (networkx.Graph or cugraph.Graph): The graph object.
    """
    if use_gpu:
        # Convert to cuGraph Graph
        G = cugraph.Graph()
        edges = np.transpose(np.nonzero(adj_matrix))
        edges_df = {"source": edges[:, 0], "destination": edges[:, 1]}
        import pandas as pd
        import cudf

        edges_pd = pd.DataFrame(edges_df)
        edges_cudf = cudf.DataFrame(edges_pd)
        G.from_cudf_edgelist(edges_cudf, source="source", destination="destination")
    else:
        import networkx as nx

        G = nx.from_numpy_array(adj_matrix)
    return G


def _generate_graph_segment(generator: GraphGenerator, n: int, m: int) -> np.ndarray:
    """
    Generates a single BA graph adjacency matrix.

    Parameters:
        generator (GraphGenerator): Instance of GraphGenerator.
        n (int): Number of nodes.
        m (int): Number of edges to attach from a new node.

    Returns:
        adj_matrix (np.ndarray): Adjacency matrix of the generated graph.
    """
    return generator.barabasi_albert(n=n, m=m)


def _calculate_change_points(sequence_length: int, num_changes: int) -> List[int]:
    """
    Randomly calculates change points within a sequence, ensuring they are well-distributed.

    Parameters:
        sequence_length (int): Total number of timesteps in the sequence.
        num_changes (int): Number of change points to generate.

    Returns:
        change_points (List[int]): Sorted list of change point indices.
    """
    min_gap = sequence_length // (num_changes + 1)
    change_points = []
    for i in range(1, num_changes + 1):
        cp = i * min_gap + random.randint(-min_gap // 4, min_gap // 4)
        cp = max(1, min(sequence_length - 1, cp))  # Ensure within bounds
        change_points.append(cp)
    return sorted(change_points)


def generate_ba_graphs_sequence(generator: GraphGenerator, config: Dict) -> Dict:
    """
    Generates a single BA graph sequence with multiple parameter changes.

    Parameters:
        generator (GraphGenerator): Instance of GraphGenerator.
        config (Dict): Configuration dictionary from BA_CONFIG.

    Returns:
        sequence_data (Dict): Contains generated graphs, change points, and m parameters.
    """
    n = config["n"]
    sequence_length = config["sequence_length"]
    num_changes = config["change_points_per_sequence"]

    # Randomly determine change points
    change_points = _calculate_change_points(sequence_length, num_changes)

    # Randomly assign m parameters for each segment
    m_initial = config["edges"]["initial"]
    m_changes = [
        random.randint(config["edges"]["change_min"], config["edges"]["change_max"])
        for _ in range(num_changes)
    ]

    # Generate all graphs
    graphs = []
    current_m = m_initial
    last_cp = 0
    for idx, cp in enumerate(change_points + [sequence_length]):
        for _ in range(last_cp, cp):
            adj = _generate_graph_segment(generator, n, current_m)
            graphs.append(adj)
        if idx < num_changes:
            current_m = m_changes[idx]
            last_cp = cp
    return {
        "graphs": graphs,
        "change_points": change_points,
        "m_parameters": [m_initial] + m_changes,
    }


def extract_features_from_sequence(
    graphs: List[np.ndarray], use_gpu: bool = GPU_AVAILABLE
) -> np.ndarray:
    """
    Extracts features for all graphs in a sequence.

    Parameters:
        graphs (List[np.ndarray]): List of adjacency matrices.
        use_gpu (bool): Whether to use GPU-accelerated computations.

    Returns:
        features (np.ndarray): Array of shape (sequence_length, n, feature_dimension).
    """
    features = []
    for adj_matrix in graphs:
        centralities = extract_centralities(adj_matrix, use_gpu=use_gpu)
        degree = centralities["degree"]  # List of length n
        betweenness = centralities["betweenness"]
        eigenvector = centralities["eigenvector"]
        closeness = centralities["closeness"]
        svd = compute_embeddings(adj_matrix, method="svd", use_gpu=use_gpu)  # 2 values
        lsvd = compute_embeddings(
            adj_matrix, method="lsvd", use_gpu=use_gpu
        )  # 2 values

        # Assemble node features
        node_features = []
        for i in range(BA_CONFIG["n"]):
            node_feat = [
                degree[i],
                betweenness[i],
                eigenvector[i],
                closeness[i],
                svd[i],  # Assuming SVD returns at least n values
                lsvd[i],  # Assuming LSVD returns at least n values
            ]
            node_features.append(node_feat)
        features.append(node_features)  # Shape: (n, feature_dimension)
    return np.array(features)  # Shape: (sequence_length, n, feature_dimension)


def compute_martingale_sum(features: np.ndarray) -> np.ndarray:
    """
    Computes the martingale sum at each timestep based on features.

    Parameters:
        features (np.ndarray): Array of shape (sequence_length, n, feature_dimension).

    Returns:
        martingale_sum (np.ndarray): Array of shape (sequence_length,) containing martingale sums.
    """
    # Placeholder implementation
    # Replace with actual martingale computation based on your framework
    # For demonstration, sum all feature values across nodes and features
    martingale_sum = features.sum(axis=(1, 2))  # Shape: (sequence_length,)
    return martingale_sum


def create_labels(
    martingale_sum: np.ndarray, threshold: float, prediction_horizon: int
) -> np.ndarray:
    """
    Creates binary labels indicating if the martingale sum exceeds the threshold within the prediction horizon.

    Parameters:
        martingale_sum (np.ndarray): Array of martingale sums at each timestep.
        threshold (float): Threshold to determine anomalies.
        prediction_horizon (int): Number of future steps to look ahead for threshold breach.

    Returns:
        labels (np.ndarray): Binary array indicating anomalies, shape (sequence_length,).
    """
    labels = np.zeros_like(martingale_sum, dtype=int)
    for t in range(len(martingale_sum)):
        # Check if any of the next M steps exceed the threshold
        future_steps = martingale_sum[t : t + prediction_horizon]
        if np.any(future_steps > threshold):
            labels[t] = 1
    return labels  # Shape: (sequence_length,)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalizes features across the entire dataset to have zero mean and unit variance.

    Parameters:
        features (np.ndarray): Array of shape (num_sequences, sequence_length, n, feature_dimension).

    Returns:
        normalized (np.ndarray): Normalized features.
    """
    # Flatten all features to compute global mean and std
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8  # Prevent division by zero
    normalized = (features - mean) / std
    return normalized


def split_dataset(sequences: np.ndarray, labels: np.ndarray, config: Dict) -> Dict:
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
        sequences (np.ndarray): Array of shape (num_sequences, sequence_length, n, feature_dimension).
        labels (np.ndarray): Array of shape (num_sequences, sequence_length).
        config (Dict): Configuration dictionary from BA_CONFIG.

    Returns:
        data_split (Dict): Dictionary containing train, validation, and test splits.
    """
    num_sequences = len(sequences)
    indices = np.arange(num_sequences)
    np.random.shuffle(indices)

    train_end = int(config["split_ratio"]["train"] * num_sequences)
    val_end = train_end + int(config["split_ratio"]["val"] * num_sequences)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train": {"sequences": sequences[train_idx], "labels": labels[train_idx]},
        "val": {"sequences": sequences[val_idx], "labels": labels[val_idx]},
        "test": {"sequences": sequences[test_idx], "labels": labels[test_idx]},
    }


def save_to_hdf5(data_split: Dict, config: Dict):
    """
    Saves the dataset splits to HDF5 files organized by train, validation, and test.

    Parameters:
        data_split (Dict): Dictionary containing train, validation, and test splits.
        config (Dict): Configuration dictionary from BA_CONFIG.
    """
    os.makedirs(config["output_dir"], exist_ok=True)
    for split_name, data in data_split.items():
        split_dir = os.path.join(config["output_dir"], split_name)
        os.makedirs(split_dir, exist_ok=True)
        with h5py.File(os.path.join(split_dir, "sequences.h5"), "w") as hf:
            hf.create_dataset("sequences", data=data["sequences"], compression="gzip")
        with h5py.File(os.path.join(split_dir, "labels.h5"), "w") as hf:
            hf.create_dataset("labels", data=data["labels"], compression="gzip")
        print(
            f"Saved {split_name} data: {data['sequences'].shape} sequences and {data['labels'].shape} labels."
        )


# -----------------------------------------------------------------------------#
#                               MAIN ENTRY POINT                               #
# -----------------------------------------------------------------------------#


def main():
    """
    Main entry point for graph sequence generation and data preparation.

    Steps:
    1. Initialize the GraphGenerator.
    2. Generate multiple graph sequences with feature extraction and anomaly labeling.
    3. Normalize the entire dataset.
    4. Split the dataset into training, validation, and testing sets.
    5. Save the processed data into HDF5 files.
    """
    print("\nGenerating Barabási-Albert Graph Sequences")
    print("----------------------------------------")
    print(f"Configuration:")
    print(f"  - Nodes per graph: {BA_CONFIG['n']}")
    print(f"  - Initial edge parameter: {BA_CONFIG['edges']['initial']}")
    print(
        f"  - Edge change range: {BA_CONFIG['edges']['change_min']} to {BA_CONFIG['edges']['change_max']}"
    )
    print(f"  - Sequence length: {BA_CONFIG['sequence_length']}")
    print(f"  - Number of sequences: {BA_CONFIG['num_sequences']}")
    print(f"  - Change points per sequence: {BA_CONFIG['change_points_per_sequence']}")
    print(f"  - Prediction horizon (M): {BA_CONFIG['prediction_horizon']}")
    print(f"  - Threshold tau: {BA_CONFIG['threshold_tau']}")
    print(f"  - Feature dimension per node: {BA_CONFIG['feature_dimension']}")
    print(f"  - Output directory: {BA_CONFIG['output_dir']}")

    generator = GraphGenerator(use_gpu=GPU_AVAILABLE)

    all_sequences = []
    all_labels = []

    print("\nGenerating sequences and extracting features...")
    for _ in tqdm(range(BA_CONFIG["num_sequences"]), desc="Sequences"):
        sequence_data = generate_ba_graphs_sequence(generator, BA_CONFIG)
        graphs = sequence_data["graphs"]
        features = extract_features_from_sequence(
            graphs, use_gpu=GPU_AVAILABLE
        )  # Shape: (sequence_length, n, feature_dim)
        martingale_sum = compute_martingale_sum(features)  # Shape: (sequence_length,)
        labels = create_labels(
            martingale_sum, BA_CONFIG["threshold_tau"], BA_CONFIG["prediction_horizon"]
        )  # Shape: (sequence_length,)
        all_sequences.append(features)
        all_labels.append(labels)

    all_sequences = np.array(
        all_sequences
    )  # Shape: (num_sequences, sequence_length, n, feature_dim)
    all_labels = np.array(all_labels)  # Shape: (num_sequences, sequence_length)

    print("\nNormalizing features...")
    all_sequences = normalize_features(all_sequences)

    print("\nSplitting dataset into train, validation, and test sets...")
    data_split = split_dataset(all_sequences, all_labels, BA_CONFIG)

    print("\nSaving datasets to HDF5 files...")
    save_to_hdf5(data_split, BA_CONFIG)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
