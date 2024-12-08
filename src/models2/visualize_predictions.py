import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from synthetic_data.create_graph_sequences import GraphType, GraphConfig, generate_graph_sequence
from src.models2.forecast import GraphTemporalForecaster
from src.utils.helpers import load_model


def create_synthetic_sequence() -> Tuple[List[nx.Graph], Dict]:
    """Create a synthetic graph sequence."""
    # Use BA graph type for demonstration
    config = GraphConfig.from_yaml(
        graph_type=GraphType.BA,
        config_path=str(project_root / "synthetic_data/configs/graph_config.yaml")
    )
    
    # Generate sequence
    result = generate_graph_sequence(config)
    
    # Convert adjacency matrices to NetworkX graphs if needed
    if isinstance(result['graphs'][0], np.ndarray):
        graphs = []
        for adj_matrix in result['graphs']:
            G = nx.from_numpy_array(adj_matrix)
            # Ensure nodes are integers
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)
        result['graphs'] = graphs
    
    return result['graphs'], result


def prepare_model_input(graphs: List[nx.Graph], config: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare input for the model from graph sequence."""
    seq_len = len(graphs)
    n_nodes = config['data']['n_nodes']
    
    # Create adjacency matrices
    adj_matrices = torch.zeros((seq_len, n_nodes, n_nodes))
    
    # Create feature matrices
    features = torch.zeros((seq_len, config['data']['n_features']))
    for i, g in enumerate(graphs):
        # Convert to NetworkX graph if needed
        if isinstance(g, np.ndarray):
            g = nx.from_numpy_array(g)
            adj_matrices[i] = torch.tensor(g)
        else:
            adj_matrices[i] = torch.tensor(nx.adjacency_matrix(g).todense())
        
        try:
            # Calculate all 6 centrality measures
            degrees = np.array([d for _, d in g.degree()])
            betweenness = np.array(list(nx.betweenness_centrality(g).values()))
            eigenvector = np.array(list(nx.eigenvector_centrality_numpy(g).values()))
            closeness = np.array(list(nx.closeness_centrality(g).values()))
            
            # Calculate SVD-based features
            adj_matrix = nx.adjacency_matrix(g).todense()
            U, S, Vt = np.linalg.svd(adj_matrix)
            svd_feat = S[0]  # Largest singular value
            lsvd_feat = np.sum(S)  # Sum of singular values
            
            # Store features in the same order as the original dataset
            features[i] = torch.tensor([
                degrees.mean(),        # Average degree
                betweenness.mean(),    # Average betweenness
                eigenvector.mean(),    # Average eigenvector
                closeness.mean(),      # Average closeness
                svd_feat,             # Largest singular value
                lsvd_feat,            # Sum of singular values
            ])
            
        except Exception as e:
            print(f"Warning: Error calculating features for graph {i}: {str(e)}")
            features[i] = torch.zeros(config['data']['n_features'])
    
    # Add batch dimension and convert to float32
    adj_matrices = adj_matrices.unsqueeze(0).float()
    features = features.unsqueeze(0).float()
    
    return {
        "adj_matrices": adj_matrices.to(device),
        "features": features.to(device)
    }


def calculate_graph_features(g: nx.Graph) -> List[float]:
    """Calculate all graph features consistently."""
    try:
        # Calculate centrality measures
        degrees = np.array([d for _, d in g.degree()])
        betweenness = np.array(list(nx.betweenness_centrality(g).values()))
        eigenvector = np.array(list(nx.eigenvector_centrality_numpy(g).values()))
        closeness = np.array(list(nx.closeness_centrality(g).values()))
        
        # Calculate SVD-based features
        adj_matrix = nx.adjacency_matrix(g).todense()
        U, S, Vt = np.linalg.svd(adj_matrix)
        svd_feat = S[0]  # Largest singular value
        lsvd_feat = np.sum(S)  # Sum of singular values
        
        return [
            degrees.mean(),        # Average degree
            betweenness.mean(),    # Average betweenness
            eigenvector.mean(),    # Average eigenvector
            closeness.mean(),      # Average closeness
            svd_feat,             # Largest singular value
            lsvd_feat,            # Sum of singular values
        ]
    except Exception as e:
        print(f"Warning: Error calculating features: {str(e)}")
        return [0.0] * 6


def visualize_predictions(
    actual_sequence: Dict,
    predicted_features: torch.Tensor,
    save_path: str = "prediction_visualization.png"
):
    """Visualize actual vs predicted sequences."""
    # Setup the plot
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True)
    axes = axes.flatten()
    
    # Add title
    fig.suptitle('Graph Sequence Analysis: Actual vs Predicted', fontsize=16, y=0.95)
    
    # Colors
    actual_color = '#2E86C1'      # Blue
    predicted_color = '#E74C3C'    # Red
    change_point_color = '#27AE60' # Green
    
    # Time points
    time_points = np.arange(len(actual_sequence['graphs']))
    forecast_points = np.arange(
        len(actual_sequence['graphs']) - predicted_features.shape[1],
        len(actual_sequence['graphs'])
    )
    
    # Calculate actual features
    actual_features = []
    for g in actual_sequence['graphs']:
        if isinstance(g, np.ndarray):
            g = nx.from_numpy_array(g)
        actual_features.append(calculate_graph_features(g))
    actual_features = np.array(actual_features)
    
    # Handle NaN values
    actual_features = np.nan_to_num(actual_features, 0.0)
    predicted_features = predicted_features.cpu().numpy()
    predicted_features = np.nan_to_num(predicted_features, 0.0)
    
    # Feature names matching the original dataset
    feature_names = [
        'Average Degree Centrality',
        'Average Betweenness Centrality',
        'Average Eigenvector Centrality',
        'Average Closeness Centrality',
        'Largest Singular Value',
        'Sum of Singular Values'
    ]
    
    # Plot each feature
    for i, (feat_actual, ax) in enumerate(zip(actual_features.T, axes)):
        # Plot actual and predicted
        ax.plot(time_points, feat_actual, '-', color=actual_color, 
                label='Actual', linewidth=2, alpha=0.8)
        ax.plot(forecast_points, predicted_features[0, :, i], '--', 
                color=predicted_color, label='Predicted', linewidth=2, alpha=0.8)
        
        # Add change points
        change_points_plotted = False
        for cp in actual_sequence['change_points']:
            ax.axvline(x=cp, color=change_point_color, linestyle=':', alpha=0.5,
                      label='Change Points' if not change_points_plotted else "")
            change_points_plotted = True
        
        # Customize plot
        ax.set_title(feature_names[i], fontsize=12, pad=10)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.1)
        
        # Set y-axis limits with padding
        ymin, ymax = feat_actual.min(), feat_actual.max()
        yrange = ymax - ymin
        ax.set_ylim([ymin - 0.1*yrange, ymax + 0.1*yrange])
    
    # Add sequence info
    info_text = (
        f"Graph Type: {actual_sequence['graph_type']}\n"
        f"Number of Nodes: {actual_sequence['n']}\n"
        f"Number of Changes: {actual_sequence['num_changes']}\n"
        f"Change Points: {actual_sequence['change_points']}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                      boxstyle='round,pad=0.5'))
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Load configuration
    with open("train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create synthetic sequence
    graphs, sequence_info = create_synthetic_sequence()
    
    # Load model
    model = GraphTemporalForecaster(
        config=config,
        num_nodes=config["data"]["n_nodes"],
        node_feat_dim=config["data"]["n_features"],
        device=device
    ).to(device)
    
    load_model(model, "checkpoints/best_model.pth", device)
    model.eval()
    
    # Prepare input
    input_data = prepare_model_input(graphs, config, device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(input_data)
        predictions = outputs["predictions"]
    
    # Visualize
    visualize_predictions(
        actual_sequence=sequence_info,
        predicted_features=predictions,
        save_path="sequence_prediction.png"
    )
    print("Visualization saved as 'sequence_prediction.png'")


if __name__ == "__main__":
    main() 