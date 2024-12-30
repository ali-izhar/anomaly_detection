import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from LSTM_GAN_GCN_torch import Generator, Discriminator, read_data, get_gcn_fact, gen_noise, get_binary_accuracy, get_mis_rate

def load_model(model_path):
    """Load saved model state and parameters."""
    model_state = torch.load(model_path)
    params = model_state['model_params']
    
    # Initialize models with saved parameters
    generator = Generator(params['node_num'], params['window_size'], params['gen_hid_num0'])
    discriminator = Discriminator(params['node_num'], params['disc_hid_num'])
    
    # Load saved states
    generator.load_state_dict(model_state['generator_state'])
    discriminator.load_state_dict(model_state['discriminator_state'])
    
    return generator, discriminator, params

def plot_adjacency_matrices(adj_true, adj_pred, title="Adjacency Matrix Comparison"):
    """Plot true vs predicted adjacency matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot true adjacency matrix
    sns.heatmap(adj_true, ax=ax1, cmap='Blues', cbar_kws={'label': 'Edge'}, 
                xticklabels=False, yticklabels=False)
    ax1.set_title('True Adjacency Matrix')
    
    # Plot predicted adjacency matrix
    sns.heatmap(adj_pred > 0.5, ax=ax2, cmap='Blues', cbar_kws={'label': 'Edge'},
                xticklabels=False, yticklabels=False)
    ax2.set_title('Predicted Adjacency Matrix')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def visualize_predictions(model_path, data_path, num_predictions=5):
    """Load model and visualize multiple predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    generator, _, params = load_model(model_path)
    generator.to(device)
    generator.eval()
    
    # Model parameters
    node_num = params['node_num']
    window_size = params['window_size']
    
    # Create predictions directory
    pred_dir = Path('predictions')
    pred_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_predictions} predictions...")
    for t in range(window_size, window_size + num_predictions):
        # Prepare data
        gcn_facts = []
        for k in range(t-window_size, t+1):
            adj = read_data(data_path, k, node_num)
            gcn_fact = get_gcn_fact(adj, node_num)
            gcn_facts.append(torch.FloatTensor(gcn_fact).to(device))
        
        # Generate prediction
        noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                       for _ in range(window_size + 1)]
        
        with torch.no_grad():
            output = generator(noise_inputs, gcn_facts)
            adj_est = output.cpu().numpy().reshape(node_num, node_num)
            adj_est = (adj_est + adj_est.T) / 2  # Ensure symmetry
            np.fill_diagonal(adj_est, 0)  # No self-loops
        
        # Get ground truth
        gnd = read_data(data_path, t+2, node_num)
        
        # Calculate metrics
        accuracy = get_binary_accuracy(adj_est, gnd)
        mis_rate = get_mis_rate(adj_est, gnd)
        
        # Plot and save comparison
        title = f"Time Step {t+2} (Accuracy: {accuracy:.4f}, Mismatch Rate: {mis_rate:.4f})"
        fig = plot_adjacency_matrices(gnd, adj_est, title)
        fig.savefig(pred_dir / f'prediction_{t+2}.png')
        plt.close(fig)
        
        print(f"Time step {t+2} - Accuracy: {accuracy:.4f}, Mismatch Rate: {mis_rate:.4f}")

def main():
    # Parameters
    model_path = 'models/lstm_gan_gcn_model.pt'  # or lstm_gan_gcn_drl2_model.pt for DrL2 version
    data_path = './data/SBM/edge_list'
    num_predictions = 5
    
    print("Starting visualization...")
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Number of predictions: {num_predictions}")
    
    visualize_predictions(model_path, data_path, num_predictions)
    print("\nVisualization complete! Check the 'predictions' directory for results.")

if __name__ == "__main__":
    main() 