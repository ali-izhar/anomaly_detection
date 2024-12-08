import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.spatiotemporal import SpatioTemporalPredictor, STModelConfig
from models.graph_dataset import GraphSequenceDataset, GraphDataConfig, create_dataloader
from train import setup_device, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(config_path: str, model_path: Path, device: torch.device):
    """Load trained model with configuration."""
    config = load_config(config_path)
    
    # Create model with same configuration
    model_config = STModelConfig(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        gnn_type=str(config["model"]["gnn_type"]),
        attention_heads=int(config["model"]["attention_heads"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    )
    
    model = SpatioTemporalPredictor(model_config)
    
    # Try different file extensions
    possible_paths = [
        model_path.with_suffix('.pt'),
        model_path.with_suffix('.model'),
        model_path
    ]
    
    loaded = False
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading model from {path}")
            try:
                # Try loading as full checkpoint first
                checkpoint = torch.load(path)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {str(e)}")
                continue
    
    if not loaded:
        raise FileNotFoundError(
            f"Could not find model file in any of these locations: {[str(p) for p in possible_paths]}"
        )
    
    model = model.to(device)
    model.eval()
    
    return model

def visualize_predictions(model, test_loader, device, save_dir: Path, num_samples=5):
    """Visualize model predictions against ground truth."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
                
            # Prepare input
            x = {k: v.to(device) for k, v in batch["x"].items()}
            y = {k: v.to(device) for k, v in batch["y"].items()}
            adj = batch["adj"].to(device)
            
            # Get predictions
            predictions = model(x, adj)
            
            # Plot each feature type
            for feat_name in predictions:
                pred = predictions[feat_name].cpu().numpy()
                true = y[feat_name].cpu().numpy()
                
                # Get dimensions
                batch_size, seq_len, num_nodes = pred.shape[:3]
                
                # Create figure with multiple subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Feature: {feat_name} - Sample {batch_idx}')
                
                # Plot heatmaps of the last timestep
                ax = axes[0, 0]
                sns.heatmap(pred[0, -1].reshape(num_nodes, -1), ax=ax)
                ax.set_title('Prediction (Last Timestep)')
                
                ax = axes[0, 1]
                sns.heatmap(true[0, -1].reshape(num_nodes, -1), ax=ax)
                ax.set_title('Ground Truth (Last Timestep)')
                
                # Plot time series for a few nodes
                nodes_to_plot = [0, 25, 50, 75, 99]  # Sample nodes
                
                # For each feature dimension
                feature_dims = pred.shape[-1] if len(pred.shape) > 3 else 1
                
                ax = axes[1, 0]
                for node in nodes_to_plot:
                    if feature_dims == 1:
                        ax.plot(pred[0, :, node], label=f'Node {node}')
                    else:
                        for dim in range(feature_dims):
                            ax.plot(pred[0, :, node, dim], 
                                  label=f'Node {node} Dim {dim}',
                                  linestyle=['solid', 'dashed'][dim % 2])
                ax.set_title('Predicted Time Series')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                ax = axes[1, 1]
                for node in nodes_to_plot:
                    if feature_dims == 1:
                        ax.plot(true[0, :, node], label=f'Node {node}')
                    else:
                        for dim in range(feature_dims):
                            ax.plot(true[0, :, node, dim],
                                  label=f'Node {node} Dim {dim}',
                                  linestyle=['solid', 'dashed'][dim % 2])
                ax.set_title('True Time Series')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{batch_idx}_{feat_name}.png', 
                          bbox_inches='tight', dpi=300)
                plt.close()
                
                # Additional plot: Feature evolution over time
                plt.figure(figsize=(15, 5))
                plt.title(f'Feature Evolution - {feat_name} - Node 0')
                
                if feature_dims > 1:
                    for dim in range(feature_dims):
                        plt.plot(pred[0, :, 0, dim], 
                               label=f'Pred Dim {dim}',
                               linestyle='solid')
                        plt.plot(true[0, :, 0, dim],
                               label=f'True Dim {dim}',
                               linestyle='dashed')
                else:
                    plt.plot(pred[0, :, 0], label='Prediction')
                    plt.plot(true[0, :, 0], label='Ground Truth')
                
                plt.xlabel('Time Step')
                plt.ylabel('Feature Value')
                plt.legend()
                plt.grid(True)
                plt.savefig(save_dir / f'evolution_{batch_idx}_{feat_name}.png',
                          bbox_inches='tight', dpi=300)
                plt.close()

def plot_loss_curves(tensorboard_path: Path, save_dir: Path):
    """Plot training and validation loss curves from tensorboard data."""
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tensorboard data
    event_acc = EventAccumulator(str(tensorboard_path))
    event_acc.Reload()
    
    # Get loss data
    train_steps, train_values = zip(*[(s.step, s.value) for s in event_acc.Scalars('Loss/train')])
    val_steps, val_values = zip(*[(s.step, s.value) for s in event_acc.Scalars('Loss/val')])
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, label='Train Loss')
    plt.plot(val_steps, val_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'loss_curves.png')
    plt.close()

def main():
    # Setup
    config_path = "train_config.yaml"
    model_dir = Path("outputs")
    model_path = model_dir / "best_model"  # Remove .pt extension here
    tensorboard_path = model_dir / "runs"
    vis_dir = model_dir / "visualizations"
    device = setup_device()
    
    try:
        # Load model
        model = load_trained_model(config_path, model_path, device)
        
        # Create test dataset
        config = load_config(config_path)
        data_config = GraphDataConfig(
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            forecast_horizon=int(config["data"]["forecast_horizon"]),
            batch_size=1,  # Use batch size 1 for visualization
            use_centrality=bool(config["data"]["use_centrality"]),
            use_spectral=bool(config["data"]["use_spectral"]),
            enable_augmentation=False,
            noise_level=0.0,
        )
        
        test_dataset = GraphSequenceDataset(
            config["paths"]["data_dir"], "test", data_config
        )
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        # Generate visualizations
        logger.info("Generating prediction visualizations...")
        visualize_predictions(model, test_loader, device, vis_dir)
        
        # Plot loss curves
        logger.info("Plotting loss curves...")
        plot_loss_curves(tensorboard_path, vis_dir)
        
        logger.info(f"Visualizations saved to {vis_dir}")
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 