# src/models2/evaluate.py

import torch
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict

from src.models2.datasets import get_dataloaders
from src.models2.forecast import GraphTemporalForecaster
from src.utils.metrics import mse_loss, mae_loss, rmse_loss
from src.utils.helpers import load_model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for input_seq, target_seq in test_loader:
            # Move data to device
            input_data = {
                "adj_matrices": input_seq["adj_matrices"].to(device),
                "features": input_seq["features"].to(device)
            }
            targets = target_seq.to(device)

            # Get predictions
            outputs = model(input_data)
            predictions = outputs["predictions"]

            # Compute metrics
            total_mse += mse_loss(predictions, targets).item() * len(predictions)
            total_mae += mae_loss(predictions, targets).item() * len(predictions)
            num_samples += len(predictions)

    # Calculate average metrics
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    rmse = rmse_loss(torch.tensor(avg_mse))

    return {
        "MSE": avg_mse,
        "MAE": avg_mae,
        "RMSE": rmse
    }


def main(config_path: str, model_path: str) -> None:
    """Main evaluation function.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Prepare data loader
    data_dir = Path("dataset")  # Updated to correct path
    _, _, test_loader = get_dataloaders(
        data_dir, 
        config,
        num_workers=1,  # Single worker for evaluation
        pin_memory=torch.cuda.is_available()
    )

    # Initialize and load model
    model = GraphTemporalForecaster(
        config=config,
        num_nodes=config["data"]["n_nodes"],  # From config
        node_feat_dim=config["data"]["n_features"],  # From config
        device=device
    ).to(device)
    
    load_model(model, model_path, device)
    logging.info(f"Loaded model from {model_path}")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Log results
    logging.info("\nTest Set Metrics:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/models2/train_config.yaml",  # Updated config path
        help="Path to the config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the trained model",
    )
    args = parser.parse_args()
    main(args.config, args.model)
