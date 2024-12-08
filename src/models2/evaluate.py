# src/models2/evaluate.py

import torch
import yaml
import argparse
from src.models2.datasets import get_dataloaders
from src.models2.forecast import ForecastingModel
from src.utils.metrics import mse_loss, mae_loss, evaluate
from src.utils.helpers import load_model


def main(config_path, model_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    data_dir = "./data/processed"
    _, _, test_loader = get_dataloaders(data_dir, config)

    # Initialize and load model
    model = ForecastingModel(config).to(device)
    load_model(model, model_path, device)

    # Define loss
    loss_fn = mse_loss

    # Evaluate
    test_loss = evaluate(model, test_loader, device, loss_fn)
    print(f"Test MSE Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to the trained model."
    )
    args = parser.parse_args()
    main(args.config, args.model)
