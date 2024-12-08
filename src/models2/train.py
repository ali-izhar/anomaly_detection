# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from src.models2.datasets import get_dataloaders
from src.models2.forecast import ForecastingModel
from src.utils.metrics import mse_loss, evaluate
from src.utils.helpers import save_model, count_parameters


def main(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    data_dir = "./data/processed"
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, config)

    # Initialize model
    model = ForecastingModel(config).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters.")

    # Define optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_fn = mse_loss

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    epochs_no_improve = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_seq, target_seq = batch
            adj_matrices = input_seq["adj_matrices"].to(device)  # (batch, T, N, N)
            features = input_seq["features"].to(device)  # (batch, T, 6)
            targets = target_seq.to(device)  # (batch, m, 6)

            optimizer.zero_grad()
            predictions = model(adj_matrices, features)  # (batch, m, 6)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * adj_matrices.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        avg_val_loss = evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}"
        )

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            save_model(model, "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    main(args.config)
