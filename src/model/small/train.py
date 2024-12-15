# src/model/small/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .model_s import SmallSTGCN


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        features = batch["features"].to(device)
        edge_indices = [[e.to(device) for e in seq] for seq in batch["edge_indices"]]
        edge_weights = [[w.to(device) for w in seq] for seq in batch["edge_weights"]]
        targets = batch["targets"].to(device)

        # Use the first timestep's graph structure
        edge_index = edge_indices[0][0]  # Take first batch, first timestep
        edge_weight = edge_weights[0][0]  # Take first batch, first timestep

        optimizer.zero_grad()
        output = model(features, edge_index, edge_weight)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            edge_indices = [
                [e.to(device) for e in seq] for seq in batch["edge_indices"]
            ]
            edge_weights = [
                [w.to(device) for w in seq] for seq in batch["edge_weights"]
            ]
            targets = batch["targets"].to(device)

            edge_index = edge_indices[0][0]
            edge_weight = edge_weights[0][0]

            output = model(features, edge_index, edge_weight)
            loss = criterion(output, targets)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = SmallSTGCN(**model_config).to(device)

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Add custom logging for learning rate changes
    def log_lr(optimizer):
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            print(f"Learning rate changed to: {current_lr:.6f}")

    # Training loop
    for epoch in range(training_config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{training_config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        if old_lr != optimizer.param_groups[0]["lr"]:
            log_lr(optimizer)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= training_config["patience"]:
                print("Early stopping triggered")
                break

    return model
