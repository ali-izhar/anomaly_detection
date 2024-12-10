import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset import DynamicGraphDataset
from link_predictor import DynamicLinkPredictor


def train_model(
    model,
    train_dataset,
    test_dataset,
    num_epochs=50,
    learning_rate=0.001,
    patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the dynamic link predictor model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience // 2)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0

        # Ensure targets are float tensors
        for i in range(len(train_dataset.targets)):
            if isinstance(train_dataset.targets[i], np.ndarray):
                train_dataset.targets[i] = torch.FloatTensor(train_dataset.targets[i])
            elif isinstance(train_dataset.targets[i], torch.Tensor):
                train_dataset.targets[i] = train_dataset.targets[i].float()
                
        for i in range(len(test_dataset.targets)):
            if isinstance(test_dataset.targets[i], np.ndarray):
                test_dataset.targets[i] = torch.FloatTensor(test_dataset.targets[i])
            elif isinstance(test_dataset.targets[i], torch.Tensor):
                test_dataset.targets[i] = test_dataset.targets[i].float()
        
        for time, snapshot in enumerate(train_dataset):
            # Move data to device
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(device)

            # Forward pass
            adj_pred = model(x, edge_index, edge_weight)

            # Ensure prediction and target have the same dtype
            adj_pred = adj_pred.float()
            y = y.float()

            # Compute loss
            loss = criterion(adj_pred, (y > 0).float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        val_batch_count = 0

        with torch.no_grad():
            for time, snapshot in enumerate(test_dataset):
                # Move data to device
                x = snapshot.x.to(device)
                edge_index = snapshot.edge_index.to(device)
                edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
                y = snapshot.y.to(device)

                # Forward pass
                adj_pred = model(x, edge_index, edge_weight)

                # Ensure prediction and target have the same dtype
                adj_pred = adj_pred.float()
                y = y.float()

                # Compute metrics
                val_loss += criterion(adj_pred, (y > 0).float()).item()
                val_preds.append(adj_pred.detach().cpu().numpy())
                val_targets.append((y > 0).detach().cpu().numpy())
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        
        # Ensure arrays are properly flattened for ROC AUC calculation
        val_preds_flat = np.concatenate([p.ravel() for p in val_preds])
        val_targets_flat = np.concatenate([t.ravel() for t in val_targets])
        val_auc = roc_auc_score(val_targets_flat, val_preds_flat)

        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)

        # Print progress
        print(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(best_model)
    return model, history


if __name__ == "__main__":
    # Create dataset
    dataset = DynamicGraphDataset()
    
    # Get train/test split for a sequence
    sequence_idx = 0  # Use first sequence for testing
    train_dataset, test_dataset = dataset.get_train_test_split(sequence_idx=sequence_idx)

    # Create model
    model = DynamicLinkPredictor(
        num_nodes=dataset.num_nodes,
        num_features=dataset.num_features,
        hidden_channels=64,
        num_layers=2,
        dropout=0.1,
    )

    # Train model
    trained_model, history = train_model(
        model,
        train_dataset,
        test_dataset,
        num_epochs=100,
        learning_rate=0.001,
        patience=15,
    )

    print("\nTraining completed!")
    print(f"Final validation AUC: {history['val_auc'][-1]:.4f}")
