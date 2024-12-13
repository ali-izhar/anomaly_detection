import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
from dataset import DynamicGraphDataset
from link_predictor import DynamicLinkPredictor
import numpy as np


def visualize_graphs(original_adj, predicted_adj, timestep, save_path=None):
    """
    Visualize original and predicted graphs side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create NetworkX graphs
    G_orig = nx.from_numpy_array(original_adj.cpu().numpy())
    G_pred = nx.from_numpy_array((predicted_adj > 0.5).cpu().numpy().astype(float))

    # Draw original graph
    pos = nx.spring_layout(G_orig)
    nx.draw(
        G_orig, pos, ax=ax1, node_color="lightblue", node_size=500, with_labels=True
    )
    ax1.set_title(f"Original Graph (t={timestep})")

    # Draw predicted graph
    nx.draw(
        G_pred, pos, ax=ax2, node_color="lightgreen", node_size=500, with_labels=True
    )
    ax2.set_title(f"Predicted Graph (t={timestep})")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def evaluate_predictions(
    model, dataset, device, phase="test", visualize=False, full_metrics=False
):
    """
    Evaluate model predictions and compute metrics.
    visualize: bool, whether to generate graph visualizations
    full_metrics: bool, whether to compute all metrics or just loss and AUC
    """
    model.eval()
    all_preds = []
    all_targets = []
    metrics = {}
    total_loss = 0
    batch_count = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for time, snapshot in enumerate(dataset):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = (
                snapshot.edge_attr.to(device)
                if snapshot.edge_attr is not None
                else None
            )
            y = snapshot.y.to(device)

            adj_pred = model(x, edge_index, edge_weight)
            loss = criterion(adj_pred, (y > 0).float())
            total_loss += loss.item()
            batch_count += 1

            all_preds.append(adj_pred.cpu())
            all_targets.append(y.cpu())

            if visualize and time in [0, len(all_preds) // 2, len(all_preds) - 1]:
                visualize_graphs(y, adj_pred, time, f"graphs_{phase}_t{time}.png")

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    all_preds_np = all_preds.numpy().ravel()
    all_targets_np = all_targets.numpy().ravel()
    preds_binary = (all_preds > 0.5).float().numpy().ravel()

    # Basic metrics (always computed)
    metrics["loss"] = total_loss / batch_count if batch_count > 0 else float("inf")
    metrics["auc"] = roc_auc_score(all_targets_np, all_preds_np)

    # Full metrics (only if requested)
    if full_metrics:
        metrics["precision"] = precision_score(
            all_targets_np, preds_binary, zero_division=0
        )
        metrics["recall"] = recall_score(all_targets_np, preds_binary, zero_division=0)
        metrics["f1"] = f1_score(all_targets_np, preds_binary, zero_division=0)

    return metrics


def train_model(
    model,
    dataset,
    sequence_indices,
    num_epochs=50,
    learning_rate=0.001,
    patience=10,
    temporal_periods=10,
    batch_size=32,
    val_ratio=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the model with temporal sequences.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience // 2)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Prepare data for all sequences
    train_data = []
    val_data = []

    for seq_idx in sequence_indices:
        # Get temporal windows for this sequence
        x, edge_indices, edge_weights, y = dataset.get_temporal_batch(
            seq_idx, temporal_periods=temporal_periods
        )

        # Split into train/val
        num_windows = len(x)
        num_val = int(num_windows * val_ratio)

        if num_val > 0:
            val_data.append(
                (
                    x[-num_val:],
                    edge_indices[-num_val:],
                    edge_weights[-num_val:],
                    y[-num_val:],
                )
            )
            train_data.append(
                (
                    x[:-num_val],
                    edge_indices[:-num_val],
                    edge_weights[:-num_val],
                    y[:-num_val],
                )
            )
        else:
            train_data.append((x, edge_indices, edge_weights, y))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # Training
        for seq_data in train_data:
            x, edge_indices, edge_weights, y = seq_data

            # Validate shapes
            print("\nInput Validation:")
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            print(f"Number of edge indices: {len(edge_indices)}")
            print(f"First edge index shape: {edge_indices[0].shape}")
            print(f"First edge weight shape: {edge_weights[0].shape}")

            # Process in batches
            num_batches = (len(x) + batch_size - 1) // batch_size
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, len(x))

                batch_x = x[start_idx:end_idx].to(device)
                batch_y = y[start_idx:end_idx].to(device)
                batch_edge_index = edge_indices[end_idx - 1].to(device)
                batch_edge_weight = edge_weights[end_idx - 1].to(device)

                # Validate batch shapes
                print("\nBatch Validation:")
                print(f"batch_x shape: {batch_x.shape}")
                print(f"batch_y shape: {batch_y.shape}")
                print(f"batch_edge_index shape: {batch_edge_index.shape}")
                print(f"batch_edge_weight shape: {batch_edge_weight.shape}")

                optimizer.zero_grad()
                adj_pred, _ = model(batch_x, batch_edge_index, batch_edge_weight)

                # Validate output shape
                print(f"adj_pred shape: {adj_pred.shape}")
                print(f"Expected shape: {batch_y.shape}")
                assert adj_pred.shape == batch_y.shape, "Prediction shape mismatch"

                loss = criterion(adj_pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

        avg_train_loss = total_loss / batch_count if batch_count > 0 else float("inf")

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for seq_data in val_data:
                x, edge_indices, edge_weights, y = seq_data

                # Process in batches
                num_batches = (len(x) + batch_size - 1) // batch_size
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, len(x))

                    batch_x = x[start_idx:end_idx].to(device)
                    batch_y = y[start_idx:end_idx].to(device)
                    batch_edge_index = edge_indices[end_idx - 1].to(device)
                    batch_edge_weight = edge_weights[end_idx - 1].to(device)

                    adj_pred, _ = model(batch_x, batch_edge_index, batch_edge_weight)

                    val_loss += criterion(adj_pred, batch_y).item()
                    val_preds.append(adj_pred.cpu())
                    val_targets.append(batch_y.cpu())

        val_loss /= len(val_data)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_auc = roc_auc_score(val_targets.numpy().ravel(), val_preds.numpy().ravel())

        print(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

    # Load best model
    model.load_state_dict(best_model)
    return model, history


if __name__ == "__main__":
    # Create dataset
    dataset = DynamicGraphDataset(variant="node_level")  # Use node-level features

    # Parameters
    temporal_periods = 10
    batch_size = 32

    # Get sequence indices for each graph type
    ba_indices = dataset.get_graph_type_indices("BA")
    er_indices = dataset.get_graph_type_indices("ER")
    nw_indices = dataset.get_graph_type_indices("NW")

    # Use a subset of sequences for training
    train_sequences = np.concatenate(
        [ba_indices[:50], er_indices[:50], nw_indices[:50]]
    )

    # Create model
    model = DynamicLinkPredictor(
        num_nodes=dataset.num_nodes,
        num_features=dataset.num_features,
        hidden_channels=64,
        num_layers=2,
        temporal_periods=temporal_periods,
        batch_size=batch_size,
    )

    # Train model
    trained_model, history = train_model(
        model,
        dataset,
        train_sequences,
        num_epochs=50,
        temporal_periods=temporal_periods,
        batch_size=batch_size,
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()
