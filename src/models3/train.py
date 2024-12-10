import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
from dataset import DynamicGraphDataset
from link_predictor import DynamicLinkPredictor


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
    nx.draw(G_orig, pos, ax=ax1, node_color='lightblue', 
            node_size=500, with_labels=True)
    ax1.set_title(f'Original Graph (t={timestep})')
    
    # Draw predicted graph
    nx.draw(G_pred, pos, ax=ax2, node_color='lightgreen',
            node_size=500, with_labels=True)
    ax2.set_title(f'Predicted Graph (t={timestep})')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def evaluate_predictions(model, dataset, device, phase='test', visualize=False, full_metrics=False):
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
            edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(device)
            
            adj_pred = model(x, edge_index, edge_weight)
            loss = criterion(adj_pred, (y > 0).float())
            total_loss += loss.item()
            batch_count += 1
            
            all_preds.append(adj_pred.cpu())
            all_targets.append(y.cpu())
            
            if visualize and time in [0, len(all_preds)//2, len(all_preds)-1]:
                visualize_graphs(y, adj_pred, time, f'graphs_{phase}_t{time}.png')
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    
    all_preds_np = all_preds.numpy().ravel()
    all_targets_np = all_targets.numpy().ravel()
    preds_binary = (all_preds > 0.5).float().numpy().ravel()
    
    # Basic metrics (always computed)
    metrics['loss'] = total_loss / batch_count if batch_count > 0 else float('inf')
    metrics['auc'] = roc_auc_score(all_targets_np, all_preds_np)
    
    # Full metrics (only if requested)
    if full_metrics:
        metrics['precision'] = precision_score(all_targets_np, preds_binary, zero_division=0)
        metrics['recall'] = recall_score(all_targets_np, preds_binary, zero_division=0)
        metrics['f1'] = f1_score(all_targets_np, preds_binary, zero_division=0)
    
    return metrics


def train_model(model, train_dataset, test_dataset, num_epochs=50,
                learning_rate=0.001, patience=10,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the model and evaluate performance.
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
        
        for time, snapshot in enumerate(train_dataset):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(device)
            
            optimizer.zero_grad()
            adj_pred = model(x, edge_index, edge_weight)
            loss = criterion(adj_pred, (y > 0).float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_loss / batch_count
        
        # Validation (basic metrics only during training)
        val_metrics = evaluate_predictions(model, test_dataset, device, 'val', 
                                         visualize=False, full_metrics=False)
        val_loss = val_metrics['loss']
        val_auc = val_metrics['auc']
        
        # Print progress (only loss and AUC)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
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
    
    # Final evaluation with full metrics and visualization
    print("\nFinal Test Set Evaluation:")
    test_metrics = evaluate_predictions(model, test_dataset, device, 'test', 
                                      visualize=True, full_metrics=True)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
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
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
