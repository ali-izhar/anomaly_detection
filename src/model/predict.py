import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import yaml
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar as np_scalar
from numpy import dtype as np_dtype

# Add safe globals
add_safe_globals([np_scalar, np_dtype])

from dataset import DynamicGraphDataset
from small.model_s import SmallSTGCN
from medium.model_m import MediumASTGCN
from large.model_l import LargeGMAN


def load_model(model_type, model_path, config):
    """Load trained model with proper safety checks."""
    if model_type == "small":
        model = SmallSTGCN(**config)
    elif model_type == "medium":
        model = MediumASTGCN(**config)
    else:  # large
        model = LargeGMAN(**config)

    try:
        # First try with weights_only=True and safe globals
        checkpoint = torch.load(model_path, weights_only=True)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True: {str(e)}")
        print("Attempting to load with map_location and weights_only=False...")
        # Fallback to safe loading with map_location
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract only the state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    return model


def predict_batch(model, batch, device):
    """Make predictions for a batch with proper normalization and safety checks."""
    model.eval()
    with torch.no_grad():
        features = batch["features"].to(device)
        edge_index = batch["edge_indices"][0][0].to(device)
        edge_weight = batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None

        # Get predictions (already normalized to [0, 1])
        predictions = model(features, edge_index, edge_weight)
        
        # Safety check for NaN values
        if torch.isnan(predictions).any():
            print("Warning: NaN values detected in predictions, replacing with safe values")
            predictions = torch.nan_to_num(predictions, nan=0.5)
        
        # Ensure predictions are in valid range
        predictions = torch.clamp(predictions, min=0.0, max=1.0)
        
        return predictions.cpu().numpy()


def plot_adjacency_comparison(true_adj, pred_adj, threshold=0.5, save_path=None):
    """Plot true vs predicted adjacency matrices with improved visualization."""
    thresholds = [0.3, 0.43, 0.5]  # Include optimal threshold from previous results
    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(20, 10))
    
    # Raw predictions visualization
    sns.heatmap(true_adj, ax=axes[0, 0], cmap="RdBu_r", center=0.5, vmin=0, vmax=1)
    axes[0, 0].set_title("True Adjacency Matrix")
    
    sns.heatmap(pred_adj, ax=axes[1, 0], cmap="RdBu_r", center=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title(f"Raw Predictions\nMean={pred_adj.mean():.3f}, Std={pred_adj.std():.3f}")

    # Thresholded predictions
    for i, thresh in enumerate(thresholds, 1):
        pred_adj_binary = (pred_adj > thresh).astype(float)
        
        # Calculate metrics
        sparsity = (pred_adj_binary == 0).mean()
        pos_ratio = (pred_adj_binary == 1).mean()
        true_pos = ((pred_adj_binary == 1) & (true_adj == 1)).sum() / (true_adj == 1).sum()
        
        sns.heatmap(pred_adj_binary, ax=axes[0, i], cmap="RdBu_r", center=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Threshold={thresh:.2f}\nSparsity={sparsity:.3f}, TP Rate={true_pos:.3f}")
        
        # Show difference from true adjacency
        diff = pred_adj_binary - true_adj
        sns.heatmap(diff, ax=axes[1, i], cmap="RdBu_r", center=0, vmin=-1, vmax=1)
        axes[1, i].set_title(f"Differences\nPos Ratio={pos_ratio:.3f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics(true_labels, pred_probs, save_dir):
    """Plot ROC, Precision-Recall curves, and sparsity analysis with safety checks."""
    # Safety check for NaN values
    if np.isnan(pred_probs).any():
        print("Warning: NaN values detected in predictions, cleaning up...")
        pred_probs = np.nan_to_num(pred_probs, nan=0.5)
    
    # Ensure predictions are in valid range
    pred_probs = np.clip(pred_probs, 0, 1)
    
    # Original ROC and PR curves
    fpr, tpr, _ = roc_curve(true_labels.flatten(), pred_probs.flatten())
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(true_labels.flatten(), pred_probs.flatten())
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(save_dir / "pr_curve.png")
    plt.close()
    
    # Additional sparsity analysis with safety checks
    thresholds = np.linspace(0.3, 0.7, 81)  # Focus on more relevant range
    sparsity_values = []
    pos_ratios = []
    
    for threshold in thresholds:
        pred_binary = (pred_probs > threshold).astype(float)
        sparsity_values.append((pred_binary == 0).mean())
        pos_ratios.append((pred_binary == 1).mean())
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sparsity_values, label='Sparsity')
    plt.plot(thresholds, pos_ratios, label='Positive Ratio')
    plt.axhline(y=0.77, color='r', linestyle='--', label='Target Sparsity')
    plt.axvline(x=0.43, color='g', linestyle='--', label='Optimal Threshold')
    plt.xlabel("Threshold")
    plt.ylabel("Ratio")
    plt.title("Sparsity Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "sparsity_analysis.png")
    plt.close()


def load_config(config_path):
    """Load model configuration from file."""
    with open(config_path, "r") as f:
        content = f.read()
        # Find the model configuration section
        config_section = content.split("Model Configuration:\n")[1].split("\n\n")[0]
        return yaml.safe_load(config_section)


def evaluate_thresholds(true_labels, pred_probs, thresholds):
    """Evaluate model performance with additional sparsity metrics."""
    results = []
    target_sparsity = 0.77  # From BA graph properties
    
    # If no thresholds provided, use relevant range
    if len(thresholds) == 0:
        thresholds = np.linspace(0.3, 0.7, 9)
    
    for threshold in thresholds:
        pred_binary = (pred_probs > threshold).astype(float)
        current_sparsity = (pred_binary == 0).mean()
        sparsity_error = abs(current_sparsity - target_sparsity)
        
        cm = confusion_matrix(true_labels.flatten(), pred_binary.flatten())
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        results.append({
            "threshold": threshold,
            "accuracy": (tp + tn) / total,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "sparsity": current_sparsity,
            "sparsity_error": sparsity_error,
            "true_positives": tp,
            "false_positives": fp
        })
    return results


def find_optimal_threshold(true_labels, pred_probs):
    """Find optimal threshold using F-beta score with sparsity constraints."""
    beta = 1.5  # Balance between precision and recall
    target_sparsity = 0.77  # Target sparsity from BA graph properties
    sparsity_tolerance = 0.05  # Acceptable deviation from target sparsity

    thresholds = np.linspace(0.3, 0.7, 81)  # Focus on more relevant range
    best_score = 0
    best_threshold = 0.43  # Start from previously found optimal
    best_sparsity_diff = float('inf')

    for threshold in thresholds:
        pred_binary = (pred_probs > threshold).astype(float)
        current_sparsity = (pred_binary == 0).mean()
        sparsity_diff = abs(current_sparsity - target_sparsity)
        
        # Stronger sparsity constraint
        if sparsity_diff > sparsity_tolerance:
            continue

        tn, fp, fn, tp = confusion_matrix(true_labels.flatten(), pred_binary.flatten()).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F-beta score with stronger sparsity penalty
        if precision + recall > 0:
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            # Stronger penalty for sparsity deviation
            f_beta *= (1 - sparsity_diff * 2)
            
            if f_beta > best_score:
                best_score = f_beta
                best_threshold = threshold
                best_sparsity_diff = sparsity_diff

    return best_threshold, best_score


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "medium"  # or "small" or "large"
    exp_dir = Path("experiments") / model_type

    # Create directories if they don't exist
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check for experiment runs
    runs = list(exp_dir.glob("*"))
    if not runs:
        raise ValueError(
            f"No experiment runs found in {exp_dir}. Please run training first."
        )

    latest_run = sorted(runs)[-1]  # Get most recent run
    print(f"Using latest run: {latest_run}")

    # Load model and config
    model_path = latest_run / "best_model.pth"
    config_path = latest_run / "model_architecture.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load model configuration
    model_config = load_config(config_path)
    print("\nModel Configuration:")
    print(yaml.dump(model_config))

    # Load dataset
    dataset = DynamicGraphDataset(variant="node_level")

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for visualization
        shuffle=False,
        num_workers=0,
        collate_fn=dataset._collate_fn,
    )

    # Load model
    model = load_model(model_type, model_path, model_config)
    model = model.to(device)

    # Create output directory
    output_dir = latest_run / "predictions"
    output_dir.mkdir(exist_ok=True)

    print(f"\nMaking predictions...")
    print(f"Output directory: {output_dir}")

    # Make predictions with additional error handling
    all_preds = []
    all_targets = []
    
    for i, batch in enumerate(test_loader):
        if i >= 5:  # Only process first 5 samples for visualization
            break

        try:
            pred_adj = predict_batch(model, batch, device)
            if np.isnan(pred_adj).any():
                print(f"Warning: NaN values in predictions for sample {i}, skipping...")
                continue
                
            true_adj = batch["targets"].numpy()

            # Plot comparison
            plot_adjacency_comparison(
                true_adj[0],  # First in batch
                pred_adj[0],  # First in batch
                save_path=output_dir / f"comparison_{i}.png",
            )
            print(f"Saved comparison plot {i}")

            all_preds.append(pred_adj)
            all_targets.append(true_adj)
            
        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
            continue

    if not all_preds:
        print("No valid predictions generated!")
        return

    # Plot metrics with cleaned data
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    try:
        plot_metrics(all_targets, all_preds, output_dir)
        print("\nSaved ROC and PR curves")
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")

    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    results = evaluate_thresholds(all_targets, all_preds, thresholds)

    # Find optimal threshold
    optimal_threshold, f_beta_score = find_optimal_threshold(all_targets, all_preds)
    print(f"\nOptimal threshold (F-beta): {optimal_threshold:.2f}")
    print(f"F-beta score: {f_beta_score:.4f}")

    # Use optimal threshold for final evaluation
    final_results = evaluate_thresholds(all_targets, all_preds, [optimal_threshold])[0]
    print("\nMetrics at optimal threshold:")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall: {final_results['recall']:.4f}")
    print(f"F1 Score: {final_results['f1']:.4f}")
    print(f"False Positive Rate: {final_results['fpr']:.4f}")

    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        plt.plot(
            [r["threshold"] for r in results],
            [r[metric] for r in results],
            label=metric.capitalize(),
        )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "threshold_analysis.png")
    plt.close()

    # Save example predictions
    np.save(
        output_dir / "sample_predictions.npy",
        {"predictions": all_preds, "targets": all_targets},
    )
    print(f"\nSaved predictions to {output_dir / 'sample_predictions.npy'}")


if __name__ == "__main__":
    main()
