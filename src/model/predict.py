import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import yaml

from dataset import DynamicGraphDataset
from small.model_s import SmallSTGCN
from medium.model_m import MediumASTGCN
from large.model_l import LargeGMAN


def load_model(model_type, model_path, config):
    """Load trained model."""
    if model_type == "small":
        model = SmallSTGCN(**config)
    elif model_type == "medium":
        model = MediumASTGCN(**config)
    else:  # large
        model = LargeGMAN(**config)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict_batch(model, batch, device):
    """Make predictions for a batch."""
    model.eval()
    with torch.no_grad():
        features = batch["features"].to(device)
        edge_index = batch["edge_indices"][0][0].to(device)
        edge_weight = (
            batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None
        )

        predictions = model(features, edge_index, edge_weight)
        predictions = torch.sigmoid(predictions)  # Add sigmoid for probabilities
        return predictions.cpu().numpy()


def plot_adjacency_comparison(true_adj, pred_adj, threshold=0.5, save_path=None):
    """Plot true vs predicted adjacency matrices with different thresholds."""
    thresholds = [0.3, 0.5, 0.7]  # Try different thresholds
    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(20, 5))

    # True adjacency
    sns.heatmap(true_adj, ax=axes[0], cmap="coolwarm", center=0.5, vmin=0, vmax=1)
    axes[0].set_title("True Adjacency Matrix")

    # Predicted adjacency at different thresholds
    for i, thresh in enumerate(thresholds, 1):
        pred_adj_binary = (pred_adj > thresh).astype(float)
        sns.heatmap(
            pred_adj_binary, ax=axes[i], cmap="coolwarm", center=0.5, vmin=0, vmax=1
        )
        axes[i].set_title(f"Predicted (threshold={thresh})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Higher quality
    plt.close()


def plot_metrics(true_labels, pred_probs, save_dir):
    """Plot ROC and Precision-Recall curves."""
    # ROC curve
    fpr, tpr, _ = roc_curve(true_labels.flatten(), pred_probs.flatten())
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_dir / "roc_curve.png")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        true_labels.flatten(), pred_probs.flatten()
    )
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(save_dir / "pr_curve.png")
    plt.close()


def load_config(config_path):
    """Load model configuration from file."""
    with open(config_path, "r") as f:
        content = f.read()
        # Find the model configuration section
        config_section = content.split("Model Configuration:\n")[1].split("\n\n")[0]
        return yaml.safe_load(config_section)


def evaluate_thresholds(true_labels, pred_probs, thresholds):
    """Evaluate model performance at different thresholds."""
    results = []
    for threshold in thresholds:
        pred_binary = (pred_probs > threshold).astype(float)
        cm = confusion_matrix(true_labels.flatten(), pred_binary.flatten())
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        results.append(
            {
                "threshold": threshold,
                "accuracy": (tp + tn) / total,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            }
        )
    return results


def find_optimal_threshold(true_labels, pred_probs):
    """Find optimal threshold using F-beta score."""
    beta = 1.5  # Adjusted to balance precision and recall better

    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0
    best_threshold = 0.5

    for threshold in thresholds:
        pred_binary = (pred_probs > threshold).astype(float)
        tn, fp, fn, tp = confusion_matrix(
            true_labels.flatten(), pred_binary.flatten()
        ).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F-beta score with additional structure penalty
        if precision + recall > 0:
            f_beta = (
                (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            )
            if f_beta > best_score:
                best_score = f_beta
                best_threshold = threshold

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

    # Make predictions
    all_preds = []
    all_targets = []

    for i, batch in enumerate(test_loader):
        if i >= 5:  # Only process first 5 samples for visualization
            break

        pred_adj = predict_batch(model, batch, device)
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

    # Plot metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    plot_metrics(all_targets, all_preds, output_dir)
    print("\nSaved ROC and PR curves")

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
