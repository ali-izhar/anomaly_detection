import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging

from dataset import DynamicGraphDataset
from model import STGCNLinkPredictor, GConvGRULinkPredictor

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LinkPredictionTrainer:
    def __init__(
        self,
        model_type: str = "stgcn",
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "cuda",
    ):
        # Initialize dataset
        self.dataset = DynamicGraphDataset(variant="node_level")

        # Model selection
        if model_type == "stgcn":
            self.model = STGCNLinkPredictor(
                num_nodes=30,
                in_channels=6,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:  # gconv_gru
            self.model = GConvGRULinkPredictor(
                num_nodes=30,
                in_channels=6,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.device = device
        self.model = self.model.to(device)

        # Optimization
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = BCEWithLogitsLoss()
        self.scaler = GradScaler()

    def train_epoch(self, sequence_idx: int, temporal_periods: int = 10):
        self.model.train()

        # Get temporal batch
        x, edge_indices, edge_weights, y = self.dataset.get_temporal_batch(
            sequence_idx=sequence_idx, temporal_periods=temporal_periods, stride=1
        )

        logger.info(f"Batch shapes - x: {x.shape}, y: {y.shape}")
        logger.info(f"Number of edge indices: {len(edge_indices)}")

        # Move to device
        x = x.to(self.device)
        edge_indices = [e.to(self.device) for e in edge_indices]
        edge_weights = [w.to(self.device) for w in edge_weights]
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            pred = self.model(x, edge_indices[-1], edge_weights[-1])
            loss = self.criterion(pred, y)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def evaluate(self, sequence_idx: int, temporal_periods: int = 10):
        self.model.eval()

        with torch.no_grad():
            x, edge_indices, edge_weights, y = self.dataset.get_temporal_batch(
                sequence_idx=sequence_idx, temporal_periods=temporal_periods, stride=1
            )

            x = x.to(self.device)
            edge_indices = [e.to(self.device) for e in edge_indices]
            edge_weights = [w.to(self.device) for w in edge_weights]
            y = y.to(self.device)

            pred = self.model(x, edge_indices[-1], edge_weights[-1])
            loss = self.criterion(pred, y)

            # Calculate metrics
            pred_links = (pred > 0.5).float()
            accuracy = (pred_links == y).float().mean()

            return {"loss": loss.item(), "accuracy": accuracy.item()}


def main():
    # Training settings
    NUM_EPOCHS = 5
    TEMPORAL_PERIODS = 10
    MODEL_TYPE = "stgcn"  # or "gconv_gru"

    # Initialize trainer
    trainer = LinkPredictionTrainer(
        model_type=MODEL_TYPE,
        hidden_channels=64,
        num_layers=2,
        dropout=0.1,
        lr=0.001,
        device="cuda",
    )

    # Get sequence indices for train/val/test
    dataset = trainer.dataset
    num_sequences = dataset.num_sequences
    indices = np.random.permutation(num_sequences)

    train_idx = indices[: int(0.7 * num_sequences)]
    val_idx = indices[int(0.7 * num_sequences) : int(0.85 * num_sequences)]
    test_idx = indices[int(0.85 * num_sequences) :]

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # Training
        train_losses = []
        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            loss = trainer.train_epoch(idx, TEMPORAL_PERIODS)
            train_losses.append(loss)

        # Validation
        val_metrics = []
        for idx in val_idx:
            metrics = trainer.evaluate(idx, TEMPORAL_PERIODS)
            val_metrics.append(metrics)

        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean([m["loss"] for m in val_metrics])
        avg_val_acc = np.mean([m["accuracy"] for m in val_metrics])

        print(f"Epoch {epoch+1:03d}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {avg_val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(trainer.model.state_dict(), f"best_{MODEL_TYPE}_model.pt")

    # Test best model
    trainer.model.load_state_dict(torch.load(f"best_{MODEL_TYPE}_model.pt"))
    test_metrics = []
    for idx in test_idx:
        metrics = trainer.evaluate(idx, TEMPORAL_PERIODS)
        test_metrics.append(metrics)

    avg_test_loss = np.mean([m["loss"] for m in test_metrics])
    avg_test_acc = np.mean([m["accuracy"] for m in test_metrics])
    print(f"\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {avg_test_acc:.4f}")


if __name__ == "__main__":
    main()
