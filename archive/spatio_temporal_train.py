# archive/spatio_temporal_train.py

"""Main training script for GCN-LSTM model on soccer player movement data."""


from pathlib import Path
import sys
import pickle
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.optim.lr_scheduler import _LRScheduler

from .spatio_temporal import GCNLSTMModel
from .gcn_lstm import (
    DataLoader,
    GraphConstructor,
    SequenceProcessor,
    ResultAnalyzer,
)


class ExperimentConfig:
    """Configuration manager for experiments."""

    def __init__(self, config_path: Path):
        """Initialize experiment configuration.

        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model_params = self._process_params(self.config["experiment_params"])
        self.training_params = self._process_params(self.config["experiment_params"])

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from yaml file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numeric string values to appropriate numeric types."""
        processed = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Try to convert string to numeric if possible
                try:
                    # Try float conversion first
                    float_val = float(value)
                    # If it's a whole number, convert to int
                    if float_val.is_integer():
                        processed[key] = int(float_val)
                    else:
                        processed[key] = float_val
                except ValueError:
                    # If conversion fails, keep original string
                    processed[key] = value
            else:
                processed[key] = value
        return processed


class DataManager:
    """Handles data loading and preprocessing."""

    def __init__(self, data_folder: Path, model_params: Dict[str, Any]):
        """Initialize data manager.

        Args:
            data_folder: Path to data directory
            model_params: Model parameters dictionary
        """
        self.data_folder = data_folder
        self.model_params = model_params

    def load_and_preprocess(self) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Load and preprocess all data."""
        # Load raw data
        soccer_data = self.data_folder / "soccer"
        soccer_matrix_filename = (
            soccer_data / f"Coords_Influence_{self.model_params['game']}.npz"
        )
        pitch_control_filename = (
            soccer_data / f"pitch_control_{self.model_params['game']}.npz"
        )

        data = DataLoader.load_data(soccer_matrix_filename)
        pitch_control_data = DataLoader.load_data(pitch_control_filename)

        # Process features
        X = (
            DataLoader.preprocess_data(
                data, pitch_control_data=pitch_control_data, feature="pitch_control"
            )
            if self.model_params["feature_name"] == "pitch_control"
            else DataLoader.preprocess_data(data, feature="influence")
        )

        # Create graph structure and process sequences
        processed_data = self._process_graph_data(X)
        return self._split_data(processed_data)

    def _process_graph_data(self, X: np.ndarray) -> Dict[str, Any]:
        """Process graph structure and create sequences."""
        # Create graph structure
        edge_indices, edge_distances = (
            GraphConstructor.create_edge_indices_and_distances(
                X, distance_threshold=self.model_params["distance_threshold"]
            )
        )

        max_edges = max(ei.size(1) for ei in edge_indices)
        padded_edge_indices, padded_edge_distances = GraphConstructor.pad_edge_data(
            edge_indices, edge_distances, max_edges
        )

        # Normalize features
        feature = X[:, :, 0]
        X_influence, _, _ = SequenceProcessor.normalize_feature(feature)

        # Generate sequences
        X_sequences, y_sequences = SequenceProcessor.create_sequences(
            X_influence,
            X_influence,
            self.model_params["n_steps_in"],
            self.model_params["n_steps_out"],
        )

        # Align edge data with sequences
        edge_indices_sequences = [
            padded_edge_indices[i : (i + self.model_params["n_steps_in"])]
            for i in range(len(X_sequences))
        ]
        edge_distances_sequences = [
            padded_edge_distances[i : (i + self.model_params["n_steps_in"])]
            for i in range(len(X_sequences))
        ]

        return {
            "X_sequences": X_sequences,
            "y_sequences": y_sequences,
            "edge_indices_sequences": edge_indices_sequences,
            "edge_distances_sequences": edge_distances_sequences,
        }

    def _split_data(
        self, processed_data: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Split data into train, validation, and test sets."""
        # Split into train and test
        (
            X_train,
            y_train,
            edge_indices_train,
            edge_distances_train,
            X_test,
            y_test,
            edge_indices_test,
            edge_distances_test,
        ) = SequenceProcessor.split_data(
            processed_data["X_sequences"],
            processed_data["y_sequences"],
            processed_data["edge_indices_sequences"],
            processed_data["edge_distances_sequences"],
            train_split=0.8,
        )

        # Convert to tensors
        X_train, y_train, X_test, y_test = SequenceProcessor.convert_to_tensors(
            X_train, y_train, X_test, y_test
        )

        # Create validation split
        n_train_new = int(0.8 * len(X_train))

        train_data = {
            "X": X_train[:n_train_new],
            "y": y_train[:n_train_new],
            "edge_indices": edge_indices_train[:n_train_new],
            "edge_weights": edge_distances_train[:n_train_new],
        }

        val_data = {
            "X": X_train[n_train_new:],
            "y": y_train[n_train_new:],
            "edge_indices": edge_indices_train[n_train_new:],
            "edge_weights": edge_distances_train[n_train_new:],
        }

        test_data = {
            "X": X_test,
            "y": y_test,
            "edge_indices": edge_indices_test,
            "edge_weights": edge_distances_test,
        }

        return train_data, val_data, test_data


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: _LRScheduler,
        device: torch.device,
    ):
        """Initialize model trainer.

        Args:
            model: The GCN-LSTM model
            optimizer: PyTorch optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            device: Torch device
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, train_data: Dict[str, torch.Tensor]) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(
            train_data["X"].to(self.device),
            [[ei.to(self.device) for ei in seq] for seq in train_data["edge_indices"]],
            [[ew.to(self.device) for ew in seq] for seq in train_data["edge_weights"]],
        )
        loss = self.criterion(outputs, train_data["y"].to(self.device))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, val_data: Dict[str, torch.Tensor]) -> float:
        """Perform validation."""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(
                val_data["X"].to(self.device),
                [
                    [ei.to(self.device) for ei in seq]
                    for seq in val_data["edge_indices"]
                ],
                [
                    [ew.to(self.device) for ew in seq]
                    for seq in val_data["edge_weights"]
                ],
            )
            val_loss = self.criterion(val_outputs, val_data["y"].to(self.device))
        return val_loss.item()

    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        num_epochs: int,
        patience: int,
    ) -> Dict[str, Any]:
        """Train the model with early stopping."""
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_data)
            val_loss = self.validate(val_data)

            self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping triggered!")
                    break

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "stopped_epoch": epoch + 1,
        }

    def test(self, test_data: Dict[str, torch.Tensor]) -> Tuple[float, torch.Tensor]:
        """Evaluate model on test data."""
        self.model.eval()
        with torch.no_grad():
            test_edge_indices = [
                [ei.to(self.device) for ei in seq] for seq in test_data["edge_indices"]
            ]
            test_edge_weights = [
                [ew.to(self.device) for ew in seq] for seq in test_data["edge_weights"]
            ]

            test_outputs = self.model(
                test_data["X"].to(self.device), test_edge_indices, test_edge_weights
            )
            test_loss = self.criterion(test_outputs, test_data["y"].to(self.device))

        return test_loss.item(), test_outputs


def save_results(
    results_folder: Path,
    model_params: Dict[str, Any],
    test_data: Dict[str, torch.Tensor],
    test_outputs: torch.Tensor,
    test_loss: float,
    training_history: Dict[str, Any],
) -> None:
    """Save experiment results."""
    results_folder.mkdir(exist_ok=True)

    model_parameters = {
        "model": model_params["model_name"],
        "game": model_params["game"],
        "y_true": test_data["y"].cpu().numpy(),
        "y_pred": test_outputs.cpu().numpy(),
        "train_loss": training_history["train_losses"],
        "val_loss": training_history["val_losses"],
        "test_loss": test_loss,
        "model_config": model_params,
    }

    filename = (
        results_folder / f"{model_params['model_name']}_{model_params['game']}.pkl"
    )
    with open(filename, "wb") as file:
        pickle.dump(model_parameters, file)

    print(f"Model parameters saved to {filename}")


def visualize_results(
    results_folder: Path, visualization_config: Dict[str, Any]
) -> None:
    """Generate visualizations based on config."""
    if not visualization_config.get("enabled", False):
        return

    # Load all results
    results = ResultAnalyzer.load_results(results_folder)

    # Generate configured plots
    for plot_config in visualization_config.get("plots", []):
        ResultAnalyzer.plot_residue(
            results=results,
            title=plot_config["title"],
            filter_game=plot_config.get("filter_game"),
            filter_models=plot_config.get("filter_models"),
        )


def main():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Initialize configuration
    config = ExperimentConfig(project_root / "tmp" / "gcn_config.yaml")

    # Setup data
    data_manager = DataManager(project_root / "data", config.model_params)
    train_data, val_data, test_data = data_manager.load_and_preprocess()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = GCNLSTMModel(
        num_node_features=config.model_params["num_node_features"],
        gcn_out_features=config.model_params["gcn_out_features"],
        lstm_hidden_size=config.model_params["lstm_hidden_size"],
        num_nodes=config.model_params["num_nodes"],
        n_steps_out=config.model_params["n_steps_out"],
        model_name=config.model_params["model_name"],
        config_path=config.config_path,
        num_lstm_layers=config.model_params["num_lstm_layers"],
    )

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training_params["learning_rate"],
        weight_decay=config.training_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=config.training_params["scheduler_factor"],
        patience=config.training_params["scheduler_patience"],
    )

    # Train and evaluate
    trainer = ModelTrainer(model, optimizer, criterion, scheduler, device)
    training_history = trainer.train(
        train_data,
        val_data,
        num_epochs=config.training_params["num_epochs"],
        patience=config.training_params["early_stopping_patience"],
    )
    test_loss, test_outputs = trainer.test(test_data)

    print(f"Test Loss: {test_loss:.4f}")

    # Save results
    results_folder = project_root / "saved_results"
    save_results(
        results_folder,
        config.model_params,
        test_data,
        test_outputs,
        test_loss,
        training_history,
    )

    # Generate visualizations if enabled
    visualize_results(results_folder, config.config.get("visualization", {}))


if __name__ == "__main__":
    main()
