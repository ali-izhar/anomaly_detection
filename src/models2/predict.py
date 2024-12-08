# src/models2/predict.py

import torch
import yaml
import argparse
from src.models2.datasets import GraphTimeSeriesDataset
from src.models2.forecast import ForecastingModel
from src.utils.normalization import Normalizer
from src.utils.helpers import load_model


def main(config_path, model_path, input_idx):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    data_dir = "./data/processed"
    dataset = GraphTimeSeriesDataset(data_dir, "test", config)  # Or any split
    sample = dataset[input_idx]
    input_seq, target_seq = sample
    adj_matrices = input_seq["adj_matrices"].unsqueeze(0).to(device)  # (1, T, N, N)
    features = input_seq["features"].unsqueeze(0).to(device)  # (1, T, 6)

    # Initialize and load model
    model = ForecastingModel(config).to(device)
    load_model(model, model_path, device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(adj_matrices, features)  # (1, m, 6)

    print("Predicted Features for Next m Steps:")
    print(prediction.squeeze(0).cpu().numpy())


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
    parser.add_argument(
        "--input_idx", type=int, default=0, help="Index of the input sample to predict."
    )
    args = parser.parse_args()
    main(args.config, args.model, args.input_idx)
