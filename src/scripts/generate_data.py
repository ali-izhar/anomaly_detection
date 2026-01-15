#!/usr/bin/env python3
"""Generate and store synthetic and MIT Reality datasets."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pickle

import numpy as np
import pandas as pd
import networkx as nx

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.graph import GraphGenerator, NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.configs import get_config, get_full_model_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 142, 241, 342, 441, 542, 642, 741, 842, 1041]
FEATURES = [
    "mean_degree", "density", "mean_clustering", "mean_betweenness",
    "mean_eigenvector", "mean_closeness", "max_singular_value", "min_nonzero_laplacian",
]


def generate_synthetic(network: str, n_sequences: int, output_dir: str, seed: int) -> dict:
    """Generate synthetic graph sequences."""
    np.random.seed(seed)
    generator = GraphGenerator(network)
    config = get_config(get_full_model_name(network))
    params = config["params"].__dict__

    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    change_points = result["change_points"]

    # Extract features
    extractor = NetworkFeatureExtractor()
    features = []
    for adj in graphs:
        graph = adjacency_to_graph(adj)
        numeric = extractor.get_numeric_features(graph)
        features.append([numeric[f] for f in FEATURES])

    return {
        "graphs": graphs,
        "features": np.array(features),
        "change_points": [int(cp) for cp in change_points],
        "params": params,
        "seed": seed,
    }


def save_synthetic_data(network: str, n_sequences: int, base_dir: str):
    """Generate and save synthetic data for a network type."""
    output_dir = os.path.join(base_dir, "synthetic", network)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {n_sequences} {network.upper()} sequences...")

    for i, seed in enumerate(SEEDS[:n_sequences]):
        data = generate_synthetic(network, n_sequences, output_dir, seed)

        # Save as pickle for efficiency
        with open(os.path.join(output_dir, f"sequence_{i:03d}.pkl"), "wb") as f:
            pickle.dump(data, f)

        # Also save features as CSV for inspection
        df = pd.DataFrame(data["features"], columns=FEATURES)
        df["change_point"] = [1 if t in data["change_points"] else 0 for t in range(len(df))]
        df.to_csv(os.path.join(output_dir, f"features_{i:03d}.csv"), index=False)

    # Save metadata
    metadata = {
        "network": network,
        "n_sequences": n_sequences,
        "seeds": SEEDS[:n_sequences],
        "features": FEATURES,
        "seq_len": data["params"].get("seq_len", 200),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {n_sequences} {network.upper()} sequences to {output_dir}")


def process_mit_reality(data_path: str, output_dir: str, threshold: float = 0.3):
    """Process MIT Reality dataset."""
    output_dir = os.path.join(output_dir, "mit_reality")
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)

    logger.info(f"Processing MIT Reality data from {data_path}...")

    # Load proximity data
    df = pd.read_csv(data_path, header=0)
    df.columns = df.columns.str.strip().str.strip('"').str.replace("\ufeff", "", regex=False)
    df = df.rename(columns={
        "user.id": "user_id", "remote.user.id.if.known": "remote_user_id",
        "time": "timestamp", "prob2": "probability"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip().str.strip('"'), errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["remote_user_id"] = pd.to_numeric(df["remote_user_id"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df = df.dropna(subset=["timestamp", "probability"])

    logger.info(f"Loaded {len(df)} records, {df['user_id'].nunique()} users")

    # Create daily graphs
    all_users = set(df["user_id"].unique()) | set(df["remote_user_id"].unique())
    daily_graphs = {}
    adjacency_matrices = []
    dates = []

    for date in sorted(df["date"].unique()):
        day_df = df[(df["date"] == date) & (df["probability"] >= threshold)]
        if len(day_df) <= 5:
            continue
        G = nx.Graph()
        G.add_nodes_from(all_users)
        G.add_edges_from(day_df[["user_id", "remote_user_id"]].values)
        if G.number_of_edges() > 1:
            daily_graphs[date.isoformat()] = G
            adjacency_matrices.append(nx.to_numpy_array(G))
            dates.append(date.isoformat())

    logger.info(f"Created {len(daily_graphs)} daily graphs")

    # Extract features
    extractor = NetworkFeatureExtractor()
    features = []
    for adj in adjacency_matrices:
        graph = adjacency_to_graph(adj)
        numeric = extractor.get_numeric_features(graph)
        features.append([numeric[f] for f in FEATURES])

    features_array = np.array(features)

    # Save graphs
    for i, (date, adj) in enumerate(zip(dates, adjacency_matrices)):
        np.save(os.path.join(output_dir, "graphs", f"graph_{i:03d}_{date}.npy"), adj)

    # Save features
    df_features = pd.DataFrame(features_array, columns=FEATURES)
    df_features["date"] = dates
    df_features.to_csv(os.path.join(output_dir, "features", "features.csv"), index=False)

    # Save as pickle
    data = {
        "adjacency_matrices": adjacency_matrices,
        "features": features_array,
        "dates": dates,
        "n_users": len(all_users),
    }
    with open(os.path.join(output_dir, "mit_reality.pkl"), "wb") as f:
        pickle.dump(data, f)

    # Save metadata
    metadata = {
        "n_days": len(dates),
        "n_users": len(all_users),
        "date_range": [dates[0], dates[-1]],
        "features": FEATURES,
        "threshold": threshold,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved MIT Reality data to {output_dir}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument("--all", action="store_true", help="Generate all datasets")
    parser.add_argument("--synthetic", nargs="*", choices=["sbm", "er", "ba", "ws"], help="Generate synthetic")
    parser.add_argument("--mit", type=str, help="MIT Reality CSV path")
    parser.add_argument("-n", "--n-sequences", type=int, default=10, help="Sequences per network")
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.all or args.synthetic is not None:
        networks = args.synthetic if args.synthetic else ["sbm", "er", "ba", "ws"]
        logger.info(f"Generating synthetic data for: {networks}")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(save_synthetic_data, net, args.n_sequences, args.output)
                for net in networks
            ]
            for f in futures:
                f.result()

    if args.all or args.mit:
        mit_path = args.mit or "archive/data/Proximity.csv"
        if os.path.exists(mit_path):
            process_mit_reality(mit_path, args.output)
        else:
            logger.warning(f"MIT Reality data not found at {mit_path}")

    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
