#!/usr/bin/env python3
"""Parameter sweep for graph change detection."""

import argparse
import copy
import itertools
import logging
import os
import sys
import time
import yaml
from pathlib import Path

import pandas as pd
from tqdm import tqdm

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.utils import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PARAM_RANGES = {
    "network": ["sbm", "ba", "er", "ws"],
    "threshold": [20, 50, 70, 100],
    "window": [5, 10],
    "horizon": [1, 3, 5, 10],
    "epsilon": [0.2, 0.5, 0.7, 0.9],
    "betting_func": ["power", "mixture", "beta"],
    "distance": ["euclidean", "mahalanobis", "cosine", "chebyshev"],
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def update_config(base: dict, params: dict) -> dict:
    config = copy.deepcopy(base)
    if "threshold" in params:
        config["detection"]["threshold"] = params["threshold"]
    if "window" in params:
        config["model"]["predictor"]["config"]["n_history"] = params["window"]
    if "horizon" in params:
        config["detection"]["prediction_horizon"] = params["horizon"]
    if "epsilon" in params:
        name = config["detection"]["betting_func_config"]["name"]
        if name == "power":
            config["detection"]["betting_func_config"]["power"]["epsilon"] = params["epsilon"]
        elif name == "mixture":
            e = params["epsilon"]
            config["detection"]["betting_func_config"]["mixture"]["epsilons"] = [max(0.1, e-0.1), e, min(0.9, e+0.1)]
    if "betting_func" in params:
        config["detection"]["betting_func_config"]["name"] = params["betting_func"]
    if "distance" in params:
        config["detection"]["distance"]["measure"] = params["distance"]
    if "network" in params:
        config["model"]["network"] = params["network"]
    return config


def calc_metrics(result: dict, true_cps: list) -> dict:
    if not result:
        return {"tpr": 0, "fpr": 0, "avg_delay": 0}
    trad = result.get("traditional_change_points", [])
    total = result.get("total_steps", 200)
    m = calculate_metrics(trad, true_cps, total, max_delay=15)
    metrics = {"tpr": m["tpr"], "fpr": m["fpr"], "avg_delay": m["avg_delay"]}
    horizon = result.get("horizon_change_points", [])
    if horizon:
        h = calculate_metrics(horizon, true_cps, total, max_delay=15)
        metrics.update({"horizon_tpr": h["tpr"], "horizon_fpr": h["fpr"], "horizon_delay": h["avg_delay"]})
    return metrics


def run_sweep(config_path: str, output_dir: str, networks: list = None, n_trials: int = 5):
    os.makedirs(output_dir, exist_ok=True)
    base = load_config(config_path)
    base["trials"]["n_trials"] = n_trials

    ranges = PARAM_RANGES.copy()
    if networks:
        ranges["network"] = networks

    combos = list(itertools.product(
        ranges["network"], ranges["threshold"], ranges["window"], ranges["horizon"],
        ranges["epsilon"], ranges["betting_func"], ranges["distance"]
    ))
    logger.info(f"Running {len(combos)} parameter combinations")

    results = []
    for network, threshold, window, horizon, epsilon, betting, distance in tqdm(combos):
        params = {"network": network, "threshold": threshold, "window": window, "horizon": horizon,
                  "epsilon": epsilon, "betting_func": betting, "distance": distance}
        config = update_config(base, params)
        ts = time.strftime("%Y%m%d_%H%M%S")
        config["output"]["directory"] = os.path.join(output_dir, f"{network}_t{threshold}_w{window}_{ts}")

        try:
            detector = GraphChangeDetection(config_dict=config)
            result = detector.run()
            if result and "aggregated" in result:
                metrics = calc_metrics(result["aggregated"], result["true_change_points"])
                results.append({**params, **metrics, "timestamp": ts})
        except Exception as e:
            logger.error(f"Failed {params}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "sweep_results.csv"), index=False)
    generate_report(df, output_dir)
    return df


def generate_report(df: pd.DataFrame, output_dir: str):
    if df.empty:
        return
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("PARAMETER SWEEP SUMMARY\n" + "=" * 40 + "\n\n")
        f.write("Best by TPR:\n")
        f.write(df.sort_values("tpr", ascending=False).head(5).to_string(index=False) + "\n\n")
        f.write("Best by Delay:\n")
        f.write(df.sort_values("avg_delay").head(5).to_string(index=False) + "\n\n")
        for net in df["network"].unique():
            sub = df[df["network"] == net]
            f.write(f"\n{net.upper()}: TPR={sub['tpr'].mean():.3f}, Delay={sub['avg_delay'].mean():.3f}\n")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument("-c", "--config", default="src/configs/algorithm.yaml")
    parser.add_argument("-o", "--output", default="results/parameter_sweep")
    parser.add_argument("-n", "--networks", nargs="+", choices=["sbm", "ba", "er", "ws"])
    parser.add_argument("-t", "--trials", type=int, default=5)
    args = parser.parse_args()

    logger.info(f"Starting sweep: {args.networks or 'all networks'}, {args.trials} trials")
    run_sweep(args.config, args.output, args.networks, args.trials)
    logger.info("Sweep completed")


if __name__ == "__main__":
    main()
