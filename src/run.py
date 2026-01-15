#!/usr/bin/env python

"""CLI for running the detection pipeline."""

import argparse
import logging
import sys
import yaml
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.utils import print_report

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Graph change point detection")
    parser.add_argument("-c", "--config", required=True, help="Config file path")
    parser.add_argument("-ll", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-n", "--n-trials", type=int, help="Number of trials")
    parser.add_argument("-p", "--prediction", type=lambda x: x.lower() == "true", help="Enable prediction")
    parser.add_argument("--network", choices=["sbm", "ba", "ws", "er"], help="Network type")
    parser.add_argument("--threshold", type=float, help="Detection threshold")
    parser.add_argument("--method", choices=["martingale", "cusum", "ewma"], help="Detection method")
    parser.add_argument("--betting", choices=["power", "mixture", "beta"], help="Betting function")
    parser.add_argument("--distance", choices=["euclidean", "mahalanobis", "cosine", "chebyshev"], help="Distance")
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.n_trials:
        config["trials"]["n_trials"] = args.n_trials
    if args.prediction is not None:
        config["execution"]["enable_prediction"] = args.prediction
    if args.network:
        config["model"]["network"] = args.network
    if args.threshold:
        config["detection"]["threshold"] = args.threshold
    if args.method:
        config["detection"]["method"] = args.method
    if args.betting:
        config["detection"]["betting_func_config"]["name"] = args.betting
    if args.distance:
        config["detection"]["distance"]["measure"] = args.distance
    if args.output:
        config["output"]["directory"] = args.output

    detector = GraphChangeDetection(config_dict=config)
    results = detector.run()
    print_report(results)


if __name__ == "__main__":
    main()
