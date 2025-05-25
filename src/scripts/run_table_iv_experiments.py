#!/usr/bin/env python3
"""
Table IV Data Collection Script

This script runs specific change scenarios for each network type to generate
the comprehensive comparison table between Traditional Martingale, Horizon Martingale,
CUSUM, and EWMA methods.

Uses optimal parameters:
- Mixture betting (0.7, 0.8, 0.9)
- Mahalanobis distance
- Threshold 50
"""

import argparse
import logging
import os
import sys
import yaml
import concurrent.futures
import threading
import time
import hashlib
import uuid
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from contextlib import contextmanager

# Platform-specific imports
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection

# Thread-safe logging setup
_log_lock = threading.Lock()


def setup_safe_logging():
    """Setup thread-safe logging for parallel execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )


setup_safe_logging()
logger = logging.getLogger(__name__)


@contextmanager
def file_lock(file_path: str):
    """Context manager for file locking to prevent race conditions (cross-platform)."""
    lock_file = f"{file_path}.lock"

    if HAS_FCNTL:
        # Unix-style file locking
        try:
            lock_fd = None
            with open(lock_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                lock_fd = f.fileno()
                yield
        except Exception as e:
            logger.error(f"File locking failed for {file_path}: {e}")
            raise
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except OSError:
                pass  # Lock file might be removed by another process
    else:
        # Windows fallback - simple existence check with retry
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Try to create lock file exclusively
                with open(lock_file, "x") as f:
                    f.write(str(os.getpid()))
                break
            except FileExistsError:
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))  # Progressive backoff
                else:
                    logger.warning(
                        f"Could not acquire lock for {file_path} after {max_attempts} attempts"
                    )
                    # Proceed without lock on Windows as fallback
                    break

        try:
            yield
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except OSError:
                pass


def generate_unique_experiment_id(exp_name: str) -> str:
    """Generate a globally unique experiment identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    process_id = os.getpid()
    random_component = str(uuid.uuid4())[:8]

    # Create hash for additional uniqueness
    unique_string = f"{exp_name}_{timestamp}_{process_id}_{random_component}"
    hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:8]

    return f"{exp_name}_{timestamp}_pid{process_id}_{hash_suffix}"


def atomic_directory_creation(directory_path: str, max_retries: int = 3) -> bool:
    """Atomically create directory with retry mechanism."""
    for attempt in range(max_retries):
        try:
            os.makedirs(directory_path, exist_ok=False)  # Fail if exists
            return True
        except FileExistsError:
            # Directory already exists, regenerate path
            logger.warning(
                f"Directory collision detected: {directory_path}, attempt {attempt + 1}"
            )
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Progressive backoff
                parent_dir = os.path.dirname(directory_path)
                base_name = os.path.basename(directory_path)
                directory_path = os.path.join(
                    parent_dir, f"{base_name}_retry{attempt + 1}"
                )
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False

    return False


def safe_file_write(file_path: str, content: Any, write_func) -> bool:
    """Safely write file with locking and atomic operations."""
    temp_file = f"{file_path}.tmp"

    try:
        with file_lock(file_path):
            # Write to temporary file first
            write_func(temp_file, content)

            # Atomic move to final location (Windows-compatible)
            if platform.system() == "Windows":
                # On Windows, remove destination file first if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rename(temp_file, file_path)
            return True

    except Exception as e:
        logger.error(f"Safe file write failed for {file_path}: {e}")
        # Clean up temporary file
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass
        return False


def get_base_config() -> Dict[str, Any]:
    """Return the base configuration for all Table IV experiments."""
    return {
        "execution": {
            "enable_prediction": False,  # Will be overridden per method
            "enable_visualization": False,
            "save_csv": True,
        },
        "trials": {
            "n_trials": 10,
            "random_seeds": [42, 142, 241, 342, 441, 542, 642, 741, 842, 1041],
        },
        "detection": {
            "method": "martingale",  # Will be overridden per method
            "threshold": 50.0,  # Optimal threshold from parameter sweep
            "batch_size": 1000,
            "reset": True,
            "reset_on_traditional": True,
            "max_window": None,
            "prediction_horizon": 5,
            "enable_pvalue_dampening": False,
            "cooldown_period": 30,
            "betting_func_config": {
                "name": "mixture",  # Optimal betting function
                "mixture": {"epsilons": [0.7, 0.8, 0.9]},  # Optimal epsilons
            },
            "distance": {
                "measure": "mahalanobis",  # Optimal distance measure
                "p": 2.0,
            },
            # CUSUM configuration
            "cusum": {
                "drift": 0.0,
                "startup_period": 40,
                "fixed_threshold": True,
                "k": 0.25,
                "h": 8.0,
                "enable_adaptive": True,
                "min_deviation": 1.5,
                "use_robust_scale": True,
                "enforce_cooldown": True,
            },
            # EWMA configuration
            "ewma": {
                "lambda": 0.15,
                "L": 4.5,
                "startup_period": 50,
                "use_var_adjust": True,
                "robust": True,
                "enforce_cooldown": True,
                "min_deviation": 2.0,
            },
        },
        "features": [
            "mean_degree",
            "density",
            "mean_clustering",
            "mean_betweenness",
            "mean_eigenvector",
            "mean_closeness",
            "max_singular_value",
            "min_nonzero_laplacian",
        ],
        "model": {
            "type": "multiview",
            "network": "sbm",  # Will be overridden per experiment
            "predictor": {
                "type": "graph",
                "config": {
                    "alpha": 0.8,
                    "gamma": 0.5,
                    "beta_init": 0.5,
                    "enforce_connectivity": True,
                    "threshold": 0.5,
                    "n_history": 10,
                },
            },
        },
        "output": {
            "save_predictions": True,
            "save_features": True,
            "save_martingales": True,
            "save_results": True,
            "save_detection_data": True,
            "results_filename": "detection_results.xlsx",
        },
    }


def create_scenario_config(
    network: str, scenario: str, method: str, enable_prediction: bool = False
) -> Dict[str, Any]:
    """Create experiment configuration for specific network scenario and method."""
    config = get_base_config()

    # Set network type
    config["model"]["network"] = network

    # Set detection method
    config["detection"]["method"] = method

    # Set prediction flag
    config["execution"]["enable_prediction"] = enable_prediction

    # Override model parameters based on scenario
    # These will be injected into the graph generator
    if network == "sbm":
        config["model"]["scenario_params"] = get_sbm_scenario_params(scenario)
    elif network == "er":
        config["model"]["scenario_params"] = get_er_scenario_params(scenario)
    elif network == "ba":
        config["model"]["scenario_params"] = get_ba_scenario_params(scenario)
    elif network == "ws":
        config["model"]["scenario_params"] = get_ws_scenario_params(scenario)

    return config


def get_sbm_scenario_params(scenario: str) -> Dict[str, Any]:
    """Get SBM parameters for specific scenarios."""
    base_params = {
        "n": 50,
        "seq_len": 200,
        "min_segment": 40,
        "num_blocks": 2,
        "min_block_size": 25,
        "max_block_size": 25,
    }

    if scenario == "community_merge":
        # Scenario: Strong communities merge into weaker structure
        return {
            **base_params,
            "min_changes": 1,
            "max_changes": 1,
            "intra_prob": 0.95,  # Start with strong communities
            "inter_prob": 0.01,  # Very low inter-community connections
            "min_intra_prob": 0.4,  # End with weaker communities
            "max_intra_prob": 0.95,
            "min_inter_prob": 0.01,
            "max_inter_prob": 0.25,  # Increase inter-community connections
        }
    elif scenario == "density_change":
        # Scenario: Overall density change while maintaining community structure
        return {
            **base_params,
            "min_changes": 1,
            "max_changes": 1,
            "intra_prob": 0.6,  # Moderate initial density
            "inter_prob": 0.05,
            "min_intra_prob": 0.3,  # Decrease density
            "max_intra_prob": 0.9,  # Or increase density
            "min_inter_prob": 0.01,
            "max_inter_prob": 0.1,
        }
    elif scenario == "mixed_changes":
        # Scenario: Multiple types of changes
        return {
            **base_params,
            "min_changes": 2,
            "max_changes": 2,
            "intra_prob": 0.8,
            "inter_prob": 0.02,
            "min_intra_prob": 0.3,
            "max_intra_prob": 0.95,
            "min_inter_prob": 0.01,
            "max_inter_prob": 0.3,
        }
    else:
        raise ValueError(f"Unknown SBM scenario: {scenario}")


def get_er_scenario_params(scenario: str) -> Dict[str, Any]:
    """Get ER parameters for specific scenarios."""
    base_params = {
        "n": 50,
        "seq_len": 200,
        "min_segment": 40,
        "min_changes": 1,
        "max_changes": 1,
    }

    if scenario == "density_increase":
        # Scenario: Sparse to dense network
        return {
            **base_params,
            "prob": 0.05,  # Start sparse
            "min_prob": 0.05,
            "max_prob": 0.3,  # End dense
        }
    elif scenario == "density_decrease":
        # Scenario: Dense to sparse network
        return {
            **base_params,
            "prob": 0.3,  # Start dense
            "min_prob": 0.05,  # End sparse
            "max_prob": 0.3,
        }
    else:
        raise ValueError(f"Unknown ER scenario: {scenario}")


def get_ba_scenario_params(scenario: str) -> Dict[str, Any]:
    """Get BA parameters for specific scenarios."""
    base_params = {
        "n": 50,
        "seq_len": 200,
        "min_segment": 40,
        "min_changes": 1,
        "max_changes": 1,
    }

    if scenario == "parameter_shift":
        # Scenario: Change in preferential attachment strength
        return {
            **base_params,
            "m": 2,  # Start with moderate attachment
            "min_m": 1,  # Decrease to tree-like
            "max_m": 5,  # Or increase to hub-dominated
        }
    elif scenario == "hub_addition":
        # Scenario: Dramatic increase in hub formation
        return {
            **base_params,
            "m": 1,  # Start tree-like
            "min_m": 1,
            "max_m": 6,  # End with strong hubs
        }
    else:
        raise ValueError(f"Unknown BA scenario: {scenario}")


def get_ws_scenario_params(scenario: str) -> Dict[str, Any]:
    """Get WS parameters for specific scenarios."""
    base_params = {
        "n": 50,
        "seq_len": 200,
        "min_segment": 40,
        "min_changes": 1,
        "max_changes": 1,
        "k_nearest": 6,
    }

    if scenario == "rewiring_increase":
        # Scenario: Regular to small-world to random
        return {
            **base_params,
            "rewire_prob": 0.05,  # Start regular
            "min_prob": 0.05,
            "max_prob": 0.4,  # End more random
            "min_k": 6,
            "max_k": 6,  # Keep k constant
        }
    elif scenario == "k_parameter_shift":
        # Scenario: Change in neighborhood size
        return {
            **base_params,
            "rewire_prob": 0.1,  # Keep rewiring constant
            "min_prob": 0.1,
            "max_prob": 0.1,
            "min_k": 4,  # Smaller neighborhoods
            "max_k": 8,  # Larger neighborhoods
        }
    else:
        raise ValueError(f"Unknown WS scenario: {scenario}")


def run_single_table_iv_experiment(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single Table IV experiment with robust parallel processing safety."""
    exp_name = experiment["exp_name"]
    unique_id = experiment["unique_id"]
    config = experiment["config"]
    output_dir = experiment["output_dir"]

    start_time = datetime.now()

    # Keep the original 10 trials with predefined seeds for robust statistics
    # No need to override the random_seeds - use all 10 trials

    try:
        logger.info(
            f"Starting Table IV {exp_name} (ID: {unique_id[:12]}...) with 10 trials"
        )

        # Atomic directory creation with retries
        if not atomic_directory_creation(output_dir):
            raise RuntimeError(
                f"Failed to create unique output directory: {output_dir}"
            )

        # Set output directory in config
        config["output"]["directory"] = output_dir

        # Save config with safe file writing
        config_path = os.path.join(output_dir, "config.yaml")

        def write_config(path, content):
            with open(path, "w") as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=True)

        if not safe_file_write(config_path, config, write_config):
            raise RuntimeError(f"Failed to save config file: {config_path}")

        # Save experiment metadata
        metadata = {
            "exp_name": exp_name,
            "unique_id": unique_id,
            "group": experiment["group"],
            "parameters": experiment["parameters"],
            "trials": 10,
            "random_seeds": config["trials"]["random_seeds"],
            "start_time": start_time.isoformat(),
            "process_id": os.getpid(),
            "table": "table_iv",
        }

        metadata_path = os.path.join(output_dir, "experiment_metadata.yaml")

        def write_metadata(path, content):
            with open(path, "w") as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=True)

        if not safe_file_write(metadata_path, metadata, write_metadata):
            logger.warning(f"Failed to save metadata for {exp_name}")

        # Run experiment using GraphChangeDetection
        detector = GraphChangeDetection(config_dict=config)
        result = detector.run()

        # Verify critical output files exist
        detection_file = os.path.join(output_dir, "detection_results.xlsx")
        if not os.path.exists(detection_file):
            raise FileNotFoundError(f"Detection results file missing: {detection_file}")

        # Verify file is not empty and readable
        if os.path.getsize(detection_file) == 0:
            raise ValueError(f"Detection results file is empty: {detection_file}")

        duration = (datetime.now() - start_time).total_seconds()

        # Update metadata with completion info
        metadata.update(
            {
                "end_time": datetime.now().isoformat(),
                "duration_seconds": duration,
                "status": "completed",
                "detection_file_size": os.path.getsize(detection_file),
            }
        )

        # Save final metadata
        safe_file_write(metadata_path, metadata, write_metadata)

        logger.info(f"✓ Table IV {exp_name} completed successfully ({duration:.1f}s)")

        return {
            "success": True,
            "exp_name": exp_name,
            "unique_id": unique_id,
            "duration": duration,
            "output_dir": output_dir,
            "trials_completed": 10,
            "process_id": os.getpid(),
            "results": result,
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"✗ Table IV {exp_name} failed after {duration:.1f}s: {str(e)}"
        logger.error(error_msg)

        # Save error information but don't remove directory for debugging
        try:
            if os.path.exists(output_dir):
                error_metadata = {
                    "exp_name": exp_name,
                    "unique_id": unique_id,
                    "error": str(e),
                    "error_time": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "status": "failed",
                    "process_id": os.getpid(),
                    "table": "table_iv",
                }

                error_path = os.path.join(output_dir, "error_log.yaml")

                def write_error(path, content):
                    with open(path, "w") as f:
                        yaml.dump(content, f, default_flow_style=False, sort_keys=True)

                safe_file_write(error_path, error_metadata, write_error)
        except Exception as save_error:
            logger.error(f"Failed to save error metadata: {save_error}")

        return {
            "success": False,
            "exp_name": exp_name,
            "unique_id": unique_id,
            "error": str(e),
            "duration": duration,
            "process_id": os.getpid(),
            "table": "table_iv",
        }


def generate_table_iv_experiments() -> List[Dict[str, Any]]:
    """Generate all Table IV experiment configurations."""
    experiments = []
    base_output = "results/table_iv_experiments"

    # Define network scenarios as shown in Table IV
    network_scenarios = {
        "sbm": ["community_merge", "density_change", "mixed_changes"],
        "er": ["density_increase", "density_decrease"],
        "ba": ["parameter_shift", "hub_addition"],
        "ws": ["rewiring_increase", "k_parameter_shift"],
    }

    # Define methods to compare
    methods = [
        (
            "traditional_martingale",
            "martingale",
            False,
        ),  # (name, method, enable_prediction)
        ("horizon_martingale", "martingale", True),
        ("cusum", "cusum", False),
        ("ewma", "ewma", False),
    ]

    # Generate experiments for each network, scenario, and method combination
    for network, scenarios in network_scenarios.items():
        for scenario in scenarios:
            for method_name, method_type, enable_prediction in methods:
                exp_name = f"{network}_{scenario}_{method_name}"
                unique_id = generate_unique_experiment_id(exp_name)
                output_dir = f"{base_output}/{unique_id}"

                config = create_scenario_config(
                    network, scenario, method_type, enable_prediction
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "unique_id": unique_id,
                        "config": config,
                        "output_dir": output_dir,
                        "group": f"{network}_{scenario}",
                        "parameters": {
                            "network": network,
                            "scenario": scenario,
                            "method": method_name,
                            "detection_method": method_type,
                            "enable_prediction": enable_prediction,
                        },
                    }
                )

    return experiments


def run_experiment_batch(
    experiments: List[Dict[str, Any]], max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Run a batch of Table IV experiments in parallel with robust monitoring."""
    logger.info(
        f"Starting {len(experiments)} Table IV experiments with {max_workers} workers"
    )
    logger.info(
        f"Expected runtime: ~{len(experiments) * 10 / max_workers:.0f} minutes (est. 10min/experiment with 10 trials)"
    )

    results = []
    completed = 0
    failed = 0
    start_time = datetime.now()

    # Track progress by group
    group_stats = {}
    for exp in experiments:
        group = exp["group"]
        if group not in group_stats:
            group_stats[group] = {"total": 0, "completed": 0, "failed": 0}
        group_stats[group]["total"] += 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_single_table_iv_experiment, exp): exp
            for exp in experiments
        }

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            try:
                result = future.result()
                exp = future_to_exp[future]
                group = exp["group"]

                results.append(result)

                if result["success"]:
                    completed += 1
                    group_stats[group]["completed"] += 1
                else:
                    failed += 1
                    group_stats[group]["failed"] += 1

                # Progress update every 5 experiments or at key milestones
                if (i + 1) % 5 == 0 or i < 5 or (i + 1) == len(experiments):
                    elapsed = (datetime.now() - start_time).total_seconds() / 60.0
                    remaining = (
                        (elapsed / (i + 1)) * (len(experiments) - i - 1) if i > 0 else 0
                    )
                    completion_pct = ((i + 1) / len(experiments)) * 100

                    logger.info(
                        f"Progress: [{i+1:3d}/{len(experiments)}] ({completion_pct:5.1f}%) - "
                        f"✓{completed} ✗{failed} - "
                        f"Elapsed: {elapsed:5.1f}min, Remaining: ~{remaining:5.1f}min"
                    )

                    # Group-wise progress
                    for group_name, stats in group_stats.items():
                        total = stats["total"]
                        comp = stats["completed"]
                        fail = stats["failed"]
                        group_pct = ((comp + fail) / total) * 100 if total > 0 else 0
                        logger.info(
                            f"  {group_name}: {comp+fail}/{total} ({group_pct:.0f}%) - ✓{comp} ✗{fail}"
                        )

            except Exception as e:
                exp = future_to_exp[future]
                failed += 1
                group_stats[exp["group"]]["failed"] += 1
                logger.error(
                    f"Exception in Table IV experiment {exp['exp_name']}: {str(e)}"
                )
                results.append(
                    {
                        "success": False,
                        "exp_name": exp["exp_name"],
                        "unique_id": exp.get("unique_id", "unknown"),
                        "error": str(e),
                        "process_id": os.getpid(),
                        "table": "table_iv",
                    }
                )

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 60.0
    success_rate = (completed / len(experiments)) * 100 if experiments else 0

    logger.info("=" * 80)
    logger.info(f"TABLE IV BATCH COMPLETED: {completed} succeeded, {failed} failed")
    logger.info(f"Success rate: {success_rate:.1f}% in {total_time:.1f} minutes")
    logger.info(
        f"Average time per experiment: {total_time*60/len(experiments):.1f} seconds"
    )

    # Group-wise final summary
    for group_name, stats in group_stats.items():
        total = stats["total"]
        comp = stats["completed"]
        fail = stats["failed"]
        group_success_rate = (comp / total) * 100 if total > 0 else 0
        logger.info(
            f"  {group_name}: {comp}/{total} succeeded ({group_success_rate:.1f}%)"
        )

    logger.info("=" * 80)

    return results


def validate_results(base_dir: str) -> Dict[str, int]:
    """Validate Table IV experiment results with detailed analysis."""
    if not os.path.exists(base_dir):
        return {
            "total_dirs": 0,
            "valid_dirs": 0,
            "config_only": 0,
            "invalid": 0,
            "errors": 0,
        }

    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    valid = 0
    config_only = 0
    invalid = 0
    errors = 0
    detailed_status = {}

    for dirname in dirs:
        dir_path = os.path.join(base_dir, dirname)

        # Check for required files
        has_config = os.path.exists(os.path.join(dir_path, "config.yaml"))
        has_results = os.path.exists(os.path.join(dir_path, "detection_results.xlsx"))
        has_metadata = os.path.exists(
            os.path.join(dir_path, "experiment_metadata.yaml")
        )
        has_error = os.path.exists(os.path.join(dir_path, "error_log.yaml"))

        if has_error:
            errors += 1
            status = "error"
        elif has_config and has_results and has_metadata:
            # Check if results file is not empty
            results_path = os.path.join(dir_path, "detection_results.xlsx")
            if os.path.getsize(results_path) > 0:
                valid += 1
                status = "valid"
            else:
                invalid += 1
                status = "empty_results"
        elif has_config and not has_results:
            config_only += 1
            status = "config_only"
        else:
            invalid += 1
            status = "invalid"

        detailed_status[dirname] = status

    return {
        "total_dirs": len(dirs),
        "valid_dirs": valid,
        "config_only": config_only,
        "invalid": invalid,
        "errors": errors,
        "detailed_status": detailed_status,
    }


def main():
    """Main entry point for Table IV data collection."""
    parser = argparse.ArgumentParser(
        description="Run Table IV experiments comparing Traditional Martingale, Horizon Martingale, CUSUM, and EWMA"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended: 4-8 for Table IV)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean results directory before starting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate experiment list but don't run experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous run by skipping completed experiments",
    )

    args = parser.parse_args()

    # Validate worker count
    if args.workers > 8:
        logger.warning(
            f"High worker count ({args.workers}) may overwhelm system resources for Table IV experiments"
        )
    if args.workers < 1:
        logger.error("Worker count must be at least 1")
        return

    logger.info("=" * 80)
    logger.info("TABLE IV DATA COLLECTION - METHOD COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(
        f"File locking: {'fcntl (Unix)' if HAS_FCNTL else 'fallback (Windows)'}"
    )
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Start time: {datetime.now().isoformat()}")

    # Clean results if requested
    if args.clean:
        import shutil

        results_dir = "results/table_iv_experiments"
        if os.path.exists(results_dir):
            logger.info(f"Cleaning {results_dir}")
            shutil.rmtree(results_dir)

    # Generate experiments
    logger.info("Generating Table IV experiment configurations...")
    experiments = generate_table_iv_experiments()

    # Group experiments by type for reporting
    by_group = {}
    for exp in experiments:
        group = exp["group"]
        if group not in by_group:
            by_group[group] = []
        by_group[group].append(exp)

    logger.info(f"Generated {len(experiments)} Table IV experiments:")
    for group, group_exps in by_group.items():
        logger.info(f"  {group}: {len(group_exps)} experiments")

    # Expected counts validation
    # Each network-scenario combination × 4 methods
    # SBM: 3 scenarios, ER: 2 scenarios, BA: 2 scenarios, WS: 2 scenarios = 9 total
    # 9 network-scenario combinations × 4 methods = 36 experiments
    expected_total = 36  # Fixed calculation
    logger.info(f"Expected: {expected_total} total experiments")

    if len(experiments) != expected_total:
        logger.error(
            f"Experiment count mismatch! Expected {expected_total}, got {len(experiments)}"
        )
        return

    # Dry run - just show what would be executed
    if args.dry_run:
        logger.info("DRY RUN - Table IV experiments that would be executed:")
        for i, exp in enumerate(experiments[:10]):  # Show first 10
            logger.info(f"  {i+1}: {exp['exp_name']} -> {exp['output_dir']}")
        if len(experiments) > 10:
            logger.info(f"  ... and {len(experiments) - 10} more")
        return

    # Resume functionality - check existing results
    if args.resume:
        logger.info("Checking for existing completed Table IV experiments...")
        validation = validate_results("results/table_iv_experiments")
        if validation["valid_dirs"] > 0:
            logger.info(f"Found {validation['valid_dirs']} completed experiments")
            # Here you could filter out completed experiments
            # For now, we'll just report and continue
        else:
            logger.info("No completed experiments found, running full batch")

    # Estimate runtime
    estimated_minutes = (
        len(experiments) * 10 / args.workers
    )  # 10 min per experiment estimate (with 10 trials)
    logger.info(
        f"Estimated runtime: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)"
    )

    # Run experiments
    start_time = datetime.now()
    try:
        results = run_experiment_batch(experiments, args.workers)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return
    except Exception as e:
        logger.error(f"Fatal error during batch execution: {e}")
        return

    # Final validation and reporting
    logger.info("Performing final validation...")
    validation = validate_results("results/table_iv_experiments")

    total_runtime = (datetime.now() - start_time).total_seconds() / 60.0

    logger.info("=" * 80)
    logger.info("TABLE IV FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Total runtime: {total_runtime:.1f} minutes ({total_runtime/60:.1f} hours)"
    )
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Valid results: {validation['valid_dirs']}")
    logger.info(f"Config only: {validation['config_only']}")
    logger.info(f"Errors: {validation['errors']}")
    logger.info(f"Invalid: {validation['invalid']}")

    success_rate = (
        (validation["valid_dirs"] / len(experiments)) * 100 if experiments else 0
    )
    logger.info(f"Overall success rate: {success_rate:.1f}%")

    if validation["valid_dirs"] == expected_total:
        logger.info("🎉 SUCCESS: All Table IV experiments completed successfully!")
        logger.info("Raw data is ready for Table IV generation")
    elif validation["valid_dirs"] >= expected_total * 0.95:  # 95% success threshold
        logger.info(
            f"✅ MOSTLY SUCCESS: {validation['valid_dirs']}/{expected_total} experiments completed"
        )
        logger.info("Sufficient data for analysis, minor failures acceptable")
    else:
        logger.warning(
            f"⚠️  PARTIAL SUCCESS: Only {validation['valid_dirs']}/{expected_total} experiments completed"
        )
        logger.warning("May need to investigate failures and rerun missing experiments")

    # Save execution summary
    summary = {
        "execution_date": datetime.now().isoformat(),
        "total_runtime_minutes": total_runtime,
        "worker_count": args.workers,
        "experiments_total": len(experiments),
        "experiments_valid": validation["valid_dirs"],
        "experiments_failed": validation["errors"],
        "success_rate_percent": success_rate,
        "by_group": {group: len(group_exps) for group, group_exps in by_group.items()},
        "table": "table_iv",
    }

    summary_path = "results/table_iv_experiments/execution_summary.yaml"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Execution summary saved to: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
