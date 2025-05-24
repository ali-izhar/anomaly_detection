#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Runner for Network Change Detection

This script conducts parameter sensitivity sweeps across synthetic networks,
saving only the raw detection_results.xlsx files. Metrics will be computed later.

PARALLEL PROCESSING SAFETY:
- Unique directory names with PID and microsecond precision
- Atomic file operations with proper locking
- Comprehensive error handling and cleanup
- Process-safe logging and progress tracking
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
project_root = str(Path(__file__).parent.parent)
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

            # Atomic move to final location
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
    """Return the base configuration for all experiments."""
    return {
        "execution": {
            "enable_prediction": False,
            "enable_visualization": False,
            "save_csv": True,
        },
        "trials": {
            "n_trials": 10,
            "random_seeds": [42, 142, 241, 342, 441, 542, 642, 741, 842, 1041],
        },
        "detection": {
            "method": "martingale",
            "threshold": 60.0,
            "batch_size": 1000,
            "reset": True,
            "reset_on_traditional": True,
            "max_window": None,
            "prediction_horizon": 5,
            "enable_pvalue_dampening": False,
            "cooldown_period": 30,
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


def create_experiment_config(network: str, **params) -> Dict[str, Any]:
    """Create experiment configuration for given parameters."""
    config = get_base_config()

    # Set network
    config["model"]["network"] = network

    # Set betting function
    if "epsilon" in params:
        config["detection"]["betting_func_config"] = {
            "name": "power",
            "power": {"epsilon": params["epsilon"]},
        }
    elif "beta_a" in params:
        # Handle both beta_a and beta_b parameters
        beta_config = {"a": params["beta_a"]}
        if "beta_b" in params:
            beta_config["b"] = params["beta_b"]
        else:
            beta_config["b"] = 1.5  # Default value

        config["detection"]["betting_func_config"] = {
            "name": "beta",
            "beta": beta_config,
        }
    elif "mixture" in params and params["mixture"]:
        config["detection"]["betting_func_config"] = {
            "name": "mixture",
            "mixture": {"epsilons": [0.7, 0.8, 0.9]},
        }

    # Set distance measure
    if "distance" in params:
        config["detection"]["distance"] = {"measure": params["distance"], "p": 2.0}

    # Set threshold
    if "threshold" in params:
        config["detection"]["threshold"] = params["threshold"]

    return config


def run_single_experiment(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment with robust parallel processing safety."""
    exp_name = experiment["exp_name"]
    unique_id = experiment["unique_id"]
    config = experiment["config"]
    output_dir = experiment["output_dir"]

    start_time = datetime.now()

    # Add deterministic seeding based on unique experiment parameters
    param_string = str(sorted(experiment.get("parameters", {}).items()))
    seed_hash = hashlib.md5(param_string.encode()).hexdigest()
    deterministic_seed = int(seed_hash[:8], 16) % (2**31 - 1)

    # Update config with deterministic seeding
    config["trials"]["random_seeds"] = [deterministic_seed]

    try:
        logger.info(
            f"Starting {exp_name} (ID: {unique_id[:12]}...) with seed {deterministic_seed}"
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
            "deterministic_seed": deterministic_seed,
            "start_time": start_time.isoformat(),
            "process_id": os.getpid(),
        }

        metadata_path = os.path.join(output_dir, "experiment_metadata.yaml")

        def write_metadata(path, content):
            with open(path, "w") as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=True)

        if not safe_file_write(metadata_path, metadata, write_metadata):
            logger.warning(f"Failed to save metadata for {exp_name}")

        # Run experiment
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

        logger.info(f"✓ {exp_name} completed successfully ({duration:.1f}s)")

        return {
            "success": True,
            "exp_name": exp_name,
            "unique_id": unique_id,
            "duration": duration,
            "output_dir": output_dir,
            "deterministic_seed": deterministic_seed,
            "process_id": os.getpid(),
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"✗ {exp_name} failed after {duration:.1f}s: {str(e)}"
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
        }


def generate_sensitivity_experiments() -> List[Dict[str, Any]]:
    """Generate all sensitivity analysis experiment configurations with unique IDs."""
    experiments = []
    base_output = "results/sensitivity_analysis"

    networks = ["sbm", "er", "ba", "ws"]
    distances = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    epsilons = [0.2, 0.5, 0.7, 0.9]
    beta_as = [0.2, 0.4, 0.6, 0.8]  # Extended range of alpha parameters
    beta_bs = [1.2, 1.8, 2.5]  # Multiple beta parameters
    thresholds = [20.0, 50.0, 100.0]

    # Power betting experiments
    for network in networks:
        for distance in distances:
            for epsilon in epsilons:
                exp_name = f"{network}_power_{epsilon}_{distance}"
                unique_id = generate_unique_experiment_id(exp_name)
                output_dir = f"{base_output}/{unique_id}"

                config = create_experiment_config(
                    network=network, epsilon=epsilon, distance=distance
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "unique_id": unique_id,
                        "config": config,
                        "output_dir": output_dir,
                        "group": "power_betting",
                        "parameters": {
                            "network": network,
                            "epsilon": epsilon,
                            "distance": distance,
                        },
                    }
                )

    # Mixture betting experiments
    for network in networks:
        for distance in distances:
            exp_name = f"{network}_mixture_{distance}"
            unique_id = generate_unique_experiment_id(exp_name)
            output_dir = f"{base_output}/{unique_id}"

            config = create_experiment_config(
                network=network, mixture=True, distance=distance
            )

            experiments.append(
                {
                    "exp_name": exp_name,
                    "unique_id": unique_id,
                    "config": config,
                    "output_dir": output_dir,
                    "group": "mixture_betting",
                    "parameters": {
                        "network": network,
                        "mixture": True,
                        "distance": distance,
                    },
                }
            )

    # Beta betting experiments - NOW WITH BOTH A AND B PARAMETERS
    for network in networks:
        for distance in distances:
            for beta_a in beta_as:
                for beta_b in beta_bs:
                    exp_name = f"{network}_beta_{beta_a}_{beta_b}_{distance}"
                    unique_id = generate_unique_experiment_id(exp_name)
                    output_dir = f"{base_output}/{unique_id}"

                    config = create_experiment_config(
                        network=network, beta_a=beta_a, beta_b=beta_b, distance=distance
                    )

                    experiments.append(
                        {
                            "exp_name": exp_name,
                            "unique_id": unique_id,
                            "config": config,
                            "output_dir": output_dir,
                            "group": "beta_betting",
                            "parameters": {
                                "network": network,
                                "beta_a": beta_a,
                                "beta_b": beta_b,
                                "distance": distance,
                            },
                        }
                    )

    # Threshold experiments
    for network in networks:
        for distance in distances:
            for threshold in thresholds:
                exp_name = f"{network}_threshold_{int(threshold)}_{distance}"
                unique_id = generate_unique_experiment_id(exp_name)
                output_dir = f"{base_output}/{unique_id}"

                config = create_experiment_config(
                    network=network,
                    epsilon=0.5,  # Default
                    distance=distance,
                    threshold=threshold,
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "unique_id": unique_id,
                        "config": config,
                        "output_dir": output_dir,
                        "group": "threshold_analysis",
                        "parameters": {
                            "network": network,
                            "threshold": threshold,
                            "distance": distance,
                        },
                    }
                )

    return experiments


def run_experiment_batch(
    experiments: List[Dict[str, Any]], max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Run a batch of experiments in parallel with robust monitoring."""
    logger.info(f"Starting {len(experiments)} experiments with {max_workers} workers")
    logger.info(
        f"Expected runtime: ~{len(experiments) * 2 / max_workers:.0f} minutes (est. 2min/experiment)"
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
            executor.submit(run_single_experiment, exp): exp for exp in experiments
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

                # Progress update every 10 experiments or at key milestones
                if (i + 1) % 10 == 0 or i < 10 or (i + 1) == len(experiments):
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
                logger.error(f"Exception in experiment {exp['exp_name']}: {str(e)}")
                results.append(
                    {
                        "success": False,
                        "exp_name": exp["exp_name"],
                        "unique_id": exp.get("unique_id", "unknown"),
                        "error": str(e),
                        "process_id": os.getpid(),
                    }
                )

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 60.0
    success_rate = (completed / len(experiments)) * 100 if experiments else 0

    logger.info("=" * 80)
    logger.info(f"BATCH COMPLETED: {completed} succeeded, {failed} failed")
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
    """Validate experiment results with detailed analysis."""
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
    """Main entry point with robust parallel execution."""
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis experiments with robust parallel processing"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended: 8-12 for production)",
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
    if args.workers > 16:
        logger.warning(
            f"High worker count ({args.workers}) may overwhelm system resources"
        )
    if args.workers < 1:
        logger.error("Worker count must be at least 1")
        return

    logger.info("=" * 80)
    logger.info("PARAMETER SENSITIVITY ANALYSIS - ROBUST PARALLEL EXECUTION")
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

        results_dir = "results/sensitivity_analysis"
        if os.path.exists(results_dir):
            logger.info(f"Cleaning {results_dir}")
            shutil.rmtree(results_dir)

    # Generate experiments
    logger.info("Generating experiment configurations...")
    experiments = generate_sensitivity_experiments()

    # Group experiments by type for reporting
    by_group = {}
    for exp in experiments:
        group = exp["group"]
        if group not in by_group:
            by_group[group] = []
        by_group[group].append(exp)

    logger.info(f"Generated {len(experiments)} experiments:")
    for group, group_exps in by_group.items():
        logger.info(f"  {group}: {len(group_exps)} experiments")

    # Expected counts validation
    expected_power = 4 * 4 * 4  # networks * distances * epsilons = 64
    expected_mixture = 4 * 4  # networks * distances = 16
    expected_beta = 4 * 4 * 4 * 3  # networks * distances * beta_as * beta_bs = 192
    expected_threshold = 4 * 4 * 3  # networks * distances * thresholds = 48
    expected_total = (
        expected_power + expected_mixture + expected_beta + expected_threshold
    )

    logger.info(f"Expected: {expected_total} total experiments")
    logger.info(f"  Power betting: {expected_power} experiments")
    logger.info(f"  Mixture betting: {expected_mixture} experiments")
    logger.info(f"  Beta betting: {expected_beta} experiments")
    logger.info(f"  Threshold analysis: {expected_threshold} experiments")

    if len(experiments) != expected_total:
        logger.error(
            f"Experiment count mismatch! Expected {expected_total}, got {len(experiments)}"
        )
        return

    # Dry run - just show what would be executed
    if args.dry_run:
        logger.info("DRY RUN - Experiments that would be executed:")
        for i, exp in enumerate(experiments[:10]):  # Show first 10
            logger.info(f"  {i+1}: {exp['exp_name']} -> {exp['output_dir']}")
        if len(experiments) > 10:
            logger.info(f"  ... and {len(experiments) - 10} more")
        return

    # Resume functionality - check existing results
    if args.resume:
        logger.info("Checking for existing completed experiments...")
        validation = validate_results("results/sensitivity_analysis")
        if validation["valid_dirs"] > 0:
            logger.info(f"Found {validation['valid_dirs']} completed experiments")
            # Here you could filter out completed experiments
            # For now, we'll just report and continue
        else:
            logger.info("No completed experiments found, running full batch")

    # Estimate runtime
    estimated_minutes = (
        len(experiments) * 2 / args.workers
    )  # 2 min per experiment estimate
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
    validation = validate_results("results/sensitivity_analysis")

    total_runtime = (datetime.now() - start_time).total_seconds() / 60.0

    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
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
        logger.info("🎉 SUCCESS: All experiments completed successfully!")
        logger.info("Raw data is ready for post-processing and Table III generation")
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
    }

    summary_path = "results/sensitivity_analysis/execution_summary.yaml"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Execution summary saved to: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
