#!/usr/bin/env python3
"""
Monitor running experiments for potential issues
"""

import os
import time
from collections import defaultdict
from pathlib import Path


def check_experiment_status():
    """Check current experiment status and potential issues."""
    results_dir = Path("results/sensitivity_analysis")

    if not results_dir.exists():
        print("❌ Results directory doesn't exist yet")
        return

    # Get all experiment directories
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    print(f"📊 Current Status: {len(exp_dirs)} experiment directories found")
    print("=" * 60)

    # Check for timestamp collisions
    timestamps = defaultdict(list)
    valid_experiments = 0
    invalid_experiments = 0

    for exp_dir in exp_dirs:
        # Extract timestamp from directory name
        parts = exp_dir.name.split("_")
        if len(parts) >= 2:
            timestamp = parts[-1]
            timestamps[timestamp].append(exp_dir.name)

        # Check if experiment has required files
        config_exists = (exp_dir / "config.yaml").exists()
        results_exists = (exp_dir / "detection_results.xlsx").exists()

        if config_exists and results_exists:
            valid_experiments += 1
            status = "✅"
        elif config_exists:
            status = "⏳"  # In progress
        else:
            invalid_experiments += 1
            status = "❌"

        print(f"{status} {exp_dir.name}")

    print("=" * 60)
    print(f"✅ Complete: {valid_experiments}")
    print(f"⏳ In Progress: {len(exp_dirs) - valid_experiments - invalid_experiments}")
    print(f"❌ Failed: {invalid_experiments}")

    # Check for timestamp collisions
    collisions = {ts: dirs for ts, dirs in timestamps.items() if len(dirs) > 1}
    if collisions:
        print("\n🚨 TIMESTAMP COLLISIONS DETECTED:")
        for timestamp, dirs in collisions.items():
            print(f"  Timestamp {timestamp}: {len(dirs)} experiments")
            for dir_name in dirs:
                print(f"    - {dir_name}")
    else:
        print("\n✅ No timestamp collisions detected")

    # Check directory creation rate
    if exp_dirs:
        latest_timestamp = max(timestamps.keys())
        print(f"\n📅 Latest experiment timestamp: {latest_timestamp}")

    return {
        "total_dirs": len(exp_dirs),
        "valid": valid_experiments,
        "invalid": invalid_experiments,
        "collisions": len(collisions),
    }


if __name__ == "__main__":
    while True:
        print(f"\n⏰ Check at {time.strftime('%H:%M:%S')}")
        status = check_experiment_status()

        if status and status["total_dirs"] >= 176:
            print(f"\n🎉 All 176 experiments created!")
            break

        print(f"\nNext check in 30 seconds...")
        time.sleep(30)
