#!/usr/bin/env python3
"""
Check and fix result file locations
"""

import os
import shutil
from pathlib import Path


def check_and_fix_results():
    """Check for misplaced result files and fix them."""
    results_dir = Path("results/sensitivity_analysis")

    if not results_dir.exists():
        print("❌ Results directory doesn't exist")
        return

    fixed_count = 0
    total_dirs = 0

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        total_dirs += 1
        print(f"\n📁 Checking: {exp_dir.name}")

        # Check if detection_results.xlsx exists in main directory
        main_results = exp_dir / "detection_results.xlsx"
        if main_results.exists():
            print("  ✅ Results file found in main directory")
            continue

        # Look for results in subdirectories
        found_file = None
        for subdir in exp_dir.iterdir():
            if subdir.is_dir():
                print(f"    🔍 Checking subdirectory: {subdir.name}")
                subdir_results = subdir / "detection_results.xlsx"
                if subdir_results.exists():
                    print(f"    ✅ Found results in subdirectory!")
                    found_file = subdir_results
                    break

                # Check for alternative names
                for alt_name in ["detection_results.csv", "results.xlsx"]:
                    alt_file = subdir / alt_name
                    if alt_file.exists():
                        print(f"    ✅ Found {alt_name} in subdirectory!")
                        found_file = alt_file
                        break
                if found_file:
                    break

        if found_file:
            # Copy to main directory
            try:
                shutil.copy2(found_file, main_results)
                print(f"    🔧 Fixed: Copied {found_file} to {main_results}")
                fixed_count += 1
            except Exception as e:
                print(f"    ❌ Error copying file: {e}")
        else:
            print("    ❌ No results file found anywhere")

    print(f"\n📊 Summary:")
    print(f"  Total directories: {total_dirs}")
    print(f"  Fixed files: {fixed_count}")
    print(f"  Missing files: {total_dirs - fixed_count}")


if __name__ == "__main__":
    check_and_fix_results()
