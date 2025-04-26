# src/utils/aggregate_csv.py

import pandas as pd
import argparse
import os


def aggregate_excel_files(input_files, output_file):
    """
    Aggregates multiple Excel files by averaging their values.
    Each input file contains averages of 5 trials, and all trials are copied to the output file.

    Args:
        input_files (list): List of paths to input Excel files
        output_file (str): Path to the output Excel file
    """
    # Columns to ignore in the aggregation
    ignore_columns = [
        "true_change_point",
        "traditional_detected_count",
        "horizon_detected_count",
        "traditional_detection_rate",
        "horizon_detection_rate",
    ]

    try:
        num_files = len(input_files)
        print(f"Reading {num_files} files: {', '.join(input_files)}")

        if num_files == 0:
            raise ValueError("No input files provided")

        # Process Aggregate sheets from all files
        aggregate_dfs = []
        for file_path in input_files:
            try:
                df = pd.read_excel(file_path, sheet_name="Aggregate")
                print(f"File {file_path} Aggregate sheet shape: {df.shape}")
                aggregate_dfs.append(df)
            except Exception as e:
                print(f"Error reading Aggregate sheet from {file_path}: {str(e)}")
                # Continue with other files

        if not aggregate_dfs:
            raise ValueError("No valid Aggregate sheets found in input files")

        # Get timestep column from the first file
        timestep = None
        if "timestep" in aggregate_dfs[0].columns:
            timestep = aggregate_dfs[0]["timestep"].copy()
            print(f"Using timestep column from first file, length: {len(timestep)}")

        # Drop columns we don't want to average
        columns_to_drop = ["timestep"] + ignore_columns
        for df in aggregate_dfs:
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

        # Find common columns across all dataframes
        common_columns = set(aggregate_dfs[0].columns)
        for df in aggregate_dfs[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        common_columns = sorted(list(common_columns))

        print(f"Number of common columns to average: {len(common_columns)}")
        if not common_columns:
            raise ValueError("No common columns found across all Aggregate sheets")

        # Reset indices and ensure all dataframes have the same columns
        for i, df in enumerate(aggregate_dfs):
            aggregate_dfs[i] = df[common_columns].reset_index(drop=True)

        # Debug: Verify we have data to average
        for i, df in enumerate(aggregate_dfs[: min(2, len(aggregate_dfs))]):
            print(
                f"First few rows of aggregate_dfs[{i}]['{common_columns[0]}']:",
                df[common_columns[0]].head().tolist(),
            )

        # Make sure all dataframes are the same length (use min length)
        min_len = min(len(df) for df in aggregate_dfs)
        for i, df in enumerate(aggregate_dfs):
            aggregate_dfs[i] = df.iloc[:min_len]

        # Create an empty result dataframe
        result_agg = pd.DataFrame()

        # Add timestep column if available
        if timestep is not None:
            result_agg["timestep"] = timestep[:min_len].reset_index(drop=True)

        # Compute averages across all dataframes
        for col in common_columns:
            # Debug output for the first column
            if col == common_columns[0]:
                print(f"Averaging column: {col}")
                for row_idx in range(min(3, min_len)):  # Print first 3 rows
                    values = [df[col].iloc[row_idx] for df in aggregate_dfs]
                    avg = sum(values) / len(values)
                    print(
                        f"  Row {row_idx}: {' + '.join(map(str, values))} / {len(values)} = {avg}"
                    )

            # Calculate average for all rows in this column
            result_agg[col] = sum(df[col] for df in aggregate_dfs) / len(aggregate_dfs)

        # Verify averaging worked correctly
        if common_columns:
            sample_col = common_columns[0]
            print(f"Verification for first 3 rows of column {sample_col}:")
            for i in range(min(3, len(result_agg))):
                values = [df[sample_col].iloc[i] for df in aggregate_dfs]
                print(
                    f"  Row {i}: ({' + '.join(map(str, values))}) / {len(values)} = {result_agg[sample_col].iloc[i]}"
                )

        print(f"Result Aggregate sheet shape: {result_agg.shape}")

        # Process ChangePointMetadata sheets
        cp_dfs = []
        for file_path in input_files:
            try:
                df = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
                print(f"File {file_path} ChangePointMetadata sheet shape: {df.shape}")
                cp_dfs.append(df)
            except Exception as e:
                print(
                    f"Error reading ChangePointMetadata sheet from {file_path}: {str(e)}"
                )
                # Continue with other files

        # Merge on change_point and average the rest
        result_cp = None
        if cp_dfs and all("change_point" in df.columns for df in cp_dfs):
            # Get all unique change points from all files
            all_change_points = set()
            for df in cp_dfs:
                all_change_points.update(df["change_point"].tolist())
            all_change_points = sorted(all_change_points)

            print(f"Total unique change points: {len(all_change_points)}")

            # Create a result dataframe with all change points
            result_cp = pd.DataFrame({"change_point": all_change_points})

            # Find common columns across all change point dataframes
            cp_common_cols = set()
            if cp_dfs:
                cp_common_cols = set(cp_dfs[0].columns)
                for df in cp_dfs[1:]:
                    cp_common_cols = cp_common_cols.intersection(set(df.columns))
                cp_common_cols = [
                    col for col in cp_common_cols if col != "change_point"
                ]

            # For each change point and each column, average the values where they exist
            for cp in all_change_points:
                for col in cp_common_cols:
                    values = []
                    for df in cp_dfs:
                        val = df.loc[df["change_point"] == cp, col].values
                        if len(val) > 0:
                            values.append(val[0])

                    if values:
                        # Average the values if any exist
                        result_cp.loc[result_cp["change_point"] == cp, col] = sum(
                            values
                        ) / len(values)
        elif cp_dfs:
            print("No change_point column found in all ChangePointMetadata sheets")
            # Find common columns across all change point dataframes
            if cp_dfs:
                cp_common_cols = set(cp_dfs[0].columns)
                for df in cp_dfs[1:]:
                    cp_common_cols = cp_common_cols.intersection(set(df.columns))
                cp_common_cols = list(cp_common_cols)

                # Prepare dataframes with only common columns
                for i, df in enumerate(cp_dfs):
                    cp_dfs[i] = df[cp_common_cols]

                # Simple average of common columns
                result_cp = sum(cp_dfs) / len(cp_dfs)
        else:
            print("No valid ChangePointMetadata sheets found")

        # Read trial sheets from all files
        trial_dfs = {}
        total_trials = 0

        for file_idx, file_path in enumerate(input_files):
            # Calculate trial offset for this file (5 trials per file)
            trial_offset = file_idx * 5

            # Read Trial1-Trial5 from this file
            for i in range(1, 6):
                sheet_name = f"Trial{i}"
                output_name = f"Trial{i + trial_offset}"
                try:
                    trial_dfs[output_name] = pd.read_excel(
                        file_path, sheet_name=sheet_name
                    )
                    print(
                        f"Read {sheet_name} from {file_path} as {output_name}, shape: {trial_dfs[output_name].shape}"
                    )
                    total_trials += 1
                except Exception as e:
                    print(f"Could not read {sheet_name} from {file_path}: {str(e)}")

        print(f"Total trial sheets found: {total_trials}")

        # Also copy over "Detection Summary" and "Detection Details" sheets if they exist
        other_sheets = ["Detection Summary", "Detection Details"]
        other_dfs = {}

        # Try to find these sheets in any of the input files (prioritizing earlier files)
        for sheet_name in other_sheets:
            for file_path in input_files:
                try:
                    other_dfs[sheet_name] = pd.read_excel(
                        file_path, sheet_name=sheet_name
                    )
                    print(
                        f"Read {sheet_name} from {file_path}, shape: {other_dfs[sheet_name].shape}"
                    )
                    # If found, move to next sheet
                    break
                except Exception as e:
                    # Try next file
                    continue

            if sheet_name not in other_dfs:
                print(f"Could not read {sheet_name} from any input file")

        # make sure output directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write results to Excel
        with pd.ExcelWriter(output_file) as writer:
            # Write aggregated sheets
            result_agg.to_excel(writer, sheet_name="Aggregate", index=False)
            if result_cp is not None:
                result_cp.to_excel(
                    writer, sheet_name="ChangePointMetadata", index=False
                )

            # Write trial sheets
            for sheet_name, df in sorted(trial_dfs.items()):
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write other sheets
            for sheet_name, df in other_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Successfully saved aggregated data to {output_file}")

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Aggregate multiple Excel files by averaging their values"
    )
    parser.add_argument("input_files", nargs="+", help="Paths to input Excel files")
    parser.add_argument(
        "--output",
        "-o",
        default="Aggregate.xlsx",
        help="Path to the output Excel file (default: Aggregate.xlsx)",
    )

    args = parser.parse_args()

    # Check if input files exist
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

    # Aggregate the Excel files
    aggregate_excel_files(args.input_files, args.output)


if __name__ == "__main__":
    main()
