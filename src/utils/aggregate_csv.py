# src/utils/aggregate_csv.py

import pandas as pd
import argparse
import os


def aggregate_excel_files(file1, file2, output_file):
    """
    Aggregates two Excel files by averaging their values.
    Each input file contains averages of 5 trials, so this produces an average of 10 trials.

    Args:
        file1 (str): Path to the first Excel file
        file2 (str): Path to the second Excel file
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
        print(f"Reading files: {file1} and {file2}")

        # Process Aggregate sheet
        df1_agg = pd.read_excel(file1, sheet_name="Aggregate")
        df2_agg = pd.read_excel(file2, sheet_name="Aggregate")

        print(f"File 1 Aggregate sheet shape: {df1_agg.shape}")
        print(f"File 2 Aggregate sheet shape: {df2_agg.shape}")

        # Get timestep column from the first file
        timestep = None
        if "timestep" in df1_agg.columns:
            timestep = df1_agg["timestep"].copy()
            print(f"Using timestep column from file 1, length: {len(timestep)}")

        # Drop columns we don't want to average
        columns_to_drop = ["timestep"] + ignore_columns

        for col in columns_to_drop:
            if col in df1_agg.columns:
                df1_agg = df1_agg.drop(col, axis=1)
            if col in df2_agg.columns:
                df2_agg = df2_agg.drop(col, axis=1)

        # Ensure both dataframes have the same columns
        common_columns = sorted(
            list(set(df1_agg.columns).intersection(set(df2_agg.columns)))
        )
        print(f"Number of common columns to average: {len(common_columns)}")

        if not common_columns:
            raise ValueError("No common columns found between the two Aggregate sheets")

        # Reset indices to ensure proper alignment for averaging
        df1_agg = df1_agg[common_columns].reset_index(drop=True)
        df2_agg = df2_agg[common_columns].reset_index(drop=True)

        # Debug: Verify we have data to average
        print(
            f"First few rows of df1_agg['{common_columns[0]}']:",
            df1_agg[common_columns[0]].head().tolist(),
        )
        print(
            f"First few rows of df2_agg['{common_columns[0]}']:",
            df2_agg[common_columns[0]].head().tolist(),
        )

        # Make sure the dataframes are the same length (use min length if different)
        min_len = min(len(df1_agg), len(df2_agg))
        df1_agg = df1_agg.iloc[:min_len]
        df2_agg = df2_agg.iloc[:min_len]

        # Create an empty result dataframe
        result_agg = pd.DataFrame()

        # Add timestep column if available
        if timestep is not None:
            result_agg["timestep"] = timestep[:min_len].reset_index(drop=True)

        # Compute averages directly with explicit iteration to ensure correct averaging
        for col in common_columns:
            # Debug output for the first column
            if col == common_columns[0]:
                print(f"Averaging column: {col}")
                for i in range(min(5, min_len)):  # Print first 5 rows
                    val1 = df1_agg[col].iloc[i]
                    val2 = df2_agg[col].iloc[i]
                    avg = (val1 + val2) / 2
                    print(f"  Row {i}: {val1} + {val2} = {avg}")

            # Calculate average for all rows in this column
            result_agg[col] = (df1_agg[col] + df2_agg[col]) / 2

        # Verify averaging worked correctly
        if common_columns:
            sample_col = common_columns[0]
            print(f"Verification for first 3 rows of column {sample_col}:")
            for i in range(min(3, len(result_agg))):
                print(
                    f"  Row {i}: ({df1_agg[sample_col].iloc[i]} + {df2_agg[sample_col].iloc[i]}) / 2 = {result_agg[sample_col].iloc[i]}"
                )

        print(f"Result Aggregate sheet shape: {result_agg.shape}")

        # Process ChangePointMetadata sheet
        df1_cp = pd.read_excel(file1, sheet_name="ChangePointMetadata")
        df2_cp = pd.read_excel(file2, sheet_name="ChangePointMetadata")

        print(f"File 1 ChangePointMetadata sheet shape: {df1_cp.shape}")
        print(f"File 2 ChangePointMetadata sheet shape: {df2_cp.shape}")

        # Merge on change_point and average the rest
        result_cp = None
        if "change_point" in df1_cp.columns and "change_point" in df2_cp.columns:
            # Get all unique change points
            all_change_points = sorted(
                set(df1_cp["change_point"].tolist() + df2_cp["change_point"].tolist())
            )

            print(f"Total unique change points: {len(all_change_points)}")

            # Create a result dataframe with all change points
            result_cp = pd.DataFrame({"change_point": all_change_points})

            # Get common columns except change_point
            cp_common_cols = [
                col
                for col in df1_cp.columns.intersection(df2_cp.columns)
                if col != "change_point"
            ]

            # For each change point and each column, average the values if they exist in both dataframes
            for cp in all_change_points:
                for col in cp_common_cols:
                    val1 = df1_cp.loc[df1_cp["change_point"] == cp, col].values
                    val2 = df2_cp.loc[df2_cp["change_point"] == cp, col].values

                    if len(val1) > 0 and len(val2) > 0:
                        # Both files have this change point, average the values
                        result_cp.loc[result_cp["change_point"] == cp, col] = (
                            val1[0] + val2[0]
                        ) / 2
                    elif len(val1) > 0:
                        # Only first file has this change point
                        result_cp.loc[result_cp["change_point"] == cp, col] = val1[0]
                    elif len(val2) > 0:
                        # Only second file has this change point
                        result_cp.loc[result_cp["change_point"] == cp, col] = val2[0]
        else:
            print("No change_point column found in ChangePointMetadata sheets")
            # If no change_point column, do a simple average of common columns
            cp_common_cols = list(set(df1_cp.columns).intersection(set(df2_cp.columns)))
            df1_cp = df1_cp[cp_common_cols]
            df2_cp = df2_cp[cp_common_cols]
            result_cp = (df1_cp + df2_cp) / 2

        # Write results to Excel
        with pd.ExcelWriter(output_file) as writer:
            result_agg.to_excel(writer, sheet_name="Aggregate", index=False)
            if result_cp is not None:
                result_cp.to_excel(
                    writer, sheet_name="ChangePointMetadata", index=False
                )

        print(f"Successfully saved aggregated data to {output_file}")

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Aggregate two Excel files by averaging their values"
    )
    parser.add_argument("file1", help="Path to the first Excel file")
    parser.add_argument("file2", help="Path to the second Excel file")
    parser.add_argument(
        "--output",
        "-o",
        default="Aggregate.xlsx",
        help="Path to the output Excel file (default: Aggregate.xlsx)",
    )

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.file1):
        print(f"Error: File {args.file1} does not exist")
        return

    if not os.path.exists(args.file2):
        print(f"Error: File {args.file2} does not exist")
        return

    # Aggregate the Excel files
    aggregate_excel_files(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
