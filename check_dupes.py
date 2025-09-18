import sys
import pandas as pd
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python check_dupes.py <csv_path> <column_name>")
        sys.exit(1)

    csv_path = sys.argv[1]
    column = sys.argv[2]

    df = pd.read_csv(csv_path, dtype=str)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    total = len(df)
    unique_vals = df[column].nunique(dropna=False)
    dup_mask = df.duplicated(subset=[column], keep=False)
    dup_rows = df[dup_mask].copy()

    num_dup_rows = len(dup_rows)
    dup_keys = (
        df[column]
        .value_counts(dropna=False)
        .loc[lambda s: s > 1]
    )
    num_dup_keys = len(dup_keys)

    print(f"File: {Path(csv_path).name}")
    print(f"Column: {column}")
    print(f"Total rows: {total}")
    print(f"Unique {column}: {unique_vals}")
    print(f"Duplicate keys (count>1): {num_dup_keys}")
    print(f"Rows involved in duplicates: {num_dup_rows}")

    # Save detailed duplicates next to the input file
    out_csv = str(Path(csv_path).with_suffix("").as_posix()) + f".{column}_duplicates.csv"
    dup_rows.sort_values(by=[column]).to_csv(out_csv, index=False)
    print(f"Detailed duplicate rows saved to {Path(out_csv).name}")

    # Show top 10 duplicate keys
    print("Top duplicate keys:")
    print(dup_keys.head(10).to_string())


if __name__ == "__main__":
    main()
