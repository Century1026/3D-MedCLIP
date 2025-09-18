import pandas as pd

# === Input files ===
file1 = "adni_apoe_plasma_combined.filtered_one_row_per_subject.csv"
file2 = "ADNI_APOE_subject_info_spreadsheet.csv"
subject_col = "Subject_Name"   # column to match on

# === Read CSVs ===
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# === Check that the column exists ===
if subject_col not in df1.columns or subject_col not in df2.columns:
    raise ValueError(f"Both CSVs must contain a '{subject_col}' column.")

# === Filter df1 to keep only matching subjects ===
filtered_df = df1[df1[subject_col].isin(df2[subject_col])]

# === Save to CSV ===
output_file = "adni_apoe_plasma_filtered.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file} with {len(filtered_df)} rows.")
