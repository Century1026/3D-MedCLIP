import pandas as pd
from pathlib import Path

INPUT_CSV = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_combined.filtered_one_row_per_subject.csv"
OUT_HAS_PTAU = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_one_row_has_ptau.csv"
OUT_NO_PTAU = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_one_row_no_ptau.csv"
PTAU_COL = "PTAU181"


df = pd.read_csv(INPUT_CSV, dtype=str)
if PTAU_COL not in df.columns:
    raise ValueError(f"Column '{PTAU_COL}' not found in {Path(INPUT_CSV).name}")

# Non-empty and non-null PTAU
mask_has_ptau = df[PTAU_COL].notna() & (df[PTAU_COL].astype(str).str.strip() != "")

df_has = df[mask_has_ptau].copy()
df_no = df[~mask_has_ptau].copy()

# Save
df_has.to_csv(OUT_HAS_PTAU, index=False)
df_no.to_csv(OUT_NO_PTAU, index=False)

print(f"Input rows: {len(df)}")
print(f"With PTAU181: {len(df_has)} -> {Path(OUT_HAS_PTAU).name}")
print(f"Without PTAU181: {len(df_no)} -> {Path(OUT_NO_PTAU).name}")
