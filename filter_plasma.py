import pandas as pd
from pathlib import Path

# === Inputs ===
PLASMA_CSV = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_combined.csv"
MRI_CSV = "/home/Student2025/3D-MedCLIP/ADNI_APOE_subject_info_spreadsheet.csv"
SUBJECT_COL = "Subject_Name"

# === Outputs ===
OUT_FILTERED = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_combined.filtered.csv"
OUT_DEDUP_VISIT = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_combined.filtered_dedup_by_visit.csv"
OUT_ONE_PER_SUBJECT = "/home/Student2025/3D-MedCLIP/adni_apoe_plasma_combined.filtered_one_row_per_subject.csv"

# Columns that define the same plasma measurement regardless of repeated imaging rows
IDENTICAL_MEASUREMENT_COLS = [
    SUBJECT_COL,
    "VISCODE",
    "dataset",
    "PTAU181",
    "ABETA42",
    "ABETA40",
    "ABETA42_40_RATIO",
    "GFAP",
]

# Helper ordering for VISCODE
VISIT_ORDER = {
    "bl": 0,
    "sc": 0,
    "m03": 3,
    "m06": 6,
    "m09": 9,
    "m12": 12,
    "m18": 18,
    "m24": 24,
    "m30": 30,
    "m36": 36,
    "m48": 48,
    "m60": 60,
}

def parse_visit_to_months(viscode: str) -> float:
    if not isinstance(viscode, str) or viscode.strip() == "":
        return float("inf")
    v = viscode.strip().lower()
    if v in VISIT_ORDER:
        return VISIT_ORDER[v]
    # try pattern like mXX
    if v.startswith("m") and v[1:].isdigit():
        return float(v[1:])
    return float("inf")


def prefer_upenn_first(group: pd.DataFrame) -> pd.DataFrame:
    # Within same subject+visit, prefer rows where dataset contains 'UPENNPLASMA'
    if "dataset" in group.columns:
        upenn_mask = group["dataset"].astype(str).str.contains("UPENNPLASMA", case=False, na=False)
        if upenn_mask.any():
            return group[upenn_mask]
    return group


def main() -> None:
    # Read as string to avoid unintended type coercions
    df_plasma = pd.read_csv(PLASMA_CSV, dtype=str)
    df_mri = pd.read_csv(MRI_CSV, dtype=str)

    if SUBJECT_COL not in df_plasma.columns or SUBJECT_COL not in df_mri.columns:
        raise ValueError(f"Both CSVs must contain a '{SUBJECT_COL}' column.")

    # Filter to MRI subjects
    mri_subjects = set(df_mri[SUBJECT_COL].dropna().astype(str).str.strip())
    df = df_plasma[df_plasma[SUBJECT_COL].astype(str).str.strip().isin(mri_subjects)].copy()

    # Save simple filtered (no dedup) for reference
    df.to_csv(OUT_FILTERED, index=False)

    # Deduplicate identical repeats coming from multiple Image.Data.ID rows for the same measurement
    for col in IDENTICAL_MEASUREMENT_COLS:
        if col not in df.columns:
            # Create missing columns as empty to allow subset dedup without errors
            df[col] = ""
    df_dedup_visit = df.drop_duplicates(subset=IDENTICAL_MEASUREMENT_COLS).copy()

    # Within each (Subject_Name, VISCODE), prefer UPENN rows over computed_ratio duplicates
    df_dedup_visit = (
        df_dedup_visit
        .groupby([SUBJECT_COL, "VISCODE"], dropna=False, as_index=False, group_keys=False)
        .apply(prefer_upenn_first)
    )
    # If still multiple rows remain within subject+visit, keep the first stable order
    df_dedup_visit = df_dedup_visit.drop_duplicates(subset=[SUBJECT_COL, "VISCODE", "dataset"])  # small cleanup
    df_dedup_visit.to_csv(OUT_DEDUP_VISIT, index=False)

    # Build one-row-per-subject by picking best visit: baseline if available else closest to baseline by time or months
    # Prepare numeric helpers
    time_col = "time.since.bl" if "time.since.bl" in df_dedup_visit.columns else None
    if time_col:
        time_vals = pd.to_numeric(df_dedup_visit[time_col], errors="coerce")
    else:
        time_vals = pd.Series([None] * len(df_dedup_visit))

    visit_months = df_dedup_visit["VISCODE"].apply(parse_visit_to_months)

    df_dedup_visit = df_dedup_visit.assign(
        _is_baseline=df_dedup_visit["VISCODE"].astype(str).str.lower().eq("bl"),
        _visit_months=visit_months,
        _time_since_bl=time_vals,
        _is_upenn=df_dedup_visit["dataset"].astype(str).str.contains("UPENNPLASMA", case=False, na=False),
    )

    def choose_one(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        # Prefer baseline
        if g["_is_baseline"].any():
            g = g.sort_values(by=["_is_baseline", "_is_upenn"], ascending=[False, False])
            return g.iloc[[0]]
        # Else prefer smallest non-null time.since.bl
        if g["_time_since_bl"].notna().any():
            g = g.sort_values(by=["_time_since_bl", "_is_upenn"], ascending=[True, False])
            return g.iloc[[0]]
        # Else prefer earliest visit by months
        g = g.sort_values(by=["_visit_months", "_is_upenn"], ascending=[True, False])
        return g.iloc[[0]]

    one_per_subject = (
        df_dedup_visit
        .groupby(SUBJECT_COL, as_index=False, group_keys=False)
        .apply(choose_one)
        .drop(columns=["_is_baseline", "_visit_months", "_time_since_bl", "_is_upenn"], errors="ignore")
    )
    one_per_subject.to_csv(OUT_ONE_PER_SUBJECT, index=False)

    # Simple report
    print(f"Input plasma rows: {len(df_plasma)}")
    print(f"MRI subjects: {len(mri_subjects)}")
    print(f"Filtered rows (subjects intersect): {len(df)} -> saved to {Path(OUT_FILTERED).name}")
    print(f"After de-duplicating repeated measurements: {len(df_dedup_visit)} -> {Path(OUT_DEDUP_VISIT).name}")
    print(f"One row per subject: {len(one_per_subject)} -> {Path(OUT_ONE_PER_SUBJECT).name}")


if __name__ == "__main__":
    main()
