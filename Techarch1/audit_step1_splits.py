
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# STEP 1: AUDIT THE EXISTING SPLIT MANIFEST
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"

EXPECTED = {
    ("Domain1", "train"): 559,
    ("Domain1", "validation"): 120,
    ("Domain1", "test"): 121,
    ("Domain2", "train"): 2104,
    ("Domain2", "validation"): 451,
    ("Domain2", "test"): 453,
}

EXPECTED_CLASS_COUNTS = {
    ("Domain1", "train"): {"Normal": 284, "Tuberculosis": 275},
    ("Domain1", "val"): {"Normal": 61, "Tuberculosis": 59},
    ("Domain1", "test"): {"Normal": 61, "Tuberculosis": 60},
    ("Domain2", "train"): {"Normal": 359, "Tuberculosis": 1745},
    ("Domain2", "val"): {"Normal": 77, "Tuberculosis": 374},
    ("Domain2", "test"): {"Normal": 78, "Tuberculosis": 375},
}


def find_column(df, candidates, description):
    """Find a column using common possible names."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        f"Cannot identify the {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: Manifest not found: {CSV_PATH}")
        print("Locate the existing dual_view_splits.csv first.")
        return

    df = pd.read_csv(CSV_PATH)

    print("\n========== MANIFEST INFORMATION ==========")
    print("File:", CSV_PATH)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    # Identify columns without changing the manifest.
    domain_col = find_column(
        df, ["domain", "domain_name", "source_domain"], "domain"
    )
    split_col = find_column(
        df, ["split", "partition", "subset", "set"], "split"
    )
    label_col = find_column(
        df, ["label", "class", "class_name", "target"], "label"
    )

    print("\n========== UNIQUE VALUES ==========")
    for col in [domain_col, split_col, label_col]:
        print(f"\n{col}:")
        print(sorted(df[col].dropna().astype(str).unique().tolist()))

    # Display the actual class distribution.
    counts = (
        df.groupby([domain_col, split_col, label_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    print("\n========== DOMAIN / SPLIT / CLASS COUNTS ==========")
    print(counts.to_string(index=False))

    print("\n========== EXPECTED SPLIT COUNT CHECK ==========")
    split_counts = df.groupby([domain_col, split_col]).size()

    for key, expected in EXPECTED.items():
        actual = split_counts.get(key, 0)
        status = "PASS" if actual == expected else "CHECK"
        print(
            f"{status}: {key[0]} / {key[1]}: "
            f"actual={actual}, expected={expected}"
        )

    print("\n========== EXPECTED CLASS COUNT CHECK ==========")
    class_counts = df.groupby(
        [domain_col, split_col, label_col]
    ).size()

    for (domain, split), expected_classes in EXPECTED_CLASS_COUNTS.items():
        for label, expected in expected_classes.items():
            key = (domain, split, label)
            actual = class_counts.get(key, 0)
            status = "PASS" if actual == expected else "CHECK"
            print(
                f"{status}: {domain}/{split}/{label}: "
                f"actual={actual}, expected={expected}"
            )

    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum().to_string())

    # Look for a likely image-path or image-ID column.
    path_candidates = [
        "image_path", "filepath", "file_path", "path",
        "image", "filename", "file_name"
    ]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    path_col = next(
        (normalized[name] for name in path_candidates if name in normalized),
        None,
    )

    if path_col:
        paths = df[path_col].dropna().astype(str)
        duplicates = paths[paths.duplicated(keep=False)]

        print("\n========== PATH DUPLICATE CHECK ==========")
        print("Path column:", path_col)
        print("Non-null paths:", len(paths))
        print("Unique paths:", paths.nunique())
        print("Repeated path rows:", len(duplicates))

        if not duplicates.empty:
            print("Example repeated paths:")
            print(duplicates.head(20).to_string(index=False))
    else:
        print(
            "\nNo standard image-path column detected. "
            "Path and file duplication checks require identifying "
            "the correct column."
        )

    # Check whether the same identifier occurs in multiple splits.
    id_candidates = [
        "image_id", "patient_id", "subject_id",
        "study_id", "uid", "id"
    ]
    id_col = next(
        (normalized[name] for name in id_candidates if name in normalized),
        None,
    )

    if id_col:
        print("\n========== IDENTIFIER / SPLIT OVERLAP CHECK ==========")
        print("Identifier column:", id_col)

        overlap = (
            df.dropna(subset=[id_col])
            .groupby(id_col)[split_col]
            .nunique()
        )
        overlapping_ids = overlap[overlap > 1]

        print("IDs appearing in multiple splits:", len(overlapping_ids))

        if not overlapping_ids.empty:
            print("Example overlapping IDs:")
            print(overlapping_ids.head(20).to_string())
    else:
        print(
            "\nNo standard patient/image ID column detected. "
            "We must inspect the manifest columns before concluding "
            "whether patient-level leakage is possible."
        )

    print("\n========== AUDIT FINISHED ==========")
    print("This script does not alter the manifest or train a model.")


if __name__ == "__main__":
    main()
