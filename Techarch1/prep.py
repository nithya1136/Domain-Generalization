
from pathlib import Path
import pandas as pd

# ==================================================
# CONFIGURATION — CHANGE THIS PATH
# ==================================================

PROJECT_ROOT = Path(r"C:\projects\FYP\Techarch1")

# Replace this with the actual folder containing Domain 1 and Domain 2
RAW_DATASET_ROOT = Path(r"C:\projects\Final Year Project\Dataset")

# ==================================================
# WORKSPACE DIRECTORIES
# ==================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

FOLDERS = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "logs",
    OUTPUT_ROOT / "01_metadata",
    OUTPUT_ROOT / "02_preprocessed",
    OUTPUT_ROOT / "03_segmentation_masks",
    OUTPUT_ROOT / "04_segmented_images",
    OUTPUT_ROOT / "05_dual_view",
    OUTPUT_ROOT / "06_reports",
]

for folder in FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)

# ==================================================
# IMAGE DISCOVERY
# ==================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".dcm"
}

DOMAIN_ALIASES = {
    "domain1": "Domain1",
    "domain_1": "Domain1",
    "domain 1": "Domain1",
    "domain2": "Domain2",
    "domain_2": "Domain2",
    "domain 2": "Domain2",
}


def identify_domain(path):
    for part in path.parts:
        # Normalize folder names: "Domain 1-NLM" -> "domain 1 nlm"
        normalized = part.lower().replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())

        if normalized.startswith("domain 1"):
            return "Domain1"

        if normalized.startswith("domain 2"):
            return "Domain2"

    return None

def identify_label(path):
    # Check folder names, not image filenames.
    # This avoids accidentally assigning labels based on file names.
    for part in reversed(path.parts[:-1]):
        name = part.lower().replace("_", " ").replace("-", " ")

        if name in {"normal", "normal images", "normal cases"}:
            return "Normal"

        if name in {
            "tb", "tuberculosis", "tuberculosis images",
            "tb images", "tb cases"
        }:
            return "Tuberculosis"

    return None


# ==================================================
# BUILD MANIFEST
# ==================================================

if not RAW_DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Raw dataset folder does not exist:\n{RAW_DATASET_ROOT}\n"
        "Edit RAW_DATASET_ROOT at the top of this script."
    )

records = []

for image_path in RAW_DATASET_ROOT.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    domain = identify_domain(image_path.relative_to(RAW_DATASET_ROOT))
    label = identify_label(image_path.relative_to(RAW_DATASET_ROOT))

    records.append({
        "image_id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "relative_path": str(
            image_path.relative_to(RAW_DATASET_ROOT)
        ),
        "domain": domain,
        "label": label,
        "extension": image_path.suffix.lower(),
        "exists": image_path.exists(),
    })

manifest = pd.DataFrame(records)

if manifest.empty:
    raise RuntimeError(
        "No supported images were found. Check RAW_DATASET_ROOT "
        "and the image file extensions."
    )

manifest_path = OUTPUT_ROOT / "01_metadata" / "master_manifest.csv"
manifest.to_csv(manifest_path, index=False)

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 55)
print("TECHNICAL ARCHITECTURE 1 — DATASET MANIFEST")
print("=" * 55)

print(f"\nDataset root: {RAW_DATASET_ROOT}")
print(f"Total images discovered: {len(manifest)}")

print("\nImages by domain:")
print(manifest["domain"].fillna("UNRECOGNIZED").value_counts().to_string())

print("\nImages by class:")
print(manifest["label"].fillna("UNRECOGNIZED").value_counts().to_string())

print("\nDomain x class:")
print(
    pd.crosstab(
        manifest["domain"].fillna("UNRECOGNIZED"),
        manifest["label"].fillna("UNRECOGNIZED")
    ).to_string()
)

unrecognized = manifest[
    manifest["domain"].isna() | manifest["label"].isna()
]

print(f"\nRows with unrecognized domain or label: {len(unrecognized)}")

if not unrecognized.empty:
    print("\nFirst 10 unrecognized paths:")
    print(unrecognized["relative_path"].head(10).to_string(index=False))

print(f"\nManifest saved to:\n{manifest_path}")
print("\nOriginal images were not modified.")
