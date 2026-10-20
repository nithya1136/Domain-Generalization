
from pathlib import Path
import pandas as pd
import numpy as np
import nrrd

BASE_DIR = Path(r"C:\projects\FYP\Techarch1")
INPUT_CSV = BASE_DIR / "outputs" / "01_metadata" / "niaid_inference_input.csv"
PREPROCESSED_MANIFEST = (
    BASE_DIR / "outputs" / "01_metadata" / "preprocessed_manifest.csv"
)
OUTPUT_CSV = BASE_DIR / "outputs" / "01_metadata" / "mask_manifest.csv"

# Load NIAID's explicit image-to-mask mapping.
pairs = pd.read_csv(INPUT_CSV)
manifest = pd.read_csv(PREPROCESSED_MANIFEST)

# Confirm that the input paths and mask paths are unique.
assert pairs["cxr_file"].is_unique, "Duplicate X-ray paths in NIAID CSV."
assert pairs["pred_seg_file"].is_unique, "Duplicate mask paths in NIAID CSV."

# Match each preprocessed image to its metadata.
manifest["preprocessed_path"] = manifest["preprocessed_path"].astype(str)
pairs["cxr_file"] = pairs["cxr_file"].astype(str)
pairs["pred_seg_file"] = pairs["pred_seg_file"].astype(str)

result = manifest.merge(
    pairs,
    left_on="preprocessed_path",
    right_on="cxr_file",
    how="left",
    validate="one_to_one"
)

# Verify every image has a corresponding mask file.
result["mask_exists"] = result["pred_seg_file"].apply(
    lambda p: isinstance(p, str) and Path(p).is_file()
)

print("Images in preprocessing manifest:", len(manifest))
print("Images mapped to masks:", result["pred_seg_file"].notna().sum())
print("Existing mask files:", result["mask_exists"].sum())
print("Missing mask mappings:", result["pred_seg_file"].isna().sum())
print("Missing mask files:", (~result["mask_exists"]).sum())

# Validate NRRD readability and record mask dimensions.
mask_shapes = []
mask_errors = []

for _, row in result.iterrows():
    mask_path = row["pred_seg_file"]

    try:
        if not isinstance(mask_path, str) or not Path(mask_path).is_file():
            raise FileNotFoundError(str(mask_path))

        mask, _ = nrrd.read(mask_path)
        mask_shapes.append("x".join(map(str, mask.shape)))
        mask_errors.append("")
    except Exception as exc:
        mask_shapes.append("")
        mask_errors.append(str(exc))

result["mask_shape"] = mask_shapes
result["mask_error"] = mask_errors

result.to_csv(OUTPUT_CSV, index=False)

print("\nUnreadable masks:", sum(bool(e) for e in mask_errors))
print("Mapping manifest saved to:", OUTPUT_CSV)
print("\nSample mapping:")
print(
    result[
        ["preprocessed_path", "domain", "label",
         "pred_seg_file", "mask_shape"]
    ].head().to_string(index=False)
)
