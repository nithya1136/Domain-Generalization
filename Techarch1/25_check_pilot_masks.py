
from pathlib import Path
import pandas as pd
import nrrd
import numpy as np

PROJECT = Path(r"C:\projects\FYP\Techarch1")
PILOT = PROJECT / "outputs" / "14_new_domain_pilot"

INPUT_CSV = PILOT / "niaid_inference_input.csv"
MASK_DIR = PILOT / "segmentation_masks"

df = pd.read_csv(INPUT_CSV)
inputs = [Path(p) for p in df["cxr_file"]]

mask_files = [
    p for p in MASK_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() == ".nrrd"
]

print("Input images:", len(inputs))
print("NRRD mask files found:", len(mask_files))
print("\nFirst 10 mask paths:")
for p in mask_files[:10]:
    print(p)

print("\n--- Mask validation ---")
valid = 0
failed = []

for mask_path in mask_files:
    try:
        mask, header = nrrd.read(str(mask_path))
        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.shape}")

        if mask.size == 0:
            raise ValueError("Empty mask")

        if not np.isfinite(mask).all():
            raise ValueError("Mask contains non-finite values")

        if np.unique(mask).size < 2:
            raise ValueError(
                f"Mask has only one value: {np.unique(mask)[:5]}"
            )

        valid += 1
        print(
            f"OK: {mask_path.name} | shape={mask.shape} "
            f"| values={np.unique(mask)[:10]}"
        )

    except Exception as exc:
        failed.append((str(mask_path), str(exc)))
        print(f"FAILED: {mask_path.name} | {exc}")

print("\n--- Summary ---")
print("Valid masks:", valid)
print("Failed masks:", len(failed))

if len(mask_files) != len(inputs):
    print(
        "WARNING: Mask count does not match input count. "
        "We must map masks to images before proceeding."
    )
