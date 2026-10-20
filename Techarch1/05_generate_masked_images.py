
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import nrrd

BASE_DIR = Path(r"C:\projects\FYP\Techarch1")
MANIFEST = BASE_DIR / "outputs" / "01_metadata" / "mask_manifest.csv"

MASKED_ROOT = BASE_DIR / "outputs" / "04_lung_masked"
OVERLAY_ROOT = BASE_DIR / "outputs" / "05_segmentation_overlays"
REPORT_PATH = BASE_DIR / "outputs" / "06_reports" / "mask_generation_report.csv"

MASKED_ROOT.mkdir(parents=True, exist_ok=True)
OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)
results = []

for index, row in df.iterrows():
    image_path = Path(row["preprocessed_path"])
    mask_path = Path(row["pred_seg_file"])

    # Preserve domain and class folder structure.
    domain = str(row["domain"])
    label = str(row["label"])

    masked_dir = MASKED_ROOT / domain / label
    overlay_dir = OVERLAY_ROOT / domain / label
    masked_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    masked_path = masked_dir / image_path.name
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read X-ray: {image_path}")

        mask, _ = nrrd.read(str(mask_path))
        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask; received shape {mask.shape}")

        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask size mismatch: {image.shape} vs {mask.shape}"
            )

        # NIAID masks may contain labels or probabilities.
        # Convert to a binary lung-region mask.
        binary_mask = mask.T > 0

        if binary_mask.sum() == 0:
            raise ValueError("Mask is empty")

        # Retain lung-region pixels; set the outside region to black.
        masked_image = np.where(binary_mask, image, 0).astype(np.uint8)

        # Create a visual overlay using the mask boundary.
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        if not cv2.imwrite(str(masked_path), masked_image):
            raise IOError(f"Could not save masked image: {masked_path}")

        if not cv2.imwrite(str(overlay_path), overlay):
            raise IOError(f"Could not save overlay: {overlay_path}")

        results.append({
            "image_id": row.get("image_id", image_path.stem),
            "preprocessed_path": str(image_path),
            "mask_path": str(mask_path),
            "masked_image_path": str(masked_path),
            "overlay_path": str(overlay_path),
            "domain": domain,
            "label": label,
            "mask_pixels": int(binary_mask.sum()),
            "mask_fraction": float(binary_mask.mean()),
            "status": "success",
            "error": ""
        })

    except Exception as exc:
        results.append({
            "image_id": row.get("image_id", image_path.stem),
            "preprocessed_path": str(image_path),
            "mask_path": str(mask_path),
            "masked_image_path": "",
            "overlay_path": "",
            "domain": domain,
            "label": label,
            "mask_pixels": 0,
            "mask_fraction": 0,
            "status": "failed",
            "error": str(exc)
        })

    if (index + 1) % 250 == 0:
        print(f"Processed {index + 1}/{len(df)} images")

report = pd.DataFrame(results)
report.to_csv(REPORT_PATH, index=False)

print("\n--- Generation Summary ---")
print("Total images:", len(report))
print("Successful:", (report["status"] == "success").sum())
print("Failed:", (report["status"] == "failed").sum())
print("Report:", REPORT_PATH)

if (report["status"] == "failed").any():
    print("\nFirst failures:")
    print(
        report.loc[report["status"] == "failed",
                   ["preprocessed_path", "error"]]
        .head(10).to_string(index=False)
    )
