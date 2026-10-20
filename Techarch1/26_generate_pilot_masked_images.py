
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import nrrd

PROJECT = Path(r"C:\projects\FYP\Techarch1")
PILOT = PROJECT / "outputs" / "14_new_domain_pilot"

SAMPLE_CSV = PILOT / "pilot_sample.csv"
PREPROCESS_REPORT = PILOT / "preprocessing_report.csv"
MASK_DIR = PILOT / "segmentation_masks"

MASKED_DIR = PILOT / "lung_masked"
OVERLAY_DIR = PILOT / "overlays"
REPORT_PATH = PILOT / "masked_image_report.csv"

MASKED_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

samples = pd.read_csv(SAMPLE_CSV)
preprocessed = pd.read_csv(PREPROCESS_REPORT)

preprocessed = preprocessed[preprocessed["status"] == "success"]
prep_by_name = {
    Path(row["original_path"]).name: Path(row["preprocessed_path"])
    for _, row in preprocessed.iterrows()
}

results = []

for _, row in samples.iterrows():
    name = row["image_name"]
    domain = row["domain"]

    image_path = prep_by_name.get(name)
    mask_path = MASK_DIR / f"{Path(name).stem}.nrrd"

    masked_folder = MASKED_DIR / domain
    overlay_folder = OVERLAY_DIR / domain
    masked_folder.mkdir(parents=True, exist_ok=True)
    overlay_folder.mkdir(parents=True, exist_ok=True)

    masked_path = masked_folder / name
    overlay_path = overlay_folder / f"{Path(name).stem}_overlay.png"

    try:
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"Preprocessed image not found for {name}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        mask, _ = nrrd.read(str(mask_path))
        mask = np.asarray(mask)

        if mask.ndim != 2 or image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: {image.shape} vs {mask.shape}"
            )

        # Check both orientations. The existing project uses mask.T.
        # We preserve that convention here, but record the mask coverage
        # so the outputs can be visually checked before inference.
        binary_mask = mask.T > 0

        mask_fraction = float(binary_mask.mean())
        if mask_fraction < 0.05 or mask_fraction > 0.95:
            raise ValueError(
                f"Implausible mask coverage: {mask_fraction:.3f}"
            )

        masked = np.where(binary_mask, image, 0).astype(np.uint8)

        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        if not cv2.imwrite(str(masked_path), masked):
            raise IOError(f"Could not save {masked_path}")
        if not cv2.imwrite(str(overlay_path), overlay):
            raise IOError(f"Could not save {overlay_path}")

        results.append({
            "domain": domain,
            "label": int(row["label"]),
            "image_name": name,
            "preprocessed_path": str(image_path),
            "mask_path": str(mask_path),
            "masked_image_path": str(masked_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "mask_fraction": mask_fraction,
            "status": "success",
            "error": ""
        })

    except Exception as exc:
        results.append({
            "domain": domain,
            "label": int(row["label"]),
            "image_name": name,
            "preprocessed_path": str(image_path) if image_path else "",
            "mask_path": str(mask_path),
            "masked_image_path": "",
            "overlay_path": "",
            "mask_fraction": 0.0,
            "status": "failed",
            "error": str(exc)
        })
        print(f"FAILED: {name}: {exc}")

report = pd.DataFrame(results)
report.to_csv(REPORT_PATH, index=False)

print("\n--- Masked Image Summary ---")
print("Total:", len(report))
print("Successful:", (report["status"] == "success").sum())
print("Failed:", (report["status"] == "failed").sum())
print("Report:", REPORT_PATH)
print("Masked images:", MASKED_DIR)
print("Overlays:", OVERLAY_DIR)

if (report["status"] == "failed").any():
    print("\nFailures:")
    print(
        report.loc[report["status"] == "failed", ["image_name", "error"]]
        .to_string(index=False)
    )
