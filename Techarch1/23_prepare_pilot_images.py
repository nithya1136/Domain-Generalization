
from pathlib import Path
import csv
import cv2
import pandas as pd
import numpy as np

PROJECT = Path(r"C:\projects\FYP\Techarch1")
SAMPLE_CSV = PROJECT / "outputs/14_new_domain_pilot/pilot_sample.csv"

OUTPUT_ROOT = PROJECT / "outputs/14_new_domain_pilot"
PREPROCESSED = OUTPUT_ROOT / "preprocessed"
INPUT_CSV = OUTPUT_ROOT / "niaid_inference_input.csv"

PREPROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SAMPLE_CSV)
results = []

def preprocess_image(source_path, destination_path, size=512):
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    h, w = image.shape
    side = max(h, w)

    # Center-pad to square without stretching the original image.
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    square = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    resized = cv2.resize(
        square, (size, size), interpolation=cv2.INTER_AREA
    )

    if not cv2.imwrite(str(destination_path), resized):
        raise IOError(f"Could not save: {destination_path}")

    return image.shape, resized.shape

for _, row in df.iterrows():
    source = Path(row["image_path"])
    domain = row["domain"]

    destination_dir = PREPROCESSED / domain
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    try:
        original_shape, output_shape = preprocess_image(source, destination)

        results.append({
            "domain": domain,
            "label": int(row["label"]),
            "image_name": source.name,
            "original_path": str(source),
            "preprocessed_path": str(destination.resolve()),
            "status": "success",
            "error": ""
        })

        print(f"OK: {domain}/{source.name}: {original_shape} -> {output_shape}")

    except Exception as exc:
        results.append({
            "domain": domain,
            "label": int(row["label"]),
            "image_name": source.name,
            "original_path": str(source),
            "preprocessed_path": "",
            "status": "failed",
            "error": str(exc)
        })
        print(f"FAILED: {source.name}: {exc}")

report_path = OUTPUT_ROOT / "preprocessing_report.csv"
pd.DataFrame(results).to_csv(report_path, index=False)

successful = [r for r in results if r["status"] == "success"]

pd.DataFrame({
    "cxr_file": [r["preprocessed_path"] for r in successful]
}).to_csv(INPUT_CSV, index=False)

print("\n--- Summary ---")
print("Selected images:", len(df))
print("Preprocessed successfully:", len(successful))
print("Failed:", len(df) - len(successful))
print("NIAID input CSV:", INPUT_CSV)
print("Report:", report_path)
