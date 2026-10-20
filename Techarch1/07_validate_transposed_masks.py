
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import nrrd

BASE = Path(r"C:\projects\FYP\Techarch1")
MANIFEST = BASE / "outputs/01_metadata/mask_manifest.csv"
OUTPUT = BASE / "outputs/06_reports/transpose_validation"

df = pd.read_csv(MANIFEST)

# One representative image for each domain/class combination.
samples = (
    df.groupby(["domain", "label"], sort=True)
      .head(1)
      .reset_index(drop=True)
)

OUTPUT.mkdir(parents=True, exist_ok=True)

for _, row in samples.iterrows():
    image = cv2.imread(
        str(row["preprocessed_path"]), cv2.IMREAD_GRAYSCALE
    )
    mask, _ = nrrd.read(str(row["pred_seg_file"]))

    if image is None:
        print("Could not read image:", row["preprocessed_path"])
        continue

    # Candidate orientation based on the previous comparison.
    mask = (mask.T > 0).astype(np.uint8)

    if image.shape != mask.shape:
        print("Shape mismatch:", row["preprocessed_path"])
        continue

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    output_name = (
        f"{row['domain']}_{row['label']}_transpose.png"
    )
    output_path = OUTPUT / output_name
    cv2.imwrite(str(output_path), overlay)

    print(output_path)

print("\nValidation overlays generated:", len(samples))
