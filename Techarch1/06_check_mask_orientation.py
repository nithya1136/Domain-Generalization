
from pathlib import Path

import cv2
import numpy as np
import nrrd

BASE = Path(r"C:\projects\FYP\Techarch1")
image_path = BASE / "outputs/02_preprocessed/Domain1/Normal/00000_Normal-1.png"
mask_path = BASE / "outputs/03_segmentation_masks/00000_Normal-1.nrrd"
output_path = BASE / "outputs/06_reports/orientation_comparison.png"

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
mask, header = nrrd.read(str(mask_path))
mask = (mask > 0).astype(np.uint8)

if image is None:
    raise FileNotFoundError(image_path)

if image.shape != mask.shape:
    raise ValueError(f"Shape mismatch: image={image.shape}, mask={mask.shape}")

# Compare the original mask and three orientation variants.
variants = [
    ("Original", mask),
    ("Flip vertical", np.flipud(mask)),
    ("Flip horizontal", np.fliplr(mask)),
    ("Transpose", mask.T),
]

panels = []

for title, candidate in variants:
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(
        np.ascontiguousarray(candidate),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel, contours, -1, (0, 255, 0), 2)

    cv2.putText(
        panel, title, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    panels.append(panel)

comparison = np.vstack([
    np.hstack(panels[:2]),
    np.hstack(panels[2:])
])

output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), comparison)

print("Saved comparison to:", output_path)
