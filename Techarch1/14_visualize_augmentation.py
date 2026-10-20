
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location(
    "dual_view_dataset",
    ROOT / "12_dual_view_dataset.py"
)
module = module_from_spec(spec)
spec.loader.exec_module(module)

manifest = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
df = pd.read_csv(manifest)

row = df[
    (df["split"] == "train") &
    (df["views_valid"].astype(str).str.lower() == "true")
].iloc[0]

with Image.open(row["original_view"]) as img:
    original = img.convert("L")

with Image.open(row["lung_view"]) as img:
    lung = img.convert("L")

# Use the same augmentation class as the training pipeline.
transform = module.DualViewAugmentation(training=True)

aug_original, aug_lung = transform(original, lung)

fig, axes = plt.subplots(2, 2, figsize=(9, 9))

axes[0, 0].imshow(original, cmap="gray")
axes[0, 0].set_title("Original X-ray — Before")

axes[0, 1].imshow(aug_original.squeeze().numpy(), cmap="gray")
axes[0, 1].set_title("Original X-ray — Augmented")

axes[1, 0].imshow(lung, cmap="gray")
axes[1, 0].set_title("Lung-masked — Before")

axes[1, 1].imshow(aug_lung.squeeze().numpy(), cmap="gray")
axes[1, 1].set_title("Lung-masked — Augmented")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

output = ROOT / "outputs" / "06_reports" / "augmentation_preview.png"
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Saved comparison to:", output)
