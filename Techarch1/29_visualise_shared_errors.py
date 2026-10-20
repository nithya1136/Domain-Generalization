
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent

csv_path = (
    ROOT / "outputs" / "15_prediction_error_analysis"
    / "images_misclassified_by_all_models.csv"
)

output_path = (
    ROOT / "outputs" / "15_prediction_error_analysis"
    / "shared_normal_errors_visual.png"
)

df = pd.read_csv(csv_path)
df = df[df["true_label_binary"] == 0].head(10).reset_index(drop=True)

if df.empty:
    raise ValueError("No shared Normal misclassifications found.")

fig, axes = plt.subplots(
    len(df), 2,
    figsize=(10, max(4, len(df) * 3)),
    squeeze=False
)

for i, row in df.iterrows():
    original_path = Path(row["original_view"])
    lung_path = Path(row["lung_view"])

    with Image.open(original_path) as image:
        original = image.convert("L").copy()

    with Image.open(lung_path) as image:
        lung = image.convert("L").copy()

    axes[i, 0].imshow(original, cmap="gray")
    axes[i, 0].set_title(f'{row["image_id"]} — Original')
    axes[i, 0].axis("off")

    axes[i, 1].imshow(lung, cmap="gray")
    axes[i, 1].set_title("Lung-masked")
    axes[i, 1].axis("off")

plt.tight_layout()
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved comparison sheet:", output_path)
print("Images displayed:", len(df))
