
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\projects\FYP\Techarch1")
INPUT = BASE / "outputs/01_metadata/mask_manifest.csv"
REPORT = BASE / "outputs/06_reports/mask_generation_report.csv"
OUTPUT = BASE / "outputs/01_metadata/dual_view_manifest.csv"

df = pd.read_csv(INPUT)
report = pd.read_csv(REPORT)

# Keep only images whose masked images and overlays were generated.
report = report[report["status"] == "success"].copy()

report["preprocessed_path"] = report["preprocessed_path"].astype(str)
df["preprocessed_path"] = df["preprocessed_path"].astype(str)

df = df.merge(
    report[["preprocessed_path", "masked_image_path", "overlay_path"]],
    on="preprocessed_path",
    how="inner",
    validate="one_to_one"
)

df["original_view"] = df["preprocessed_path"]
df["lung_view"] = df["masked_image_path"]

df["original_exists"] = df["original_view"].map(
    lambda p: Path(p).is_file()
)
df["lung_view_exists"] = df["lung_view"].map(
    lambda p: Path(p).is_file()
)

df["views_valid"] = df["original_exists"] & df["lung_view_exists"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Rows in dual-view manifest:", len(df))
print("Valid original views:", int(df["original_exists"].sum()))
print("Valid lung views:", int(df["lung_view_exists"].sum()))
print("Rows with both views:", int(df["views_valid"].sum()))
print("Domains:", df["domain"].value_counts().to_dict())
print("Classes:", df["label"].value_counts().to_dict())
print("Saved to:", OUTPUT)
