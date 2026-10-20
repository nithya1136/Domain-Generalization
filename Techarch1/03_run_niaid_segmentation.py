
from pathlib import Path
import subprocess
import sys
import pandas as pd
import os

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\projects\FYP\Techarch1")

MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "01_metadata"
    / "preprocessed_manifest.csv"
)

MASK_DIR = PROJECT_ROOT / "outputs" / "03_segmentation_masks"

SEGMENTED_IMAGE_DIR = (
    PROJECT_ROOT / "outputs" / "04_segmented_images"
)

INPUT_CSV = (
    PROJECT_ROOT / "outputs" / "01_metadata"
    / "niaid_inference_input.csv"
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Existing NIAID repository and checkpoint
NIAID_ROOT = Path(
    r"C:\projects\Final Year Project\Models\NIAID"
)

INFERENCE_SCRIPT = (
    NIAID_ROOT / "segment_lung_cxr"
    / "inference" / "inference_lung_segment.py"
)

CHECKPOINT = (
    NIAID_ROOT / "segment_lung_cxr"
    / "data" / "weights"
    / "cxr_lung_segment_including_heart_512.pt"
)

# Use the existing NIAID virtual environment
PYTHON = Path(
    r"C:\projects\Final Year Project\venv_niaid\Scripts\python.exe"
)

# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

for path in [MANIFEST_PATH, INFERENCE_SCRIPT, CHECKPOINT, PYTHON]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "preprocessed_path", "domain", "label", "image_id"
}

missing = required_columns - set(manifest.columns)

if missing:
    raise ValueError(
        f"Manifest is missing columns: {sorted(missing)}"
    )

manifest = manifest.dropna(subset=["preprocessed_path"]).copy()
manifest["preprocessed_path"] = (
    manifest["preprocessed_path"].astype(str)
)

missing_images = [
    p for p in manifest["preprocessed_path"]
    if not Path(p).is_file()
]

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} preprocessed images are missing. "
        f"First examples: {missing_images[:5]}"
    )

# ============================================================
# PREPARE NIAID INPUT
# ============================================================

MASK_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# The official NIAID script expects a CSV column named cxr_file.
pd.DataFrame({
    "cxr_file": manifest["preprocessed_path"]
}).to_csv(INPUT_CSV, index=False)

print(f"Images submitted to NIAID: {len(manifest)}")
print(f"Input CSV: {INPUT_CSV}")
print(f"Mask output directory: {MASK_DIR}")

# ============================================================
# RUN OFFICIAL NIAID INFERENCE
# ============================================================

env = os.environ.copy()
env["PYTHONPATH"] = (
    str(NIAID_ROOT)
    + os.pathsep
    + env.get("PYTHONPATH", "")
)

command = [
    str(PYTHON),
    "-m",
    "segment_lung_cxr.inference.inference_lung_segment",
    str(INPUT_CSV),
    str(CHECKPOINT),
    "--img_size", "512", "512",
    "--patch_size", "512", "512",
    "--post_process", "true",
    "--output_pred_dir", str(MASK_DIR),
]

log_path = LOG_DIR / "niaid_segmentation.log"

print("\nStarting NIAID segmentation...")
print("This may take time because inference is using the configured environment.")

with open(log_path, "w", encoding="utf-8") as log_file:
    result = subprocess.run(
        command,
        cwd=str(NIAID_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

print(f"\nNIAID process exit code: {result.returncode}")
print(f"Execution log: {log_path}")

if result.returncode != 0:
    print(
        "\nSegmentation did not complete successfully. "
        "Read the log file for the actual error before retrying."
    )
    sys.exit(result.returncode)

print(
    "\nInference process finished. "
    "Next, we must map the generated masks to the original images."
)
