
from pathlib import Path
import subprocess
import os
import sys
import pandas as pd

PROJECT_ROOT = Path(r"C:\projects\FYP\Techarch1")
PILOT_ROOT = PROJECT_ROOT / "outputs" / "14_new_domain_pilot"

INPUT_CSV = PILOT_ROOT / "niaid_inference_input.csv"
MASK_DIR = PILOT_ROOT / "segmentation_masks"
LOG_PATH = PILOT_ROOT / "niaid_segmentation.log"

NIAID_ROOT = Path(
    r"C:\projects\Final Year Project\Models\NIAID"
)

CHECKPOINT = (
    NIAID_ROOT / "segment_lung_cxr" / "data" / "weights"
    / "cxr_lung_segment_including_heart_512.pt"
)

PYTHON = Path(
    r"C:\projects\Final Year Project\venv_niaid\Scripts\python.exe"
)

for path in [INPUT_CSV, CHECKPOINT, PYTHON]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

df = pd.read_csv(INPUT_CSV)
if "cxr_file" not in df.columns:
    raise ValueError("Input CSV must contain a 'cxr_file' column.")

if len(df) != 40:
    raise ValueError(f"Expected 40 images, found {len(df)}.")

missing = [p for p in df["cxr_file"] if not Path(p).is_file()]
if missing:
    raise FileNotFoundError(f"Missing input images: {missing[:5]}")

MASK_DIR.mkdir(parents=True, exist_ok=True)

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

env = os.environ.copy()
env["PYTHONPATH"] = (
    str(NIAID_ROOT)
    + os.pathsep
    + env.get("PYTHONPATH", "")
)

print("Images:", len(df))
print("NIAID repository:", NIAID_ROOT)
print("Output masks:", MASK_DIR)
print("Starting segmentation...")

with open(LOG_PATH, "w", encoding="utf-8") as log:
    result = subprocess.run(
        command,
        cwd=str(NIAID_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )

print("NIAID exit code:", result.returncode)
print("Log:", LOG_PATH)

if result.returncode != 0:
    print("Segmentation failed. Check the log before retrying.")
    print("\nLast 30 log lines:")
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        print("\n".join(lines[-30:]))
    sys.exit(result.returncode)

print("NIAID inference finished. Next we will verify the mask files.")
