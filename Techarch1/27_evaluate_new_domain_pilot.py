
from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import confusion_matrix, accuracy_score


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "14_new_domain_pilot"

SAMPLE_CSV = OUTPUT_DIR / "pilot_sample.csv"
MASKED_REPORT = OUTPUT_DIR / "masked_image_report.csv"

RESULTS_DIR = OUTPUT_DIR / "model_predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "BCE": ROOT / "outputs/12_four_experiments/01_BCE/existing_baseline_checkpoint.pt",
    "BCE_ACDSA": ROOT / "outputs/12_four_experiments/02_BCE_ACDSA/best_model.pt",
    "BCE_SupCon": ROOT / "outputs/12_four_experiments/03_BCE_SupCon_Run2/best_model.pt",
    "BCE_ACDSA_SupCon": ROOT / "outputs/12_four_experiments/04_BCE_ACDSA_SupCon/best_model.pt",
}

IMAGE_SIZE = 224
THRESHOLD = 0.5
BATCH_SIZE = 8


# ---------------------------------------------------------
# 2. LOAD THE EXISTING MODEL CLASS
# ---------------------------------------------------------
MODEL_FILE = ROOT / "15_joint_learning_model.py"

spec = importlib.util.spec_from_file_location(
    "joint_learning_model",
    MODEL_FILE,
)
model_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_module
spec.loader.exec_module(model_module)

JointLearningModel = model_module.JointLearningModel


# ---------------------------------------------------------
# 3. DEVICE
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("NEW-DOMAIN DUAL-VIEW PILOT EVALUATION")
print("=" * 65)
print(f"Device: {DEVICE}")
print(f"Threshold: {THRESHOLD}")
print(f"Output folder: {RESULTS_DIR}")


# ---------------------------------------------------------
# 4. VALIDATE INPUT FILES
# ---------------------------------------------------------
for path in [SAMPLE_CSV, MASKED_REPORT, MODEL_FILE]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

for name, path in CHECKPOINTS.items():
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint for {name} not found: {path}"
        )

samples = pd.read_csv(SAMPLE_CSV)
mask_report = pd.read_csv(MASKED_REPORT)

required_sample_cols = {
    "domain", "label", "image_path", "image_name"
}
missing = required_sample_cols - set(samples.columns)
if missing:
    raise ValueError(
        f"Sample manifest missing columns: {sorted(missing)}"
    )

required_mask_cols = {
    "domain", "image_name", "masked_image_path", "status"
}
missing = required_mask_cols - set(mask_report.columns)
if missing:
    raise ValueError(
        f"Mask report missing columns: {sorted(missing)}"
    )

# Keep only successfully generated masked views.
mask_report["status"] = mask_report["status"].astype(str).str.lower()
mask_report = mask_report[mask_report["status"].isin(
    ["success", "ok", "true"]
)].copy()

# Use domain + filename to match each original image to its lung view.
samples["image_name"] = samples["image_name"].astype(str)
mask_report["image_name"] = mask_report["image_name"].astype(str)

samples = samples.merge(
    mask_report[["domain", "image_name", "masked_image_path"]],
    on=["domain", "image_name"],
    how="left",
    validate="one_to_one",
)

if samples["masked_image_path"].isna().any():
    missing_rows = samples[samples["masked_image_path"].isna()]
    raise ValueError(
        "Some pilot images have no matching lung-masked view:\n"
        + missing_rows[["domain", "image_name"]].to_string(index=False)
    )

# Confirm every image pair exists.
for _, row in samples.iterrows():
    for col in ["image_path", "masked_image_path"]:
        path = Path(row[col])
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

print(f"Pilot samples loaded: {len(samples)}")
print("\nSamples by domain:")
print(samples.groupby(["domain", "label"]).size().to_string())


# ---------------------------------------------------------
# 5. PREPROCESS EXACTLY LIKE VALIDATION/TEST
# ---------------------------------------------------------
def load_grayscale_tensor(path):
    with Image.open(path) as image:
        image = image.convert("L")
        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )
        tensor = TF.to_tensor(image)
    return tensor


# ---------------------------------------------------------
# 6. CHECKPOINT LOADING
# ---------------------------------------------------------
def load_checkpoint(model, checkpoint_path):
    # weights_only=False supports checkpoints that contain
    # additional training metadata. Only load checkpoints you trust.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # A plain state_dict is also a dictionary.
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    # Handle checkpoints saved from DataParallel, if applicable.
    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------
# 7. RUN INFERENCE
# ---------------------------------------------------------
all_results = []

for model_name, checkpoint_path in CHECKPOINTS.items():
    print("\n" + "-" * 65)
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_path.name}")

    model = JointLearningModel(
        pretrained=False,
        dropout=0.3,
    )
    model = load_checkpoint(model, checkpoint_path)

    model_results = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch_rows = samples.iloc[start:start + BATCH_SIZE]

        original_batch = torch.stack([
            load_grayscale_tensor(path)
            for path in batch_rows["image_path"]
        ])

        lung_batch = torch.stack([
            load_grayscale_tensor(path)
            for path in batch_rows["masked_image_path"]
        ])

        original_batch = original_batch.to(DEVICE)
        lung_batch = lung_batch.to(DEVICE)

        with torch.inference_mode():
            output = model(original_batch, lung_batch)
            logits = output["logits"].reshape(-1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        for (_, row), probability in zip(
            batch_rows.iterrows(), probabilities
        ):
            predicted_label = int(probability >= THRESHOLD)

            record = {
                "model": model_name,
                "domain": row["domain"],
                "image_name": row["image_name"],
                "true_label": int(row["label"]),
                "tb_probability": float(probability),
                "predicted_label": predicted_label,
                "threshold": THRESHOLD,
                "correct": predicted_label == int(row["label"]),
                "original_path": row["image_path"],
                "lung_masked_path": row["masked_image_path"],
            }

            model_results.append(record)
            all_results.append(record)

    model_df = pd.DataFrame(model_results)
    per_model_path = RESULTS_DIR / f"{model_name}_predictions.csv"
    model_df.to_csv(per_model_path, index=False)

    print(f"Saved: {per_model_path}")
    print(
        f"Predicted TB for "
        f"{int(model_df['predicted_label'].sum())}/"
        f"{len(model_df)} samples"
    )

    del model


# ---------------------------------------------------------
# 8. SAVE ALL PER-IMAGE PREDICTIONS
# ---------------------------------------------------------
all_df = pd.DataFrame(all_results)
all_predictions_path = RESULTS_DIR / "all_model_predictions.csv"
all_df.to_csv(all_predictions_path, index=False)


# ---------------------------------------------------------
# 9. PILOT METRICS
# ---------------------------------------------------------
summary_rows = []

for model_name in CHECKPOINTS:
    model_df = all_df[all_df["model"] == model_name]

    for domain_name, domain_df in model_df.groupby("domain"):
        y_true = domain_df["true_label"].to_numpy(dtype=int)
        y_pred = domain_df["predicted_label"].to_numpy(dtype=int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        accuracy = accuracy_score(y_true, y_pred)

        # These are meaningful for the single-class pilot domains:
        # Domain3_Normal: specificity = TN / (TN + FP)
        # Domain4_TB: sensitivity = TP / (TP + FN)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan

        summary_rows.append({
            "model": model_name,
            "domain": domain_name,
            "n_samples": len(domain_df),
            "correct": int((y_true == y_pred).sum()),
            "accuracy_on_this_single_class_sample": float(accuracy),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "false_positive_rate": (
                float(fp / (fp + tn)) if (fp + tn) else np.nan
            ),
            "false_negative_rate": (
                float(fn / (fn + tp)) if (fn + tp) else np.nan
            ),
            "threshold": THRESHOLD,
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_DIR / "pilot_metrics_summary.csv"
summary_df.to_csv(summary_path, index=False)


# ---------------------------------------------------------
# 10. HUMAN-READABLE REPORT
# ---------------------------------------------------------
report_path = RESULTS_DIR / "pilot_metrics_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("NEW-DOMAIN PILOT EVALUATION\n")
    report.write("=" * 65 + "\n")
    report.write(f"Device: {DEVICE}\n")
    report.write(f"Threshold: {THRESHOLD}\n")
    report.write(f"Total pilot images: {len(samples)}\n\n")

    report.write("Samples by domain and label:\n")
    report.write(
        samples.groupby(["domain", "label"])
        .size()
        .to_string()
        + "\n\n"
    )

    report.write("Pilot metrics by model and domain:\n")
    report.write(summary_df.to_string(index=False))
    report.write("\n\nINTERPRETATION NOTES\n")
    report.write(
        "- Domain3_Normal contains Normal images only. "
        "Its specificity and false-positive rate are the main metrics.\n"
    )
    report.write(
        "- Domain4_TB contains TB images only. "
        "Its sensitivity and false-negative rate are the main metrics.\n"
    )
    report.write(
        "- These two single-class samples do not support a full "
        "two-class accuracy, balanced accuracy, or AUROC assessment.\n"
    )
    report.write(
        "- This is a small pilot sample, not a definitive "
        "cross-domain generalization result.\n"
    )

print("\n" + "=" * 65)
print("PILOT INFERENCE COMPLETED")
print("=" * 65)
print(f"All predictions: {all_predictions_path}")
print(f"Metrics CSV:     {summary_path}")
print(f"Text report:     {report_path}")
print("\nSummary:")
print(summary_df.to_string(index=False))
