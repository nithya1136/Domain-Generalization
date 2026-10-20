
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
OUT = ROOT / "outputs" / "13_model_analysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
THRESHOLD = 0.5

CHECKPOINTS = {
    "BCE": ROOT / "outputs/12_four_experiments/01_BCE/existing_baseline_checkpoint.pt",
    "BCE_ACDSA": ROOT / "outputs/12_four_experiments/02_BCE_ACDSA/best_model.pt",
    "BCE_SupCon": ROOT / "outputs/12_four_experiments/03_BCE_SupCon_Run2/best_model.pt",
    "BCE_ACDSA_SupCon": ROOT / "outputs/12_four_experiments/04_BCE_ACDSA_SupCon/best_model.pt",
}


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path):
    try:
        checkpoint = torch.load(
            path, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        f"Could not identify model weights in {path}. "
        f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
    )


def calculate_metrics(y, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "sensitivity_TB": recall_score(y, predictions, zero_division=0),
        "specificity_Normal": tn / (tn + fp) if tn + fp else np.nan,
        "precision_TB": precision_score(y, predictions, zero_division=0),
        "F1_TB": f1_score(y, predictions, zero_division=0),
        "AUROC": roc_auc_score(y, probabilities),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "mean_prob_Normal": probabilities[y == 0].mean(),
        "mean_prob_TB": probabilities[y == 1].mean(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)

    # Restrict the manifest to source-domain validation BEFORE loading.
    df = pd.read_csv(MANIFEST)
    val_df = df[
        (df["domain"] == "Domain1")
        & (df["split"] == "validation")
        & (df["views_valid"].astype(str).str.lower() == "true")
    ].copy()

    if val_df.empty:
        raise RuntimeError("No Domain1 validation images found.")

    val_manifest = OUT / "domain1_validation_only.csv"
    val_df.to_csv(val_manifest, index=False)

    dataset_module = load_module(
        "12_dual_view_dataset.py", "analysis_dataset"
    )
    model_module = load_module(
        "15_joint_learning_model.py", "analysis_model"
    )
    DatasetClass = dataset_module.DualViewChestXrayDataset
    ModelClass = model_module.JointLearningModel

    # The dataset class accepts a manifest and split, so use a
    # temporary source-validation-only manifest to prevent domain leakage.
    dataset = DatasetClass(
        manifest_path=val_manifest,
        split="validation",
        image_size=224,
        verify_paths=True,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    labels, image_ids = [], []
    original_batches, lung_batches = [], []

    for batch in loader:
        labels.extend(batch["label"].numpy().astype(int).tolist())
        image_ids.extend([str(x) for x in batch["image_id"]])
        original_batches.append(batch["original"])
        lung_batches.append(batch["lung"])

    y = np.asarray(labels, dtype=int)
    original_all = torch.cat(original_batches)
    lung_all = torch.cat(lung_batches)

    print("Domain1 validation samples:", len(y))
    print("Normal:", int((y == 0).sum()), "| TB:", int((y == 1).sum()))

    if len(np.unique(y)) != 2:
        raise RuntimeError("Validation set must contain both classes.")

    results = []
    all_probs = {}

    for name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nLoading {name}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        model = ModelClass(pretrained=False).to(DEVICE)
        state = load_checkpoint(checkpoint_path)

        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        model.eval()

        probs = []
        with torch.no_grad():
            for start in range(0, len(y), BATCH_SIZE):
                original = original_all[start:start + BATCH_SIZE].to(DEVICE)
                lung = lung_all[start:start + BATCH_SIZE].to(DEVICE)
                output = model(original, lung)
                probabilities = torch.sigmoid(
                    output["logits"].reshape(-1)
                )
                probs.extend(probabilities.cpu().numpy().tolist())

        probs = np.asarray(probs)
        all_probs[name] = probs
        metrics = calculate_metrics(y, probs, THRESHOLD)
        results.append({"model": name, **metrics})

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Balanced accuracy={metrics['balanced_accuracy']:.4f} | "
            f"Sensitivity={metrics['sensitivity_TB']:.4f} | "
            f"Specificity={metrics['specificity_Normal']:.4f} | "
            f"AUROC={metrics['AUROC']:.4f}"
        )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(
        OUT / "validation_metrics_threshold_0.5.csv", index=False
    )

    predictions_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y,
    })
    for name, probs in all_probs.items():
        predictions_df[f"{name}_prob_TB"] = probs
        predictions_df[f"{name}_prediction_at_0.5"] = (
            probs >= THRESHOLD
        ).astype(int)

    predictions_df.to_csv(
        OUT / "validation_predictions.csv", index=False
    )

    # ROC curves: source validation only.
    plt.figure(figsize=(8, 6))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Domain1 Validation ROC Curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "validation_roc_curves.png", dpi=160)
    plt.close()

    # Probability distributions, separated by true class.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (name, probs) in zip(axes.ravel(), all_probs.items()):
        ax.hist(probs[y == 0], bins=20, alpha=0.65, label="Normal")
        ax.hist(probs[y == 1], bins=20, alpha=0.65, label="TB")
        ax.axvline(THRESHOLD, linestyle="--", label="Threshold 0.5")
        ax.set_title(name)
        ax.set_xlabel("Predicted TB probability")
        ax.set_ylabel("Number of images")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "validation_probability_distributions.png", dpi=160)
    plt.close(fig)

    print("\nAnalysis complete. Results saved to:")
    print(OUT)
    print("\nMetrics at threshold 0.5:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
0