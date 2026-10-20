
"""
Experiment 4: BCE + ACDSA + Supervised Contrastive Learning

Protocol:
    Train       : Domain1 train only
    Checkpoint  : Domain1 validation BCE only
    Final test  : Domain2 test only

Previous experiment outputs are preserved.
"""

from pathlib import Path
import importlib.util
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15
PATIENCE = 4

LAMBDA_ACDSA = 0.1
SUPCON_WEIGHT = 0.1
TEMPERATURE = 0.07

SOURCE_DOMAIN = "Domain1"
TARGET_DOMAIN = "Domain2"

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
OUTPUT = ROOT / "outputs" / "12_four_experiments" / "04_BCE_ACDSA_SupCon"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 3. LOAD EXISTING PROJECT MODULES
# =========================================================

def load_module(filename, module_name):
    path = ROOT / filename

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================
# 4. SUPCON PROJECTION HEAD
# =========================================================

class ProjectionHead(nn.Module):
    """Project 512-dimensional features into a 128-D embedding."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, features):
        return self.network(features)


# =========================================================
# 5. ACDSA PAIRED-VIEW ALIGNMENT LOSS
# =========================================================

def acdsa_alignment_loss(original_features, lung_features):
    """
    Align normalized original-image and lung-view features.
    """

    original_norm = F.normalize(
        original_features, p=2, dim=1, eps=1e-8
    )
    lung_norm = F.normalize(
        lung_features, p=2, dim=1, eps=1e-8
    )

    return ((original_norm - lung_norm) ** 2).sum(dim=1).mean()


# =========================================================
# 6. SUPERVISED CONTRASTIVE LOSS
# =========================================================

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Same-class embeddings are positives.
    Different-class embeddings are negatives.
    Self-comparisons are excluded.
    """

    features = F.normalize(features, dim=1)
    labels = labels.reshape(-1, 1)

    similarity = torch.matmul(features, features.T) / temperature

    self_mask = torch.eye(
        features.size(0),
        device=features.device,
        dtype=torch.bool,
    )

    similarity = similarity.masked_fill(self_mask, -1e9)

    positive_mask = (
        (labels == labels.T) & (~self_mask)
    ).float()

    log_prob = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True
    )

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0

    if not valid.any():
        return features.sum() * 0.0

    mean_positive_log_prob = (
        (positive_mask * log_prob).sum(dim=1)
        / positive_count.clamp_min(1.0)
    )

    return -mean_positive_log_prob[valid].mean()


# =========================================================
# 7. EVALUATION HELPERS
# =========================================================

def unpack_batch(batch):
    return batch["original"], batch["lung"], batch["label"]


@torch.no_grad()
def validation_bce(model, loader, criterion):
    """Use source-validation BCE only for checkpoint selection."""

    model.eval()
    loss_sum = 0.0
    count = 0

    for batch in loader:
        original, lung, labels = unpack_batch(batch)

        original = original.to(DEVICE, dtype=torch.float32)
        lung = lung.to(DEVICE, dtype=torch.float32)
        labels = labels.to(DEVICE, dtype=torch.float32).reshape(-1)

        output = model(original, lung)
        logits = output["logits"].reshape(-1)
        loss = criterion(logits, labels)

        n = labels.size(0)
        loss_sum += loss.item() * n
        count += n

    return loss_sum / count


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()

    all_labels = []
    all_probabilities = []

    for batch in loader:
        original, lung, labels = unpack_batch(batch)

        original = original.to(DEVICE, dtype=torch.float32)
        lung = lung.to(DEVICE, dtype=torch.float32)

        output = model(original, lung)
        probabilities = torch.sigmoid(output["logits"].reshape(-1))

        all_labels.extend(labels.cpu().numpy().reshape(-1).astype(int))
        all_probabilities.extend(probabilities.cpu().numpy().reshape(-1))

    return np.asarray(all_labels), np.asarray(all_probabilities)


def calculate_metrics(labels, probabilities):
    # Fixed threshold; never tune it using target test labels.
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(labels, predictions, zero_division=0),
        "auroc": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# =========================================================
# 8. MAIN TRAINING PIPELINE
# =========================================================

def main():
    set_seed(SEED)

    print("=" * 65)
    print("EXPERIMENT 4: BCE + ACDSA + SUPCON")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Manifest:", MANIFEST)
    print("Output:", OUTPUT)

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    # Do not overwrite any existing experiment output.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty:\n{OUTPUT}\n"
            "Preserve existing results and choose a new directory "
            "if a separate run is required."
        )

    df = pd.read_csv(MANIFEST)

    required = {
        "domain", "split", "label", "views_valid",
        "original_view", "lung_view", "image_id",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df = df[
        df["views_valid"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    source_train = df[
        (df["domain"] == SOURCE_DOMAIN)
        & (df["split"] == "train")
    ].copy()

    source_val = df[
        (df["domain"] == SOURCE_DOMAIN)
        & (df["split"] == "validation")
    ].copy()

    target_test = df[
        (df["domain"] == TARGET_DOMAIN)
        & (df["split"] == "test")
    ].copy()

    subsets = {
        "source train": source_train,
        "source validation": source_val,
        "target test": target_test,
    }

    for name, part in subsets.items():
        if part.empty:
            raise ValueError(f"No samples found for {name}.")

        if not part["label"].isin(["Normal", "Tuberculosis"]).all():
            raise ValueError(f"Unexpected class labels in {name}.")

        missing_paths = (
            ~part["original_view"].map(lambda p: Path(p).is_file())
            | ~part["lung_view"].map(lambda p: Path(p).is_file())
        )

        if missing_paths.any():
            raise FileNotFoundError(
                f"Missing images in {name}. Example IDs: "
                f"{part.loc[missing_paths, 'image_id'].head().tolist()}"
            )

    # Guard against accidental image overlap between source and target.
    train_ids = set(source_train["image_id"])
    val_ids = set(source_val["image_id"])
    test_ids = set(target_test["image_id"])

    if train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Source and target image IDs overlap.")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Create isolated manifests without modifying the master manifest.
    split_paths = {
        "train": OUTPUT / "source_train_manifest.csv",
        "validation": OUTPUT / "source_validation_manifest.csv",
        "test": OUTPUT / "target_test_manifest.csv",
    }

    for name, frame in [
        ("train", source_train),
        ("validation", source_val),
        ("test", target_test),
    ]:
        isolated = frame.copy()
        isolated["split"] = name
        isolated.to_csv(split_paths[name], index=False)

    print("\nDomain1 training images:", len(source_train))
    print("Domain1 validation images:", len(source_val))
    print("Domain2 test images:", len(target_test))
    print("ACDSA weight:", LAMBDA_ACDSA)
    print("SupCon weight:", SUPCON_WEIGHT)
    print("SupCon temperature:", TEMPERATURE)

    dataset_module = load_module(
        "12_dual_view_dataset.py", "dual_view_dataset_exp4"
    )
    model_module = load_module(
        "15_joint_learning_model.py", "joint_learning_model_exp4"
    )

    DatasetClass = dataset_module.DualViewChestXrayDataset
    ModelClass = model_module.JointLearningModel

    train_dataset = DatasetClass(
        manifest_path=split_paths["train"],
        split="train",
    )
    val_dataset = DatasetClass(
        manifest_path=split_paths["validation"],
        split="validation",
    )
    test_dataset = DatasetClass(
        manifest_path=split_paths["test"],
        split="test",
    )

    print("\nLoaded paired-view datasets:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ModelClass(pretrained=True).to(DEVICE)
    projection = ProjectionHead().to(DEVICE)

    # Compute class weighting from source training labels only.
    normal_count = int((source_train["label"] == "Normal").sum())
    tb_count = int((source_train["label"] == "Tuberculosis").sum())

    if normal_count == 0 or tb_count == 0:
        raise ValueError("Both classes must be present in source training.")

    pos_weight = torch.tensor(
        [normal_count / tb_count],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    checkpoint_path = OUTPUT / "best_model.pt"

    # -------------------- TRAIN --------------------
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        projection.train()

        total_count = 0
        sum_total = 0.0
        sum_bce = 0.0
        sum_acdsa = 0.0
        sum_supcon = 0.0

        for batch in train_loader:
            original, lung, labels = unpack_batch(batch)

            original = original.to(DEVICE, dtype=torch.float32)
            lung = lung.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.float32).reshape(-1)

            optimizer.zero_grad(set_to_none=True)

            output = model(original, lung)
            logits = output["logits"].reshape(-1)

            # Objective 1: classification.
            bce_loss = criterion(logits, labels)

            # Objective 2: align paired-view features.
            acdsa_loss = acdsa_alignment_loss(
                output["original_features"],
                output["lung_features"],
            )

            # Objective 3: supervised contrastive learning.
            original_projection = projection(output["original_features"])
            lung_projection = projection(output["lung_features"])

            embeddings = torch.cat(
                [original_projection, lung_projection], dim=0
            )
            contrastive_labels = torch.cat(
                [labels.long(), labels.long()], dim=0
            )

            supcon_loss = supervised_contrastive_loss(
                embeddings,
                contrastive_labels,
                temperature=TEMPERATURE,
            )

            total_loss = (
                bce_loss
                + LAMBDA_ACDSA * acdsa_loss
                + SUPCON_WEIGHT * supcon_loss
            )

            total_loss.backward()
            optimizer.step()

            n = labels.size(0)
            total_count += n
            sum_total += total_loss.item() * n
            sum_bce += bce_loss.item() * n
            sum_acdsa += acdsa_loss.item() * n
            sum_supcon += supcon_loss.item() * n

        train_total = sum_total / total_count
        train_bce = sum_bce / total_count
        train_acdsa = sum_acdsa / total_count
        train_supcon = sum_supcon / total_count

        # Checkpoint selection uses source-validation BCE only.
        val_loss = validation_bce(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_bce_loss": train_bce,
            "train_acdsa_loss": train_acdsa,
            "train_supcon_loss": train_supcon,
            "source_validation_bce": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Total={train_total:.4f} | "
            f"BCE={train_bce:.4f} | "
            f"ACDSA={train_acdsa:.4f} | "
            f"SupCon={train_supcon:.4f} | "
            f"Val BCE={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "projection_state_dict": projection.state_dict(),
                "best_validation_bce": best_val_loss,
                "seed": SEED,
                "lambda_acdsa": LAMBDA_ACDSA,
                "supcon_weight": SUPCON_WEIGHT,
                "temperature": TEMPERATURE,
                "source_domain": SOURCE_DOMAIN,
                "target_domain": TARGET_DOMAIN,
            }, checkpoint_path)

            print("  Saved best source-validation checkpoint.")
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(
        OUTPUT / "training_history.csv", index=False
    )

    # -------------------- FINAL TARGET TEST --------------------
    # Domain2 is used only after training and checkpoint selection.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    projection.load_state_dict(checkpoint["projection_state_dict"])

    labels, probabilities = get_predictions(model, test_loader)
    metrics = calculate_metrics(labels, probabilities)

    lines = [
        "EXPERIMENT 4: BCE + ACDSA + SUPCON",
        f"Source domain: {SOURCE_DOMAIN}",
        f"Target domain: {TARGET_DOMAIN}",
        f"Best source-validation epoch: {best_epoch}",
        f"Best Domain1 validation BCE: {best_val_loss:.6f}",
        f"Source training counts: Normal={normal_count}, TB={tb_count}",
        f"ACDSA weight: {LAMBDA_ACDSA}",
        f"SupCon weight: {SUPCON_WEIGHT}",
        f"SupCon temperature: {TEMPERATURE}",
        "Target threshold: 0.5 (fixed, not tuned on target test)",
        "",
        "DOMAIN2 TARGET TEST METRICS",
    ]

    for name, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            lines.append(f"{name}: {value:.6f}")
        else:
            lines.append(f"{name}: {value}")

    lines.extend([
        "",
        "Confusion matrix (rows=true, columns=predicted):",
        str(np.array([
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ])),
    ])

    report_path = OUTPUT / "target_test_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("\nSaved checkpoint:", checkpoint_path)
    print("Saved training history:", OUTPUT / "training_history.csv")
    print("Saved report:", report_path)
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
