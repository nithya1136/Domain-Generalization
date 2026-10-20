
"""
Experiment 3: BCE + Supervised Contrastive Learning (SupCon)

Protocol:
    Train       : Domain1 train only
    Checkpoint  : Domain1 validation BCE only
    Final test  : Domain2 test only

Existing BCE and ACDSA results are not modified.
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

SUPCON_WEIGHT = 0.1
TEMPERATURE = 0.07

SOURCE_DOMAIN = "Domain1"
TARGET_DOMAIN = "Domain2"

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
OUTPUT = ROOT / "outputs" / "12_four_experiments" / "03_BCE_SupCon_Run2"

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
# 3. IMPORT EXISTING PROJECT MODULES
# =========================================================

def load_module(filename, module_name):
    path = ROOT / filename

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================
# 4. PROJECTION HEAD
# =========================================================

class ProjectionHead(nn.Module):
    """
    Shared projection head applied independently to the
    original-view and lung-view features.

    Input:  512 dimensions
    Output: 128 dimensions
    """

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
# 5. SUPERVISED CONTRASTIVE LOSS
# =========================================================

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    features: [N, embedding_dimension]
    labels:   [N]

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

    # Exclude each embedding from its own denominator.
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
# 6. DATASET HELPERS
# =========================================================

def create_dataset_manifest(df, path):
    """
    Save an isolated manifest containing only the intended
    domain and split. The existing master manifest is unchanged.
    """
    df.to_csv(path, index=False)


def unpack_batch(batch):
    original = batch["original"]
    lung = batch["lung"]
    labels = batch["label"]

    return original, lung, labels


# =========================================================
# 7. TRAINING
# =========================================================

def train_one_epoch(model, projection, loader, optimizer, criterion):
    model.train()
    projection.train()

    total = 0
    sum_total = 0.0
    sum_bce = 0.0
    sum_supcon = 0.0

    for batch in loader:
        original, lung, labels = unpack_batch(batch)

        original = original.to(DEVICE, dtype=torch.float32)
        lung = lung.to(DEVICE, dtype=torch.float32)
        labels = labels.to(DEVICE, dtype=torch.float32).reshape(-1)

        optimizer.zero_grad(set_to_none=True)

        output = model(original, lung)

        logits = output["logits"].reshape(-1)
        original_features = output["original_features"]
        lung_features = output["lung_features"]

        # Classification objective.
        bce_loss = criterion(logits, labels)

        # Project the two views separately using one shared head.
        original_projection = projection(original_features)
        lung_projection = projection(lung_features)

        # Two embeddings per image, with the same image label.
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

        total_loss = bce_loss + SUPCON_WEIGHT * supcon_loss

        total_loss.backward()
        optimizer.step()

        n = labels.size(0)
        total += n
        sum_total += total_loss.item() * n
        sum_bce += bce_loss.item() * n
        sum_supcon += supcon_loss.item() * n

    return {
        "total_loss": sum_total / total,
        "bce_loss": sum_bce / total,
        "supcon_loss": sum_supcon / total,
    }


@torch.no_grad()
def validation_bce(model, loader, criterion):
    """
    Source validation BCE is the only checkpoint-selection
    criterion. SupCon is not used to select the checkpoint.
    """
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


# =========================================================
# 8. FINAL EVALUATION
# =========================================================

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
        all_probabilities.extend(
            probabilities.cpu().numpy().reshape(-1)
        )

    return np.asarray(all_labels), np.asarray(all_probabilities)


def calculate_metrics(labels, probabilities):
    # Fixed threshold; no tuning on target test.
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
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
# 9. MAIN
# =========================================================

def main():
    set_seed(SEED)

    print("=" * 60)
    print("EXPERIMENT 3: BCE + SUPERVISED CONTRASTIVE LEARNING")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Manifest:", MANIFEST)
    print("Output:", OUTPUT)

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    # Never overwrite a previous experiment.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty:\n{OUTPUT}\n"
            "Preserve its contents. Choose a new output directory "
            "if you intentionally want a separate run."
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

    # Split the source data and target test before constructing datasets.
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

    if source_train.empty or source_val.empty or target_test.empty:
        raise ValueError(
            "Required splits not found. Expected Domain1 train, "
            "Domain1 validation, and Domain2 test."
        )

    # Validate path existence before training.
    for name, part in [
        ("source train", source_train),
        ("source validation", source_val),
        ("target test", target_test),
    ]:
        missing_paths = (
            ~part["original_view"].map(lambda p: Path(p).is_file())
            | ~part["lung_view"].map(lambda p: Path(p).is_file())
        )
        if missing_paths.any():
            raise FileNotFoundError(
                f"{name} contains missing image paths. "
                f"Example IDs: {part.loc[missing_paths, 'image_id'].head().tolist()}"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_manifest = OUTPUT / "source_train_manifest.csv"
    val_manifest = OUTPUT / "source_validation_manifest.csv"
    test_manifest = OUTPUT / "target_test_manifest.csv"

    create_dataset_manifest(source_train, train_manifest)
    create_dataset_manifest(source_val, val_manifest)
    create_dataset_manifest(target_test, test_manifest)

    print(f"\nDomain1 training images: {len(source_train)}")
    print(f"Domain1 validation images: {len(source_val)}")
    print(f"Domain2 test images: {len(target_test)}")

    # Import the actual class names and APIs from your files.
    dataset_module = load_module(
        "12_dual_view_dataset.py", "dual_view_dataset_module"
    )
    model_module = load_module(
        "15_joint_learning_model.py", "joint_learning_model_module"
    )

    DatasetClass = dataset_module.DualViewChestXrayDataset
    ModelClass = model_module.JointLearningModel

    train_dataset = DatasetClass(
        manifest_path=train_manifest,
        split="train",
    )
    val_dataset = DatasetClass(
        manifest_path=val_manifest,
        split="validation",
    )
    test_dataset = DatasetClass(
        manifest_path=test_manifest,
        split="test",
    )

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

    # Use the same model architecture and pretrained initialization
    # as the existing project.
    model = ModelClass(pretrained=True).to(DEVICE)
    projection = ProjectionHead().to(DEVICE)

    # Class weighting is calculated from Domain1 training labels only.
    normal_count = int((source_train["label"] == "Normal").sum())
    tb_count = int((source_train["label"] == "Tuberculosis").sum())

    if normal_count == 0 or tb_count == 0:
        raise ValueError(
            "Expected both 'Normal' and 'Tuberculosis' in source training."
        )

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

    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []

    checkpoint_path = OUTPUT / "best_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_stats = train_one_epoch(
            model, projection, train_loader, optimizer, criterion
        )
        val_loss = validation_bce(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_bce_loss": train_stats["bce_loss"],
            "train_supcon_loss": train_stats["supcon_loss"],
            "source_validation_bce": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Total={train_stats['total_loss']:.4f} | "
            f"BCE={train_stats['bce_loss']:.4f} | "
            f"SupCon={train_stats['supcon_loss']:.4f} | "
            f"Val BCE={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "best_validation_bce": best_val,
                    "seed": SEED,
                    "supcon_weight": SUPCON_WEIGHT,
                    "temperature": TEMPERATURE,
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(
        OUTPUT / "training_history.csv", index=False
    )

    # Restore the best source-validation checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Domain2 is evaluated only after checkpoint selection is complete.
    labels, probabilities = get_predictions(model, test_loader)
    metrics = calculate_metrics(labels, probabilities)

    lines = [
        "EXPERIMENT 3: BCE + SUPERVISED CONTRASTIVE LEARNING",
        f"Source domain: {SOURCE_DOMAIN}",
        f"Target domain: {TARGET_DOMAIN}",
        f"Best epoch: {best_epoch}",
        f"Best Domain1 validation BCE: {best_val:.6f}",
        f"Source training class counts: Normal={normal_count}, TB={tb_count}",
        f"SupCon weight: {SUPCON_WEIGHT}",
        f"Temperature: {TEMPERATURE}",
        "Target threshold: 0.5 (fixed, not tuned on target test)",
        "",
        "DOMAIN2 TARGET TEST METRICS",
    ]

    for name, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            lines.append(f"{name}: {value:.6f}")
        else:
            lines.append(f"{name}: {value}")

    report = "\n".join(lines)
    (OUTPUT / "target_test_report.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n" + report)
    print("\nSaved checkpoint:", checkpoint_path)
    print("Saved report:", OUTPUT / "target_test_report.txt")
    print("Experiment completed.")


if __name__ == "__main__":
    main()
