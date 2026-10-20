
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from importlib.util import spec_from_file_location, module_from_spec


# --------------------------------------------------
# 1. Reproducibility and paths
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "06_reports"
CHECKPOINT_DIR = OUTPUT_DIR / "07_checkpoints"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = OUTPUT_DIR / "01_metadata" / "dual_view_splits.csv"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
PATIENCE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# --------------------------------------------------
# 2. Load existing project modules
# --------------------------------------------------
def load_module(module_name, file_path):
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "dual_view_dataset",
    ROOT / "12_dual_view_dataset.py",
)

model_module = load_module(
    "joint_learning_model",
    ROOT / "15_joint_learning_model.py",
)

DualViewChestXrayDataset = (
    dataset_module.DualViewChestXrayDataset
)

JointLearningModel = model_module.JointLearningModel


# --------------------------------------------------
# 3. Create datasets and loaders
# --------------------------------------------------
train_dataset = DualViewChestXrayDataset(MANIFEST, "train")
val_dataset = DualViewChestXrayDataset(MANIFEST, "validation")

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


# --------------------------------------------------
# 4. Model, loss, optimizer
# --------------------------------------------------
model = JointLearningModel().to(DEVICE)

# Initial baseline: classification loss only.
# Use training-set class counts to reduce imbalance bias.
train_labels = train_dataset.df["label"].map(
    dataset_module.DualViewChestXrayDataset.LABEL_MAP
).to_numpy()

normal_count = int((train_labels == 0).sum())
tb_count = int((train_labels == 1).sum())

pos_weight = torch.tensor(
    [normal_count / max(tb_count, 1)],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Normal: {normal_count} | TB: {tb_count}")
print("Positive-class loss weight:", pos_weight.item())


# --------------------------------------------------
# 5. Validation helper
# --------------------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            original = batch["original"].to(DEVICE)
            lung = batch["lung"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)

            outputs = model(original, lung)
            logits = outputs["logits"].reshape(-1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_probs.extend(
                torch.sigmoid(logits).cpu().numpy()
            )

    labels = np.asarray(all_labels)
    probs = np.asarray(all_probs)
    predictions = (probs >= 0.5).astype(int)

    auc = (
        roc_auc_score(labels, probs)
        if len(np.unique(labels)) == 2
        else float("nan")
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": auc,
        "labels": labels,
        "probs": probs,
        "predictions": predictions,
    }


# --------------------------------------------------
# 6. Training loop with early stopping
# --------------------------------------------------
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

checkpoint_path = CHECKPOINT_DIR / "baseline_best.pt"
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        original = batch["original"].to(DEVICE)
        lung = batch["lung"].to(DEVICE)
        labels = batch["label"].float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(original, lung)
        logits = outputs["logits"].reshape(-1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * labels.size(0)

    train_loss = total_train_loss / len(train_dataset)
    val_metrics = evaluate(model, val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        "val_auc": val_metrics["auc"],
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Val accuracy: {val_metrics['accuracy']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val AUROC: {val_metrics['auc']:.4f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "seed": SEED,
        }, checkpoint_path)

        print("  Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


# --------------------------------------------------
# 7. Save training history and validation report
# --------------------------------------------------
pd.DataFrame(history).to_csv(
    REPORT_DIR / "baseline_training_history.csv",
    index=False,
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
val_metrics = evaluate(model, val_loader)

cm = confusion_matrix(
    val_metrics["labels"],
    val_metrics["predictions"],
    labels=[0, 1],
)

report = classification_report(
    val_metrics["labels"],
    val_metrics["predictions"],
    labels=[0, 1],
    target_names=["Normal", "Tuberculosis"],
    zero_division=0,
)

with open(REPORT_DIR / "baseline_validation_report.txt", "w") as f:
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Validation loss: {val_metrics['loss']:.4f}\n")
    f.write(f"Validation accuracy: {val_metrics['accuracy']:.4f}\n")
    f.write(f"Validation F1: {val_metrics['f1']:.4f}\n")
    f.write(f"Validation AUROC: {val_metrics['auc']:.4f}\n\n")
    f.write("Confusion matrix (rows=true, columns=predicted):\n")
    f.write(str(cm))
    f.write("\n\n")
    f.write(report)

print("\nBaseline training complete.")
print("Best epoch:", best_epoch)
print("Checkpoint:", checkpoint_path)
print("History:", REPORT_DIR / "baseline_training_history.csv")
print("Validation report:", REPORT_DIR / "baseline_validation_report.txt")
