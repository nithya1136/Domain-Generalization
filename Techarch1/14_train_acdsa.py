from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, recall_score,
)
from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
EXPERIMENT_DIR = ROOT / "outputs" / "12_four_experiments" / "02_BCE_ACDSA"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
PATIENCE = 4
SEED = 42
LAMBDA_ACDSA = 0.1


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module("dual_view_dataset_acdsa", ROOT / "12_dual_view_dataset.py")
model_module = load_module("joint_learning_model_acdsa", ROOT / "15_joint_learning_model.py")
DatasetClass = dataset_module.DualViewChestXrayDataset
ModelClass = model_module.JointLearningModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split_manifests():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    required = {"domain", "split", "label", "views_valid", "original_view", "lung_view", "image_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    valid = df[df["views_valid"].astype(str).str.lower().eq("true")].copy()
    subsets = {
        "train": valid[(valid["domain"] == "Domain1") & (valid["split"] == "train")].copy(),
        "validation": valid[(valid["domain"] == "Domain1") & (valid["split"] == "validation")].copy(),
        "test": valid[(valid["domain"] == "Domain2") & (valid["split"] == "test")].copy(),
    }
    for name, frame in subsets.items():
        if frame.empty:
            raise ValueError(f"No samples found for {name}")
        if not frame["label"].isin(["Normal", "Tuberculosis"]).all():
            raise ValueError(f"Unexpected labels in {name}")
        for col in ("original_view", "lung_view"):
            missing_paths = ~frame[col].map(lambda p: Path(p).is_file())
            if missing_paths.any():
                raise FileNotFoundError(
                    f"{name} has missing {col} paths: {frame.loc[missing_paths, col].head().tolist()}"
                )

    assert set(subsets["train"]["image_id"]).isdisjoint(set(subsets["test"]["image_id"]))
    assert set(subsets["validation"]["image_id"]).isdisjoint(set(subsets["test"]["image_id"]))

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, frame in subsets.items():
        frame = frame.copy()
        frame["split"] = split_name
        path = EXPERIMENT_DIR / f"manifest_{split_name}.csv"
        frame.to_csv(path, index=False)
        paths[split_name] = path
    return paths


def build_dataset(path, split):
    return DatasetClass(manifest_path=path, split=split, verify_paths=True)


def acdsa_alignment_loss(original_features, lung_features):
    """Proposed paired-view alignment loss using L2-normalized features."""
    original_norm = F.normalize(original_features, p=2, dim=1, eps=1e-8)
    lung_norm = F.normalize(lung_features, p=2, dim=1, eps=1e-8)
    return ((original_norm - lung_norm) ** 2).sum(dim=1).mean()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            original = batch["original"].to(device)
            lung = batch["lung"].to(device)
            labels = batch["label"].float().to(device)
            outputs = model(original, lung)
            logits = outputs["logits"].reshape(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "labels": y_true, "probabilities": y_prob, "predictions": y_pred,
    }


def main():
    set_seed(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = EXPERIMENT_DIR / "best_model.pt"
    history_path = EXPERIMENT_DIR / "training_history.csv"
    report_path = EXPERIMENT_DIR / "target_test_report.txt"

    # Do not silently overwrite a previous run.
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            f"ACDSA outputs already exist in {EXPERIMENT_DIR}. "
            "Archive/rename the existing run or choose a new output directory before rerunning."
        )

    paths = make_split_manifests()
    train_dataset = build_dataset(paths["train"], "train")
    val_dataset = build_dataset(paths["validation"], "validation")
    test_dataset = build_dataset(paths["test"], "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Experiment: Domain1 train/validation -> Domain2 test")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Target test samples:", len(test_dataset))
    print("ACDSA lambda:", LAMBDA_ACDSA)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    model = ModelClass(pretrained=True).to(device)
    labels = train_dataset.df["label"].map(DatasetClass.LABEL_MAP).to_numpy()
    normal_count = int((labels == 0).sum())
    tb_count = int((labels == 1).sum())
    pos_weight = torch.tensor([normal_count / max(tb_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_total = running_bce = running_align = 0.0
        for batch in train_loader:
            original = batch["original"].to(device)
            lung = batch["lung"].to(device)
            y = batch["label"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(original, lung)
            logits = outputs["logits"].reshape(-1)
            bce_loss = criterion(logits, y)
            align_loss = acdsa_alignment_loss(outputs["original_features"], outputs["lung_features"])
            loss = bce_loss + LAMBDA_ACDSA * align_loss
            loss.backward()
            optimizer.step()
            n = y.size(0)
            running_total += loss.item() * n
            running_bce += bce_loss.item() * n
            running_align += align_loss.item() * n

        train_total = running_total / len(train_dataset)
        train_bce = running_bce / len(train_dataset)
        train_align = running_align / len(train_dataset)
        # Keep checkpoint selection based on source-validation BCE, like the baseline.
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch, "train_total_loss": train_total, "train_bce_loss": train_bce,
            "train_acdsa_loss": train_align, "val_bce_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"], "val_auc": val_metrics["auc"],
        })
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train total={train_total:.4f} | BCE={train_bce:.4f} | "
              f"ACDSA={train_align:.4f} | Source val BCE={val_metrics['loss']:.4f} | "
              f"Val balanced acc={val_metrics['balanced_accuracy']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss, "source_domain": "Domain1",
                "target_domain": "Domain2", "seed": SEED, "lambda_acdsa": LAMBDA_ACDSA,
            }, checkpoint_path)
            print("  Saved best source-validation checkpoint.")
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    # Domain2 test is evaluated only after checkpoint selection is complete.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    report = classification_report(test_metrics["labels"], test_metrics["predictions"],
                                   labels=[0, 1], target_names=["Normal", "Tuberculosis"],
                                   zero_division=0)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Experiment: Domain1_to_Domain2_BCE_ACDSA\n")
        f.write(f"Best source-validation epoch: {best_epoch}\n")
        f.write(f"Target test samples: {len(test_dataset)}\n")
        f.write(f"ACDSA lambda: {LAMBDA_ACDSA}\n")
        for key in ["loss", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                    "f1", "auc", "tn", "fp", "fn", "tp"]:
            f.write(f"{key}: {test_metrics[key]}\n")
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(str(np.array([[test_metrics["tn"], test_metrics["fp"]],
                              [test_metrics["fn"], test_metrics["tp"]]])))
        f.write("\n\n" + report)

    print("\nTarget-domain test results")
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                "f1", "auc", "tn", "fp", "fn", "tp"]:
        print(f"{key}: {test_metrics[key]}")
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
