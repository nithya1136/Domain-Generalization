
from pathlib import Path
import argparse
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
    recall_score,
)

from importlib.util import spec_from_file_location, module_from_spec


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "01_metadata" / "dual_view_splits.csv"
REPORT_ROOT = ROOT / "outputs" / "08_domain_generalization"
CHECKPOINT_ROOT = ROOT / "outputs" / "09_domain_checkpoints"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
PATIENCE = 4
SEED = 42


def load_module(name, path):
    spec = spec_from_file_location(name, path)
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

DatasetClass = dataset_module.DualViewChestXrayDataset
ModelClass = model_module.JointLearningModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_domain_manifest(source_domain, target_domain):
    df = pd.read_csv(MANIFEST)

    required = {
        "domain", "split", "label", "views_valid",
        "original_view", "lung_view", "image_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    valid = df[
        df["views_valid"].astype(str).str.lower().eq("true")
    ].copy()

    # Keep source-domain train/validation only.
    source_train = valid[
        (valid["domain"] == source_domain)
        & (valid["split"] == "train")
    ].copy()

    source_val = valid[
        (valid["domain"] == source_domain)
        & (valid["split"] == "validation")
    ].copy()

    # Target-domain test images are never used for training or selection.
    target_test = valid[
        (valid["domain"] == target_domain)
        & (valid["split"] == "test")
    ].copy()

    for name, subset in [
        ("source train", source_train),
        ("source validation", source_val),
        ("target test", target_test),
    ]:
        if subset.empty:
            raise ValueError(f"No samples found for {name}")

        if not subset["label"].isin(["Normal", "Tuberculosis"]).all():
            raise ValueError(f"Unexpected labels in {name}")

        for col in ["original_view", "lung_view"]:
            missing_paths = ~subset[col].map(
                lambda p: Path(p).is_file()
            )
            if missing_paths.any():
                raise FileNotFoundError(
                    f"{name}: missing {col} paths: "
                    f"{subset.loc[missing_paths, col].head().tolist()}"
                )

    # Check that no target-domain images enter source training/validation.
    assert set(source_train["image_id"]).isdisjoint(
        set(target_test["image_id"])
    )
    assert set(source_val["image_id"]).isdisjoint(
        set(target_test["image_id"])
    )

    return source_train, source_val, target_test


def build_dataset(frame, split_name):
    # The existing dataset class reads a CSV with its split column.
    # Create a temporary split manifest containing only this experiment's
    # allowed samples.
    temp_path = REPORT_ROOT / f"_temporary_{split_name}.csv"
    frame.to_csv(temp_path, index=False)

    return DatasetClass(
        manifest_path=temp_path,
        split=split_name,
        verify_paths=True,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_domains = [], [], []

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
            all_domains.extend(batch["domain"].cpu().numpy().astype(int))

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    print("Probability summary:")
    print("  Min:", float(y_prob.min()))
    print("  25th percentile:", float(np.percentile(y_prob, 25)))
    print("  Median:", float(np.median(y_prob)))
    print("  75th percentile:", float(np.percentile(y_prob, 75)))
    print("  Max:", float(y_prob.max()))
    print("  Predicted Normal:", int((y_pred == 0).sum()))
    print("  Predicted TB:", int((y_pred == 1).sum()))

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) == 2
        else float("nan")
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "specificity": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "labels": y_true,
        "probabilities": y_prob,
        "predictions": y_pred,
    }


def main(source_domain, target_domain):
    set_seed(SEED)

    if source_domain == target_domain:
        raise ValueError("Source and target domains must be different.")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    experiment = f"{source_domain}_to_{target_domain}"
    report_dir = REPORT_ROOT / experiment
    checkpoint_dir = CHECKPOINT_ROOT / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_train, source_val, target_test = make_domain_manifest(
        source_domain, target_domain
    )

    # Give the existing dataset class the split names it expects.
    train_frame = source_train.copy()
    train_frame["split"] = "train"
    val_frame = source_val.copy()
    val_frame["split"] = "validation"
    test_frame = target_test.copy()
    test_frame["split"] = "test"

    train_dataset = build_dataset(train_frame, "train")
    val_dataset = build_dataset(val_frame, "validation")
    test_dataset = build_dataset(test_frame, "test")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Experiment:", experiment)
    print("Source training:", len(train_dataset))
    print("Source validation:", len(val_dataset))
    print("Target test:", len(test_dataset))
    print("Initializing fresh ImageNet-pretrained model.")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    model = ModelClass(pretrained=True).to(device)

    # Compute class weighting from source training data only.
    labels = train_dataset.df["label"].map(
        DatasetClass.LABEL_MAP
    ).to_numpy()
    normal_count = int((labels == 0).sum())
    tb_count = int((labels == 1).sum())

    pos_weight = torch.tensor(
        [normal_count / max(tb_count, 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = checkpoint_dir / "best_model.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            original = batch["original"].to(device)
            lung = batch["lung"].to(device)
            y = batch["label"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(original, lung)
            logits = outputs["logits"].reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)

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
            f"Train loss {train_loss:.4f} | "
            f"Source val loss {val_metrics['loss']:.4f} | "
            f"Accuracy {val_metrics['accuracy']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"AUROC {val_metrics['auc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "seed": SEED,
            }, checkpoint_path)
            print("  Saved best source-validation checkpoint.")
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        report_dir / "training_history.csv", index=False
    )

    # Load the selected source-validation checkpoint.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Target domain is evaluated only after model selection is complete.
    test_metrics = evaluate(model, test_loader, criterion, device)
    probs = test_metrics["probabilities"]

    print("\nTarget-test probability summary:")
    print("Min:", float(np.min(probs)))
    print("25th percentile:", float(np.percentile(probs, 25)))
    print("Median:", float(np.median(probs)))
    print("75th percentile:", float(np.percentile(probs, 75)))
    print("Max:", float(np.max(probs)))
    print("Predicted Normal:", int((probs < 0.5).sum()))
    print("Predicted TB:", int((probs >= 0.5).sum()))
    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[0, 1],
        target_names=["Normal", "Tuberculosis"],
        zero_division=0,
    )

    with open(report_dir / "target_test_report.txt", "w") as f:
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Best source-validation epoch: {best_epoch}\n")
        f.write(f"Target test samples: {len(test_dataset)}\n")
        for key in [
            "loss", "accuracy", "sensitivity", "specificity",
            "f1", "auc", "tn", "fp", "fn", "tp"
        ]:
            f.write(f"{key}: {test_metrics[key]}\n")
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(str(np.array([
            [test_metrics["tn"], test_metrics["fp"]],
            [test_metrics["fn"], test_metrics["tp"]],
        ])))
        f.write("\n\n")
        f.write(report)

    print("\nTarget-domain test results")
    for key in [
        "accuracy", "sensitivity", "specificity", "f1", "auc",
        "tn", "fp", "fn", "tp"
    ]:
        print(f"{key}: {test_metrics[key]}")

    print("Checkpoint:", checkpoint_path)
    print("Report:", report_dir / "target_test_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["Domain1", "Domain2"], required=True
    )
    parser.add_argument(
        "--target", choices=["Domain1", "Domain2"], required=True
    )
    args = parser.parse_args()
    main(args.source, args.target)
