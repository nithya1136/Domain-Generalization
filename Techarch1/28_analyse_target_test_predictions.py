
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "15_prediction_error_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = (
    ROOT / "outputs" / "12_four_experiments"
    / "02_BCE_ACDSA" / "manifest_test.csv"
)

EXPERIMENTS = {
    "BCE": ROOT / "outputs" / "12_four_experiments"
    / "01_BCE" / "existing_baseline_checkpoint.pt",

    "BCE_ACDSA": ROOT / "outputs" / "12_four_experiments"
    / "02_BCE_ACDSA" / "best_model.pt",

    "BCE_SupCon": ROOT / "outputs" / "12_four_experiments"
    / "03_BCE_SupCon_Run2" / "best_model.pt",

    "BCE_ACDSA_SupCon": ROOT / "outputs" / "12_four_experiments"
    / "04_BCE_ACDSA_SupCon" / "best_model.pt",
}

BATCH_SIZE = 16
THRESHOLD = 0.5
SEED = 42

sys.path.insert(0, str(ROOT))
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location(
    "joint_learning_model", ROOT / "15_joint_learning_model.py"
)
model_module = module_from_spec(spec)
spec.loader.exec_module(model_module)
JointLearningModel = model_module.JointLearningModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Support raw state_dicts and common checkpoint dictionary formats.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state = checkpoint["model_state"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    # Remove DataParallel prefix if present.
    state = {
        k.removeprefix("module."): v
        for k, v in state.items()
        if isinstance(k, str)
    }

    model.load_state_dict(state, strict=True)
    return model


class TestDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        original_path = Path(row["original_view"])
        lung_path = Path(row["lung_view"])

        with Image.open(original_path) as im:
            original = self.transform(im.convert("L"))

        with Image.open(lung_path) as im:
            lung = self.transform(im.convert("L"))

        label = 1 if str(row["label"]).strip().lower() in {
            "tuberculosis", "tb", "1"
        } else 0

        return original, lung, label, index


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    df = df[
        (df["split"].astype(str).str.lower() == "test")
        & (df["domain"].astype(str).str.lower() == "domain2")
    ].copy()

    if "views_valid" in df.columns:
        valid = df["views_valid"].astype(str).str.lower().isin(
            ["true", "1"]
        )
        df = df[valid].copy()

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid Domain2 test rows found in manifest.")

    missing = []
    for col in ["original_view", "lung_view"]:
        for p in df[col].astype(str):
            if not Path(p).exists():
                missing.append(p)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing. First few:\n"
            + "\n".join(missing[:10])
        )

    labels = np.array([
        1 if str(x).strip().lower() in {"tuberculosis", "tb", "1"}
        else 0
        for x in df["label"]
    ])

    print(f"Domain2 test images: {len(df)}")
    print("Ground-truth counts:", {
        "Normal": int((labels == 0).sum()),
        "Tuberculosis": int((labels == 1).sum()),
    })

    dataset = TestDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_predictions = df[[
        "image_id", "domain", "label",
        "original_view", "lung_view"
    ]].copy()

    all_predictions["true_label_binary"] = labels

    for name, checkpoint_path in EXPERIMENTS.items():
        print(f"\nEvaluating {name}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # pretrained=False avoids downloading ImageNet weights.
        model = JointLearningModel(
            pretrained=False, dropout=0.3
        ).to(device)

        load_checkpoint(model, checkpoint_path)
        model.eval()

        logits_list = []
        index_list = []

        with torch.no_grad():
            for original, lung, batch_labels, indices in loader:
                original = original.to(device)
                lung = lung.to(device)

                output = model(original, lung)
                logits = output["logits"].reshape(-1)

                logits_list.extend(logits.cpu().numpy().tolist())
                index_list.extend(indices.numpy().tolist())

        logits_array = np.asarray(logits_list)
        probabilities = 1.0 / (1.0 + np.exp(-logits_array))
        predictions = (probabilities >= THRESHOLD).astype(int)

        # Store by dataset index to retain exact manifest row alignment.
        probs_by_index = np.empty(len(df), dtype=float)
        preds_by_index = np.empty(len(df), dtype=int)
        probs_by_index[index_list] = probabilities
        preds_by_index[index_list] = predictions

        all_predictions[f"{name}_prob_TB"] = probs_by_index
        all_predictions[f"{name}_pred_binary"] = preds_by_index
        all_predictions[f"{name}_pred_label"] = np.where(
            preds_by_index == 1, "Tuberculosis", "Normal"
        )
        all_predictions[f"{name}_correct"] = (
            preds_by_index == labels
        )

        tn, fp, fn, tp = confusion_matrix(
            labels, preds_by_index, labels=[0, 1]
        ).ravel()

        metrics = {
            "experiment": name,
            "n": int(len(labels)),
            "threshold": THRESHOLD,
            "accuracy": float(accuracy_score(labels, preds_by_index)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, preds_by_index)
            ),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1_score(labels, preds_by_index, zero_division=0)),
            "auc": float(roc_auc_score(labels, probs_by_index)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        print(json.dumps(metrics, indent=2))

        with open(OUT / f"{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(classification_report(
            labels,
            preds_by_index,
            labels=[0, 1],
            target_names=["Normal", "Tuberculosis"],
            zero_division=0,
        ))

    # Summarize which experiments make mistakes on the same images.
    correct_columns = [f"{name}_correct" for name in EXPERIMENTS]
    all_predictions["num_models_wrong"] = (
        ~all_predictions[correct_columns].astype(bool)
    ).sum(axis=1)

    all_predictions["all_models_wrong"] = (
        all_predictions["num_models_wrong"] == len(EXPERIMENTS)
    )

    all_predictions.to_csv(
        OUT / "all_target_test_predictions.csv", index=False
    )

    errors = all_predictions[
        all_predictions["num_models_wrong"] > 0
    ].copy()
    errors.to_csv(OUT / "misclassified_images.csv", index=False)

    shared_errors = all_predictions[
        all_predictions["all_models_wrong"]
    ].copy()
    shared_errors.to_csv(
        OUT / "images_misclassified_by_all_models.csv", index=False
    )

    print("\nAnalysis complete.")
    print("Output folder:", OUT)
    print("All predictions:", OUT / "all_target_test_predictions.csv")
    print("Misclassified images:", OUT / "misclassified_images.csv")
    print("Wrong in all models:", len(shared_errors))


if __name__ == "__main__":
    main()
