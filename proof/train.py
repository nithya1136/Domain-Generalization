import os
import random
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# ------------------------------------------------------------
# CHANGE THESE TWO PATHS
# ------------------------------------------------------------

DOMAIN_1_PATH = r"C:\projects\Final Year Project\Dataset\Domain 1-NLM"
DOMAIN_2_PATH = r"C:\projects\Final Year Project\Dataset\Domain 2-Mendeley"

# ------------------------------------------------------------

MODEL_SAVE_PATH = "resnet18_domain1_baseline.pth"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESNET-18 CROSS-DOMAIN BASELINE")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)
print("Device          :", device)

if torch.cuda.is_available():
    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )

print("=" * 70)


# ============================================================
# 4. CHECK DATASET PATHS
# ============================================================

if not os.path.exists(DOMAIN_1_PATH):
    raise FileNotFoundError(
        f"Domain 1 path not found:\n{DOMAIN_1_PATH}"
    )

if not os.path.exists(DOMAIN_2_PATH):
    raise FileNotFoundError(
        f"Domain 2 path not found:\n{DOMAIN_2_PATH}"
    )


# ============================================================
# 5. SIMPLE PREPROCESSING
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD DOMAIN 1
# ============================================================

print("\nLoading Domain 1...")

domain1_full = datasets.ImageFolder(
    root=DOMAIN_1_PATH,
    transform=train_transform
)

print("Domain 1 classes:")
print(domain1_full.class_to_idx)

print(
    "Total Domain 1 images:",
    len(domain1_full)
)


# ============================================================
# 7. LOAD DOMAIN 2
# ============================================================

print("\nLoading Domain 2...")

domain2_dataset = datasets.ImageFolder(
    root=DOMAIN_2_PATH,
    transform=test_transform
)

print("Domain 2 classes:")
print(domain2_dataset.class_to_idx)

print(
    "Total Domain 2 images:",
    len(domain2_dataset)
)


# ============================================================
# 8. VERIFY CLASS MAPPING
# ============================================================

if domain1_full.class_to_idx != domain2_dataset.class_to_idx:

    raise ValueError(
        "\nERROR: Domain 1 and Domain 2 have different "
        "class mappings!\n"
        f"Domain 1: {domain1_full.class_to_idx}\n"
        f"Domain 2: {domain2_dataset.class_to_idx}"
    )

print("\nClass mapping verified.")

print(
    "Normal       =",
    domain1_full.class_to_idx["Normal"]
)

print(
    "Tuberculosis =",
    domain1_full.class_to_idx["Tuberculosis"]
)


# ============================================================
# 9. SPLIT DOMAIN 1
# ============================================================

train_size = int(
    0.8 * len(domain1_full)
)

val_size = (
    len(domain1_full) - train_size
)

train_dataset, domain1_eval_dataset = random_split(
    domain1_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print("\nDomain 1 split:")
print("Training:", len(train_dataset))
print("Evaluation:", len(domain1_eval_dataset))


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

domain1_eval_loader = DataLoader(
    domain1_eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

domain2_test_loader = DataLoader(
    domain2_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# 11. LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# ============================================================
# 12. CHANGE FINAL CLASSIFIER
# ============================================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

print("ResNet-18 loaded successfully.")


# ============================================================
# 13. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 15. TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_train_accuracy = 0.0


for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100.0 * correct / total
    )

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Loss     : {epoch_loss:.4f}"
    )

    print(
        f"Accuracy : {epoch_accuracy:.2f}%"
    )

    # Save best training model
    if epoch_accuracy > best_train_accuracy:

        best_train_accuracy = epoch_accuracy

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            "✓ Best model saved"
        )


print("\nTraining completed.")

print(
    "Best model:",
    MODEL_SAVE_PATH
)


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=device
    )
)

model.to(device)

print("\nBest trained model loaded.")


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    dataset_name
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc=f"Evaluating {dataset_name}"
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # TN, FP, FN, TP
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    # Sensitivity = Recall
    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # ROC-AUC
    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1-score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Normal",
                "Tuberculosis"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Tuberculosis"
        ],
        yticklabels=[
            "Normal",
            "Tuberculosis"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix - {dataset_name}"
    )

    plt.tight_layout()

    safe_name = (
        dataset_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }


# ============================================================
# 18. EVALUATE DOMAIN 1
# ============================================================

domain1_results = evaluate_model(
    model,
    domain1_eval_loader,
    "DOMAIN 1 EVALUATION"
)


# ============================================================
# 19. TEST + EVALUATE DOMAIN 2
# ============================================================

domain2_results = evaluate_model(
    model,
    domain2_test_loader,
    "DOMAIN 2 UNSEEN TEST"
)


# ============================================================
# 20. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CROSS-DOMAIN COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<15}"
    f"{'Domain 1':>15}"
    f"{'Domain 2':>15}"
)

print("-" * 45)

print(
    f"{'Accuracy':<15}"
    f"{domain1_results['accuracy']:>15.4f}"
    f"{domain2_results['accuracy']:>15.4f}"
)

print(
    f"{'Precision':<15}"
    f"{domain1_results['precision']:>15.4f}"
    f"{domain2_results['precision']:>15.4f}"
)

print(
    f"{'Recall':<15}"
    f"{domain1_results['recall']:>15.4f}"
    f"{domain2_results['recall']:>15.4f}"
)

print(
    f"{'Sensitivity':<15}"
    f"{domain1_results['sensitivity']:>15.4f}"
    f"{domain2_results['sensitivity']:>15.4f}"
)

print(
    f"{'Specificity':<15}"
    f"{domain1_results['specificity']:>15.4f}"
    f"{domain2_results['specificity']:>15.4f}"
)

print(
    f"{'F1-score':<15}"
    f"{domain1_results['f1']:>15.4f}"
    f"{domain2_results['f1']:>15.4f}"
)

print(
    f"{'ROC-AUC':<15}"
    f"{domain1_results['auc']:>15.4f}"
    f"{domain2_results['auc']:>15.4f}"
)

print("=" * 70)

accuracy_gap = (
    domain1_results["accuracy"]
    -
    domain2_results["accuracy"]
)

f1_gap = (
    domain1_results["f1"]
    -
    domain2_results["f1"]
)

print(
    f"\nAccuracy generalization gap: "
    f"{accuracy_gap:.4f}"
)

print(
    f"F1 generalization gap: "
    f"{f1_gap:.4f}"
)

print("\nExperiment completed.")