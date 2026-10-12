import random
import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

DOMAIN_1_PATH = r"C:\projects\Final Year Project\Dataset\Domain 1-NLM"
DOMAIN_2_PATH = r"C:\projects\Final Year Project\Dataset\Domain 2-Mendeley"


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GRAYSCALE RESNET-18")
print("TRAIN BALANCED DOMAIN 2 → TEST DOMAIN 2 + DOMAIN 1")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)
print("Device          :", device)

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))

print("=" * 70)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD DOMAIN 1
# ============================================================

print("\nLoading Domain 1...")

domain1_dataset = datasets.ImageFolder(
    DOMAIN_1_PATH,
    transform=transform
)

print("Domain 1 classes:")
print(domain1_dataset.class_to_idx)
print("Total Domain 1 images:", len(domain1_dataset))


# ============================================================
# LOAD DOMAIN 2
# ============================================================

print("\nLoading Domain 2...")

domain2_dataset = datasets.ImageFolder(
    DOMAIN_2_PATH,
    transform=transform
)

print("Domain 2 classes:")
print(domain2_dataset.class_to_idx)
print("Total Domain 2 images:", len(domain2_dataset))


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

assert domain1_dataset.class_to_idx == domain2_dataset.class_to_idx

print("\nClass mapping verified.")

normal_class = domain2_dataset.class_to_idx["Normal"]
tb_class = domain2_dataset.class_to_idx["Tuberculosis"]

print("Normal       =", normal_class)
print("Tuberculosis =", tb_class)


# ============================================================
# GET DOMAIN 2 CLASS INDICES
# ============================================================

normal_indices = [
    i
    for i, (_, label) in enumerate(domain2_dataset.samples)
    if label == normal_class
]

tb_indices = [
    i
    for i, (_, label) in enumerate(domain2_dataset.samples)
    if label == tb_class
]

print("\nDomain 2 original distribution:")
print("Normal       :", len(normal_indices))
print("Tuberculosis :", len(tb_indices))


# ============================================================
# BALANCE DOMAIN 2 TRAINING DATA
# ============================================================

random.shuffle(normal_indices)
random.shuffle(tb_indices)

num_per_class = min(
    len(normal_indices),
    len(tb_indices)
)

selected_normal = normal_indices[:num_per_class]
selected_tb = tb_indices[:num_per_class]

balanced_indices = (
    selected_normal +
    selected_tb
)

random.shuffle(balanced_indices)

domain2_train = Subset(
    domain2_dataset,
    balanced_indices
)

print("\nBalanced Domain 2 training:")
print("Normal       :", len(selected_normal))
print("Tuberculosis :", len(selected_tb))
print("Total        :", len(domain2_train))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    domain2_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

domain2_test_loader = DataLoader(
    domain2_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

domain1_test_loader = DataLoader(
    domain1_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet-18...")

model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

old_conv = model.conv1

model.conv1 = nn.Conv2d(
    1,
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# Convert pretrained RGB weights → grayscale
with torch.no_grad():

    model.conv1.weight[:] = old_conv.weight.mean(
        dim=1,
        keepdim=True
    )

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

print("ResNet-18 loaded successfully.")
print("Input channels: 1")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING ON BALANCED DOMAIN 2")
print("=" * 70)

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Loss     : {epoch_loss:.4f}")
    print(f"Accuracy : {epoch_accuracy * 100:.2f}%")

    if epoch_accuracy > best_accuracy:

        best_accuracy = epoch_accuracy

        torch.save(
            model.state_dict(),
            "resnet18_gray_domain2.pth"
        )

        print("✓ Best model saved")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best trained model...")

model.load_state_dict(
    torch.load(
        "resnet18_gray_domain2.pth",
        map_location=device
    )
)

model.to(device)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, loader, name):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
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

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

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

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# TEST DOMAIN 2
# ============================================================

domain2_results = evaluate_model(
    model,
    domain2_test_loader,
    "DOMAIN 2 TEST"
)


# ============================================================
# TEST DOMAIN 1
# ============================================================

domain1_results = evaluate_model(
    model,
    domain1_test_loader,
    "DOMAIN 1 UNSEEN TEST"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CROSS-DOMAIN COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'Domain 2':>15}"
    f"{'Domain 1':>15}"
)

print("-" * 50)

metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("F1-score", "f1"),
    ("ROC-AUC", "roc_auc")
]

for name, key in metrics:

    print(
        f"{name:<20}"
        f"{domain2_results[key]:>15.4f}"
        f"{domain1_results[key]:>15.4f}"
    )

print("\nExperiment completed.")
print("Saved model: resnet18_gray_domain2.pth")