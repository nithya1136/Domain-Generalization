
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(r"C:\projects\FYP\Techarch1")
INPUT = BASE / "outputs/01_metadata/dual_view_manifest.csv"
OUTPUT = BASE / "outputs/01_metadata/dual_view_splits.csv"

df = pd.read_csv(INPUT)

# Do not split rows whose two views are not available.
df = df[df["views_valid"] == True].copy()

# First split: 70% train, 30% temporary.
train_parts = []
temp_parts = []

for (domain, label), group in df.groupby(["domain", "label"]):
    train, temp = train_test_split(
        group,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )
    train_parts.append(train)
    temp_parts.append(temp)

train_df = pd.concat(train_parts, ignore_index=True)

# Split the remaining 30% equally into validation and test.
val_parts = []
test_parts = []

for (domain, label), group in pd.concat(
    temp_parts, ignore_index=True
).groupby(["domain", "label"]):
    val, test = train_test_split(
        group,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )
    val_parts.append(val)
    test_parts.append(test)

train_df["split"] = "train"
val_df = pd.concat(val_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

val_df["split"] = "validation"
test_df["split"] = "test"

splits = pd.concat(
    [train_df, val_df, test_df], ignore_index=True
)

# Verify each source image occurs exactly once.
assert splits["preprocessed_path"].is_unique
assert len(splits) == len(df)
assert splits.groupby("split").size().sum() == len(df)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
splits.to_csv(OUTPUT, index=False)

print("Total:", len(splits))
print("\nSplit sizes:")
print(splits["split"].value_counts())

print("\nDomain/class distribution:")
print(pd.crosstab(
    [splits["domain"], splits["label"]],
    splits["split"]
))

print("\nSaved to:", OUTPUT)
