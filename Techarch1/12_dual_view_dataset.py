
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from importlib.util import spec_from_file_location, module_from_spec


# Load the augmentation module from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
AUGMENTATION_PATH = PROJECT_ROOT / "10_dual_view_augmentation.py"

spec = spec_from_file_location("dual_view_augmentation", AUGMENTATION_PATH)
augmentation_module = module_from_spec(spec)
spec.loader.exec_module(augmentation_module)

DualViewAugmentation = augmentation_module.DualViewAugmentation


class DualViewChestXrayDataset(Dataset):
    """
    Dataset for paired original and lung-masked chest X-rays.

    Returns:
        original_tensor: (1, 224, 224)
        lung_tensor:     (1, 224, 224)
        label:           0 = Normal, 1 = Tuberculosis
        domain:          0 = Domain1, 1 = Domain2
    """

    LABEL_MAP = {"Normal": 0, "Tuberculosis": 1}
    DOMAIN_MAP = {"Domain1": 0, "Domain2": 1}

    def __init__(
        self,
        manifest_path,
        split,
        image_size=224,
        verify_paths=True,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        df = pd.read_csv(self.manifest_path)

        required_columns = {
            "original_view",
            "lung_view",
            "label",
            "domain",
            "split",
            "views_valid",
            "image_id",
        }

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")

        df = df[
            (df["split"] == split)
            & (df["views_valid"].astype(str).str.lower() == "true")
        ].copy()

        if df.empty:
            raise ValueError(f"No valid samples found for split: {split}")

        if not df["label"].isin(self.LABEL_MAP).all():
            raise ValueError("Unexpected class label in manifest")

        if not df["domain"].isin(self.DOMAIN_MAP).all():
            raise ValueError("Unexpected domain name in manifest")

        if verify_paths:
            valid_rows = []
            missing_rows = []

            for _, row in df.iterrows():
                original_path = Path(row["original_view"])
                lung_path = Path(row["lung_view"])

                if original_path.is_file() and lung_path.is_file():
                    valid_rows.append(row)
                else:
                    missing_rows.append(row["image_id"])

            if missing_rows:
                raise FileNotFoundError(
                    f"{len(missing_rows)} samples have missing views. "
                    f"Examples: {missing_rows[:5]}"
                )

            df = pd.DataFrame(valid_rows)

        self.df = df.reset_index(drop=True)

        self.transform = DualViewAugmentation(
            training=(split == "train"),
            image_size=image_size,
        )

        print(
            f"{split}: {len(self.df)} paired samples | "
            f"augmentation={'on' if split == 'train' else 'off'}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["original_view"]) as img:
            original = img.convert("L")

        with Image.open(row["lung_view"]) as img:
            lung = img.convert("L")

        original_tensor, lung_tensor = self.transform(original, lung)

        label = torch.tensor(
            self.LABEL_MAP[row["label"]],
            dtype=torch.long,
        )

        domain = torch.tensor(
            self.DOMAIN_MAP[row["domain"]],
            dtype=torch.long,
        )

        return {
            "original": original_tensor,
            "lung": lung_tensor,
            "label": label,
            "domain": domain,
            "image_id": row["image_id"],
        }


if __name__ == "__main__":
    manifest = (
        PROJECT_ROOT
        / "outputs"
        / "01_metadata"
        / "dual_view_splits.csv"
    )

    dataset = DualViewChestXrayDataset(
        manifest_path=manifest,
        split="train",
    )

    sample = dataset[0]

    print("\nSample verification")
    print("Image ID:", sample["image_id"])
    print("Original shape:", tuple(sample["original"].shape))
    print("Lung shape:", tuple(sample["lung"].shape))
    print("Label:", sample["label"].item())
    print("Domain:", sample["domain"].item())
