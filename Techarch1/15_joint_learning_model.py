
import torch
import torch.nn as nn

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Load the shared ResNet-18 implementation.
spec = spec_from_file_location(
    "shared_resnet18",
    ROOT / "13_shared_resnet18.py",
)
module = module_from_spec(spec)
spec.loader.exec_module(module)

SharedResNet18 = module.SharedResNet18


class JointLearningModel(nn.Module):
    """
    Initial dual-view baseline model.

    Original X-ray  ─┐
                     ├─ Shared ResNet-18
    Lung-masked X-ray┘

    Both feature vectors are concatenated and passed to a
    binary classification head.

    Output:
        logits: (B, 1)
        original_features: (B, 512)
        lung_features: (B, 512)
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()

        self.backbone = SharedResNet18(pretrained=pretrained)

        # 512 original-view features + 512 lung-view features.
        self.feature_dim = 512
        self.fusion_dim = 1024

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, original, lung):
        original_features, lung_features = self.backbone(
            original,
            lung,
        )

        fused_features = torch.cat(
            [original_features, lung_features],
            dim=1,
        )

        logits = self.classifier(fused_features)

        return {
            "logits": logits,
            "original_features": original_features,
            "lung_features": lung_features,
            "fused_features": fused_features,
        }


if __name__ == "__main__":
    model = JointLearningModel(pretrained=False)
    model.eval()

    # Simulate a batch of two grayscale image pairs.
    original = torch.randn(2, 1, 224, 224)
    lung = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        output = model(original, lung)

    print("Joint Learning Model verification passed")
    print("Original features:", tuple(output["original_features"].shape))
    print("Lung features:", tuple(output["lung_features"].shape))
    print("Fused features:", tuple(output["fused_features"].shape))
    print("Classification logits:", tuple(output["logits"].shape))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
