
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SharedResNet18(nn.Module):
    """
    One ResNet-18 backbone shared by both X-ray views.

    Inputs:
        original: (B, 1, 224, 224)
        lung:     (B, 1, 224, 224)

    Outputs:
        original_features: (B, 512)
        lung_features:     (B, 512)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt RGB first convolution to grayscale.
        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        backbone.conv1 = new_conv

        # Remove the final ImageNet classification layer.
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

    def forward(self, original, lung):
        # The same module processes both views; weights are shared.
        original_features = self.feature_extractor(original).flatten(1)
        lung_features = self.feature_extractor(lung).flatten(1)

        return original_features, lung_features


if __name__ == "__main__":
    model = SharedResNet18(pretrained=True)
    model.eval()

    original = torch.randn(2, 1, 224, 224)
    lung = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        original_features, lung_features = model(original, lung)

    print("Shared ResNet-18 verification passed")
    print("Original features:", tuple(original_features.shape))
    print("Lung features:", tuple(lung_features.shape))
    print("Feature dimension:", model.feature_dim)
    print("Shared backbone parameters:", sum(
        p.numel() for p in model.feature_extractor.parameters()
    ))
