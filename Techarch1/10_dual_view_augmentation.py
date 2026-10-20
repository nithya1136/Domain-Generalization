
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class DualViewAugmentation:
    """
    Apply synchronized augmentation to an original X-ray and its
    corresponding lung-masked image.

    Input:
        original: grayscale PIL image
        lung:     grayscale PIL image

    Output:
        original_tensor, lung_tensor
    """

    def __init__(self, training=True, image_size=224):
        self.training = training
        self.image_size = image_size

    def __call__(self, original, lung):
        # Both views must have the same spatial dimensions.
        if original.size != lung.size:
            raise ValueError(
                f"View dimensions differ: {original.size} vs {lung.size}"
            )

        # Resize both views identically.
        original = TF.resize(
            original,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        lung = TF.resize(
            lung,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        if self.training:
            # Synchronized horizontal flip.
            if random.random() < 0.5:
                original = TF.hflip(original)
                lung = TF.hflip(lung)

            # Synchronized small rotation.
            angle = random.uniform(-5.0, 5.0)
            original = TF.rotate(
                original,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            lung = TF.rotate(
                lung,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            # Conservative intensity augmentation.
            # Use the same parameters for both views.
            brightness = random.uniform(0.95, 1.05)
            contrast = random.uniform(0.95, 1.05)

            original = TF.adjust_brightness(original, brightness)
            lung = TF.adjust_brightness(lung, brightness)

            original = TF.adjust_contrast(original, contrast)
            lung = TF.adjust_contrast(lung, contrast)

        # Convert grayscale images to tensors in [0, 1].
        original = TF.to_tensor(original)
        lung = TF.to_tensor(lung)

        # ResNet-18 can accept one-channel input after its first
        # convolution is adapted to one input channel.
        return original, lung


if __name__ == "__main__":
    print("Dual-view augmentation module loaded successfully.")
    print("Training: synchronized geometric and conservative intensity augmentation.")
    print("Validation/test: resize and tensor conversion only.")
    print("Output: two grayscale tensors with shape (1, 224, 224).")
