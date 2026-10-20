
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\projects\Final Year Project\Dataset\Temp")

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

for folder_name in ["Normal", "Tuberculosis"]:
    folder = ROOT / folder_name

    print(f"\n{'=' * 50}")
    print(f"Folder: {folder_name}")
    print(f"Path: {folder}")

    if not folder.exists():
        print("ERROR: Folder not found!")
        continue

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Total images: {len(images)}")
    print("Extensions:", dict(Counter(
        p.suffix.lower() for p in images
    )))

    print("\nFirst 5 image paths:")
    for image_path in images[:5]:
        print(image_path)

print("\nInspection completed.")
