
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageFile

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\projects\FYP\Techarch1")

MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "01_metadata" / "master_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_preprocessed"

OUTPUT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "01_metadata"
    / "preprocessed_manifest.csv"
)

IMAGE_SIZE = 512

# Allow Pillow to load images that may be truncated.
# Such images are still logged if processing fails.
ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

# ============================================================
# HELPER: PAD TO SQUARE, THEN RESIZE
# ============================================================

def preprocess_image(image_path, output_path):
    with Image.open(image_path) as image:
        # Convert to a single grayscale channel.
        image = image.convert("L")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image.size}")

        # Preserve aspect ratio by padding the shorter dimension.
        side = max(width, height)

        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top

        # Use black padding; do not stretch the anatomy.
        image = ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=0
        )

        # Resize using high-quality interpolation.
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Preserve grayscale intensity values in [0, 255].
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")

        return {
            "original_width": width,
            "original_height": height,
            "processed_width": IMAGE_SIZE,
            "processed_height": IMAGE_SIZE,
            "channels": 1,
            "pixel_min": int(np.array(image).min()),
            "pixel_max": int(np.array(image).max()),
        }

# ============================================================
# MAIN
# ============================================================

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_path", "domain", "label", "image_id"
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    total = len(manifest)

    for index, row in manifest.iterrows():
        source_path = Path(str(row["image_path"]))

        if not source_path.is_file():
            failures.append({
                "image_path": str(source_path),
                "error": "Source file does not exist"
            })
            continue

        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            failures.append({
                "image_path": str(source_path),
                "error": "Unsupported image extension"
            })
            continue

        # Use domain/class subfolders to keep the dataset organized.
        domain = str(row["domain"])
        label = str(row["label"])

        # Include a unique index so repeated filenames cannot overwrite.
        output_name = (
            f"{index:05d}_{source_path.stem}.png"
        )

        output_path = OUTPUT_DIR / domain / label / output_name

        try:
            image_info = preprocess_image(source_path, output_path)

            results.append({
                "image_id": row["image_id"],
                "original_path": str(source_path),
                "preprocessed_path": str(output_path),
                "domain": domain,
                "label": label,
                "status": "success",
                **image_info,
            })

        except Exception as error:
            failures.append({
                "image_path": str(source_path),
                "error": repr(error)
            })

        processed_count = len(results) + len(failures)

        if processed_count % 100 == 0 or processed_count == total:
            print(f"Progress: {processed_count}/{total}")

    # Save the successful image mappings.
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_MANIFEST, index=False)

    # Save failures separately.
    failure_path = (
        PROJECT_ROOT / "outputs" / "06_reports"
        / "preprocessing_failures.csv"
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["image_path", "error"]
    ).to_csv(failure_path, index=False)

    print("\n" + "=" * 55)
    print("IMAGE PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"Images in source manifest: {total}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output manifest: {OUTPUT_MANIFEST}")
    print(f"Failure log: {failure_path}")

    if not result_df.empty:
        print("\nSuccessful outputs by domain and class:")
        print(
            pd.crosstab(
                result_df["domain"],
                result_df["label"]
            ).to_string()
        )

    if failures:
        print(
            "\nSome images failed. Review the failure CSV "
            "before proceeding."
        )
    else:
        print("\nAll images were processed successfully.")

if __name__ == "__main__":
    main()
