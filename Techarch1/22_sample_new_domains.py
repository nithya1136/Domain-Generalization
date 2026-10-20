
from pathlib import Path
import random
import csv

ROOT = Path(r"C:\projects\Final Year Project\Dataset\Temp")
OUTPUT = Path("outputs/14_new_domain_pilot")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20
SEED = 42

random.seed(SEED)

domains = {
    "Domain3_Normal": ROOT / "Normal",
    "Domain4_TB": ROOT / "Tuberculosis",
}

rows = []

for domain_name, folder in domains.items():
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    if len(images) < SAMPLE_SIZE:
        raise ValueError(
            f"{domain_name}: only {len(images)} images found"
        )

    selected = random.sample(images, SAMPLE_SIZE)

    for path in selected:
        rows.append({
            "domain": domain_name,
            "label": 0 if domain_name == "Domain3_Normal" else 1,
            "image_path": str(path.resolve()),
            "image_name": path.name,
        })

    print(f"\n{domain_name}: selected {len(selected)} images")
    for path in selected:
        print(path.name)

csv_path = OUTPUT / "pilot_sample.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTotal selected: {len(rows)}")
print(f"Sample manifest saved to: {csv_path.resolve()}")
print("Original images were not modified.")
