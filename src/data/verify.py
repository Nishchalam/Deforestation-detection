from pathlib import Path

dataset = Path("data/raw/EuroSAT_RGB")

assert dataset.exists(), "Dataset not found!"

classes = sorted(
    [p for p in dataset.iterdir() if p.is_dir()]
)

print("=" * 60)
print("Dataset Summary")
print("=" * 60)

total = 0

for cls in classes:

    n = len(list(cls.glob("*.jpg")))

    total += n

    print(f"{cls.name:<25}{n}")

print("=" * 60)
print(f"Classes : {len(classes)}")
print(f"Images  : {total}")
print("=" * 60)