from pathlib import Path
import subprocess
import zipfile
import shutil

DATA_ROOT = Path("data/raw")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DATASET = "apollo2506/eurosat-dataset"

print("=" * 60)
print("Downloading EuroSAT Dataset")
print("=" * 60)

subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(DATA_ROOT),
    ],
    check=True,
)

zip_file = DATA_ROOT / "eurosat-dataset.zip"

print("\nExtracting dataset...")

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(DATA_ROOT)

zip_file.unlink()

print("\nCleaning directory...")

# The dataset extracts to data/raw/2750
source = DATA_ROOT / "2750"
target = DATA_ROOT / "EuroSAT_RGB"

if source.exists():
    source.rename(target)

print("\nDataset Ready!")

print(target)