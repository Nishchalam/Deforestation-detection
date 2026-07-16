# 🛠️ Implementation Guide: Build Deforestation Detection from Scratch

This guide is a step-by-step tutorial to help beginners build the entire deforestation detection system from scratch.

---

## 1. Prerequisites & Environment Setup
Before writing code, ensure you have Python 3.10+ installed.

### Setup Steps:
```bash
# 1. Create a project directory
mkdir Deforestation-detection && cd Deforestation-detection

# 2. Setup a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Create requirements.txt and install
pip install --upgrade pip
pip install torch torchvision numpy pandas matplotlib opencv-python scikit-learn seaborn pyyaml pytest
```

---

## 2. Dataset Download & Verification
We use the EuroSAT RGB dataset, which contains 27,000 satellite patches categorized into 10 classes.

### Code (`src/data/download.py`):
Implement script utilizing Kaggle API or direct URL download.
* **Why it exists**: Automates dataset retrieval and extracts files to `data/raw/EuroSAT_RGB/`.
* **Common Mistakes**: Forgetting to configure Kaggle API credentials (`~/.kaggle/kaggle.json`).
* **Verification**: Verify that `data/raw/EuroSAT_RGB/Forest/` contains `.jpg` files.

---

## 3. Exploratory Data Analysis (EDA)
Create `notebooks/01_EDA.ipynb`.
* **Why it exists**: Helps understand class balances, calculate image mean/std across channels, and plot sample grids.
* **Verification**: Ensure all 10 categories contain approximately 2,000–3,000 images (balanced dataset).

---

## 4. Preprocessing & DataLoader
Create `src/data/transforms.py` and `src/data/dataloader.py`.
* **Why it exists**: Resizes inputs to 224x224, applies random flips/rotations to prevent overfitting, and normalizes channels.
* **Common Mistakes**: Applying training augmentations to the validation or testing splits.
* **Verification**: Write a quick script to load a batch and verify shape is `(BatchSize, 3, 224, 224)`.

---

## 5. Model Zoo & Reusable Trainer
Create model definitions in `src/models/` and the trainer in `src/training/trainer.py`.
* **Why it exists**: Separates model construction from training execution.
* **CNN implementation order**:
  1. **LeNet-5**: Simple convolutions and pooling.
  2. **AlexNet**: Deep layers with Dropout.
  3. **VGG16**: Deep stacks of 3x3 filters.
  4. **GoogLeNet**: Multi-scale Inception blocks.
  5. **ResNet18 / ResNet50**: Skip connections to avoid vanishing gradients.
  6. **EfficientNet-B0**: compound scaling.
* **Common Mistakes**: Incorrect computation of flattened dimensions before linear layers. Use `nn.AdaptiveAvgPool2d((1, 1))` to standardize feature sizes.
* **Verification**: Run pytest unit tests to check model compilation: `pytest tests/`.

---

## 6. Sentinel-2 Inference & Mapping
Create `src/inference/patch_generator.py` and `src/inference/landcover_mapper.py`.
* **Why it exists**: Slices large temporal TIF files into 64x64 grids, runs classification, and stitches labels back into maps.
* **Verification**: Run sliding window inference on a mock image and confirm the output matches the original dimensions.

---

## 7. Change Detection & Validation
Create `src/change_detection/` and `src/validation/` packages.
* **Why it exists**: Detects Forest → Non-Forest transitions, counts losses/gains, and evaluates predictions against ground truth.
* **Verification**: Confirm `statistics.json` and `validation_report.md` are correctly generated under `reports/validation/`.

---

## 8. Model Explainability (XAI)
Create `src/explainability/gradcam.py` and `src/explainability/saliency.py`.
* **Why it exists**: Explains model decisions using Grad-CAM heatmaps and Saliency maps.
* **Verification**: Ensure overlay heatmaps are correctly exported to `outputs/explainability/`.
