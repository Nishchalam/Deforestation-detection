# A Tutorial-Style Deep Dive: Deep Learning for Amazon Deforestation Detection

> **Who this is for.** You've written some Python and maybe touched a Keras or PyTorch tutorial once, but you don't know what a spectral band is, what "Hansen" means to a remote-sensing person, or how a classifier that only knows 64×64 photos ends up producing a map of forest loss over 22 km of the Amazon. This document walks through the whole pipeline in that repo, from first principles, and tells you where the honest cracks are.
>
> **What this is not.** A restatement of the README. The README is the "how do I run it" page. This is the "what is actually going on" page.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Remote Sensing in Ten Minutes](#2-remote-sensing-in-ten-minutes)
3. [The Three Datasets This Project Uses](#3-the-three-datasets-this-project-uses)
4. [A Crash Course in Convolutional Neural Networks](#4-a-crash-course-in-convolutional-neural-networks)
5. [The Model Zoo — Six Architectures, One Task](#5-the-model-zoo--six-architectures-one-task)
6. [The Training Pipeline](#6-the-training-pipeline)
7. [From Classifier to Map: Sliding-Window "Segmentation"](#7-from-classifier-to-map-sliding-window-segmentation)
8. [Change Detection: What Turned Into What](#8-change-detection-what-turned-into-what)
9. [Hansen Validation: The Ground Truth](#9-hansen-validation-the-ground-truth)
10. [What Actually Happened When We Ran It](#10-what-actually-happened-when-we-ran-it)
11. [Honest Limitations](#11-honest-limitations)
12. [Where to Go Next](#12-where-to-go-next)
13. [Glossary](#13-glossary)

---

## 1. The Problem

The Amazon rainforest has been losing tree cover for decades. In the Brazilian state of **Rondônia**, in the country's south-west, the pattern is unusually clear from space: a road (like BR-364) is cut into the jungle, and settlers clear rectangular parcels perpendicular to the road for pasture and crops. Seen from a satellite, the pattern looks like a fish skeleton — the road is the spine, the cleared plots are the ribs. This is the famous **"fishbone" deforestation pattern**, and it makes Rondônia one of the best-studied testbeds for satellite-based forest monitoring on Earth.

The concrete task in this repo is: **given two Sentinel-2 satellite images of the same region taken years apart, produce a map that highlights where forest was replaced with something else, and quantify how much forest was lost.**

That sentence hides a stack of subproblems:

- **What is "forest" in a pixel?** Pixels don't have labels. Something needs to classify them.
- **How do you handle images that are millions of pixels wide?** Neural networks eat fixed-size inputs.
- **How do you distinguish "the trees got cut down" from "the season changed" or "a cloud moved"?**
- **How do you know whether your map is right?** You need an independent ground-truth to compare against.

The rest of this document unpacks each of those.

---

## 2. Remote Sensing in Ten Minutes

### 2.1 Satellites see the world in "bands"

Your phone camera captures three colour channels — Red, Green, Blue — because human eyes have three types of cone cell. That's a design choice, not a law of physics. A satellite sensor is free to record any wavelength of light it wants.

An Earth-observation satellite typically records several **spectral bands**, each covering a narrow range of the electromagnetic spectrum. Plants, water, soil, and asphalt reflect very differently outside the visible range, so extra bands help distinguish them. For example, healthy vegetation is *dim* in visible red (chlorophyll absorbs it) and *very bright* in near-infrared (leaves scatter it). That contrast is the basis of the **NDVI** (Normalized Difference Vegetation Index):

```
NDVI = (NIR - Red) / (NIR + Red)
```

NDVI is high (near +1) over healthy vegetation, near 0 over bare soil, and negative over water. Whole industries run on this one ratio.

### 2.2 Three kinds of resolution

Every satellite involves trade-offs across three resolutions:

- **Spatial resolution** — how large a real-world patch each pixel represents. Sentinel-2's RGB bands are **10 m per pixel**. A 64×64 patch therefore covers 640 m × 640 m on the ground — about six city blocks.
- **Temporal resolution / revisit time** — how often the satellite passes over the same spot. Sentinel-2 is roughly **every 5 days** at the equator (two satellites, A and B).
- **Spectral resolution** — how many bands, and how narrow they are. Sentinel-2 has 13 bands; MODIS has 36 but at 250 m resolution; a phone camera has 3 broad ones.

You can rarely have all three at once. The satellite designer picks a corner of that triangle.

### 2.3 Sentinel-2 specifically

Sentinel-2 is a pair of ESA (European Space Agency) satellites launched in 2015 and 2017. Data is free and open. This project uses only three of its 13 bands — B4 (red), B3 (green), B2 (blue) — to make a natural-colour RGB image, because that's what EuroSAT ships. In principle you could use all 13, and you would get a better model. That's future work.

### 2.4 Composites and cloud filtering

A single Sentinel-2 scene often has clouds. To get a usable image of a place, remote-sensing pipelines build a **composite**: for every pixel, look at all the scenes taken over a year, throw out the cloudy ones, and take the median (or a similar summary). What you get is a "typical clear day" view — no clouds, no shadows, and averaged over seasonal variation.

That's exactly what `download_region.py` does. Look at `get_s2_composite`:

```python
s2_col = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(region)
      .filterDate(f"{year}-01-01", f"{year}-12-31")
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
)
composite = s2_col.median()
rgb = composite.visualize(bands=['B4','B3','B2'], min=0, max=3000)
```

Three moves:

1. Pull every Sentinel-2 Level-2A (surface-reflectance, harmonized) scene inside the bounding box during the target year.
2. Keep only scenes with < 10 % cloud cover — widen to 25 % if we found nothing.
3. Take the pixel-wise **median** of what's left. Median is the standard choice over mean because it rejects outliers (residual cloud, aircraft, sun-glint) that would drag a mean.

The final `visualize(min=0, max=3000)` step converts raw reflectance (nominally 0–10000) into 8-bit RGB by clipping at 3000. This is a cosmetic choice — it makes vegetation look green rather than washed-out — but it also means the pixel intensities the model sees at inference are **not on the same scale** as the EuroSAT training images. Hold that thought; it comes back in §11.

### 2.5 What is "Google Earth Engine"?

Google Earth Engine (GEE) is a hosted platform that stores petabytes of satellite imagery and lets you run queries against it from Python without ever downloading the raw scenes. `download_region.py` uses it to do all the compositing server-side and only pulls back a single 1024×1024 PNG at the end. That's why running the demo needs a GEE auth flow and a Cloud project — you're using their compute.

---

## 3. The Three Datasets This Project Uses

### 3.1 EuroSAT — the training set

**EuroSAT** is a labelled land-cover dataset built from Sentinel-2 imagery over Europe. There are two variants; this project uses the **RGB variant**:

- **27,000 images**, each 64 × 64 pixels, 3 channels
- **10 classes**, each with 2,000–3,000 images:
  `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake`
- Each image covers 640 m × 640 m on the ground
- Downloaded from Kaggle (`nilesh789/eurosat-rgb`, 90 MB zipped)

The images are already curated: someone drew polygons over land-use maps of Europe and cut 64×64 chips from Sentinel-2. That's why it's a clean classification task — a whole 64×64 chip is one thing.

### 3.2 The three demo regions in Rondônia

The pipeline is deployed on three non-overlapping regions to test that it generalizes:

| Region | Purpose | Notes |
| :--- | :--- | :--- |
| Ji-Paraná | Single-date land-cover mapping | Heavy agriculture / pasture mixing |
| Porto Velho Frontier | Temporal change + validation | Active clearing frontier |
| Ariquemes Corridor | End-to-end demo | Urbanizing corridor, default bounding box |

Bounding boxes and cloud thresholds live in `src/regions.py` and `download_region.py`. The default demo box is `[-63.0, -10.5, -62.8, -10.3]` — roughly 22 km × 22 km around Ariquemes.

### 3.3 Hansen Global Forest Change — the reference truth

This one deserves its own section: see [§9](#9-hansen-validation-the-ground-truth).

---

## 4. A Crash Course in Convolutional Neural Networks

If you've never trained one, here's the minimum you need to follow the rest of this document.

### 4.1 What a convolution is

A **convolution** is a small learnable filter — say 3×3 numbers — that slides across the input image and computes a weighted sum at every position. If the filter looks like `[[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]]` (Sobel), it responds strongly to vertical edges. In a CNN, the filters are *learned* from data; you don't handcraft them.

A single convolution *layer* usually applies dozens of such filters in parallel, producing a stack of "feature maps" — one per filter.

### 4.2 Why stacking helps

- **Layer 1** learns tiny local patterns: edges, colour blobs.
- **Layer 3** starts learning combinations: corners, small textures.
- **Layer 10+** learns things like "a rectangular field with parallel furrows" or "the crown of a broadleaf tree."

Between layers, **pooling** (usually taking the max over 2×2 windows) halves the resolution but doubles the receptive field. By the end, each neuron "sees" a large portion of the input.

### 4.3 From features to a class

The last convolutional feature map is flattened (or globally average-pooled) into a vector, run through one or two **fully-connected (FC)** layers, and finally a **softmax** turns 10 raw scores into 10 probabilities that sum to 1.

### 4.4 How the network learns

- **Loss function**: `CrossEntropyLoss` compares the predicted probability distribution to the true one-hot label. It is minimized when the model puts all its probability mass on the correct class.
- **Optimizer**: `Adam` computes gradients of the loss with respect to every weight and takes small steps in the direction that reduces the loss. Its per-parameter adaptive learning rate is why it "just works" on most vision tasks.
- **Backpropagation** is the chain rule applied efficiently across the network — you don't have to implement it; PyTorch does.

### 4.5 Preventing overfitting

A model with 11 million parameters can memorize 21,600 training images perfectly and still be useless on new ones. Three levers push back:

- **Data augmentation** — flip / rotate / colour-jitter each training image on every epoch so the model can't rote-learn pixel positions. See `src/preprocessing.py`.
- **Early stopping** — if validation loss stops improving for N epochs, stop training and revert to the best checkpoint.
- **Weight decay / dropout** — not heavily used in this project; the augmentation + early stopping combo is enough.

---

## 5. The Model Zoo — Six Architectures, One Task

Six CNNs were trained on EuroSAT from scratch (no ImageNet pretraining). Each is a milestone in the history of computer vision.

### 5.1 LeNet-5 (1998)

Yann LeCun's original CNN, designed for handwritten digit recognition. Two conv layers, two pooling layers, two FC layers. **62,006 parameters** — smaller than a JPEG of your face. Included as a baseline; anything below its accuracy means something is broken.

### 5.2 AlexNet (2012)

The paper that started the deep-learning boom by winning ImageNet 2012 by a huge margin. Introduced ReLU activations, GPU training, dropout, and a deep-for-its-time 8-layer stack. **57 M parameters**, mostly in three enormous FC layers at the end. On EuroSAT it does OK (83 %) but its "bulk in the head" design is the least parameter-efficient in the zoo.

### 5.3 VGG-16 (2014)

Very deep, very simple: sixteen weight layers, all with 3×3 convolutions and 2×2 max-pooling. **134 M parameters**. It set the pattern for "make convolutional networks deeper by stacking small filters." The catch: it is memory-hungry to train, needs a good initialization to converge, and can collapse to trivial predictions when trained from scratch with defaults on smaller datasets.

In this project's benchmark **VGG-16 failed to converge** — test accuracy 11.11 % (one class out of ten = random on a balanced test set with tie-breaking), F1 = 0.02. The README's table notes this and points at the notebook. A future attempt with pretrained weights, smaller learning rate, or LR-warmup would almost certainly fix it.

### 5.4 GoogLeNet / Inception v1 (2014)

Instead of one filter size per layer, an **Inception module** runs 1×1, 3×3, and 5×5 convs in parallel and concatenates the outputs. The model chooses at each spot which scale is most useful. Much more parameter-efficient than VGG (**6 M** vs 134 M). Reaches 87 % on EuroSAT.

### 5.5 ResNet-18 (2015)

**The deployment model.** ResNet introduced **residual connections** ("skip connections"): every couple of layers, the input is added back to the output. Formally, instead of learning `H(x)`, the layer learns the *residual* `F(x) = H(x) - x`, and the output is `F(x) + x`. This tiny change makes very deep networks trainable — gradients flow back through the skips without vanishing.

ResNet-18 is the small member of the ResNet family: 18 weight layers, **11.2 M parameters**, ~43 MB on disk. On this project's EuroSAT split it scores **96.04 %** — the best in the zoo — and is also the *fastest* at inference (276 img/s), because it is deep-but-narrow and the residuals help it converge in fewer epochs.

### 5.6 EfficientNet-B0 (2019)

Uses "compound scaling" — jointly scaling depth, width, and resolution by a search-tuned formula — to hit high accuracy at low parameter count. **4.0 M parameters**, ~15 MB. In this project it lands around 87 %, slightly under ResNet-18 despite being 3× smaller. It's the natural choice if you need to run on a phone.

### 5.7 Summary — why ResNet-18 wins here

| Model | Test Acc | Params | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-18** | **96.04 %** | 11.2 M | Best accuracy AND best throughput |
| GoogLeNet | 87.00 % | 5.98 M | Runner-up |
| EfficientNet-B0 | 86.63 % | 4.02 M | Best accuracy-per-parameter |
| AlexNet | 83.26 % | 57.0 M | Heavy FC head |
| LeNet-5 | 76.26 % | 62 K | Baseline |
| VGG-16 | 11.11 % | 134.3 M | Failed to converge from scratch |

ResNet-18 is Pareto-optimal along accuracy, speed, and size. Nothing else in the zoo dominates it on any axis. That's why it's picked as the deployment model for the whole downstream pipeline.

---

## 6. The Training Pipeline

### 6.1 The splits

`notebooks/02_Preprocessing.ipynb` (and the `train.py` script here) build a **stratified 80 / 10 / 10 split** of the 27,000 EuroSAT images: 21,600 train, 2,700 validation, 2,700 test. "Stratified" means every class is represented in every split in its original proportion — you don't accidentally end up with all the "Highway" images in the training set.

Splits are written to `data/processed/{train,validation,test}.csv` as `image_path,label,class_name` triples. The `EuroSATDataset` in `src/dataset.py` just reads a CSV row, opens the image, applies the transform, and returns a dict.

### 6.2 Augmentation (train only)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # upsize for ResNet-family expectations
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),     # legal for aerial imagery! no up/down
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15,
                           saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),  # ImageNet stats
])
```

Two RS-specific things to notice:

- **Vertical flip is turned on.** For a photo of a cat that would be wrong (cats have "up"). For satellite imagery there is no "up" — a field looks the same whether you view it from north-up or south-up. Vertical flip is free training data.
- **Normalization uses ImageNet mean/std.** That's a bit dodgy — EuroSAT isn't ImageNet. But because we train from scratch, the model just learns to compensate, and it's simpler than re-computing statistics.

The `test_transform` skips all augmentation: `Resize → ToTensor → Normalize`. You never augment your test set.

### 6.3 Training loop, in words

Per epoch, the `Trainer` in `src/training.py`:

1. Sets the model to `train()` mode (enables dropout, updates batch-norm running stats).
2. For each batch: zero the gradients, forward pass, compute loss, backward pass, optimizer step. Optionally wrap in `torch.amp.autocast` for mixed precision.
3. After the epoch: switch to `eval()` mode, run the whole validation set with `torch.no_grad()`, record loss and accuracy.
4. Step the scheduler. Default is `ReduceLROnPlateau` — if val loss plateaus for 2 epochs, cut the learning rate by 10×.
5. If val loss improved, save `best_model.pth` and reset the patience counter. Otherwise increment. If patience runs out, early-stop.
6. Free CUDA cache (`torch.cuda.empty_cache()`) — this is the small change that helped on the low-VRAM laptop.

Every checkpoint contains **not just weights** but the optimizer state, scheduler state, epoch number, and full history — so training resumes exactly where it left off.

### 6.4 What "AMP" is and why it helps

**Automatic Mixed Precision** keeps the model weights in float32 but casts activations and gradients to float16 during the forward and backward passes. Two effects:

- **Memory** — activations dominate VRAM usage; halving their precision roughly halves peak VRAM. That's how you fit VGG-16 on a 5 GB laptop GPU.
- **Speed** — modern NVIDIA GPUs have dedicated Tensor Cores that run fp16 matmuls 2–4× faster than fp32.

The `--amp` flag on `train.py` turns this on. The `GradScaler` handles the tricky bit: fp16 gradients can underflow to zero, so we multiply the loss by a large factor before the backward pass and unscale before the optimizer step.

### 6.5 Reproduction numbers

Running `train.py --model resnet18 --epochs 20 --batch_size 32 --patience 3` on the fresh Kaggle-downloaded EuroSAT:

- Early-stopped at epoch 16 (best val loss at epoch 13).
- Best val loss **0.121** (notebook: 0.117).
- Test set: **95.15 %** accuracy, F1 = **0.951** (notebook: 96.04 %, F1 = 0.959).
- Wall clock: **6.1 minutes** on an RTX 5080.
- Peak VRAM during training: **2.26 GB / 16 GB**.

Within a percentage point of the notebook — the gap is the different split seed and one fewer LR-drop cycle before early stopping.

---

## 7. From Classifier to Map: Sliding-Window "Segmentation"

This is the part that most CV newcomers find surprising, so slow down here.

### 7.1 The mismatch

The classifier we just trained eats **one 64×64 patch** and returns **one class label**. But the deployment target is a **1024×1024 satellite image of a whole region**, and what we want back is a **map** — a per-location prediction — not a single label.

There are two families of solutions:

- **Semantic segmentation** — train a different kind of network (U-Net, DeepLab, SegFormer, etc.) that outputs a per-pixel class map in one forward pass. Requires per-pixel labels for training. Slow to train, fast to infer.
- **Patch-based classification with a sliding window** — reuse the patch classifier, but apply it to a grid of patches carved out of the big image, and paint each cell of the output map with the predicted class. Requires only patch-level labels (which we already have). Simple, but coarse-grained.

**This project takes the second path.** So "segmentation" in the loose sense means "produce a segmented-looking map of land cover", but the underlying operation is patch classification with tessellation. There is no per-pixel neural network anywhere.

### 7.2 The sliding window

`PatchGenerator` in `src/inference.py`:

```python
for y in range(0, height - patch_size + 1, stride):
    for x in range(0, width - patch_size + 1, stride):
        patch = image.crop((x, y, x+patch_size, y+patch_size))
        yield patch, (x, y)
```

With `patch_size=64` and `stride=64` on a 1024×1024 image, that's `(1024 // 64)² = 256` non-overlapping patches — a 16×16 grid.

Setting `stride < patch_size` gives overlapping patches; the reconstruction step averages predictions in overlap zones, which reduces edge artefacts. It also increases compute proportionally to `(patch_size / stride)²`. The demo defaults to non-overlapping for speed.

### 7.3 Batched inference

`LandCoverMapper.generate_map` collects all patches, then feeds them to the model in batches of size `--batch_size`. For a 16×16 grid at batch 16, that's 16 forward passes. On a modern GPU this is milliseconds.

Each patch yields:

- `predicted_class` — the argmax over 10 EuroSAT classes.
- `confidence` — the softmax probability of that class.
- `probability_distribution` — full 10-vector, kept for potential downstream analysis.

### 7.4 Painting the map

Each class has a fixed RGB colour (in `LandCoverMapper.COLOR_MAP`):

- Forest → dark green
- Pasture → yellow
- AnnualCrop → light pink
- River, SeaLake → blue / cyan
- Residential → orange
- Industrial → red
- Highway → grey
- HerbaceousVegetation → light green
- PermanentCrop → brown

The reconstruction step fills every patch's bounding box in a blank canvas with its class colour. Overlap regions average. Same trick, in parallel, builds a **confidence map** (greyscale — brighter = more confident).

The output for a 1024×1024 input at stride 64 is two 1024×1024 PNGs: `landcover_map_2018.png` and `confidence_map_2018.png`. To the eye it looks like a low-resolution mosaic — because that's exactly what it is.

### 7.5 Why this is coarse

Every 64×64 pixel patch in the output takes **one label**. If a patch straddles a road, a forest edge, and a field, you get whichever of the three dominates in the classifier's opinion. That's the fundamental resolution of this map: **one class per 64-pixel cell**. Ways to improve it, without switching to a segmentation network:

- **Smaller patches at inference** (e.g. 32×32). Requires training the model at 32×32 (or accepting worse performance from resizing).
- **Overlapping stride** (e.g. stride=32 on 64 patches) plus averaged prediction. Softens boundaries but doesn't add resolution.
- **A per-pixel network** (U-Net trained on labelled polygons). Different data requirement, different training loop, different notebook. Left as future work.

---

## 8. Change Detection: What Turned Into What

Given a land-cover map at Year A (2018) and Year B (2022), how do you extract "deforestation"?

### 8.1 Post-classification comparison

The naive baseline is to subtract the two RGB images pixel-wise and threshold — "look for pixels that changed a lot". This is called **image-differencing change detection** and it is *terrible* for real deforestation monitoring because:

- Sun angle differs between years.
- Vegetation phenology differs (wet vs dry season).
- Atmospheric conditions differ.
- Clouds and cloud shadows differ.

All of those produce false positives.

The alternative — and what this project does — is **post-classification change detection**: classify each date independently into land-cover categories, then compare the *categorical* maps. If a patch is "Forest" in 2018 and "Pasture" in 2022, that's a real class change, not just a lighting change. This throws away lighting information entirely and works on labels.

### 8.2 The transition matrix

`ChangeDetector.compute_transition_matrix(changes)` returns a 10×10 matrix `T[i, j]` = number of patches that were class `i` in Year A and class `j` in Year B. The diagonal is "stable" patches (no class change). Off-diagonal entries are transitions. A well-visible bright cell at `T[Forest, Pasture]` is the smoking gun of deforestation.

The heatmap is written to `reports/demo_results/transition_matrix_heatmap.png`.

### 8.3 The "Forest → non-Forest" filter

`DeforestationDetector` in `src/change_detection.py` just applies a one-line predicate:

```python
was_forest = (class_a == "Forest")
now_not_forest = (class_b != "Forest")
deforested = was_forest and now_not_forest
```

Optionally gated by a `--confidence_threshold` so that low-confidence flips don't count. The result is a per-patch binary mask over the 16×16 grid: 1 = deforested, 0 = stable.

Two visualizations follow:

- `detected_deforestation_mask.png` — a black-and-white patch grid at the input resolution (white = deforested).
- `deforestation_overlay_visual.png` — the Year B RGB with red bounding boxes drawn around each deforested patch.

### 8.4 What this misses

Deforestation is not the only interesting land-cover change. This project fixes the definition as *any* Forest→non-Forest transition. It ignores:

- **Reforestation** (non-Forest → Forest) — sometimes just as important.
- **Forest degradation** (still forest, but thinner canopy) — invisible to a classifier that only sees "Forest / not Forest".
- **Seasonal deciduous** — some forests look thin in dry season and are misclassified as Pasture in some years but not others.

For a first pass this is fine. For research use you'd want a more nuanced definition.

---

## 9. Hansen Validation: The Ground Truth

Now the part you asked about specifically.

### 9.1 What is "Hansen"?

The **Hansen Global Forest Change** dataset is the standard reference for global forest-loss monitoring. It was first published in *Science* in 2013 by a team led by **Matthew Hansen** at the University of Maryland (UMD), and is updated annually. In Earth Engine it lives at `UMD/hansen/global_forest_change_YYYY_vN_M`.

It ships several raster layers at **30 m per pixel** (roughly one Landsat pixel). The ones this project uses are:

- `treecover2000` — percent tree cover in year 2000. The baseline.
- `lossyear` — 8-bit integer per pixel: `0` if no loss detected, otherwise `1..N` encoding the year of loss (`1` = 2001, `2` = 2002, …, `23` = 2023 in the 2023 version). Values > 0 mean the pixel had a "stand-replacement disturbance" — the tree canopy went away entirely, whatever the cause.
- `gain` — pixels that gained forest cover between 2000 and 2012.

The dataset is built from Landsat time series, not Sentinel-2, using a supervised classifier trained on human-labelled reference points. It is not a ground survey — it is a global machine-learning product too — but it's the most widely accepted forest-loss reference in the world, so treating it as ground truth is the community standard.

### 9.2 What "loss" means to Hansen

Hansen defines forest loss as a **stand-replacement disturbance** — the canopy at that 30 m pixel is essentially gone. That includes:

- Clear-cutting for agriculture (most Rondônia loss).
- Logging.
- Fire.
- Windthrow.
- Some — but not all — selective logging (which often leaves the canopy partly intact and evades detection).

Hansen does *not* separate anthropogenic clearing from natural disturbance. If you want "human-caused deforestation only" you have to overlay other layers (roads, protected areas, etc.). This project doesn't; it treats any Hansen loss inside the year range as ground-truth deforestation.

### 9.3 Extracting the mask for our window

`get_hansen_loss_mask(year1, year2, region)` in `download_region.py`:

```python
hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
lossyear = hansen.select('lossyear')
y1_code = year1 - 2000   # 2018 → 18
y2_code = year2 - 2000   # 2022 → 22
loss_mask = lossyear.gte(y1_code).And(lossyear.lte(y2_code))
```

That gives a 30 m-per-pixel binary mask: 1 where loss occurred between year1 and year2 inclusive, 0 elsewhere. Rendered as a 1024×1024 PNG (white = loss, black = stable) and saved as `hansen_validation_mask.png`. Because we're rendering a 22 km × 22 km region at 1024 px, each output pixel covers ~22 m — coarser than Hansen's native 30 m, so the mask *undersamples* the real Hansen data slightly. For a proper study you would work in the native Landsat grid.

### 9.4 The alignment problem

We now have two maps of the same region:

- Model predictions on a **64×64-pixel patch grid**: a 16×16 grid of binary "deforested / not" cells.
- Hansen loss mask on a **1024×1024 pixel grid**: fine-grained binary.

They are at *very* different resolutions. Comparing them requires bringing Hansen down to the model's grid.

### 9.5 The 10 % threshold rule

`run_demo.py`:

```python
gt_patches = []
for bbox in map_b["bboxes"]:              # each 64×64 patch in the grid
    x, y, w, h = bbox
    patch_pixels = gt_array[y:y+h, x:x+w] # 64×64 slice of Hansen mask
    loss_ratio = np.mean(patch_pixels == 255)
    gt_patches.append(1 if loss_ratio > 0.10 else 0)
```

For each patch in the 16×16 grid we ask: *what fraction of the underlying Hansen pixels are marked as loss?* If more than 10 % of the patch area was cleared, we call that patch "deforested" in the ground truth.

Why 10 %? It's a defensible choice, not a discovered law. Reasoning:

- 0 % (any loss counts): captures every specked-in Hansen pixel, including thin strips along roads. High recall, but you flag patches that are 99 % stable forest with one edge trimmed.
- 50 % (majority-loss counts): only patches that are mostly cleared. High precision, but you miss the beginning of a clearing.
- 10 % is a middle ground and is the value the project ships. You should treat it as a hyperparameter and try 5 % / 20 % if you want to see the sensitivity.

After this step you have two 256-length binary vectors that live on the same grid:

- `pred_defor_mask[i]` — 1 if the model predicts patch `i` transitioned Forest → non-Forest.
- `gt_patches[i]` — 1 if Hansen says patch `i` lost more than 10 % of its area between year A and year B.

### 9.6 The confusion counts

For each patch you now have one of four outcomes:

| Model \ Hansen | Loss (1) | No loss (0) |
| :--- | :---: | :---: |
| **Detects (1)** | True Positive (TP) | False Positive (FP) |
| **Skips (0)** | False Negative (FN) | True Negative (TN) |

These four numbers are the basis of every metric that follows. `validate_deforestation()` in `src/change_detection.py` computes them and then derives:

- **Precision** = TP / (TP + FP). Of the patches the model flagged, how many really were cleared?
- **Recall** = TP / (TP + FN). Of the patches Hansen says were cleared, how many did the model catch?
- **F1** = harmonic mean of precision and recall. Punishes imbalance between them.
- **Intersection over Union (IoU)** = TP / (TP + FP + FN). Popular in segmentation. Numerator is patches the two agree are cleared; denominator is patches either side thinks are cleared.

The report also computes:

- `forest_area_lost_pred_ha` = `sum(pred) × (patch_size_m² / 10000)`.
- `forest_area_lost_gt_ha` = `sum(gt) × (patch_size_m² / 10000)`.

(where `patch_size_m` is passed as `float(args.patch_size)` — treating one pixel as one metre, which is a placeholder unit. For real hectares you would use the actual ground area per patch.)

### 9.7 The spatial error map

If instead of counting you *paint* the four categories:

- Green = True Positive
- Red = False Positive
- Blue = False Negative
- Black = True Negative

you get the **spatial error map** shown in `12_Validation.ipynb`. This is the single most useful diagnostic in the pipeline: red splotches tell you where the model is over-eager, blue splotches tell you where it misses real clearings, and the geometry usually explains why.

---

## 10. What Actually Happened When We Ran It

We reproduced the whole pipeline on a fresh Kaggle download of EuroSAT and a fresh GEE fetch of the Ariquemes region.

**Training + evaluation match the notebooks cleanly:**

| Step | Result | Notebook | Peak VRAM |
| :--- | :---: | :---: | :---: |
| `train.py` ResNet-18, 20 ep bs=32 | val loss 0.121, early stopped ep 16 | 0.117, ep 21 | 2.26 GB |
| `evaluate.py` on test set | acc **95.15 %**, F1 **0.951** | 96.04 %, 0.959 | 1.32 GB |

The tiny gap is a different split seed and one fewer LR-drop cycle.

**The deforestation demo runs end-to-end but scores near zero:**

| Metric | Value |
| :--- | :---: |
| Total patches evaluated | 256 |
| Model-detected deforested patches | 1 |
| Hansen ground-truth deforested patches | 24 |
| IoU / Precision / Recall / F1 | 0.00 / 0.00 / 0.00 / 0.00 |

We predicted 1 patch cleared, Hansen said 24 were cleared, and our one guess wasn't among Hansen's 24. Why? See the next section.

---

## 11. Honest Limitations

### 11.1 The scale mismatch (the big one)

**EuroSAT patches are 64 px × 10 m/px = 640 m on the ground.**
**Our demo renders 22 km × 22 km at 1024 px, so a 64-px patch covers ≈ 1.4 km.**

The classifier is being asked to look at things that are more than **twice the size** of what it ever trained on. A 640-m EuroSAT "Pasture" is a single field; a 1.4-km patch is often a road + a field + a treeline. Everything the classifier learned about what "Forest" looks like at 640 m is wrong at 1.4 km.

Fix: render the region at 3072 px instead of 1024 px so that 64 px = ~450 m — closer to the training scale. Or crop the region to a smaller box (say 5 km × 5 km) and keep 1024 px.

### 11.2 The colour balance mismatch

EuroSAT images were saved as standard 8-bit PNGs; whatever normalization they went through, it wasn't `visualize(min=0, max=3000)`. Our GEE pipeline stretches Sentinel-2 reflectance with that fixed range for aesthetic reasons. The result is that a "Forest" patch in a GEE composite has a slightly different colour distribution than any Forest patch the model saw during training.

Fix: use the same normalization as EuroSAT (whatever it is — the dataset paper documents it), or fine-tune the model on a few hundred hand-labelled Sentinel-2 patches from Rondônia.

### 11.3 No pretraining

None of the six models were initialized with ImageNet weights. That's why VGG-16 collapsed and why nothing exceeds 96 %. Loading `torchvision.models.resnet18(weights=IMAGENET1K_V1)` and fine-tuning would probably push accuracy over 98 % and greatly help downstream deployment.

### 11.4 Post-classification vs pixel-wise

The whole "segmentation" is really tessellated classification. A U-Net trained on labelled Rondônia polygons would produce a per-pixel forest / non-forest mask with fluid boundaries. Much more expensive to build (needs pixel labels) but a much better product.

### 11.5 The 10 % Hansen threshold

Pulled from thin air. A patch with 9 % loss is called "no deforestation" and a patch with 11 % loss is called "deforestation". That threshold moves TP/FP/FN counts around a lot and is worth a sweep in any serious use.

### 11.6 The 22-m Hansen re-sampling

Rendering Hansen at 22 m/px on a native-30 m raster does *some* nearest-neighbour aliasing. Better to keep everything in the Landsat grid (30 m/px) and do the loss counting there.

### 11.7 One year, one composite

The composite is an annual median. Loss detection is on a fixed year pair. A real monitoring system would use monthly composites and quarterly change detection, which catches clearings sooner and is more robust to seasonal misclassification.

---

## 12. Where to Go Next

Roughly in order of effort:

1. **Rebalance the demo scale** — render 5 km × 5 km at 1024 px so 64-px patches match EuroSAT's 640 m. Rerun the demo and compare.
2. **Fine-tune with ImageNet weights** — swap `create_model(...)` for `torchvision.models.resnet18(weights=IMAGENET1K_V1)` and start training with `lr=1e-4`. Expect > 98 % on the test set.
3. **Add all Sentinel-2 bands.** EuroSAT has an MS (multispectral) variant with 13 bands. Retrain on that; deploy on 13-band GEE composites. NIR alone should improve Forest / non-Forest separation dramatically.
4. **Sweep the 10 % Hansen threshold.** Plot precision-recall as a function of the threshold and report the F1-optimal value with confidence bounds via bootstrap.
5. **Ship a small U-Net baseline.** Even a poorly trained U-Net on a few hundred manually labelled polygons will show whether pixel-wise is worth the effort. In this domain it usually is.
6. **Temporal smoothing.** Compute per-year land-cover maps and require two consecutive years of "not Forest" before calling a patch "deforested". Kills a lot of one-year misclassifications.
7. **Ingest INPE PRODES.** Brazil's national deforestation inventory (INPE PRODES) is a stronger reference than Hansen for Amazon-specific validation. Its native grid is 30 m too and it's in GEE.

---

## 13. Glossary

- **Band.** One narrow-wavelength channel recorded by a satellite sensor. Sentinel-2 has 13.
- **CNN.** Convolutional Neural Network. A neural network that uses learnable spatial filters.
- **Composite.** A single image summarizing many satellite scenes over a time window, usually by pixel-wise median.
- **EuroSAT.** A 27,000-image labelled land-cover dataset built from Sentinel-2 over Europe. Ten classes, 64×64 RGB.
- **False positive / negative.** Model said Yes when truth said No / model said No when truth said Yes.
- **F1 score.** Harmonic mean of precision and recall. 0 means at least one is 0.
- **Fine-tuning.** Starting from pretrained weights (usually ImageNet) and doing a short training run on your task-specific data.
- **GEE.** Google Earth Engine, a hosted platform for satellite-imagery analysis.
- **Hansen.** UMD-produced global forest-loss dataset. Standard remote-sensing reference truth.
- **Inference.** Running a trained model on new data. As opposed to training.
- **IoU.** Intersection over Union. TP / (TP + FP + FN). A common segmentation metric.
- **NDVI.** Normalized Difference Vegetation Index. `(NIR - Red) / (NIR + Red)`. Vegetation health proxy.
- **Overfitting.** Model does well on training data and badly on new data.
- **PKCE.** The OAuth flow GEE's `notebook` auth mode uses. Involves a code challenge and a verifier.
- **Post-classification change detection.** Compare class labels between dates rather than pixel intensities.
- **Precision / recall.** TP / (TP + FP) and TP / (TP + FN).
- **ReLU.** Rectified Linear Unit. `f(x) = max(0, x)`. The default activation in modern CNNs.
- **Reflectance.** Fraction of incoming light reflected back. Sentinel-2 SR is atmospherically corrected surface reflectance in the range 0–10000.
- **Residual connection.** `output = layer(input) + input`. The one trick that made deep networks trainable.
- **Semantic segmentation.** Per-pixel classification.
- **Sentinel-2.** ESA satellite pair, 13 bands, 10 m in RGB, ~5-day revisit.
- **Sliding window.** Move a fixed-size window across a large image and process each position independently.
- **Softmax.** Turns raw scores into a probability distribution that sums to 1.
- **Stratified split.** A train/val/test split that preserves the per-class proportions.
- **Transition matrix.** `T[i,j]` = number of patches that went from class `i` to class `j` between two dates.

---

*Version 1.0 — corresponds to commit at the time of writing. See `README.md` for how to run the pipeline and `report.md` for a lighter overview.*
