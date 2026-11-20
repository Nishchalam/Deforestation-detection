# Deforestation detection in Rondônia from Sentinel-2

Per-pixel forest mapping for **Ariquemes, Rondônia, Brazil** (bbox −63.0, −10.5, −62.8, −10.3) in **2018 and 2022**, built on real Sentinel-2 imagery and labelled with [Hansen Global Forest Change](https://glad.earthengine.app/view/global-forest-change). Forest loss is defined as forest(2018) AND NOT forest(2022). Every number in this README is generated from `reports/metrics/*.json` by [`scripts/readme_tables.py`](scripts/readme_tables.py).

> **Status: v2 is in progress on branch `v2-real-data`.** Data, labels, split, baselines, U-Nets and change detection run end to end via `scripts/` and [`notebooks/end_to_end.ipynb`](notebooks/end_to_end.ipynb). Deployment (ONNX, API, Docker) comes next. The v1 "deforestation" results (IoU 0.975) were computed on synthetic scenes stitched from EuroSAT tiles and have been removed (see [PR #2](https://github.com/Nishchalam/Deforestation-detection/pull/2)).

## Pipeline

```mermaid
flowchart LR
    A[GEE: Sentinel-2 L1C<br/>Jun-Sep 2018 & 2022] -->|s2cloudless mask,<br/>median| B[6-band composites<br/>10 m, EPSG:32720]
    H[GEE: Hansen GFC<br/>treecover2000, lossyear] -->|nearest, same grid| L[labels: forest 2018,<br/>forest 2022, loss]
    B --> T[256 px tiles,<br/>blocked train/val/test]
    L --> T
    T --> M[forest masks: NDVI,<br/>random forest, U-Net]
    T --> D2[direct change models:<br/>12-band U-Net, random forest]
    M -->|2018 AND NOT 2022| C[loss map, hectares,<br/>error analysis, PRODES check]
    D2 --> C
    C --> D[ONNX + FastAPI + Docker<br/><i>next</i>]
```

## Data

<!-- BEGIN:data -->
| | |
|---|---|
| Sentinel-2 collection | `COPERNICUS/S2_HARMONIZED` + s2cloudless cloud mask |
| Hansen GFC version | `UMD/hansen/global_forest_change_2025_v1_13` |
| Grid | EPSG:32720, 10 m, 2190 x 2213 px |
| Scenes 2018 (Jun-Sep) | 48 images / 24 dates; clear obs per pixel min 14, median 20 |
| Scenes 2022 (Jun-Sep) | 50 images / 24 dates; clear obs per pixel min 13, median 17 |
| Forest fraction 2018 / 2022 | 39.7% / 37.2% |
| Hansen loss 2019-2022 | 120,395 px = 1,204 ha |
<!-- END:data -->

![Composites and Hansen layers on one grid](reports/figures/alignment.png)

The bottom row zooms into the yellow box. The red line traces the 2019–2022 Hansen loss: it sits on intact forest in 2018 and on clearings in 2022, which shows the imagery and labels share one grid.

### Spatial split

Whole 256×256 px tiles are assigned to a split. Test is one contiguous 3×3 block and val one 2×4 block that doesn't touch it, so test pixels aren't surrounded by training pixels ([why](https://doi.org/10.1111/ecog.02881)). The blocks are chosen by a fixed, label-only rule before any training ([`tiles.choose_split`](src/deforest/tiles.py)).

<!-- BEGIN:split -->
| Split | Tiles | Forest 2018 | Forest 2022 | Loss 2019-2022 |
|---|---:|---:|---:|---:|
| train | 47 | 40.5% | 38.2% | 2.31% |
| val | 8 | 39.7% | 37.5% | 2.20% |
| test | 9 | 40.4% | 38.4% | 1.98% |
| all | 64 | 40.4% | 38.1% | 2.25% |
<!-- END:split -->

![Tile split](reports/figures/split.png)

## Results (9 test tiles, never used for training or selection)

**Forest mask** (pixels of 9 test tiles × 2 years; CI = 95% bootstrap over tiles):

<!-- BEGIN:forest -->
| Method | IoU | IoU 95% CI | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| NDVI threshold | 0.701 | [0.607, 0.770] | 0.824 | 0.815 | 0.833 |
| Random forest (pixel) | 0.739 | [0.649, 0.803] | 0.850 | 0.894 | 0.810 |
| U-Net R18, ImageNet init | 0.756 | [0.675, 0.815] | 0.861 | 0.883 | 0.841 |
| U-Net R18, random init | 0.746 | [0.664, 0.806] | 0.855 | 0.870 | 0.840 |
<!-- END:forest -->

**Forest loss 2018 → 2022**, scored against Hansen loss 2019–2022. There are two approaches:
- *Differencing*: predicted forest 2018 AND NOT predicted forest 2022.
- *Direct*: one model sees both years stacked (12 bands) and predicts loss.

The minimum mapping unit (MMU), and the probability threshold for direct models, are chosen on val tiles:

<!-- BEGIN:change -->
| Method | Approach | MMU (px) | IoU | F1 | F1 95% CI | Precision | Recall | Loss area (ha) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NDVI threshold | differencing | 100 | 0.229 | 0.372 | [0.132, 0.507] | 0.343 | 0.407 | 138.9 |
| Random forest (pixel) | differencing | 50 | 0.377 | 0.548 | [0.327, 0.632] | 0.569 | 0.528 | 108.5 |
| U-Net R18, ImageNet init | differencing | 50 | 0.280 | 0.438 | [0.190, 0.564] | 0.343 | 0.603 | 205.1 |
| U-Net R18, random init | differencing | 100 | 0.220 | 0.361 | [0.108, 0.491] | 0.320 | 0.413 | 150.9 |
| Random forest, direct (12 bands) | direct | 100 | 0.411 | 0.582 | [0.361, 0.685] | 0.489 | 0.719 | 171.8 |
| U-Net R18, direct (12 bands) | direct | 25 | 0.451 | 0.622 | [0.375, 0.719] | 0.594 | 0.652 | 128.1 |
| *Hansen reference* | | | | | | | | 116.8 |
| *PRODES (independent)* | | | | | | | | 10.8 |
<!-- END:change -->

**Independent check against [PRODES](https://terrabrasilis.dpi.inpe.br/)**, which maps only clear-cut of primary forest, so recall is the meaningful number:

<!-- BEGIN:prodes -->
| vs PRODES 2019-2022, test tiles unless noted | Precision | Recall | F1 |
|---|---:|---:|---:|
| Hansen loss (the training labels) | 0.012 | 0.127 | 0.021 |
| Hansen loss (the training labels) — whole AOI | 0.184 | 0.562 | 0.278 |
| NDVI threshold | 0.038 | 0.490 | 0.071 |
| Random forest (pixel) | 0.050 | 0.500 | 0.091 |
| U-Net R18, ImageNet init | 0.014 | 0.262 | 0.026 |
| U-Net R18, random init | 0.040 | 0.558 | 0.075 |
| Random forest, direct (12 bands) | 0.037 | 0.585 | 0.069 |
| U-Net R18, direct (12 bands) | 0.034 | 0.400 | 0.062 |
<!-- END:prodes -->

![Loss error map](reports/figures/change_error_map.png)

What the numbers say, in short:
- The U-Net is the best forest-mask model, but the confidence intervals of all three learned methods overlap.
- For loss, differencing two masks turns 1–3 pixel edge disagreements into false loss slivers. The direct 12-band U-Net avoids most of them: it has the best loss F1, and its loss area is closest to Hansen. Its CI still overlaps with the direct random forest. See the [error analysis](notebooks/end_to_end.ipynb) (section 7).
- The test block contains very little PRODES loss, so the PRODES rows on test tiles rest on few pixels.

## Data sources

- [Sentinel-2 L1C, harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED) (ESA Copernicus), via [Google Earth Engine](https://earthengine.google.com/)
- [s2cloudless cloud probability](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) and the [masking recipe](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless)
- [PRODES yearly deforestation](https://terrabrasilis.dpi.inpe.br/en/home-page/) (INPE), via the TerraBrasilis WFS
- [Hansen et al. Global Forest Change](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html) ([paper](https://doi.org/10.1126/science.1244693))

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA/CPU build
pip install -r requirements.txt && pip install -e .
earthengine authenticate          # once per machine; set EE_PROJECT=<gcp-project> if needed
```

## Reproduce

```bash
python scripts/download_data.py --config configs/ariquemes.yaml --counts-only   # scene counts only
python scripts/download_data.py --config configs/ariquemes.yaml                 # GeoTIFFs -> data/raw/
python scripts/check_alignment.py --config configs/ariquemes.yaml               # alignment figure + data_stats.json
python scripts/make_labels.py --config configs/ariquemes.yaml                   # labels.tif
python scripts/make_tiles.py --config configs/ariquemes.yaml                    # split.json + figure
python scripts/train.py                                                         # baselines + 3 U-Nets -> outputs/models/
python scripts/evaluate.py                                                      # test metrics -> reports/metrics/, deploy.json
python scripts/readme_tables.py                                                 # refresh README tables
pytest
```

The [notebook](notebooks/end_to_end.ipynb) runs the same `train`/`evaluate` code from [`src/deforest/experiment.py`](src/deforest/experiment.py) and adds the figures and error analysis. It is generated by [`notebooks/build_notebook.py`](notebooks/build_notebook.py):

```bash
python notebooks/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/end_to_end.ipynb
FAST_DEV_RUN=1 python scripts/train.py && FAST_DEV_RUN=1 python scripts/evaluate.py   # CPU smoke run -> outputs/fast_dev/
```

**Predict** forest masks, a loss map and area stats for any two composites on one grid (or for a bbox; it downloads them from GEE):

```bash
python scripts/predict.py --before s2_2018.tif --after s2_2022.tif --out outputs/predict/demo
python scripts/predict.py --bbox -63.0 -10.5 -62.8 -10.3 --years 2018 2022 --out outputs/predict/ariquemes
```

All settings are in [`configs/ariquemes.yaml`](configs/ariquemes.yaml). Data, imagery and weights are not committed.

## Design decisions

- **L1C instead of L2A for both years:** GEE has only 6 L2A dates for Jun–Sep 2018 here vs 24 for L1C, so L1C keeps processing and seasonal sampling identical across years ([`download_data.py --counts-only`](scripts/download_data.py)).
- **Dry-season median composite:** fewest clouds; the median rejects leftover cloud and shadow ([GEE compositing](https://developers.google.com/earth-engine/guides/ic_composite_mosaic)).
- **One shared UTM grid:** S2 and Hansen are requested on an identical 10 m grid snapped to the native S2 pixels, so labels line up pixel for pixel ([`gee.make_grid`](src/deforest/gee.py)).
- **Tiled `getDownloadURL` instead of Export to Drive:** the area is small (~60 MB), so synchronous 1024 px requests under the 32 MB cap keep the download a single scripted step ([limits](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl)).
- **Loss window 2019–2022:** it equals forest(2018) AND NOT forest(2022) under the label rule. Caveat: loss in Oct–Dec 2018 is missed and Oct–Dec 2022 is included.
- **U-Net with ResNet-18 encoder, `in_channels=6`:** a standard segmentation baseline; the pretrained RGB filters are adapted to 6 bands ([smp](https://smp.readthedocs.io/en/latest/insights.html#input-channels)).
- **Direct change model (both years stacked):** the network compares the dates itself instead of subtracting two independent masks ([early-fusion change detection, Daudt et al. 2018](https://arxiv.org/abs/1810.08462)).
- **Minimum mapping unit and threshold chosen on val:** removes speckle without touching test data.
- **Blocked spatial split:** avoids inflated scores from neighbouring, near-identical tiles ([Roberts et al. 2017](https://doi.org/10.1111/ecog.02881)).

## Limitations (so far)

- Labels come from Hansen, so agreement with Hansen is partly circular. PRODES gives an independent check, but it uses a stricter loss definition.
- Hansen has label noise: some pixels it calls forest in 2022 already look cleared (e.g. loss it records only in 2023 or later).
- L1C is top-of-atmosphere reflectance (no atmospheric correction). Both years are treated the same way.
- With 64 tiles, the test set is 9 tiles, so the confidence intervals are wide.
- Remaining loss errors are mostly 1–3 pixel boundary disagreements with Hansen's 30 m grid, plus degraded (not cleared) forest, which Hansen sometimes counts as loss.

## Legacy: EuroSAT CNN benchmark (v1)

The v1 project classified 64×64 [EuroSAT](https://github.com/phelber/eurosat) RGB patches into 10 land-cover classes with CNNs trained from scratch. Numbers come from the test-set metrics files in [`legacy/eurosat/`](legacy/eurosat/):

<!-- BEGIN:legacy -->
| Model | Test acc. | Macro P | Macro R | Macro F1 | img/s |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 96.04% | 0.960 | 0.960 | 0.959 | 276 |
| GoogLeNet | 87.00% | 0.880 | 0.869 | 0.868 | 139 |
| EfficientNet-B0 | 86.63% | 0.889 | 0.869 | 0.866 | 153 |
| AlexNet | 83.26% | 0.840 | 0.833 | 0.828 | 189 |
| LeNet-5 | 64.41% | 0.655 | 0.631 | 0.620 | 214 |
<!-- END:legacy -->

## License

[Apache 2.0](LICENSE)
