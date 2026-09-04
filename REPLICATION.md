# Replication Attempt — `train.py`, `evaluate.py`, `run_demo.py`

Environment: Linux container, 4 CPU cores, no GPU, no internet access beyond
PyPI. Date of run: 2026-09-04.

## Summary

| Script | Runs? | Published numbers reproduced? |
| :--- | :---: | :--- |
| `train.py` | yes | not verifiable — EuroSAT is unobtainable in this environment |
| `evaluate.py` | yes | not verifiable — same reason |
| `run_demo.py` | only after a bug fix | no; see below |

The three CLIs are mechanically sound (one crash fixed, see below), but none of
the accuracy figures in `README.md` can be checked here, because the dataset
they depend on cannot be downloaded.

## What blocked a true replication

`data/raw/` and `data/processed/` are gitignored, so the repository ships no
EuroSAT data and no checkpoints. Every documented mirror was unreachable from
this sandbox (`madm.dfki.de`, `zenodo.org`, `huggingface.co`,
`isgwww.cs.uni-magdeburg.de` — all refused). `download_region.py` additionally
needs Google Earth Engine credentials.

To exercise the code paths regardless, `scripts/make_synthetic_eurosat.py`
generates a EuroSAT-shaped stand-in: 10 classes, 64x64 JPEGs, same folder
layout, split with the same stratified 80/10/10 logic as
`notebooks/02_Preprocessing.ipynb`. **The images are procedural colour/texture
patterns, not satellite imagery.** Metrics on them measure whether the code
works, nothing else.

## Runs

```
python scripts/make_synthetic_eurosat.py --per-class 200   # 2000 imgs, 1600/200/200
python train.py --model resnet18 --epochs 3 --lr 0.001 --batch_size 32
python evaluate.py --model resnet18 --checkpoint outputs/checkpoints/resnet18/best_model.pth
python run_demo.py --model resnet18 --checkpoint outputs/checkpoints/resnet18/best_model.pth \
    --data_dir <dir with sentinel2_2018.png, sentinel2_2022.png, hansen_validation_mask.png>
```

`train.py` — completed 3 epochs in 4.4 min on CPU, best val acc 93.00% (epoch 1).
Checkpointing, `history.json`, TensorBoard logging and `training_summary.md` all
behave as documented.

`evaluate.py` — test accuracy 0.8850, precision 0.9156, recall 0.8850,
F1 0.8769, 55.2 img/s on CPU. Loads both raw and `model_state_dict`-wrapped
checkpoints correctly.

`run_demo.py` — after the fix below, completed end to end on the shipped
Ariquemes images: 256 patches, 5 predicted deforested vs 16 in the reference
mask, IoU 0.3125, precision 1.0, recall 0.3125. All 9 figures/reports written.
The low recall is expected: the classifier was trained on synthetic patterns, so
its land-cover predictions on the demo tiles are largely meaningless.

## Defects found

1. **`run_demo.py` crashed with `NameError: name 'Path' is not defined`**
   (line 175, `output_dir=Path(args.output_dir)`), `pathlib` was never imported.
   The crash happens at step 8 of 8, after all the compute. Fixed in this commit.
   As shipped, `run_demo.py` could never have completed a run.

2. **`run_demo.py` filenames do not match the shipped data.** It looks for
   `sentinel2_<year>.png` and `hansen_validation_mask.png`; `data/demo/` contains
   region-suffixed names (`sentinel2_ariquemes_2018.png`, …). Out of the box the
   script therefore falls through to `download_region.py`, which requires Earth
   Engine auth. Not fixed here — it needs a decision about whether the CLI should
   take a `--region` argument like `src/regions.py` already models.

3. **Hectare figures differ by 100x between the notebook and the CLI.**
   `13_Project_Demo.ipynb` uses `(64 * 10) ** 2` m² per patch (10 m Sentinel-2
   pixels) and reports 614.4 ha of loss; `validate_deforestation` in
   `src/change_detection.py` and `run_demo.py` use `patch_size ** 2 / 10000`,
   i.e. 1 m pixels, giving 15.97 ha for the same kind of scene. At Sentinel-2
   resolution the CLI's areas are 100x too small. The local variable is also
   named `pixel_area_ha` when it holds a patch area.

## Two things the README overstates

**The benchmark table does not match the committed metrics.** `README.md`
reports EfficientNet-B0 94.10%, GoogLeNet 90.10%, AlexNet 84.10%, LeNet-5
74.20%. The JSON files under `notebooks/reports/metrics/` — the actual run
output — say 86.63%, 87.00%, 83.26% and 64.41%. Only ResNet-18 (96.04%)
matches. `13_Project_Demo.ipynb` hardcodes the README's inflated values as
`defaults` and then overrides them from the JSON, which is why the notebook's
own printed table shows the correct, lower numbers.

**The validation is not against real Sentinel-2 or Hansen data.**
`src/regions.py::generate_region_demo_data` builds the region images by tiling
64x64 EuroSAT chips, and the shipped `data/demo/*.png` are exactly that: the
mean absolute pixel gradient at 64-pixel column boundaries is 30.1 versus 2.95
elsewhere, i.e. hard seams on a 16x16 grid. The "Hansen" mask is pure 0/255 and
covers exactly 6.25% of the image = 16 of 256 patches, perfectly block-aligned.

So the classifier is evaluated on chips drawn from its own training
distribution, on a grid that coincides exactly with its inference stride, against
a reference mask laid out on that same grid. The reported IoU of 0.975 is a
property of that construction, not evidence of deforestation-detection skill on
real imagery. The README and `report.md` describe this as Sentinel-2 composites
validated against Hansen Global Forest Change; that claim is not supported by
what the repository actually runs.
