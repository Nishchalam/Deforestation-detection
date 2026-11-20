"""Fill the generated tables in README.md from reports/metrics/*.json.

    python scripts/readme_tables.py            # rewrite README.md in place
    python scripts/readme_tables.py --check    # exit 1 if README.md is out of date (CI)

Tables live between <!-- BEGIN:name --> and <!-- END:name --> markers; never edit them by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"


def load(name: str) -> dict:
    return json.loads((METRICS / name).read_text())


def data_table() -> str:
    m, s = load("data_manifest.json"), load("data_stats.json")
    g = m["grid"]
    years = sorted(m["sentinel2"]["scenes"])
    y1, y2 = years
    loss_key = next(k for k in s if k.startswith("loss_"))
    rows = [
        ("Sentinel-2 collection", f"`{m['sentinel2']['collection']}` + {m['sentinel2']['cloud_mask']} cloud mask"),
        ("Hansen GFC version", f"`{m['hansen_asset']}`"),
        ("Grid", f"{g['crs']}, {g['scale']} m, {g['width']} x {g['height']} px"),
    ]
    for y in years:
        sc = m["sentinel2"]["scenes"][y]
        n = s[f"n_obs_{y}_min_median"]
        rows.append((f"Scenes {y} (Jun-Sep)", f"{sc['n_images']} images / {sc['n_dates']} dates; "
                                              f"clear obs per pixel min {n[0]}, median {n[1]:g}"))
    rows += [
        (f"Forest fraction {y1} / {y2}", f"{100 * s[f'forest_frac_{y1}']:.1f}% / {100 * s[f'forest_frac_{y2}']:.1f}%"),
        (f"Hansen loss {int(y1) + 1}-{y2}", f"{s[loss_key]:,} px = {s[loss_key] * g['scale'] ** 2 / 1e4:,.0f} ha"),
    ]
    return "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def split_table() -> str:
    sp = load("split.json")
    keys = list(sp["all"])
    fk = [k for k in keys if k.startswith("forest_frac_")]
    lk = next(k for k in keys if k.startswith("loss_frac_"))
    head = "| Split | Tiles | " + " | ".join(f"Forest {k[-4:]}" for k in fk) + f" | Loss {lk[-9:].replace('_', '-')} |"
    lines = [head, "|---|---:|" + "---:|" * (len(fk) + 1)]
    for name in ("train", "val", "test", "all"):
        st = sp[name] if name == "all" else sp[name]["stats"]
        lines.append(f"| {name} | {st['n_tiles']} | " + " | ".join(f"{100 * st[k]:.1f}%" for k in fk)
                     + f" | {100 * st[lk]:.2f}% |")
    return "\n".join(lines)


def legacy_table() -> str:
    out = subprocess.run([sys.executable, str(ROOT / "legacy/eurosat/make_table.py")],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


NAMES = {"ndvi": "NDVI threshold", "random_forest": "Random forest (pixel)",
         "unet_imagenet": "U-Net R18, ImageNet init", "unet_scratch": "U-Net R18, random init", "hansen": "Hansen loss (the training labels)",
         "rf_change": "Random forest, direct (12 bands)", "unet_change": "U-Net R18, direct (12 bands)"}


def _ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def forest_table() -> str:
    r = load("forest_mask_test.json")
    lines = ["| Method | IoU | IoU 95% CI | F1 | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for m, v in r["results"].items():
        p = v["pooled"]
        lines.append(f"| {NAMES.get(m, m)} | {p['iou']:.3f} | {_ci(v['ci95']['iou'])} | {p['f1']:.3f} | "
                     f"{p['precision']:.3f} | {p['recall']:.3f} |")
    return "\n".join(lines)


def change_table() -> str:
    r, a = load("change_test.json"), load("area_test.json")
    lines = [f"| Method | Approach | MMU (px) | IoU | F1 | F1 95% CI | Precision | Recall | Loss area (ha) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, v in r["results"].items():
        p = v["pooled"]
        lines.append(f"| {NAMES.get(m, m)} | {v['approach']} | {v['mmu_px']} | {p['iou']:.3f} | {p['f1']:.3f} | {_ci(v['ci95']['f1'])} | "
                     f"{p['precision']:.3f} | {p['recall']:.3f} | {a[m + '_ha']:.1f} |")
    lines.append(f"| *Hansen reference* | | | | | | | | {a['hansen_ha']:.1f} |")
    if "prodes_ha" in a:
        lines.append(f"| *PRODES (independent)* | | | | | | | | {a['prodes_ha']:.1f} |")
    return "\n".join(lines)


def prodes_table() -> str:
    r = load("prodes_test.json")
    lines = ["| vs PRODES 2019-2022, test tiles unless noted | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for k, v in r.items():
        if isinstance(v, dict):
            base = k.replace("_vs_prodes", "").replace("_whole_aoi", "")
            name = NAMES.get(base, base) + (" — whole AOI" if k.endswith("_whole_aoi") else "")
            lines.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    return "\n".join(lines)


TABLES = {"data": data_table, "split": split_table, "forest": forest_table, "change": change_table,
          "prodes": prodes_table, "legacy": legacy_table}


def render(text: str) -> str:
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    path = Path(args.readme)
    old = path.read_text()
    new = render(old)
    if args.check:
        if new != old:
            sys.exit("README.md tables are out of date: run python scripts/readme_tables.py")
        print("README tables up to date")
    else:
        path.write_text(new)
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
