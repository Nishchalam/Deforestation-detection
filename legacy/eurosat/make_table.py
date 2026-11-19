"""Print the EuroSAT benchmark table from metrics/*_metrics.json (the only source of the numbers)."""
import json
from pathlib import Path

NAMES = {"resnet18": "ResNet-18", "googlenet": "GoogLeNet", "efficientnetb0": "EfficientNet-B0",
         "alexnet": "AlexNet", "lenet": "LeNet-5"}

rows = []
for f in sorted(Path(__file__).parent.glob("metrics/*_metrics.json")):
    m = json.loads(f.read_text())
    rows.append((m["accuracy"], NAMES.get(f.stem.replace("_metrics", ""), f.stem), m))

print("| Model | Test acc. | Macro P | Macro R | Macro F1 | img/s |")
print("|---|---:|---:|---:|---:|---:|")
for acc, name, m in sorted(rows, reverse=True):
    print(f"| {name} | {100 * acc:.2f}% | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['images_per_second']:.0f} |")
