"""
report.py

Exports validation results (Markdown reports, CSV tables, and JSON files).
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def export_validation_reports(
    metrics: Dict[str, Any],
    stats: Dict[str, Any],
    classes: List[str],
    output_dir: Path
):
    """
    Saves validation metrics and statistics to Markdown, CSV, and JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 2. Save Statistics JSON
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    # 3. Save Summary CSV
    summary_data = {
        "Metric": list(metrics.keys()),
        "Value": [str(v) for v in metrics.values()]
    }
    pd.DataFrame(summary_data).to_csv(output_dir / "summary.csv", index=False)
    
    # 4. Save Classification Report CSV if present
    if "classification_report" in metrics:
        try:
            report_df = pd.DataFrame(metrics["classification_report"]).transpose()
            report_df.to_csv(output_dir / "classification_report.csv")
        except Exception:
            pass
            
    # 5. Generate validation_report.md
    md_lines = [
        "# Deforestation Detection Validation and Performance Analysis\n\n",
        "This automated report summarizes the performance of the deforestation classification pipeline.\n\n",
        "## 📊 Quantitative Metrics\n\n",
        "| Metric | Value |\n",
        "| :--- | :---: |\n"
    ]
    
    for key, val in metrics.items():
        if key != "classification_report" and key != "confusion_matrix":
            md_lines.append(f"| {key.replace('_', ' ').title()} | {val} |\n")
            
    md_lines.append("\n## 🌲 Forest Cover Changes\n\n")
    if "area_before_ha" in stats:
        area_before = stats["area_before_ha"].get("Forest", 0.0)
        area_after = stats["area_after_ha"].get("Forest", 0.0)
        diff = stats["difference_ha"].get("Forest", 0.0)
        
        md_lines.extend([
            f"* **Initial Forest Area (Year A)**: {area_before:.2f} ha\n",
            f"* **Final Forest Area (Year B)**:   {area_after:.2f} ha\n",
            f"* **Net Forest Cover Change**:      {diff:.2f} ha\n"
        ])
        
    with open(output_dir / "validation_report.md", "w") as f:
        f.writelines(md_lines)
        
    print(f"Validation reports exported to {output_dir}")
