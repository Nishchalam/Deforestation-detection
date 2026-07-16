import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def aggregate_experiments(experiments_dir: Path) -> List[Dict[str, Any]]:
    """
    Scans the experiments directory and gathers configurations, histories, and metadata.
    """
    experiments_data = []
    
    if not experiments_dir.exists():
        return experiments_data
        
    for exp_folder in sorted(experiments_dir.iterdir()):
        if not exp_folder.is_dir():
            continue
            
        config_path = exp_folder / "config.yaml"
        metadata_path = exp_folder / "metadata.json"
        history_path = exp_folder / "history.csv"
        best_model_path = exp_folder / "best_model.pth"
        
        if not config_path.exists():
            continue
            
        # 1. Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            
        model_name = config.get("model", {}).get("name", exp_folder.name)
        
        # Initialize record
        record = {
            "Experiment": exp_folder.name,
            "Model": model_name,
            "Params": 0,
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Train Time": "0s",
            "Train Time (s)": 0.0,
            "Infer Time": "0.0ms",
            "Infer Time (s)": 0.0,
            "Images/Sec": 0.0,
            "Model Size (MB)": 0.0,
            "Epochs": config.get("training", {}).get("epochs", 0),
        }
        
        # 2. Load model weight size
        if best_model_path.exists():
            record["Model Size (MB)"] = round(best_model_path.stat().st_size / (1024 * 1024), 2)
            
        # 3. Load tracker metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            record["Train Time (s)"] = metadata.get("training_time", 0.0)
            record["Train Time"] = f"{record['Train Time (s)']:.2f}s"
            
        # 4. Load history metrics
        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
                if not history_df.empty:
                    # Retrieve best validation performance row
                    best_row = history_df.loc[history_df["val_accuracy"].idxmax()]
                    record["Accuracy"] = round(float(best_row.get("val_accuracy", 0.0)), 4)
                    record["Precision"] = round(float(best_row.get("val_precision", 0.0)), 4)
                    record["Recall"] = round(float(best_row.get("val_recall", 0.0)), 4)
                    record["F1"] = round(float(best_row.get("val_f1", 0.0)), 4)
            except Exception as e:
                print(f"Error reading history for {exp_folder.name}: {e}")
                
        # 5. Fallback parameter counting using PyTorch model loading
        # Let's count parameters using the registry
        try:
            from src.experiments.registry import create_model
            model_cfg = config.get("model", {})
            model = create_model(model_cfg.get("name"), in_channels=model_cfg.get("in_channels", 3), num_classes=model_cfg.get("num_classes", 10))
            record["Params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            pass
            
        experiments_data.append(record)
        
    return experiments_data

def generate_comparison_reports(experiments_data: List[Dict[str, Any]], output_dir: Path):
    """
    Saves aggregated experiment results into Markdown, CSV, and JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not experiments_data:
        print("No experiments found to compare.")
        return
        
    df = pd.DataFrame(experiments_data)
    
    # Save CSV
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison report saved to CSV: {csv_path}")
    
    # Save JSON
    summary_path = output_dir / "summary.json"
    summary_data = {
        "experiments": experiments_data,
        "best_accuracy_model": df.loc[df["Accuracy"].idxmax()]["Model"] if not df.empty else None,
        "lightest_model": df.loc[df["Model Size (MB)"].idxmin()]["Model"] if not df.empty else None
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary JSON saved to: {summary_path}")
    
    # Generate Markdown Table
    md_path = output_dir / "comparison.md"
    md_lines = [
        "# Model Performance Comparison Report\n",
        "This report compiles evaluation and latency metrics across all trained architectures.\n",
        "## Performance Table\n",
        "| Model | Params | Accuracy | Precision | Recall | F1 | Train Time | Model Size (MB) |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    ]
    
    for rec in experiments_data:
        md_lines.append(
            f"| {rec['Model']} | {rec['Params']:,} | {rec['Accuracy']:.4f} | {rec['Precision']:.4f} | "
            f"{rec['Recall']:.4f} | {rec['F1']:.4f} | {rec['Train Time']} | {rec['Model Size (MB)']} MB |\n"
        )
        
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"Comparison report saved to MD: {md_path}")

def main():
    repo_root = Path(__file__).resolve().parents[2]
    experiments_dir = repo_root / "outputs/experiments"
    output_dir = repo_root / "reports/comparison"
    
    print("Aggregating experiment metrics...")
    experiments_data = aggregate_experiments(experiments_dir)
    generate_comparison_reports(experiments_data, output_dir)

if __name__ == "__main__":
    main()
