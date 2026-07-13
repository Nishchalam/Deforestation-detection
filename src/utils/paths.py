"""
paths.py

Centralized path management for the Deforestation Detection project.
"""

from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EUROSAT_DIR = RAW_DATA_DIR / "EuroSAT"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Ensure essential directories exist
for d in [
    DATA_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    OUTPUTS_DIR, 
    EXPERIMENTS_DIR,
    CHECKPOINTS_DIR, 
    LOGS_DIR, 
    FIGURES_DIR, 
    PREDICTIONS_DIR,
    CONFIGS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)
