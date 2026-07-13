"""
train.py

Main entry point for training models using the ExperimentRunner and configuration files.
"""

import argparse
from src.experiments.runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Train a Deforestation Detection model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    runner.run()

if __name__ == "__main__":
    main()
