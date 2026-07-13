"""
zoo.py

Model zoo interface to instantiate models via the registry.
"""

from src.experiments.registry import create_model

__all__ = ["create_model"]
