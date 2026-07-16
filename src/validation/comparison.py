"""
comparison.py

Validation comparison helpers.
"""

from typing import List

def compare_predictions(gt: List[str], pred: List[str]) -> List[bool]:
    """Returns a list of boolean match flags."""
    return [g == p for g, p in zip(gt, pred)]
