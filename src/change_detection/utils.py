"""
utils.py

Change detection utility helpers.
"""

import numpy as np
from typing import List

def print_transition_matrix(matrix: np.ndarray, classes: List[str]):
    """Prints a formatted text transition matrix to stdout."""
    header = f"{'From/To':<20}" + "".join([f"{c[:10]:>12}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(matrix):
        row_str = f"{classes[i]:<20}" + "".join([f"{val:>12}" for val in row])
        print(row_str)
