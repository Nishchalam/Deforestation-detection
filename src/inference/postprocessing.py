"""
postprocessing.py

Implements spatial postprocessing (e.g. majority filter) to smooth classification maps.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter

def majority_filter_grid(grid_classes: List[str], nx: int, ny: int, kernel_size: int = 3) -> List[str]:
    """
    Applies a spatial majority filter on a 2D grid layout of predicted class names.
    
    Parameters
    ----------
    grid_classes : list of str
        Flat list of predicted classes in row-major order.
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    kernel_size : int
        Odd integer size of the local neighborhood window.
        
    Returns
    -------
    list of str
        The smoothed class list.
    """
    if len(grid_classes) != nx * ny:
        raise ValueError("grid_classes size must match grid dimensions nx * ny.")
        
    grid = np.array(grid_classes).reshape((ny, nx))
    smoothed_grid = grid.copy()
    
    offset = kernel_size // 2
    
    for r in range(ny):
        for c in range(nx):
            # Extract neighborhood window boundary
            r_start = max(0, r - offset)
            r_end = min(ny, r + offset + 1)
            c_start = max(0, c - offset)
            c_end = min(nx, c + offset + 1)
            
            neighborhood = grid[r_start:r_end, c_start:c_end].flatten()
            
            # Find majority class
            most_common = Counter(neighborhood).most_common(1)[0][0]
            smoothed_grid[r, c] = most_common
            
    return smoothed_grid.flatten().tolist()
