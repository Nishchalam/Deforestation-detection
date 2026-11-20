"""Tiling and spatially blocked train/val/test split.

Whole tiles go to one split. Test and val are each one contiguous block of tiles, so
test pixels are not surrounded by training pixels (limits spatial-autocorrelation
leakage; see Roberts et al. 2017, https://doi.org/10.1111/ecog.02881).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Tile:
    row: int  # tile index
    col: int
    size: int

    @property
    def window(self) -> tuple[slice, slice]:
        return (slice(self.row * self.size, (self.row + 1) * self.size),
                slice(self.col * self.size, (self.col + 1) * self.size))

    @property
    def id(self) -> str:
        return f"r{self.row:02d}c{self.col:02d}"


def tile_grid(height: int, width: int, size: int) -> tuple[int, int]:
    """Number of full tiles (rows, cols); the partial strip at bottom/right is dropped."""
    return height // size, width // size


def block_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _touches(a: set, b: set) -> bool:
    """True if any cell of a is in, or 8-neighbour of, a cell of b."""
    return any((r + dr, c + dc) in b for r, c in a for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _score(cells: set, forest: np.ndarray, loss: np.ndarray, valid: np.ndarray) -> float:
    """Relative distance of a block's forest and loss fractions from the AOI-wide ones."""
    idx = tuple(np.array(sorted(cells)).T)
    f_all = forest.sum() / valid.sum()
    l_all = loss.sum() / valid.sum()
    v = valid[idx].sum()
    if v == 0:
        return np.inf
    return abs(forest[idx].sum() / v - f_all) / f_all + abs(loss[idx].sum() / v - l_all) / l_all


def choose_split(forest: np.ndarray, loss: np.ndarray, valid: np.ndarray,
                 test_block: tuple[int, int], val_block: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    """Deterministic blocked split over a (rows, cols) grid of per-tile pixel counts.

    forest/loss/valid: (n_rows, n_cols) arrays of per-tile pixel counts.
    test: the test_block-shaped block whose forest and loss fractions are closest to the
    AOI-wide fractions. val: same rule over val_block (either orientation), restricted to
    blocks that neither overlap nor touch the test block. Everything else is train.
    Uses labels only, never model outputs, and is fixed before any training."""
    n_rows, n_cols = forest.shape

    def candidates(h: int, w: int):
        for r0, c0 in product(range(n_rows - h + 1), range(n_cols - w + 1)):
            yield block_cells(r0, c0, h, w)

    test = min(candidates(*test_block), key=lambda b: (_score(b, forest, loss, valid), sorted(b)))
    shapes = {val_block, val_block[::-1]}
    val_cands = [b for s in sorted(shapes) for b in candidates(*s) if not _touches(b, test)]
    if not val_cands:
        raise ValueError("No val block fits without touching the test block")
    val = min(val_cands, key=lambda b: (_score(b, forest, loss, valid), sorted(b)))
    train = set(product(range(n_rows), range(n_cols))) - test - val
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def per_tile_counts(mask: np.ndarray, size: int) -> np.ndarray:
    """Sum of a boolean mask inside each full tile -> (n_rows, n_cols)."""
    n_r, n_c = tile_grid(*mask.shape, size)
    m = mask[: n_r * size, : n_c * size].astype(np.int64)
    return m.reshape(n_r, size, n_c, size).sum(axis=(1, 3))
