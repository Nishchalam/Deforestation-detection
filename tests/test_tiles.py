import numpy as np

from deforest import tiles


def test_tile_grid_drops_partial_strips():
    assert tiles.tile_grid(2213, 2190, 256) == (8, 8)


def test_per_tile_counts():
    m = np.zeros((4, 6), bool)
    m[0, 0] = m[3, 5] = m[2, 2] = True
    assert tiles.per_tile_counts(m, 2).tolist() == [[1, 0, 0], [0, 1, 1]]


def _split(seed=0):
    rng = np.random.default_rng(seed)
    valid = np.full((8, 8), 100)
    forest = rng.integers(0, 100, (8, 8))
    loss = rng.integers(0, 10, (8, 8))
    return tiles.choose_split(forest, loss, valid, (3, 3), (2, 4))


def test_split_partitions_grid_without_overlap():
    s = _split()
    cells = [set(s[k]) for k in ("train", "val", "test")]
    assert sum(map(len, cells)) == 64
    assert set.union(*cells) == {(r, c) for r in range(8) for c in range(8)}
    assert len(cells[1]) == 8 and len(cells[2]) == 9


def test_test_and_val_are_contiguous_blocks_that_do_not_touch():
    s = _split()
    for key, n in (("test", 9), ("val", 8)):
        rows = {r for r, _ in s[key]}
        cols = {c for _, c in s[key]}
        assert len(rows) * len(cols) == n  # a full rectangle
        assert max(rows) - min(rows) + 1 == len(rows) and max(cols) - min(cols) + 1 == len(cols)
    test = set(s["test"])
    for r, c in s["val"]:
        assert all((r + dr, c + dc) not in test for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def test_split_is_deterministic():
    assert _split(1) == _split(1)
