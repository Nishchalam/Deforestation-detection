import numpy as np

from deforest import change


def test_loss_map_is_before_and_not_after():
    before = np.array([1, 1, 0, 0], bool)
    after = np.array([1, 0, 1, 0], bool)
    assert change.loss_map(before, after).tolist() == [False, True, False, False]


def test_remove_small_drops_blobs_below_mmu_with_8_connectivity():
    m = np.zeros((6, 6), bool)
    m[0, 0] = True                       # 1 px blob
    m[2, 2] = m[3, 3] = m[4, 4] = True   # 3 px diagonal blob (8-connected)
    out = change.remove_small(m, 2)
    assert not out[0, 0] and out[2, 2] and out[3, 3] and out[4, 4]
    assert (change.remove_small(m, 0) == m).all()


def test_error_map_codes():
    pred = np.array([1, 1, 0, 0], bool)
    true = np.array([1, 0, 1, 0], bool)
    assert change.error_map(pred, true).tolist() == [change.TP, change.FP, change.FN, change.TN]


def test_categorize_fp_priority_order():
    fp = np.ones((1, 4), bool)
    lossyear = np.array([[23, 15, 0, 0]])
    treecover = np.array([[80, 80, 10, 80]])
    forest18 = np.ones((1, 4), bool)
    cat = change.categorize_fp(fp, lossyear, treecover, 30, forest18, 2018, 2022)
    assert cat[0, 0] == "Hansen loss 2023+"
    assert cat[0, 1] == "Hansen loss <=2018"
    assert cat[0, 2] == "canopy2000 < threshold"
