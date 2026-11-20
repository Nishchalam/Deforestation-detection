import numpy as np

from deforest import labels

# lossyear codes: 0 none, 18 -> 2018, 19 -> 2019, 22 -> 2022, 23 -> 2023
TC = np.array([[80, 80, 80, 80, 80, 10]])
LY = np.array([[0, 18, 19, 22, 23, 0]])


def test_forest_mask_respects_threshold_and_loss_year():
    assert labels.forest_mask(TC, LY, 2018, 30).tolist() == [[True, False, True, True, True, False]]
    assert labels.forest_mask(TC, LY, 2022, 30).tolist() == [[True, False, False, False, True, False]]


def test_threshold_is_inclusive():
    assert labels.forest_mask(np.array([30]), np.array([0]), 2018, 30).tolist() == [True]
    assert labels.forest_mask(np.array([29]), np.array([0]), 2018, 30).tolist() == [False]


def test_loss_is_forest_y1_and_not_forest_y2():
    # only the 2019 and 2022 losses fall in (2018, 2022]
    assert labels.loss_mask(TC, LY, 2018, 2022, 30).tolist() == [[False, False, True, True, False, False]]


def test_valid_mask_needs_land_and_observations():
    dm = np.array([1, 1, 2, 1])
    n1 = np.array([5, 0, 5, 5])
    n2 = np.array([5, 5, 5, 0])
    assert labels.valid_mask(dm, n1, n2).tolist() == [True, False, False, False]
