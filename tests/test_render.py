import io

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from app import render

CRS_UTM = CRS.from_epsg(32720)
TF = from_origin(500000, 8900000, 10, 10)


def _img(h=100, w=100):
    return np.random.default_rng(1).integers(500, 3000, (6, h, w)).astype(np.uint16)


def _open(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png))


def _wgs84_bounds(h=100, w=100):
    return transform_bounds(CRS_UTM, "EPSG:4326", 500000, 8900000 - 10 * h, 500000 + 10 * w, 8900000)


def test_mask_png_is_rgba_with_color():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    png, _ = render.mask_png(mask, TF, CRS_UTM, render.LAYER_COLORS["loss"], alpha=170)
    im = _open(png)
    assert im.mode == "RGBA"
    a = np.asarray(im)
    on = a[..., 3] == 170
    assert on.any()
    assert (a[on][:, :3] == (230, 30, 30)).all()
    assert set(np.unique(a[..., 3])) <= {0, 170}


def test_mask_all_false_is_fully_transparent():
    png, _ = render.mask_png(np.zeros((100, 100), bool), TF, CRS_UTM, (1, 2, 3))
    assert (np.asarray(_open(png))[..., 3] == 0).all()


def test_bounds_contain_source_wgs84_bounds():
    west, south, east, north = _wgs84_bounds()
    for png, b in (
        render.mask_png(np.ones((100, 100), bool), TF, CRS_UTM, (1, 2, 3)),
        render.rgb_png(_img(), np.ones((100, 100), bool), TF, CRS_UTM),
    ):
        (s, w), (n, e) = b
        assert s <= south + 1e-4 and w <= west + 1e-4 and n >= north - 1e-4 and e >= east - 1e-4
        assert south - s < 1e-3 and n - north < 1e-3 and west - w < 1e-3 and e - east < 1e-3


def test_rgb_nodata_is_transparent():
    rng = np.random.default_rng(0)
    img = rng.integers(500, 3000, (6, 100, 100)).astype(np.uint16)
    valid = np.ones((100, 100), bool)
    img[:, :, :30] = 0
    valid[:, :30] = False
    png, _ = render.rgb_png(img, valid, TF, CRS_UTM)
    im = _open(png)
    assert im.mode == "RGBA"
    a = np.asarray(im)
    assert (a[..., 3] == 0).any() and (a[..., 3] == 255).any()
    # left strip (nodata) is transparent; centre of the valid area is opaque
    h, w = a.shape[:2]
    assert (a[h // 2, : w // 10, 3] == 0).all()
    assert a[h // 2, int(w * 0.8), 3] == 255


def test_rgb_outside_footprint_transparent():
    # far from the UTM central meridian the grid is visibly rotated in Web Mercator,
    # so the warped rectangle has empty corners
    tf = from_origin(800000, 8900000, 10, 10)
    png, _ = render.rgb_png(_img(1000, 1000), np.ones((1000, 1000), bool), tf, CRS_UTM)
    a = np.asarray(_open(png))[..., 3]
    assert a[a.shape[0] // 2, a.shape[1] // 2] == 255
    assert any(a[r, c] == 0 for r in (0, -1) for c in (0, -1))
