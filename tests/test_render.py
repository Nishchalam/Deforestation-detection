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


def test_orientation_and_bounds_order():
    # block in the north-west corner of the source: must land in the top-left of the PNG
    mask = np.zeros((100, 100), bool)
    mask[:30, :30] = True
    png, ((s, w), (n, e)) = render.mask_png(mask, TF, CRS_UTM, (1, 2, 3))
    assert s < n and w < e
    a = np.asarray(_open(png))[..., 3] > 0
    h, wd = a.shape
    assert a[: h // 2, : wd // 2].any()
    assert not a[h // 2 :, :].any() and not a[:, wd // 2 :].any()
    # and rgb: a bright north-east block lands top-right
    img = _img()
    valid = np.zeros((100, 100), bool)
    valid[:30, 70:] = True
    rgb_png, _ = render.rgb_png(img, valid, TF, CRS_UTM)
    b = np.asarray(_open(rgb_png))[..., 3] > 0
    h, wd = b.shape
    assert b[: h // 2, wd // 2 :].any()
    assert not b[h // 2 :, :].any() and not b[:, : wd // 2].any()


def test_rgb_png_is_capped_and_bounds_still_contain_source():
    h, w = 2000, 3000
    img = np.random.default_rng(2).integers(500, 3000, (6, h, w)).astype(np.uint16)
    png, ((s, west), (n, e)) = render.rgb_png(img, np.ones((h, w), bool), TF, CRS_UTM)
    im = _open(png)
    assert max(im.size) <= 1600 and max(im.size) > 1500
    src_w, src_s, src_e, src_n = transform_bounds(CRS_UTM, "EPSG:4326", 500000, 8900000 - 10 * h, 500000 + 10 * w, 8900000)
    assert s <= src_s + 1e-4 and west <= src_w + 1e-4 and n >= src_n - 1e-4 and e >= src_e - 1e-4
    assert src_s - s < 1e-2 and n - src_n < 1e-2
    assert np.asarray(im)[im.size[1] // 2, im.size[0] // 2, 3] == 255
    small, _ = render.rgb_png(_img(), np.ones((100, 100), bool), TF, CRS_UTM)  # small inputs are not resized
    assert max(_open(small).size) <= 100 + 2


def test_shared_stretch_keeps_a_real_brightness_change():
    """M2: independent 2-98% stretches hide a uniform brightening; a shared reference shows it."""
    before = _img()
    after = (before * 1.5).astype(np.uint16)
    valid = np.ones((100, 100), bool)

    def mean_rgb(img, **kw):
        png, _ = render.rgb_png(img, valid, TF, CRS_UTM, **kw)
        return np.asarray(_open(png))[..., :3].astype(float).mean()

    assert abs(mean_rgb(before) - mean_rgb(after)) < 2  # independent: the change disappears
    ref = render.stretch_ref([before, after], valid)
    assert mean_rgb(after, ref=ref) > mean_rgb(before, ref=ref) + 10  # shared: it is visible
    assert render.rgb_png(before, valid, TF, CRS_UTM)[0] == render.rgb_png(before, valid, TF, CRS_UTM, ref=None)[0]


def test_stretch_ref_uses_valid_pixels_only():
    img = _img()
    img[:, :, :50] = 60000  # huge values in an invalid strip must not set the stretch
    valid = np.ones((100, 100), bool)
    valid[:, :50] = False
    ref = render.stretch_ref([img], valid)
    assert ref.max() < 3000
