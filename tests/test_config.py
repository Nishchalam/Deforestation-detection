from deforest.config import load_config


def test_overrides_parse_yaml_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\nyears: [2018, 2022]\n")
    cfg = load_config(p, ["a.b=0.5", "years=[2019, 2023]", "new.key=null"])
    assert cfg.a.b == 0.5 and cfg.years == [2019, 2023] and cfg.new.key is None


def test_repo_configs_load():
    cfg = load_config("configs/ariquemes.yaml")
    ucfg = load_config("configs/unet.yaml")
    assert len(cfg.aoi.bbox) == 4 and cfg.labels.treecover_threshold == 30
    assert ucfg.unet.in_channels == len(cfg.sentinel2.bands)
