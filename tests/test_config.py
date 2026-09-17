import pytest
import yaml

from lfpaudit.config import BANDS, ExperimentConfig, dump_config, load_config, set_seed


def test_defaults_match_the_paper():
    config = ExperimentConfig()
    assert config.data.window_s == 3.0
    assert config.data.fs == 1250.0
    assert config.data.target_fs == 16000
    assert config.train.learning_rate == 3e-5
    assert config.train.warmup_ratio == 0.1
    assert config.train.epochs == 10


def test_yaml_roundtrip(tmp_path):
    config = ExperimentConfig(name="demo", notes="x")
    config.train.seed = 11
    path = tmp_path / "e.yaml"
    dump_config(config, path)
    assert load_config(path) == config


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {"learning_rate": 1e-4, "nonsense": 2}}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_bands_are_contiguous_and_ordered():
    edges = list(BANDS.values())
    assert all(low < high for low, high in edges)
    assert all(a[1] == b[0] for a, b in zip(edges[:-1], edges[1:], strict=True))


def test_set_seed_makes_numpy_and_torch_deterministic():
    import numpy as np
    import torch

    set_seed(4)
    a, ta = np.random.rand(3), torch.rand(3)
    set_seed(4)
    np.testing.assert_array_equal(np.random.rand(3), a)
    torch.testing.assert_close(torch.rand(3), ta)
