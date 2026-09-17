import numpy as np

from lfpaudit import REGIONS
from lfpaudit.data.synthetic import REGION_SIGNATURES, SyntheticSpec, build_synthetic_store


def test_store_shape_matches_spec(tmp_path):
    spec = SyntheticSpec(sessions_per_dataset=2, channels_per_session=5, chunks_per_channel=4)
    store = build_synthetic_store(str(tmp_path / "s"), spec)
    expected = 1 * 2 * 5 * 4
    assert len(store.index) == expected
    assert store.array.shape == (expected, int(spec.window_s * spec.fs))


def test_generation_is_deterministic(tmp_path):
    a = build_synthetic_store(str(tmp_path / "a"), SyntheticSpec(seed=5))
    b = build_synthetic_store(str(tmp_path / "b"), SyntheticSpec(seed=5))
    np.testing.assert_array_equal(np.asarray(a.array), np.asarray(b.array))


def test_different_seeds_differ(tmp_path):
    a = build_synthetic_store(str(tmp_path / "a"), SyntheticSpec(seed=1))
    b = build_synthetic_store(str(tmp_path / "b"), SyntheticSpec(seed=2))
    assert not np.array_equal(np.asarray(a.array), np.asarray(b.array))


def test_every_region_is_represented(synthetic_store):
    assert set(synthetic_store.index["region"]) == set(REGIONS)
    assert set(REGION_SIGNATURES) == set(REGIONS)


def test_two_datasets_are_present(synthetic_store):
    assert set(synthetic_store.index["dataset"]) == {"synthA", "synthB"}


def test_chunks_are_normalised(synthetic_store):
    sample = synthetic_store.take(range(20))
    np.testing.assert_allclose(sample.mean(axis=1), 0.0, atol=1e-2)
    np.testing.assert_allclose(sample.std(axis=1), 1.0, atol=1e-2)
