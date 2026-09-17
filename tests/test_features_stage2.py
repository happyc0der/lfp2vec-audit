"""Tests for the Stage 2 feature sets and their cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit.features.amplitude import amplitude_features
from lfpaudit.features.cache import FeatureCache, index_fingerprint
from lfpaudit.features.embed import prepare_waveforms
from lfpaudit.features.geometry import geometry_features


@pytest.fixture
def index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": np.arange(6),
            "group": ["a", "a", "a", "b", "b", "b"],
            "channel": [0, 5, 10, 2, 4, 6],
            "depth_um": [20.0, 120.0, 220.0, 40.0, 60.0, 80.0],
            "lateral_um": [11.0, 27.0, 43.0, 11.0, 27.0, 43.0],
            "region": ["CA1", "DG", "VIS", "CA1", "CA1", "DG"],
            "t0_s": [0.0, 3.0, 6.0, 0.0, 3.0, 6.0],
            "scale_std_uv": [100.0, 200.0, 50.0, 10.0, 1000.0, 120.0],
            "scale_mean_uv": [0.5, -0.5, 0.0, 1.0, -1.0, 0.0],
        }
    )


def test_geometry_shape_and_columns(index):
    features = geometry_features(index)
    assert features.shape == (6, 4)
    np.testing.assert_allclose(features[:, 0], index["depth_um"])
    np.testing.assert_allclose(features[:, 3], index["channel"])


def test_depth_fraction_is_normalised_within_each_probe(index):
    """Absolute depth cannot transfer between insertions; relative position at least might."""
    fraction = geometry_features(index)[:, 2]
    np.testing.assert_allclose(fraction[:3], [0.0, 0.5, 1.0])
    # Group b spans a much narrower depth range but still covers the full zero to one.
    np.testing.assert_allclose(fraction[3:], [0.0, 0.5, 1.0])


def test_geometry_tolerates_a_missing_lateral_column(index):
    features = geometry_features(index.drop(columns=["lateral_um"]))
    np.testing.assert_allclose(features[:, 1], 0.0)


def test_geometry_requires_depth(index):
    with pytest.raises(ValueError, match="missing columns"):
        geometry_features(index.drop(columns=["depth_um"]))


def test_amplitude_features_are_log_scaled(index):
    features = amplitude_features(index)
    assert features.shape == (6, 2)
    np.testing.assert_allclose(features[:, 0], np.log10(index["scale_std_uv"]))
    np.testing.assert_allclose(features[:, 1], index["scale_mean_uv"])


def test_amplitude_requires_the_stored_scale(index):
    with pytest.raises(ValueError, match="amplitude columns"):
        amplitude_features(index.drop(columns=["scale_std_uv"]))


def test_cache_round_trip(index, tmp_path):
    cache = FeatureCache(tmp_path, index)
    values = np.arange(12, dtype=float).reshape(6, 2)
    cache.save("demo", values)
    np.testing.assert_allclose(cache.load("demo"), values)


def test_cache_misses_when_absent(index, tmp_path):
    assert FeatureCache(tmp_path, index).load("nothing") is None


def test_cache_refuses_itself_when_the_store_changes(index, tmp_path):
    """A rebuilt store with stale features would make every later number quietly wrong."""
    cache = FeatureCache(tmp_path, index)
    cache.save("demo", np.zeros((6, 2)))

    changed = index.copy()
    changed.loc[0, "region"] = "CA3"
    assert FeatureCache(tmp_path, changed).load("demo") is None


def test_fingerprint_ignores_unrelated_columns(index):
    """Adding a column later should not invalidate every cached feature."""
    extended = index.copy()
    extended["some_new_column"] = 1
    assert index_fingerprint(index) == index_fingerprint(extended)


def test_get_or_build_calls_the_builder_once(index, tmp_path):
    cache = FeatureCache(tmp_path, index)
    calls = []

    def builder():
        calls.append(1)
        return np.ones((6, 3))

    cache.get_or_build("demo", builder)
    cache.get_or_build("demo", builder)
    assert len(calls) == 1


def test_cache_rejects_a_wrong_length_array(index, tmp_path):
    with pytest.raises(ValueError, match="rows for an index"):
        FeatureCache(tmp_path, index).save("demo", np.zeros((3, 2)))


def test_prepare_waveforms_resamples_and_normalises():
    rng = np.random.default_rng(0)
    chunks = rng.normal(loc=50.0, scale=20.0, size=(3, 3750))
    prepared = prepare_waveforms(chunks, fs=1250.0)
    assert prepared.shape == (3, 48000)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(prepared.std(axis=1), 1.0, atol=1e-3)


def test_prepare_waveforms_lowpass_removes_high_frequencies():
    fs, n = 1250.0, 3750
    t = np.arange(n) / fs
    # A 5 Hz component that must survive plus a 200 Hz one that must not.
    signal = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t))[None, :]

    filtered = prepare_waveforms(signal, fs=fs, lowpass_hz=100.0)
    spectrum = np.abs(np.fft.rfft(filtered[0]))
    freqs = np.fft.rfftfreq(filtered.shape[1], d=1 / 16000)
    high = spectrum[freqs > 150].max()
    low = spectrum[(freqs > 2) & (freqs < 20)].max()
    assert high < 0.01 * low


def test_prepare_waveforms_rejects_lowpass_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        prepare_waveforms(np.zeros((1, 100)), fs=1250.0, lowpass_hz=700.0)
