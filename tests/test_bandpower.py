import numpy as np
import pytest
from scipy.signal import welch

from lfpaudit.config import BANDS
from lfpaudit.features.bandpower import band_feature_names, band_power, power_spectrum


def test_feature_shape_and_names():
    rng = np.random.default_rng(0)
    features = band_power(rng.normal(size=(7, 3750)), fs=1250.0)
    assert features.shape == (7, len(BANDS))
    assert len(band_feature_names()) == len(BANDS)
    assert band_feature_names()[0].startswith("delta_")


def test_relative_power_sums_to_one_before_log():
    rng = np.random.default_rng(1)
    features = band_power(rng.normal(size=(4, 3750)), fs=1250.0, relative=True, log=False)
    np.testing.assert_allclose(features.sum(axis=1), 1.0, atol=1e-9)


def test_pure_tone_lands_in_the_right_band():
    fs, duration = 1250.0, 3.0
    t = np.arange(int(fs * duration)) / fs
    for band, (low, high) in BANDS.items():
        if high > fs / 2:
            continue
        freq = (low + high) / 2
        features = band_power(np.sin(2 * np.pi * freq * t)[None, :], fs=fs, log=False)
        assert list(BANDS)[int(features.argmax())] == band


def test_matches_a_manual_welch_integration():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(1, 3750))
    fs = 1250.0
    freqs, psd = welch(signal, fs=fs, nperseg=int(fs), axis=-1)
    df = float(freqs[1] - freqs[0])

    manual = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        manual.append(psd[..., mask].sum(axis=-1) * df)
    manual_arr = np.stack(manual, axis=-1)
    manual_arr = manual_arr / manual_arr.sum(axis=-1, keepdims=True)

    computed = band_power(signal, fs=fs, relative=True, log=False)
    np.testing.assert_allclose(computed, manual_arr, rtol=1e-12, atol=1e-12)


def test_bands_above_nyquist_become_zero_not_missing():
    rng = np.random.default_rng(3)
    # At 300 Hz the ripple band (100-250 Hz) is partly above Nyquist but still present;
    # at 120 Hz it is entirely gone and must yield a zero column rather than vanish.
    features = band_power(rng.normal(size=(2, 360)), fs=120.0, relative=False, log=False)
    assert features.shape[1] == len(BANDS)
    assert features[:, list(BANDS).index("ripple")].sum() == pytest.approx(0.0)


def test_power_spectrum_returns_matching_lengths():
    freqs, psd = power_spectrum(np.zeros((3, 1250)), fs=1250.0)
    assert psd.shape[0] == 3
    assert psd.shape[-1] == len(freqs)


def test_planted_region_signatures_are_recoverable(synthetic_store):
    """DG is generated with the strongest theta boost; it should show it."""
    index = synthetic_store.index
    features = band_power(synthetic_store.take(index["chunk_id"].to_numpy()), fs=synthetic_store.fs)
    theta = features[:, list(BANDS).index("theta")]
    by_region = {
        r: theta[(index["region"] == r).to_numpy()].mean() for r in index["region"].unique()
    }
    assert max(by_region, key=by_region.get) == "DG"


def test_log_spectrum_is_amplitude_free_and_band_limited():
    from lfpaudit.features.bandpower import log_spectrum

    rng = np.random.default_rng(0)
    t = np.arange(3750) / 1250.0
    chunks = rng.normal(size=(4, 3750)) + 3 * np.sin(2 * np.pi * 8 * t)
    features = log_spectrum(chunks, fs=1250.0)
    assert features.shape == (4, 100)
    assert np.allclose(features, log_spectrum(chunks * 7.0, fs=1250.0), atol=1e-5)
    assert np.all(features.argmax(axis=1) == 7)  # the 8 Hz bin, counting from 1 Hz
    # Energy above the range cannot reach the features except through normalisation.
    loud = chunks + 50 * np.sin(2 * np.pi * 400 * t)
    assert np.allclose(features, log_spectrum(loud, fs=1250.0), atol=1e-3)
