"""Tests for the test-time input transforms.

Each transform makes a claim about what it does to a signal. These check the claim directly on
synthetic data with known spectral content, because an ablation that does not do what its name
says would produce a plausible accuracy drop and an entirely wrong conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfpaudit.config import BANDS
from lfpaudit.eval.ablations import (
    ABLATIONS,
    amplitude_scale,
    band_stop,
    phase_randomise,
    temporal_mask,
)
from lfpaudit.features.bandpower import band_power
from lfpaudit.features.embed import prepare_waveforms

FS = 1250.0
N = 3750


def _tones(freqs, n=4):
    t = np.arange(N) / FS
    return np.stack([sum(np.sin(2 * np.pi * f * t) for f in freqs) for _ in range(n)])


class TestBandStop:
    @pytest.mark.parametrize("band", [b for b, (_, hi) in BANDS.items() if hi < FS / 2])
    def test_removes_the_named_band(self, band):
        low, high = BANDS[band]
        centre = 0.5 * (low + high)
        signal = _tones([centre])
        before = band_power(signal, fs=FS, relative=False, log=False)
        after = band_power(band_stop(signal, FS, band), fs=FS, relative=False, log=False)
        index = list(BANDS).index(band)
        assert after[:, index].mean() < 0.05 * before[:, index].mean()

    def test_leaves_other_bands_alone(self):
        # A theta tone and a gamma tone; removing gamma must not touch theta.
        signal = _tones([6.0, 60.0])
        before = band_power(signal, fs=FS, relative=False, log=False)
        after = band_power(band_stop(signal, FS, "gamma"), fs=FS, relative=False, log=False)
        theta = list(BANDS).index("theta")
        assert after[:, theta].mean() > 0.9 * before[:, theta].mean()

    def test_unknown_band_is_rejected(self):
        with pytest.raises(ValueError, match="unknown band"):
            band_stop(_tones([10.0]), FS, "nonsense")

    def test_band_above_nyquist_is_rejected(self):
        with pytest.raises(ValueError, match="above Nyquist"):
            band_stop(_tones([10.0]), 100.0, "ripple")


class TestPhaseRandomise:
    def test_preserves_the_power_spectrum(self):
        """The whole point: same spectrum, different waveform."""
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(6, N))
        surrogate = phase_randomise(signal, seed=1)
        before = np.abs(np.fft.rfft(signal, axis=-1))
        after = np.abs(np.fft.rfft(surrogate, axis=-1))
        np.testing.assert_allclose(after, before, rtol=1e-8, atol=1e-8)

    def test_preserves_band_power(self):
        signal = _tones([6.0, 60.0])
        before = band_power(signal, fs=FS)
        after = band_power(phase_randomise(signal, seed=2), fs=FS)
        np.testing.assert_allclose(after, before, rtol=1e-6)

    def test_destroys_the_waveform(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(size=(8, N))
        surrogate = phase_randomise(signal, seed=4)
        for original, scrambled in zip(signal, surrogate, strict=True):
            assert abs(np.corrcoef(original, scrambled)[0, 1]) < 0.2

    def test_output_is_real_and_finite(self):
        surrogate = phase_randomise(np.random.default_rng(5).normal(size=(3, N)))
        assert np.isrealobj(surrogate)
        assert np.isfinite(surrogate).all()

    def test_is_deterministic_given_a_seed(self):
        signal = np.random.default_rng(6).normal(size=(3, N))
        np.testing.assert_allclose(phase_randomise(signal, seed=7), phase_randomise(signal, seed=7))


class TestAmplitudeScale:
    def test_is_cancelled_by_normalisation(self):
        """The control. Per-chunk normalisation must make this exactly null."""
        signal = np.random.default_rng(8).normal(loc=30.0, scale=12.0, size=(4, N))
        plain = prepare_waveforms(signal, fs=FS)
        scaled = prepare_waveforms(amplitude_scale(signal, 7.5), fs=FS)
        np.testing.assert_allclose(scaled, plain, atol=1e-5)

    def test_rejects_a_non_positive_factor(self):
        with pytest.raises(ValueError, match="must be positive"):
            amplitude_scale(np.zeros((2, 10)), 0.0)


class TestTemporalMask:
    def test_zeroes_the_right_amount(self):
        signal = np.ones((5, 1000))
        masked = temporal_mask(signal, 0.25, seed=0)
        assert (masked == 0).sum(axis=1).tolist() == [250] * 5

    def test_masked_span_is_contiguous(self):
        masked = temporal_mask(np.ones((1, 1000)), 0.2, seed=1)[0]
        zeros = np.flatnonzero(masked == 0)
        assert zeros.max() - zeros.min() + 1 == len(zeros)

    def test_leaves_the_rest_untouched(self):
        signal = np.random.default_rng(9).normal(size=(3, 1000))
        masked = temporal_mask(signal, 0.1, seed=2)
        kept = masked != 0
        np.testing.assert_allclose(masked[kept], signal[kept])

    def test_rejects_an_out_of_range_fraction(self):
        for bad in (0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="strictly between"):
                temporal_mask(np.ones((2, 100)), bad)


class TestRegistry:
    def test_covers_every_band_plus_the_controls(self):
        for band in BANDS:
            assert f"stop_{band}" in ABLATIONS
        assert {"none", "phase_randomised", "mask_25pct"} <= set(ABLATIONS)
        assert [a.name for a in ABLATIONS.values() if a.is_control] == [
            "amplitude_x0.5",
            "amplitude_x2",
        ]

    def test_none_is_the_identity(self):
        signal = np.random.default_rng(10).normal(size=(3, N))
        np.testing.assert_allclose(ABLATIONS["none"].apply(signal, FS), signal)

    def test_every_entry_is_callable_and_shape_preserving(self):
        signal = np.random.default_rng(11).normal(size=(4, N))
        for name, ablation in ABLATIONS.items():
            out = ablation.apply(signal, FS)
            assert out.shape == signal.shape, name
            assert np.isfinite(out).all(), name
