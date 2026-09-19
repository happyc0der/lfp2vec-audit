"""Tests for per-probe spectral whitening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from lfpaudit.features.whiten import (
    GroupWhitener,
    probe_mean_spectrum,
    whiten_by_group,
    whiten_chunks,
)

FS = 1250.0
N = 3750


def _coloured(rng, n_chunks, tilt, n=N):
    """1/f-like noise whose spectral slope is set by ``tilt``: two labs with different filters."""
    white = rng.standard_normal((n_chunks, n))
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1 / FS)
    spectrum[:, 1:] /= np.maximum(freqs[1:], 1.0) ** tilt
    return np.fft.irfft(spectrum, n=n, axis=-1)


def test_whitened_probe_has_a_flat_mean_spectrum():
    rng = np.random.default_rng(0)
    chunks = _coloured(rng, 60, tilt=0.8)
    _, reference = probe_mean_spectrum(chunks, FS)
    white = whiten_chunks(chunks, reference, FS)
    freqs, power = welch(white, fs=FS, nperseg=512, axis=-1)
    mean = power.mean(axis=0)
    band = (freqs > 5) & (freqs < 500)
    # Flat to within a factor of two across the band, where the original spanned decades.
    assert mean[band].max() / mean[band].min() < 2.5
    _, original = welch(chunks, fs=FS, nperseg=512, axis=-1)
    assert original.mean(axis=0)[band].max() / original.mean(axis=0)[band].min() > 50


def test_whitening_preserves_between_channel_differences():
    """A channel with extra theta must keep it after whitening, relative to its probe."""
    rng = np.random.default_rng(1)
    base = _coloured(rng, 40, tilt=0.8)
    t = np.arange(N) / FS
    theta = 3.0 * np.sin(2 * np.pi * 7 * t)
    chunks = base.copy()
    chunks[:20] += theta  # first half of the probe carries theta, second half does not
    _, reference = probe_mean_spectrum(chunks, FS)
    white = whiten_chunks(chunks, reference, FS)

    freqs, power = welch(white, fs=FS, nperseg=512, axis=-1)
    theta_bin = np.argmin(np.abs(freqs - 7))
    with_theta = power[:20, theta_bin].mean()
    without = power[20:, theta_bin].mean()
    assert with_theta > 3 * without


def test_two_labs_with_different_filters_become_indistinguishable_in_mean_spectrum():
    """The point of the lever: remove each recording's fingerprint, not the signal."""
    rng = np.random.default_rng(2)
    lab_a = _coloured(rng, 50, tilt=0.5)
    lab_b = _coloured(rng, 50, tilt=1.5)  # much steeper roll-off, like a band-passed pipeline
    groups = np.array(["a"] * 50 + ["b"] * 50)
    white = whiten_by_group(np.concatenate([lab_a, lab_b]), groups, FS)

    freqs, power = welch(white, fs=FS, nperseg=512, axis=-1)
    band = (freqs > 5) & (freqs < 500)
    # Compare spectral shape, not level: each chunk keeps its own energy by design, and the model
    # z-scores every chunk anyway, so level carries nothing while shape carries the fingerprint.
    mean_a = power[:50].mean(axis=0)[band]
    mean_b = power[50:].mean(axis=0)[band]
    shape_a = mean_a / mean_a.mean()
    shape_b = mean_b / mean_b.mean()
    ratio = shape_a / shape_b
    # Before whitening the ratio spans orders of magnitude; after, it is near one throughout.
    assert ratio.max() / ratio.min() < 3


def test_whitening_a_matching_chunk_preserves_scale():
    rng = np.random.default_rng(3)
    chunks = _coloured(rng, 30, tilt=0.8)
    _, reference = probe_mean_spectrum(chunks, FS)
    white = whiten_chunks(chunks, reference, FS)
    # Normalised gain means overall variance is of the same order, not blown up or crushed.
    assert 0.2 < white.std() / chunks.std() < 5


def test_output_is_real_and_same_length():
    rng = np.random.default_rng(4)
    chunks = rng.standard_normal((5, N))
    _, reference = probe_mean_spectrum(chunks, FS)
    white = whiten_chunks(chunks, reference, FS)
    assert white.shape == chunks.shape
    assert np.isrealobj(white)
    assert np.isfinite(white).all()


def test_floor_prevents_blow_up_on_empty_bins():
    rng = np.random.default_rng(5)
    chunks = rng.standard_normal((10, N))
    reference = np.zeros(257)
    reference[:10] = 1.0  # power only at the lowest bins; everything else "empty"
    white = whiten_chunks(chunks, reference, FS)
    assert np.isfinite(white).all()
    assert white.std() < 100 * chunks.std()


def test_group_whitener_fits_references_and_applies_by_group():
    rng = np.random.default_rng(6)
    table = np.concatenate([_coloured(rng, 30, 0.5), _coloured(rng, 30, 1.5)])
    index = pd.DataFrame({"chunk_id": np.arange(60), "group": ["a"] * 30 + ["b"] * 30})
    whitener = GroupWhitener(FS).fit(lambda ids: table[np.asarray(ids)], index, per_group=20)
    assert set(whitener.reference) == {"a", "b"}
    out = whitener.apply(table[[0, 45]], np.array(["a", "b"]))
    assert out.shape == (2, N)
    with pytest.raises(KeyError, match="no whitening reference"):
        whitener.apply(table[:1], np.array(["c"]))


def test_whiten_by_group_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="group labels"):
        whiten_by_group(np.zeros((4, 100)), np.array(["a", "b"]), FS)
