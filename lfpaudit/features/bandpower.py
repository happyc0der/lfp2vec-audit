"""Band-limited power features.

These reproduce the feature definition of LFP-LOC (Perna et al., Frontiers in Neuroscience,
2026), the training-free method whose stated advantage over LFP2Vec is interpretability. Using
them as a supervised baseline turns that rhetorical contrast into a measurable one: if a six-
number summary of a chunk's spectrum classifies regions nearly as well as a 95M-parameter
transformer, the interpretability argument has teeth; if it does not, the transformer is doing
something a power spectrum cannot.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from lfpaudit.config import BANDS

BAND_NAMES: tuple[str, ...] = tuple(BANDS)


def power_spectrum(
    chunks: np.ndarray, fs: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density for a batch of chunks.

    Returns ``(freqs, psd)`` where ``psd`` has shape ``(n_chunks, n_freqs)``.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.ndim == 1:
        chunks = chunks[None, :]
    if nperseg is None:
        # ~1 s segments give roughly 1 Hz resolution, enough to separate delta from theta.
        nperseg = min(chunks.shape[-1], int(round(fs)))
    freqs, psd = welch(chunks, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def band_power(
    chunks: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = True,
    log: bool = True,
    nperseg: int | None = None,
) -> np.ndarray:
    """Per-chunk power in each canonical band.

    With ``relative=True`` each band is expressed as a fraction of total in-band power, which
    removes overall amplitude; with ``log=True`` the result is log-transformed, which is what
    makes the features roughly linearly separable. Shape is ``(n_chunks, len(bands))``.
    """
    bands = bands or BANDS
    freqs, psd = power_spectrum(chunks, fs=fs, nperseg=nperseg)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    columns = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            # Band lies outside the Nyquist range of this sampling rate; emit zeros rather than
            # silently dropping a column, so feature dimensionality never depends on fs.
            columns.append(np.zeros(psd.shape[0]))
            continue
        columns.append(psd[..., mask].sum(axis=-1) * df)
    features = np.stack(columns, axis=-1)

    if relative:
        total = features.sum(axis=-1, keepdims=True)
        features = features / np.maximum(total, 1e-20)
    if log:
        features = np.log10(features + 1e-12)
    return features


def band_feature_names(bands: dict[str, tuple[float, float]] | None = None) -> list[str]:
    """Column names matching :func:`band_power`, for coefficient tables and attribution plots."""
    return [f"{name}_{low:g}-{high:g}Hz" for name, (low, high) in (bands or BANDS).items()]
