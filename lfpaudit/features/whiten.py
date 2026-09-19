"""Per-probe spectral whitening: remove each recording's spectral fingerprint before the model.

Stage 4 established that the model reads spectral power and little else, and that the two labs'
spectra diverge by orders of magnitude above 100 Hz because their pipelines filter differently.
Stage 5's centering lever then showed the cross-lab failure is not a displacement in embedding
space: once each probe's mean embedding is removed, the lab signature goes but region decoding
does not come back. The region-discriminating content itself is lab-specific.

Whitening acts one step earlier, on the input. Each probe's chunks are divided by that probe's
own mean power spectrum, so every probe presents a flat average spectrum to the model and only the
deviation of each channel from its probe's norm survives. That is the part a within-probe region
label can depend on, and it is the part a lab's filtering cannot touch, since filtering scales
every channel of a probe alike.

The transform is applied identically at training and test time, uses no labels, and has no
hyper-parameters beyond the smoothing that keeps a division by near-zero power from exploding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


def probe_mean_spectrum(
    chunks: np.ndarray, fs: float, nperseg: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch power spectrum over every chunk of one probe."""
    chunks = np.asarray(chunks, dtype=np.float64)
    freqs, power = welch(chunks, fs=fs, nperseg=min(nperseg, chunks.shape[-1]), axis=-1)
    return freqs, power.mean(axis=0)


def whiten_chunks(
    chunks: np.ndarray, reference_power: np.ndarray, fs: float, floor: float = 1e-9
) -> np.ndarray:
    """Divide each chunk's spectrum by the square root of a reference power spectrum.

    Works in the rFFT domain so the result is a real waveform of the original length. The
    reference is interpolated onto the FFT bins and floored at ``floor`` times its peak so bins
    with essentially no power do not amplify noise. The floor has to sit far below the peak: a
    1/f spectrum spans six or more decades, and a floor at a thousandth of the peak left everything
    above the first three unwhitened, which is exactly the region the two labs differ in.
    Each chunk is then rescaled to its original energy, so only the spectral shape changes.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    n = chunks.shape[-1]
    bins = np.fft.rfftfreq(n, d=1.0 / fs)
    ref_freqs = np.linspace(0, fs / 2, len(reference_power))
    gain = np.interp(bins, ref_freqs, np.asarray(reference_power, dtype=np.float64))

    # Floor relative to the reference's peak, not its median: a sparse reference has a median
    # of zero, which made the floor zero and the division produce NaN.
    peak = float(np.max(gain))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("reference power spectrum has no positive power")
    gain = np.maximum(gain, floor * peak)

    spectrum = np.fft.rfft(chunks, axis=-1) / np.sqrt(gain)
    white = np.fft.irfft(spectrum, n=n, axis=-1)

    # Whiten the shape, keep the energy. Dividing by any global constant cannot do this for a
    # coloured spectrum, since the bins holding the energy are exactly the ones scaled down
    # most; so each chunk is rescaled to its original root-mean-square instead.
    before = np.sqrt(np.mean(chunks**2, axis=-1, keepdims=True))
    after = np.sqrt(np.mean(white**2, axis=-1, keepdims=True))
    return white * (before / np.maximum(after, 1e-12))


def whiten_by_group(chunks: np.ndarray, groups: pd.Series | np.ndarray, fs: float) -> np.ndarray:
    """Whiten every chunk against the mean spectrum of the probe it came from."""
    chunks = np.asarray(chunks, dtype=np.float64)
    groups = np.asarray(groups)
    if len(chunks) != len(groups):
        raise ValueError(f"{len(chunks)} chunks for {len(groups)} group labels")
    out = np.empty_like(chunks)
    for group in pd.unique(groups):
        mask = groups == group
        _, reference = probe_mean_spectrum(chunks[mask], fs)
        out[mask] = whiten_chunks(chunks[mask], reference, fs)
    return out


class GroupWhitener:
    """Reference spectra computed once per probe, applied to chunks fetched later.

    A dataset that converts chunks one at a time cannot see a whole probe at once, so the
    references are fitted up front from a sample of each probe's chunks and looked up by group.
    """

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self.reference: dict = {}

    def fit(self, take, index: pd.DataFrame, per_group: int = 400, seed: int = 0) -> GroupWhitener:
        rng = np.random.default_rng(seed)
        for group, part in index.groupby("group"):
            chosen = rng.choice(part.index.to_numpy(), min(per_group, len(part)), replace=False)
            _, self.reference[str(group)] = probe_mean_spectrum(
                take(index.loc[chosen, "chunk_id"].to_numpy()), self.fs
            )
        return self

    def apply(self, chunks: np.ndarray, groups: np.ndarray) -> np.ndarray:
        chunks = np.asarray(chunks, dtype=np.float64)
        out = np.empty_like(chunks)
        for group in pd.unique(np.asarray(groups)):
            if str(group) not in self.reference:
                raise KeyError(f"no whitening reference fitted for group {group!r}")
            mask = np.asarray(groups) == group
            out[mask] = whiten_chunks(chunks[mask], self.reference[str(group)], self.fs)
        return out
