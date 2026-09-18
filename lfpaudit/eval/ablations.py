"""Test-time input transforms that ask what a model actually listens to.

Each transform is a pure function on stored chunks, applied before ``prepare_waveforms`` so that
every model sees the modified signal through exactly the pipeline it was trained on. Three of the
four ask a real question. The fourth is a control whose answer is known in advance, and which
exists so that a pipeline that has stopped normalising where it claims to would be caught here
rather than misread as a finding.

``band_stop`` removes one canonical band and measures what it cost. ``phase_randomise`` keeps
every chunk's power spectrum and scrambles its waveform: a model whose accuracy survives it was
only ever reading the spectrum, however many parameters it has. ``temporal_mask`` zeroes a
contiguous span. ``amplitude_scale`` multiplies by a constant that per-chunk normalisation must
cancel exactly, so its effect must be null to three decimals on every model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from lfpaudit.config import BANDS


def band_stop(chunks: np.ndarray, fs: float, band: str, order: int = 4) -> np.ndarray:
    """Remove one canonical band with a zero-phase Butterworth band-stop filter."""
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; expected one of {sorted(BANDS)}")
    low, high = BANDS[band]
    nyquist = 0.5 * fs
    high = min(high, nyquist * 0.999)
    if low >= high:
        raise ValueError(f"band {band!r} lies above Nyquist for fs {fs}")
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandstop", output="sos")
    return sosfiltfilt(sos, np.asarray(chunks, dtype=np.float64), axis=-1)


def phase_randomise(chunks: np.ndarray, seed: int = 0) -> np.ndarray:
    """Surrogate chunks with the same power spectrum and a random waveform.

    Keeps the magnitude of every Fourier coefficient and draws each phase uniformly, so band
    power, the full periodogram, and every spectral feature are unchanged while the time-domain
    signal is unrecognisable. The DC and Nyquist terms are kept real so the output is real.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(chunks, axis=-1)
    magnitude = np.abs(spectrum)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phases[..., 0] = 0.0
    if chunks.shape[-1] % 2 == 0:
        phases[..., -1] = 0.0
    surrogate = np.fft.irfft(magnitude * np.exp(1j * phases), n=chunks.shape[-1], axis=-1)
    return surrogate


def amplitude_scale(chunks: np.ndarray, factor: float) -> np.ndarray:
    """Multiply by a constant. A control: normalisation must cancel this exactly."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    return np.asarray(chunks, dtype=np.float64) * factor


def temporal_mask(chunks: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Zero one contiguous span covering ``fraction`` of each chunk, at a random offset."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")
    chunks = np.array(chunks, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    n = chunks.shape[-1]
    span = int(round(fraction * n))
    starts = rng.integers(0, n - span + 1, size=chunks.shape[0])
    for row, start in enumerate(starts):
        chunks[row, start : start + span] = 0.0
    return chunks


@dataclass(frozen=True)
class Ablation:
    """A named transform plus the question it answers, for tables and figures."""

    name: str
    apply: Callable[[np.ndarray, float], np.ndarray]
    question: str
    is_control: bool = False


def _registry() -> dict[str, Ablation]:
    entries = [
        Ablation("none", lambda x, fs: np.asarray(x, dtype=np.float64), "unmodified reference")
    ]
    for band in BANDS:
        entries.append(
            Ablation(
                f"stop_{band}",
                lambda x, fs, band=band: band_stop(x, fs, band),
                f"accuracy cost of removing the {band} band",
            )
        )
    entries += [
        Ablation(
            "phase_randomised",
            lambda x, fs: phase_randomise(x, seed=0),
            "does anything beyond the power spectrum matter",
        ),
        Ablation(
            "mask_25pct",
            lambda x, fs: temporal_mask(x, 0.25, seed=0),
            "reliance on a quarter of the window being present",
        ),
        Ablation(
            "amplitude_x0.5",
            lambda x, fs: amplitude_scale(x, 0.5),
            "control: must be null under per-chunk normalisation",
            is_control=True,
        ),
        Ablation(
            "amplitude_x2",
            lambda x, fs: amplitude_scale(x, 2.0),
            "control: must be null under per-chunk normalisation",
            is_control=True,
        ),
    ]
    return {a.name: a for a in entries}


#: Every ablation this stage runs, in the order tables report them.
ABLATIONS: dict[str, Ablation] = _registry()
