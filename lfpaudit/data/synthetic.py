"""A deterministic fake dataset with a known answer.

Nothing here is neuroscience. The point is that every module -- chunking, splits, features,
metrics, the trainer -- can be exercised end to end in continuous integration without a single
byte of real ephys, and that a smoke run has a ground truth we can assert against.

Each region gets a distinct spectral signature on top of a 1/f background, loosely inspired by
real hippocampal physiology (ripples in CA1, strong theta in DG) so that band-power features are
genuinely informative and a broken pipeline is visible as chance-level accuracy. A per-dataset
gain and noise level stands in for a lab-to-lab acquisition shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfpaudit import REGIONS
from lfpaudit.config import BANDS
from lfpaudit.data.chunk import ChunkStore, ChunkWriter, chunk_signal, zscore_chunks

#: Which bands are boosted for each region, and by how much (multiplicative, on amplitude).
REGION_SIGNATURES: dict[str, dict[str, float]] = {
    "CA1": {"theta": 2.0, "ripple": 3.0},
    "CA2": {"theta": 1.2},
    "CA3": {"theta": 1.6, "gamma": 2.2},
    "DG": {"theta": 3.0, "gamma": 1.4},
    "VIS": {"alpha": 2.4, "beta": 2.0},
}

#: Fake acquisition differences between "labs", used to exercise the cross-lab split.
DATASET_PROFILES: dict[str, dict[str, float]] = {
    "synthA": {"gain": 1.0, "noise": 1.0, "drift_hz": 0.0},
    "synthB": {"gain": 0.6, "noise": 1.6, "drift_hz": 0.4},
}


@dataclass(frozen=True)
class SyntheticSpec:
    """Size of the fake dataset to generate."""

    datasets: tuple[str, ...] = ("synthA",)
    sessions_per_dataset: int = 3
    channels_per_session: int = 10
    chunks_per_channel: int = 8
    fs: float = 1250.0
    window_s: float = 3.0
    seed: int = 0


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    return np.fft.irfft(spectrum * scale, n=n)


def synth_channel(
    region: str,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    profile: dict[str, float] | None = None,
) -> np.ndarray:
    """Generate one channel of fake LFP carrying ``region``'s spectral signature."""
    if region not in REGION_SIGNATURES:
        raise ValueError(f"unknown region {region!r}")
    profile = profile or DATASET_PROFILES["synthA"]
    t = np.arange(n_samples) / fs

    signal = profile["noise"] * _pink_noise(n_samples, rng)
    for band, boost in REGION_SIGNATURES[region].items():
        low, high = BANDS[band]
        # A handful of jittered sinusoids per band is enough to raise that band's power without
        # producing a single sharp line that a classifier could latch onto trivially.
        for _ in range(4):
            freq = rng.uniform(low, high)
            phase = rng.uniform(0, 2 * np.pi)
            signal += boost * rng.uniform(0.6, 1.0) * np.sin(2 * np.pi * freq * t + phase)
    if profile["drift_hz"] > 0:
        signal += 3.0 * np.sin(2 * np.pi * profile["drift_hz"] * t + rng.uniform(0, 2 * np.pi))
    return profile["gain"] * signal


def build_synthetic_store(path: str, spec: SyntheticSpec | None = None) -> ChunkStore:
    """Write a complete fake :class:`~lfpaudit.data.chunk.ChunkStore` to ``path``.

    Channels are laid out in depth order and regions are assigned in blocks, so the
    depth-only baseline is informative within a session and meaningless across sessions with a
    different region ordering -- exactly the confound the real experiments test for.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    window_samples = int(round(spec.window_s * spec.fs))
    # Enough signal for the requested chunks, with a margin so chunk_signal's start offset works.
    n_samples = window_samples * (spec.chunks_per_channel + 1)

    writer = ChunkWriter(path, n_samples=window_samples, fs=spec.fs)
    for dataset in spec.datasets:
        profile = DATASET_PROFILES.get(dataset, DATASET_PROFILES["synthA"])
        for session_i in range(spec.sessions_per_dataset):
            session = f"{dataset}-s{session_i:02d}"
            # Rotate the region ordering per session so depth alone cannot generalise.
            order = np.roll(np.arange(len(REGIONS)), session_i)
            for channel in range(spec.channels_per_session):
                region = REGIONS[order[channel % len(REGIONS)]]
                raw = synth_channel(region, n_samples, spec.fs, rng, profile)
                chunks, t0 = chunk_signal(
                    raw,
                    fs=spec.fs,
                    window_s=spec.window_s,
                    start_s=0.0,
                    n_chunks=spec.chunks_per_channel,
                )
                writer.append(
                    zscore_chunks(chunks),
                    metadata={
                        "dataset": dataset,
                        "session": session,
                        "probe": f"{session}-p0",
                        "channel": channel,
                        "depth_um": float(channel * 20),
                        "acronym": region,
                        "region": region,
                        "group": session,
                    },
                    t0_s=t0,
                )
    return writer.close()
