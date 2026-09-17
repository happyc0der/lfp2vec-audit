"""Feeding stored chunks to the model, and choosing which ones.

Two things happen here that affect results rather than plumbing.

Chunks are converted on demand rather than up front. A three-second window is 3 750 samples at
the stored rate and 48 000 at the rate the model expects, so materialising a training set in
model form would cost twenty times the memory of the cache for no added information. Conversion
follows the upstream order exactly: resample first, normalise second.

Training samples are drawn class-balanced. The corpus runs from 49 000 visual-cortex chunks to
5 900 CA3 chunks in one dataset, and a model trained on that mixture learns the prior at least as
much as the physiology. Balancing costs some data and buys a number that means what it appears to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from lfpaudit.features.embed import prepare_waveforms


class ChunkDataset(Dataset):
    """Chunks as model input, converted one item at a time.

    ``take`` is any callable mapping chunk ids to stored waveforms, so this works against a
    :class:`~lfpaudit.data.corpus.Corpus` or a single store without knowing which.
    """

    def __init__(
        self,
        take,
        chunk_ids: np.ndarray,
        labels: np.ndarray,
        fs: float,
        lowpass_hz: float | None = None,
    ) -> None:
        self.take = take
        self.chunk_ids = np.asarray(chunk_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.fs = float(fs)
        self.lowpass_hz = lowpass_hz
        if len(self.chunk_ids) != len(self.labels):
            raise ValueError(f"{len(self.chunk_ids)} chunks for {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def __getitem__(self, position: int):
        import torch

        waveform = prepare_waveforms(
            self.take(self.chunk_ids[position : position + 1]),
            fs=self.fs,
            lowpass_hz=self.lowpass_hz,
        )[0]
        return torch.from_numpy(waveform), int(self.labels[position])


def balanced_sample(
    labels: np.ndarray,
    chunk_ids: np.ndarray,
    max_per_class: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw up to ``max_per_class`` chunks of each class, without replacement.

    Classes with fewer chunks than the cap contribute everything they have, so the result is
    balanced where the data allows and honest about it where it does not. Returns positions into
    the arrays passed in, not chunk ids, so the caller keeps control of its own indexing.
    """
    labels = np.asarray(labels)
    if len(labels) != len(chunk_ids):
        raise ValueError(f"{len(labels)} labels for {len(chunk_ids)} chunks")
    rng = np.random.default_rng(seed)

    chosen: list[np.ndarray] = []
    for value in np.unique(labels):
        candidates = np.flatnonzero(labels == value)
        take = min(len(candidates), max_per_class)
        chosen.append(rng.choice(candidates, size=take, replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)


def class_counts(labels: np.ndarray, names: list[str]) -> dict[str, int]:
    """Readable class histogram for the manifest and the log."""
    counts = pd.Series(labels).value_counts().to_dict()
    return {names[int(k)]: int(v) for k, v in sorted(counts.items())}
