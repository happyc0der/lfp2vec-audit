"""Turning continuous LFP into the fixed-length chunks every model consumes.

The on-disk format is deliberately dull: one float16 memmap holding ``(n_chunks, n_samples)``
at the *native* 1250 Hz rate, plus a parquet index describing where each row came from.
Upsampling to 16 kHz for wav2vec2 happens in the dataset at load time, because storing 48 000
float32 samples per chunk would inflate the cache roughly twentyfold for no added information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

#: Columns every chunk index carries. ``group`` is the unit that splits must not straddle.
#: ``scale_mean_uv`` and ``scale_std_uv`` record what per-chunk normalisation removed, so the
#: original microvolt waveform is recoverable from the stored z-scored one. Absolute amplitude is
#: a plausible acquisition-specific cue, and an ablation that cannot restore it cannot test that.
INDEX_COLUMNS = (
    "chunk_id",
    "dataset",
    "session",
    "probe",
    "channel",
    "depth_um",
    "acronym",
    "region",
    "t0_s",
    "group",
    "scale_mean_uv",
    "scale_std_uv",
)

#: Columns carried when the source provides them, filled with NaN otherwise. Kept separate from
#: :data:`INDEX_COLUMNS` because a store is still valid without any of them.
OPTIONAL_COLUMNS = (
    "lateral_um",
    "ccf_ap_um",
    "ccf_dv_um",
    "ccf_lr_um",
)

_ARRAY_NAME = "chunks.f16"
_INDEX_NAME = "index.parquet"
_META_NAME = "store.json"


def chunk_signal(
    signal: np.ndarray,
    fs: float,
    window_s: float = 3.0,
    start_s: float = 200.0,
    n_chunks: int | None = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a 1-D signal into consecutive non-overlapping windows.

    Returns the ``(n, window_samples)`` array and the start time of each window in seconds.
    Windows that would run past the end of the signal are not emitted, so the caller always
    receives complete chunks and can see from the returned length how many were available.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    window_samples = int(round(window_s * fs))
    if window_samples <= 0:
        raise ValueError("window_s must be positive")
    first = int(round(start_s * fs))
    if first < 0:
        raise ValueError("start_s must be non-negative")

    available = (len(signal) - first) // window_samples
    available = max(available, 0)
    count = available if n_chunks is None else min(n_chunks, available)
    if count == 0:
        return np.empty((0, window_samples), dtype=np.float64), np.empty(0, dtype=np.float64)

    offsets = first + np.arange(count) * window_samples
    chunks = np.stack([signal[o : o + window_samples] for o in offsets])
    return chunks, offsets / fs


def zscore_chunks(chunks: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise each chunk to zero mean and unit variance.

    Per-chunk (rather than per-channel or per-session) normalisation is what the upstream code
    does. It also removes absolute amplitude as a cue, which matters for the amplitude-scaling
    ablation in :mod:`lfpaudit.eval.ablations`.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.ndim != 2:
        raise ValueError(f"expected a 2-D array of chunks, got shape {chunks.shape}")
    mean = chunks.mean(axis=1, keepdims=True)
    std = chunks.std(axis=1, keepdims=True)
    return (chunks - mean) / (std + eps)


def resample_to(chunks: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    """Fourier-resample chunks to ``target_fs``, as the upstream pipeline does before wav2vec2."""
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.ndim == 1:
        chunks = chunks[None, :]
    if float(fs) == float(target_fs):
        return chunks
    n_out = int(round(chunks.shape[-1] * target_fs / fs))
    return resample(chunks, n_out, axis=-1)


@dataclass(frozen=True)
class ChunkStore:
    """A read-only view over a chunk cache written by :class:`ChunkWriter`."""

    path: Path
    index: pd.DataFrame
    n_samples: int
    fs: float

    @classmethod
    def open(cls, path: str | Path) -> ChunkStore:
        path = Path(path)
        meta = json.loads((path / _META_NAME).read_text())
        index = pd.read_parquet(path / _INDEX_NAME)
        return cls(path=path, index=index, n_samples=int(meta["n_samples"]), fs=float(meta["fs"]))

    @property
    def array(self) -> np.memmap:
        """Memory-mapped ``(n_chunks, n_samples)`` float16 view of the waveforms."""
        return np.memmap(
            self.path / _ARRAY_NAME,
            dtype=np.float16,
            mode="r",
            shape=(len(self.index), self.n_samples),
        )

    def take(self, rows: np.ndarray | list[int], microvolts: bool = False) -> np.ndarray:
        """Load the given row positions as float32.

        With ``microvolts`` the stored z-scored waveform is rescaled back to its original
        amplitude using the per-chunk statistics in the index, which is what the amplitude
        ablation needs. Rows are positions in the index, not chunk ids.
        """
        rows = np.asarray(rows, dtype=np.int64)
        payload = np.asarray(self.array[rows], dtype=np.float32)
        if not microvolts:
            return payload
        frame = self.index.iloc[rows]
        std = frame["scale_std_uv"].to_numpy(dtype=np.float32)[:, None]
        mean = frame["scale_mean_uv"].to_numpy(dtype=np.float32)[:, None]
        return payload * std + mean


class ChunkWriter:
    """Append chunks to a cache directory, then finalise it into a :class:`ChunkStore`.

    Chunks are appended per channel so a long recording never has to be held in memory at once.
    """

    def __init__(self, path: str | Path, n_samples: int, fs: float) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.n_samples = int(n_samples)
        self.fs = float(fs)
        self._rows: list[dict] = []
        self._optional: set[str] = set()
        self._handle = open(self.path / _ARRAY_NAME, "wb")

    def append(
        self,
        chunks: np.ndarray,
        metadata: dict,
        t0_s: np.ndarray,
        normalise: bool = True,
        drop_flat: bool = True,
    ) -> int:
        """Write one channel's chunks and record one index row each.

        ``metadata`` supplies the non-time-varying columns (dataset, session, probe, channel,
        depth, acronym, region, group, and any of :data:`OPTIONAL_COLUMNS`). Chunks arrive in
        microvolts; with ``normalise`` they are z-scored before storage and the mean and standard
        deviation that were removed are recorded per row.

        ``drop_flat`` skips chunks with zero variance rather than storing NaNs or dividing by
        zero. Allen ships probe files that are entirely zeros while still advertising LFP data,
        and upstream silently discarded those as all-zero trials; here they are dropped at the
        boundary and counted. Returns the number of rows actually written.
        """
        chunks = np.asarray(chunks, dtype=np.float64)
        if chunks.ndim != 2 or chunks.shape[1] != self.n_samples:
            raise ValueError(f"expected chunks of shape (n, {self.n_samples}), got {chunks.shape}")
        if len(t0_s) != len(chunks):
            raise ValueError("t0_s must have one entry per chunk")
        if not np.isfinite(chunks).all():
            raise ValueError(f"non-finite values in chunks for {metadata}")

        mean = chunks.mean(axis=1)
        std = chunks.std(axis=1)
        keep = std > 0 if drop_flat else np.ones(len(chunks), dtype=bool)
        if not keep.any():
            return 0
        chunks, mean, std = chunks[keep], mean[keep], std[keep]
        t0_s = np.asarray(t0_s, dtype=np.float64)[keep]

        payload = (chunks - mean[:, None]) / std[:, None] if normalise else chunks
        self._handle.write(np.asarray(payload, dtype=np.float16).tobytes(order="C"))

        self._optional.update(k for k in OPTIONAL_COLUMNS if k in metadata)
        for start, chunk_mean, chunk_std in zip(t0_s, mean, std, strict=True):
            row = {
                "chunk_id": len(self._rows),
                "t0_s": float(start),
                "scale_mean_uv": float(chunk_mean) if normalise else 0.0,
                "scale_std_uv": float(chunk_std) if normalise else 1.0,
            }
            row.update(metadata)
            self._rows.append(row)
        return int(keep.sum())

    def close(self) -> ChunkStore:
        """Flush the array, write the index and metadata, and return the resulting store."""
        self._handle.close()
        index = pd.DataFrame(self._rows)
        missing = [c for c in INDEX_COLUMNS if c not in index.columns]
        if missing and len(index):
            raise ValueError(f"chunk index is missing columns: {missing}")
        columns = list(INDEX_COLUMNS) + sorted(self._optional)
        if len(index):
            index = index[columns]
        else:
            index = pd.DataFrame(columns=columns)
        index.to_parquet(self.path / _INDEX_NAME, index=False)
        (self.path / _META_NAME).write_text(
            json.dumps({"n_samples": self.n_samples, "fs": self.fs, "n_chunks": len(index)})
        )
        return ChunkStore.open(self.path)

    def __enter__(self) -> ChunkWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if not self._handle.closed:
            self.close()
