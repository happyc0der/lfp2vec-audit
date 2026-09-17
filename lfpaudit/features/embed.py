"""Frozen wav2vec2 embeddings: what the audio prior gives before any LFP training.

LFP2Vec starts from ``facebook/wav2vec2-base``, a model trained on speech, and the paper's own
ablation reports that this initialisation is worth a great deal: audio initialisation with about
six thousand LFP trials matched random initialisation with over four hundred thousand. That
raises a question the paper does not ask. If the audio prior carries most of the benefit, how much
of the final performance is available with no LFP training at all?

This module answers it by running the untouched audio checkpoint forward, mean-pooling its last
hidden state, and handing the result to a linear classifier. Nothing is fine-tuned. Whatever that
scores is the part of the method attributable to the pretrained representation rather than to
anything learned from brain data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from scipy.signal import butter, sosfiltfilt

from lfpaudit.data.chunk import resample_to

#: The audio checkpoint LFP2Vec adapts, and the rate it expects.
DEFAULT_MODEL = "facebook/wav2vec2-base"
AUDIO_FS = 16000


def _batches(total: int, size: int) -> Iterator[slice]:
    for start in range(0, total, size):
        yield slice(start, min(start + size, total))


def prepare_waveforms(chunks: np.ndarray, fs: float, lowpass_hz: float | None = None) -> np.ndarray:
    """Turn stored chunks into model input: optionally band-limited, resampled, re-normalised.

    The upstream pipeline resamples to 16 kHz and only then normalises each chunk, so that order
    is kept here. ``lowpass_hz`` supports the harmonised-band variant, where both datasets are
    restricted below the corner where their preprocessing diverges.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    if lowpass_hz is not None:
        if lowpass_hz >= 0.5 * fs:
            raise ValueError(f"lowpass_hz {lowpass_hz} is at or above Nyquist for fs {fs}")
        sos = butter(4, lowpass_hz / (0.5 * fs), btype="low", output="sos")
        chunks = sosfiltfilt(sos, chunks, axis=-1)

    upsampled = resample_to(chunks, fs=fs, target_fs=AUDIO_FS)
    mean = upsampled.mean(axis=-1, keepdims=True)
    std = upsampled.std(axis=-1, keepdims=True)
    return ((upsampled - mean) / (std + 1e-10)).astype(np.float32)


def load_encoder(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the audio checkpoint in evaluation mode, with gradients disabled."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model


def embed_chunks(
    take: Callable[[np.ndarray], np.ndarray],
    chunk_ids: np.ndarray,
    fs: float,
    model=None,
    device: str = "cpu",
    batch_size: int = 32,
    lowpass_hz: float | None = None,
    model_name: str = DEFAULT_MODEL,
    progress_every: int = 20,
) -> np.ndarray:
    """Mean-pooled hidden states for a set of chunks, as ``(n_chunks, hidden)``.

    ``take`` is any callable mapping chunk ids to waveforms, so this works against either a
    :class:`~lfpaudit.data.chunk.ChunkStore` or a :class:`~lfpaudit.data.corpus.Corpus` without
    knowing which. Chunks are fetched per batch rather than all at once: at 16 kHz a batch of 32
    is already six million floats, and the whole corpus would not fit in memory.
    """
    import torch

    chunk_ids = np.asarray(chunk_ids)
    model = model or load_encoder(model_name, device)
    outputs: list[np.ndarray] = []

    total_batches = (len(chunk_ids) + batch_size - 1) // batch_size
    for number, window in enumerate(_batches(len(chunk_ids), batch_size)):
        waveforms = prepare_waveforms(take(chunk_ids[window]), fs=fs, lowpass_hz=lowpass_hz)
        tensor = torch.from_numpy(waveforms).to(device)
        hidden = model(tensor).last_hidden_state
        outputs.append(hidden.mean(dim=1).float().cpu().numpy())
        if progress_every and number % progress_every == 0:
            print(f"  embedded batch {number + 1}/{total_batches}", flush=True)

    return np.concatenate(outputs, axis=0).astype(np.float32)
