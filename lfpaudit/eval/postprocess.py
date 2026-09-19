"""The paper's post-processing, as its code does it rather than as its text describes it.

The paper says predictions are smoothed over a temporal window with a class prior on
probabilities, then each channel takes the majority label of its five nearest neighbours. The
notebook that produced the published numbers does something more specific. It averages raw
*logits* over every test trial of a channel, multiplies by a hand-set class weight, and argmaxes,
so every chunk of a channel receives one label. It then replaces each channel's label with the
mode over itself and its two index-neighbours either side on the same shank.

Two consequences matter here. First, the temporal step is not a window at all; it is a collapse
to one prediction per channel, which is a large denoiser when a channel has a hundred chunks. And
second, the spatial step is a geometry prior: it assumes neighbouring contacts share a region,
which is the same assumption the electrode-position baseline makes explicit. Applying this
pipeline to every model alike, position included, is what lets the fixes table say whether a gain
came from the representation or from the prior.

The class weight `[1, 1, 5, 2, 1]` in their notebook is hand-set and its class ordering cannot be
determined from the repository, so the default here is the uniform prior the paper explicitly
permits. Anything else would be a knob tuned by looking at results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lfpaudit import REGIONS


@dataclass
class PostProcessed:
    """Per-chunk labels after each stage, plus the per-channel table they came from."""

    raw: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray
    channels: pd.DataFrame

    def stages(self) -> dict[str, np.ndarray]:
        return {"raw": self.raw, "temporal": self.temporal, "spatial": self.spatial}


def temporal_aggregate(
    logits: np.ndarray, channel_key: np.ndarray, prior: np.ndarray | None = None
) -> tuple[dict, np.ndarray]:
    """Average logits over every chunk of a channel and take one label for the channel.

    Their notebook averages logits, not probabilities, and the paper's "preserves softmax
    semantics" does not describe it. Logits are averaged here to match the code that produced the
    published numbers; the difference is noted in the deviations file.

    Returns the per-channel label lookup and the per-chunk labels after broadcasting it back.
    """
    logits = np.asarray(logits, dtype=np.float64)
    channel_key = np.asarray(channel_key)
    if len(logits) != len(channel_key):
        raise ValueError(f"{len(logits)} logit rows for {len(channel_key)} channel keys")
    prior = np.ones(logits.shape[1]) if prior is None else np.asarray(prior, dtype=np.float64)
    if prior.shape != (logits.shape[1],):
        raise ValueError(f"prior has shape {prior.shape}, expected ({logits.shape[1]},)")

    per_channel: dict = {}
    for key in pd.unique(channel_key):
        mean = logits[channel_key == key].mean(axis=0)
        per_channel[key] = int(np.argmax(mean * prior))

    broadcast = np.array([per_channel[k] for k in channel_key], dtype=np.int64)
    return per_channel, broadcast


def spatial_vote(
    channel_labels: dict,
    ordering: dict[str, list],
    window: int = 2,
) -> dict:
    """Replace each channel's label with the mode over itself and ``window`` neighbours each side.

    ``ordering`` maps each shank (here, each probe group) to its channels in physical order along
    the shank. Their notebook indexes by raw channel number within a 128-contact shank; for
    Neuropixels stores where only some contacts are kept, rank along the shank is the faithful
    analogue, since "five nearest neighbours" is a statement about physical adjacency.

    Ties resolve to the smallest label, which is what ``scipy.stats.mode`` does and what their
    notebook therefore did.
    """
    smoothed: dict = {}
    for _shank, keys in ordering.items():
        labels = [channel_labels[k] for k in keys]
        for position, key in enumerate(keys):
            lo, hi = max(0, position - window), min(len(keys), position + window + 1)
            neighbourhood = labels[lo:hi]
            values, counts = np.unique(neighbourhood, return_counts=True)
            smoothed[key] = int(values[np.argmax(counts)])
    return smoothed


def apply_paper_pipeline(
    frame: pd.DataFrame,
    logits: np.ndarray,
    prior: np.ndarray | None = None,
    window: int = 2,
) -> PostProcessed:
    """Run temporal then spatial post-processing over a prediction table.

    ``frame`` needs ``group``, ``channel`` and ``depth_um`` columns aligned with ``logits``; the
    channel key is (group, channel) and the shank ordering is by depth within group.
    """
    for column in ("group", "channel", "depth_um"):
        if column not in frame.columns:
            raise ValueError(f"prediction table is missing {column!r}")
    if len(frame) != len(logits):
        raise ValueError(f"{len(frame)} rows for {len(logits)} logit rows")

    keys = np.array(
        [f"{g}::{c}" for g, c in zip(frame["group"], frame["channel"], strict=True)],
        dtype=object,
    )
    raw = np.asarray(logits).argmax(axis=1).astype(np.int64)
    per_channel, temporal = temporal_aggregate(logits, keys, prior=prior)

    positions = (
        frame.assign(key=keys).drop_duplicates("key").sort_values(["group", "depth_um", "channel"])
    )
    ordering = {
        str(group): list(part["key"]) for group, part in positions.groupby("group", sort=False)
    }
    smoothed = spatial_vote(per_channel, ordering, window=window)
    spatial = np.array([smoothed[k] for k in keys], dtype=np.int64)

    channels = positions[["key", "group", "channel", "depth_um"]].copy()
    channels["temporal"] = channels["key"].map(per_channel)
    channels["spatial"] = channels["key"].map(smoothed)
    if "label" in frame.columns:
        truth = frame.assign(key=keys).groupby("key")["label"].first()
        channels["label"] = channels["key"].map(truth)

    return PostProcessed(raw=raw, temporal=temporal, spatial=spatial, channels=channels)


def channel_level_scores(channels: pd.DataFrame, stage: str) -> dict[str, float]:
    """Balanced and raw accuracy with one row per channel, the unit their pipeline reports at."""
    if "label" not in channels.columns:
        raise ValueError("channel table has no labels")
    truth = channels["label"].to_numpy(dtype=np.int64)
    predicted = channels[stage].to_numpy(dtype=np.int64)
    present = np.unique(truth)
    recalls = [float((predicted[truth == c] == c).mean()) for c in present]
    counts = np.bincount(truth, minlength=len(REGIONS))
    return {
        "n_channels": int(len(truth)),
        "raw_accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "chance": 1.0 / len(present),
        "majority": float(counts.max() / counts.sum()),
    }
