"""Electrode position as a feature, which is to say as a suspect.

This module exists to attack the project's own result. Probes are lowered into the brain along
stereotyped trajectories, so depth correlates with anatomy before any signal is considered: on a
typical hippocampal insertion the deepest channels are in one structure and the shallowest in
another simply because that is the order the tissue comes in. A model given only raw position,
with the voltage discarded entirely, therefore sets a floor that any claim about decoding
physiology has to clear.

The interesting comparison is not the within-lab number but the gap between within-lab and
cross-lab. Position generalises only while trajectories do, so a geometry baseline that holds up
within a lab and collapses across labs is evidence that part of what looks like anatomy decoding
is really trajectory memorisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "depth_um",
    "lateral_um",
    "depth_fraction",
    "channel_index",
)


def geometry_features(index: pd.DataFrame) -> np.ndarray:
    """Position-only features for every row of a chunk index.

    Returns ``(n_chunks, 4)``: absolute depth along the shank, lateral offset within the shank,
    depth rescaled to the span of channels kept on that probe, and the channel number.

    ``depth_fraction`` is the one that could plausibly transfer between probes, since absolute
    depth depends on how far a given insertion was advanced while relative position within the
    recorded span is at least comparable. It is computed per group, so a probe that sampled a
    narrow depth range still spans the full zero to one.
    """
    missing = [c for c in ("depth_um", "channel", "group") if c not in index.columns]
    if missing:
        raise ValueError(f"chunk index is missing columns needed for geometry: {missing}")

    depth = index["depth_um"].to_numpy(dtype=np.float64)
    lateral = (
        index["lateral_um"].to_numpy(dtype=np.float64)
        if "lateral_um" in index.columns
        else np.zeros(len(index))
    )
    channel = index["channel"].to_numpy(dtype=np.float64)

    frame = pd.DataFrame({"group": index["group"].to_numpy(), "depth": depth})
    low = frame.groupby("group")["depth"].transform("min").to_numpy()
    high = frame.groupby("group")["depth"].transform("max").to_numpy()
    span = np.where(high > low, high - low, 1.0)
    fraction = (depth - low) / span

    return np.column_stack([depth, np.nan_to_num(lateral, nan=0.0), fraction, channel]).astype(
        np.float64
    )
