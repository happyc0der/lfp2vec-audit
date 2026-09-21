"""Electrode position as a feature, which is to say as a suspect, and a correction.

Probes are lowered along stereotyped trajectories, so depth correlates with anatomy before any
signal is considered. A model given only position, with the voltage discarded, sets a floor that
any claim about decoding physiology has to clear, and the gap between its within-lab and
cross-lab scores says how much of that floor is trajectory memorisation.

**The first version of this module leaked test labels, and results built on it were wrong.** It
included depth rescaled to the span of channels kept on each probe. Kept channels are the ones
whose histology label falls in the five target regions, so that span is set by the test probe's
own labels: a relative depth of zero means "the deepest in-scope structure" and one means "the
shallowest", which gives the anatomy away. That single feature scored 0.81 and 0.76 across labs
on its own, while absolute position scored 0.09 and 0.30, at or below chance. Nothing a user
would know before histology can produce it.

`position_features` is the honest baseline: where the contact sits along the shank, which the
probe's geometry gives for free. `labelled_span_fraction` is kept only so the leak stays
measurable and documented; it must never appear in a comparison as though it were available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = ("depth_um", "lateral_um")


def position_features(index: pd.DataFrame) -> np.ndarray:
    """Absolute contact position along and across the shank, as ``(n_chunks, 2)``.

    Both values come from the probe's geometry alone and are known before any recording is made.
    Channel number is deliberately left out: within one probe type it duplicates depth, and
    across probe types its scale differs, so it would only add a lab-identity cue.
    """
    if "depth_um" not in index.columns:
        raise ValueError("chunk index is missing 'depth_um', needed for position features")
    depth = index["depth_um"].to_numpy(dtype=np.float64)
    lateral = (
        index["lateral_um"].to_numpy(dtype=np.float64)
        if "lateral_um" in index.columns
        else np.zeros(len(index))
    )
    return np.column_stack([depth, np.nan_to_num(lateral, nan=0.0)])


def labelled_span_fraction(index: pd.DataFrame) -> np.ndarray:
    """Depth rescaled to the span of kept channels on each probe. **Leaks test labels.**

    Diagnostic only. Which channels are kept is decided by their histology labels, so this
    feature encodes where the in-scope anatomy starts and stops on the very probe being scored.
    It exists so that the size of the leak can be reported, not so that it can be used.
    """
    missing = [c for c in ("depth_um", "group") if c not in index.columns]
    if missing:
        raise ValueError(f"chunk index is missing columns: {missing}")
    depth = index["depth_um"].to_numpy(dtype=np.float64)
    frame = pd.DataFrame({"group": index["group"].to_numpy(), "depth": depth})
    low = frame.groupby("group")["depth"].transform("min").to_numpy()
    high = frame.groupby("group")["depth"].transform("max").to_numpy()
    span = np.where(high > low, high - low, 1.0)
    return ((depth - low) / span).reshape(-1, 1)
