"""Signal amplitude as a feature, also as a suspect.

Chunks are stored normalised, so no model in this project sees absolute amplitude. That is the
right default, and it is also what makes amplitude worth testing separately: the scale removed by
normalisation was recorded per chunk, so it can be handed to a classifier on its own.

There are two reasons to care. Within a probe, amplitude varies with anatomy, since field
potentials are larger near dense cell layers than in white matter, so amplitude alone may decode
regions to a degree. Across labs it varies with acquisition: the two stores here differ in median
amplitude by roughly a quarter. A feature that carries anatomy within a lab and lab identity
across labs is exactly the kind of thing that flatters a cross-lab result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = ("log_scale_uv", "mean_uv")


def amplitude_features(index: pd.DataFrame) -> np.ndarray:
    """Per-chunk amplitude features recovered from the index.

    Returns ``(n_chunks, 2)``: the base-ten logarithm of the standard deviation in microvolts,
    and the mean offset. The logarithm is used because amplitudes span roughly an order of
    magnitude and a linear model should see that range as additive.
    """
    missing = [c for c in ("scale_std_uv", "scale_mean_uv") if c not in index.columns]
    if missing:
        raise ValueError(f"chunk index is missing amplitude columns: {missing}")

    scale = index["scale_std_uv"].to_numpy(dtype=np.float64)
    mean = index["scale_mean_uv"].to_numpy(dtype=np.float64)
    return np.column_stack([np.log10(np.maximum(scale, 1e-12)), mean])
