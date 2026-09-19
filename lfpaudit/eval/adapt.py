"""Per-probe embedding centering: the smallest unsupervised adaptation that fits the geometry.

Stage 4 found that a model handed recordings from a lab it never saw squeezes them into a tight
cluster displaced from its own training distribution, and that a decision boundary fitted in one
region of representation space says nothing about inputs that all fall in another. The obvious
response is to move the cluster back: subtract each probe's mean embedding, so every probe sits
at the origin regardless of which rig produced it.

This uses no labels from anywhere, has no hyper-parameters, and touches nothing the model
learned. If it works, the acquisition structure was mostly a shift; if it does not, the structure
is in the shape of the cloud rather than its position, and a translation cannot reach it. Either
answer is informative.

The classifier head has to be refit on training embeddings centred the same way, since the
original head was fitted on uncentred ones. That refit sees only training-lab data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lfpaudit import REGIONS
from lfpaudit.eval.metrics import expand_probabilities
from lfpaudit.models.baselines import build_model


def center_per_group(embeddings: np.ndarray, groups: pd.Series | np.ndarray) -> np.ndarray:
    """Subtract each group's mean embedding from its members."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    groups = np.asarray(groups)
    if len(embeddings) != len(groups):
        raise ValueError(f"{len(embeddings)} embeddings for {len(groups)} group labels")
    centred = embeddings.copy()
    for group in pd.unique(groups):
        mask = groups == group
        centred[mask] -= embeddings[mask].mean(axis=0)
    return centred


def refit_head(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_groups: pd.Series | np.ndarray,
    seed: int = 0,
):
    """A linear head on per-group-centred training embeddings.

    Returns the fitted model; use :func:`predict_centered` to apply it, so the test side is
    centred by exactly the same rule.
    """
    centred = center_per_group(train_embeddings, train_groups)
    model = build_model("logreg", seed=seed)
    model.fit(centred, np.asarray(train_labels, dtype=np.int64))
    return model


def predict_centered(
    model,
    test_embeddings: np.ndarray,
    test_groups: pd.Series | np.ndarray,
    n_classes: int = len(REGIONS),
) -> np.ndarray:
    """Probabilities in the full label space for per-group-centred test embeddings."""
    centred = center_per_group(test_embeddings, test_groups)
    classes = np.asarray(model[-1].classes_, dtype=np.int64)
    return expand_probabilities(model.predict_proba(centred), classes, n_classes)
