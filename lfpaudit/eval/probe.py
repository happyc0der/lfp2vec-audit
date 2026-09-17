"""Measuring how much a representation knows about where a recording came from.

Stage 2 found that a linear model recovers the source dataset from frozen audio embeddings at
area under the curve 1.000, on probes it had never seen, and that this is why cross-lab transfer
collapses: a decision boundary fitted in one lab's region of that space says nothing about where
the other lab's chunks fall.

The natural follow-up is whether fine-tuning on region labels removes that structure. For the
answer to mean anything, the two numbers have to be produced the same way, so both the frozen and
the fine-tuned measurement call this module rather than each rolling its own probe.

Folds hold out one group from each dataset. Holding out a single group would leave a test set
containing only one label, where area under the curve is undefined and accuracy is a tautology,
and the question is precisely whether the separation survives on unseen probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lfpaudit.models.baselines import fit_predict


@dataclass
class ProbeResult:
    """Per-fold scores and their summary."""

    folds: pd.DataFrame

    @property
    def auc(self) -> float:
        return float(self.folds["auc"].mean())

    @property
    def auc_sd(self) -> float:
        return float(self.folds["auc"].std())

    def summary_line(self, label: str) -> str:
        return f"{label}: area under the curve {self.auc:.3f} +/- {self.auc_sd:.3f}"


def paired_holdout_folds(groups: pd.Series, datasets: pd.Series) -> list[tuple[str, str]]:
    """Pairs of groups, one from each dataset, to hold out together."""
    names = sorted(datasets.unique())
    if len(names) != 2:
        raise ValueError(f"expected exactly two datasets, got {sorted(names)}")

    by_dataset = {name: sorted(groups[datasets == name].astype(str).unique()) for name in names}
    count = max(len(v) for v in by_dataset.values())
    first, second = names
    return [
        (
            by_dataset[first][i % len(by_dataset[first])],
            by_dataset[second][i % len(by_dataset[second])],
        )
        for i in range(count)
    ]


def lab_identity_auc(
    features: np.ndarray,
    groups: pd.Series,
    datasets: pd.Series,
    seed: int = 0,
) -> ProbeResult:
    """How well a linear model identifies the source dataset from a representation.

    ``features`` are one row per chunk, in the same order as ``groups`` and ``datasets``. Chance
    is 0.5; a score near 1.0 means the representation encodes acquisition rather than, or as well
    as, whatever it was built for.
    """
    features = np.asarray(features, dtype=np.float64)
    groups = groups.astype(str).reset_index(drop=True)
    datasets = datasets.astype(str).reset_index(drop=True)
    if not (len(features) == len(groups) == len(datasets)):
        raise ValueError(
            f"{len(features)} rows, {len(groups)} groups and {len(datasets)} dataset labels"
        )

    names = sorted(datasets.unique())
    target = (datasets == names[1]).astype(int).to_numpy()

    rows = []
    for held in paired_holdout_folds(groups, datasets):
        is_test = groups.isin(held).to_numpy()
        train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        if len(np.unique(target[train])) < 2 or len(np.unique(target[test])) < 2:
            continue
        probs = fit_predict(
            "logreg", features[train], target[train], features[test], seed=seed, n_classes=2
        )
        predicted = probs.argmax(axis=1)
        per_class = [float((predicted[target[test] == c] == c).mean()) for c in (0, 1)]
        rows.append(
            {
                "held_out": "+".join(held),
                "n_test": int(len(test)),
                "auc": float(roc_auc_score(target[test], probs[:, 1])),
                "balanced_accuracy": float(np.mean(per_class)),
            }
        )

    if not rows:
        raise ValueError("no fold had both datasets present in train and test")
    return ProbeResult(folds=pd.DataFrame(rows))
