"""Cross-validation folds, built on the split machinery that already verifies itself.

Stage 1 evaluated on a single held-out insertion and got 0.628 balanced accuracy. That number is
almost meaningless on its own, because insertions differ enormously: one passes through three
structures and another through four, one sits squarely in a cell layer and another skims it. A
single split reports whichever insertion the seed happened to choose.

Leaving out one session at a time instead produces a distribution, and because features are
cached the seventeen folds cost what one did. Every fold here is an ordinary
:class:`~lfpaudit.data.splits.Split`, so the existing leakage verifier applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lfpaudit.data.splits import Split, verify_no_leakage


@dataclass(frozen=True)
class Fold:
    """One fold: a split plus the name of what was held out."""

    name: str
    scheme: str
    split: Split

    def sizes(self) -> dict[str, int]:
        return self.split.sizes()


def _ids(index: pd.DataFrame, groups: list[str]) -> list[int]:
    mask = index["group"].astype(str).isin(groups)
    return sorted(int(i) for i in index.loc[mask, "chunk_id"])


def leave_one_group_out(
    index: pd.DataFrame,
    dataset: str | None = None,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> list[Fold]:
    """One fold per group, each holding that group out for test.

    A validation group is drawn from the remaining training groups, so model selection never
    touches the held-out session. With only a handful of groups the validation set is a whole
    session rather than a fraction of one, which is coarse but keeps the grouping honest.
    """
    frame = index if dataset is None else index[index["dataset"] == dataset]
    if frame.empty:
        raise ValueError(f"no chunks for dataset {dataset!r}")
    groups = sorted(frame["group"].astype(str).unique())
    if len(groups) < 3:
        raise ValueError(f"need at least 3 groups for leave-one-out, got {len(groups)}")

    rng = np.random.default_rng(seed)
    folds: list[Fold] = []
    for held_out in groups:
        remaining = [g for g in groups if g != held_out]
        n_val = max(1, int(round(val_fraction * len(remaining))))
        shuffled = [str(g) for g in rng.permutation(np.asarray(remaining, dtype=object))]
        val_groups, train_groups = shuffled[:n_val], shuffled[n_val:]

        split = Split(
            kind="cross_session",
            seed=seed,
            train=_ids(frame, train_groups),
            val=_ids(frame, val_groups),
            test=_ids(frame, [held_out]),
            train_groups=sorted(train_groups),
            val_groups=sorted(val_groups),
            test_groups=[held_out],
        )
        verify_no_leakage(split, index)
        scheme = f"loso_{dataset}" if dataset else "loso"
        folds.append(Fold(name=held_out, scheme=scheme, split=split))
    return folds


def cross_lab_folds(
    index: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> list[Fold]:
    """Train on one dataset, and score each group of the other separately.

    Pooling the whole target dataset into one test set would give a single number with no sense
    of its spread. Scoring per target group instead makes cross-lab results comparable with the
    within-lab folds, and shows whether transfer fails everywhere or only on some probes.
    """
    source = index[index["dataset"] == train_dataset]
    target = index[index["dataset"] == test_dataset]
    if source.empty or target.empty:
        raise ValueError(f"cross-lab needs both datasets; got {train_dataset} and {test_dataset}")
    if train_dataset == test_dataset:
        raise ValueError("cross-lab train and test datasets must differ")

    source_groups = sorted(source["group"].astype(str).unique())
    if len(source_groups) < 2:
        raise ValueError("need at least 2 source groups to hold out a validation group")
    rng = np.random.default_rng(seed)
    shuffled = [str(g) for g in rng.permutation(np.asarray(source_groups, dtype=object))]
    n_val = max(1, int(round(val_fraction * len(shuffled))))
    val_groups, train_groups = shuffled[:n_val], shuffled[n_val:]

    folds: list[Fold] = []
    for held_out in sorted(target["group"].astype(str).unique()):
        split = Split(
            kind="cross_lab",
            seed=seed,
            train=_ids(source, train_groups),
            val=_ids(source, val_groups),
            test=_ids(target, [held_out]),
            train_groups=sorted(train_groups),
            val_groups=sorted(val_groups),
            test_groups=[held_out],
        )
        verify_no_leakage(split, index)
        folds.append(
            Fold(name=held_out, scheme=f"cross_lab_{train_dataset}_to_{test_dataset}", split=split)
        )
    return folds


def build_schemes(index: pd.DataFrame, seed: int = 0) -> dict[str, list[Fold]]:
    """Every evaluation scheme this stage reports, keyed by name."""
    datasets = sorted(index["dataset"].astype(str).unique())
    schemes: dict[str, list[Fold]] = {}
    for dataset in datasets:
        schemes[f"loso_{dataset}"] = leave_one_group_out(index, dataset, seed=seed)
    if len(datasets) == 2:
        first, second = datasets
        schemes[f"cross_lab_{first}_to_{second}"] = cross_lab_folds(index, first, second, seed=seed)
        schemes[f"cross_lab_{second}_to_{first}"] = cross_lab_folds(index, second, first, seed=seed)
    return schemes
