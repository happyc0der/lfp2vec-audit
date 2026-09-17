"""Group-aware train/val/test splits, and the verifier that proves they are clean.

Three kinds exist and they answer different questions:

``in_session``
    Chunks from the same channel land in train and test. This *is* leakage; it is included
    only as a deliberately inflated upper bound, and is labelled as such everywhere it appears.
``cross_session``
    Whole sessions are held out. This is the honest within-lab number.
``cross_lab``
    Train on one dataset, test on another. This is the generalisation claim under audit.

Splits are plain JSON holding chunk ids so that a run manifest can pin the exact partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_KINDS = ("in_session", "cross_session", "cross_lab")


@dataclass(frozen=True)
class Split:
    """Chunk ids assigned to each partition, plus the provenance needed to re-derive it."""

    kind: str
    seed: int
    train: list[int]
    val: list[int]
    test: list[int]
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> Split:
        return cls(**json.loads(Path(path).read_text()))

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _assign_groups(
    groups: list[str], val_fraction: float, test_fraction: float, rng: np.random.Generator
) -> tuple[list[str], list[str], list[str]]:
    """Partition group names, guaranteeing at least one group in val and test when possible."""
    shuffled = list(rng.permutation(np.asarray(groups, dtype=object)))
    n = len(shuffled)
    if n < 3:
        raise ValueError(
            f"need at least 3 groups for a group-aware split, got {n}: {sorted(groups)}"
        )
    n_test = max(1, int(round(test_fraction * n)))
    n_val = max(1, int(round(val_fraction * n)))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return [str(g) for g in train], [str(g) for g in val], [str(g) for g in test]


def make_split(
    index: pd.DataFrame,
    kind: str = "cross_session",
    seed: int = 0,
    val_fraction: float = 0.30,
    test_fraction: float = 0.15,
    train_datasets: list[str] | None = None,
    test_datasets: list[str] | None = None,
) -> Split:
    """Build a split over a chunk index.

    For ``cross_lab``, ``train_datasets`` and ``test_datasets`` must be disjoint; validation is
    carved out of the training datasets by session, so no target-lab data is ever seen during
    training or model selection.
    """
    if kind not in SPLIT_KINDS:
        raise ValueError(f"kind must be one of {SPLIT_KINDS}, got {kind!r}")
    rng = np.random.default_rng(seed)
    index = index.reset_index(drop=True)

    if kind == "in_session":
        ids = index["chunk_id"].to_numpy()
        shuffled = rng.permutation(ids)
        n_test = max(1, int(round(test_fraction * len(shuffled))))
        n_val = max(1, int(round(val_fraction * len(shuffled))))
        test, val, train = (
            shuffled[:n_test],
            shuffled[n_test : n_test + n_val],
            shuffled[n_test + n_val :],
        )
        groups = sorted(index["group"].astype(str).unique())
        return Split(
            kind,
            seed,
            sorted(int(i) for i in train),
            sorted(int(i) for i in val),
            sorted(int(i) for i in test),
            groups,
            groups,
            groups,
        )

    if kind == "cross_session":
        groups = sorted(index["group"].astype(str).unique())
        train_g, val_g, test_g = _assign_groups(groups, val_fraction, test_fraction, rng)
    else:
        if not train_datasets or not test_datasets:
            raise ValueError("cross_lab requires train_datasets and test_datasets")
        overlap = set(train_datasets) & set(test_datasets)
        if overlap:
            raise ValueError(f"cross_lab datasets must be disjoint, shared: {sorted(overlap)}")
        source = index[index["dataset"].isin(train_datasets)]
        target = index[index["dataset"].isin(test_datasets)]
        if source.empty or target.empty:
            raise ValueError("cross_lab split has an empty source or target dataset")
        source_groups = sorted(source["group"].astype(str).unique())
        if len(source_groups) < 2:
            raise ValueError("cross_lab needs at least 2 source sessions to hold out a val set")
        shuffled = list(rng.permutation(np.asarray(source_groups, dtype=object)))
        n_val = max(1, int(round(val_fraction * len(shuffled))))
        val_g = [str(g) for g in shuffled[:n_val]]
        train_g = [str(g) for g in shuffled[n_val:]]
        test_g = sorted(target["group"].astype(str).unique())

    group_col = index["group"].astype(str)
    select = lambda names: sorted(  # noqa: E731 - a local alias keeps the three lines symmetric
        int(i) for i in index.loc[group_col.isin(names), "chunk_id"]
    )
    return Split(kind, seed, select(train_g), select(val_g), select(test_g), train_g, val_g, test_g)


def verify_no_leakage(split: Split, index: pd.DataFrame) -> dict[str, int]:
    """Raise if a split is unsound; return partition sizes when it is clean.

    Checks performed:

    1. No chunk id appears in more than one partition.
    2. Every partition is non-empty.
    3. For group-aware kinds, no ``group`` value appears in more than one partition.
    4. For ``cross_lab``, train and test share no ``dataset`` value.
    """
    partitions = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}
    for name, ids in partitions.items():
        if not ids:
            raise AssertionError(f"partition {name!r} is empty")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = partitions[a] & partitions[b]
        if shared:
            raise AssertionError(
                f"{len(shared)} chunk ids appear in both {a} and {b}, e.g. {sorted(shared)[:5]}"
            )

    if split.kind != "in_session":
        lookup = index.set_index("chunk_id")
        groups = {
            n: set(lookup.loc[sorted(ids), "group"].astype(str)) for n, ids in partitions.items()
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = groups[a] & groups[b]
            if shared:
                raise AssertionError(f"groups leak between {a} and {b}: {sorted(shared)[:5]}")
        if split.kind == "cross_lab":
            datasets = {
                n: set(lookup.loc[sorted(ids), "dataset"].astype(str))
                for n, ids in partitions.items()
            }
            shared = datasets["train"] & datasets["test"]
            if shared:
                raise AssertionError(f"cross_lab train and test share datasets: {sorted(shared)}")

    return split.sizes()
