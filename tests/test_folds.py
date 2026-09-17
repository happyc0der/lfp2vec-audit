"""Tests for the cross-validation folds."""

from __future__ import annotations

import pytest

from lfpaudit.data.splits import verify_no_leakage
from lfpaudit.eval.folds import build_schemes, cross_lab_folds, leave_one_group_out


def test_one_fold_per_group(synthetic_store):
    folds = leave_one_group_out(synthetic_store.index)
    groups = sorted(synthetic_store.index["group"].astype(str).unique())
    assert len(folds) == len(groups)
    assert sorted(f.name for f in folds) == groups


def test_each_group_is_held_out_exactly_once(synthetic_store):
    folds = leave_one_group_out(synthetic_store.index)
    held_out = [g for fold in folds for g in fold.split.test_groups]
    assert sorted(held_out) == sorted(set(held_out))


def test_every_fold_is_leakage_free(synthetic_store):
    for fold in leave_one_group_out(synthetic_store.index):
        verify_no_leakage(fold.split, synthetic_store.index)


def test_validation_never_touches_the_held_out_group(synthetic_store):
    for fold in leave_one_group_out(synthetic_store.index):
        assert fold.name not in fold.split.val_groups
        assert fold.name not in fold.split.train_groups


def test_dataset_filter_restricts_the_folds(synthetic_store):
    folds = leave_one_group_out(synthetic_store.index, dataset="synthA")
    assert all(g.startswith("synthA") for fold in folds for g in fold.split.test_groups)
    assert folds[0].scheme == "loso_synthA"


def test_cross_lab_folds_score_each_target_group(synthetic_store):
    folds = cross_lab_folds(synthetic_store.index, "synthA", "synthB")
    target = sorted(
        synthetic_store.index[synthetic_store.index["dataset"] == "synthB"]["group"].unique()
    )
    assert sorted(f.name for f in folds) == target
    for fold in folds:
        verify_no_leakage(fold.split, synthetic_store.index)


def test_cross_lab_train_and_test_come_from_different_datasets(synthetic_store):
    lookup = synthetic_store.index.set_index("chunk_id")
    for fold in cross_lab_folds(synthetic_store.index, "synthA", "synthB"):
        assert set(lookup.loc[fold.split.train, "dataset"]) == {"synthA"}
        assert set(lookup.loc[fold.split.test, "dataset"]) == {"synthB"}


def test_cross_lab_rejects_identical_datasets(synthetic_store):
    with pytest.raises(ValueError, match="must differ"):
        cross_lab_folds(synthetic_store.index, "synthA", "synthA")


def test_cross_lab_rejects_an_absent_dataset(synthetic_store):
    with pytest.raises(ValueError, match="needs both datasets"):
        cross_lab_folds(synthetic_store.index, "synthA", "nowhere")


def test_build_schemes_covers_both_directions(synthetic_store):
    schemes = build_schemes(synthetic_store.index)
    assert set(schemes) == {
        "loso_synthA",
        "loso_synthB",
        "cross_lab_synthA_to_synthB",
        "cross_lab_synthB_to_synthA",
    }
    for folds in schemes.values():
        assert folds


def test_folds_are_deterministic(synthetic_store):
    first = leave_one_group_out(synthetic_store.index, seed=3)
    second = leave_one_group_out(synthetic_store.index, seed=3)
    assert [f.split for f in first] == [f.split for f in second]


def test_too_few_groups_is_rejected(synthetic_store):
    small = synthetic_store.index[synthetic_store.index["group"].isin(["synthA-s00"])]
    with pytest.raises(ValueError, match="at least 3 groups"):
        leave_one_group_out(small)
