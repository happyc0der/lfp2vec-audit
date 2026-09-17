"""Tests for the lab-identity probe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit.eval.probe import lab_identity_auc, paired_holdout_folds


@pytest.fixture
def setting():
    """Six groups, three per dataset, forty chunks each."""
    groups = pd.Series(np.repeat([f"{d}-{i}" for d in ("ibl", "allen") for i in range(3)], 40))
    datasets = pd.Series(np.repeat(["ibl", "allen"], 120))
    return groups, datasets


def test_folds_hold_out_one_group_from_each_dataset(setting):
    groups, datasets = setting
    folds = paired_holdout_folds(groups, datasets)
    assert len(folds) == 3
    for first, second in folds:
        assert first.startswith("allen") or first.startswith("ibl")
        assert first.split("-")[0] != second.split("-")[0]


def test_folds_require_exactly_two_datasets(setting):
    groups, _ = setting
    with pytest.raises(ValueError, match="exactly two datasets"):
        paired_holdout_folds(groups, pd.Series(["only"] * len(groups)))


def test_separable_representation_scores_near_one(setting):
    """A feature that encodes the dataset should be caught, which is the point of the probe."""
    groups, datasets = setting
    rng = np.random.default_rng(0)
    offset = (datasets == "allen").to_numpy().astype(float)[:, None] * 8.0
    features = rng.normal(size=(len(groups), 4)) + offset

    result = lab_identity_auc(features, groups, datasets)
    assert result.auc > 0.99
    assert len(result.folds) == 3


def test_uninformative_representation_scores_near_chance(setting):
    groups, datasets = setting
    rng = np.random.default_rng(1)
    features = rng.normal(size=(len(groups), 4))

    result = lab_identity_auc(features, groups, datasets)
    assert 0.3 < result.auc < 0.7


def test_probe_generalises_to_unseen_groups(setting):
    """Every fold's score comes from probes the classifier never trained on."""
    groups, datasets = setting
    rng = np.random.default_rng(2)
    features = (
        rng.normal(size=(len(groups), 3))
        + (datasets == "allen").to_numpy().astype(float)[:, None] * 5.0
    )

    result = lab_identity_auc(features, groups, datasets)
    for row in result.folds.itertuples():
        held = set(row.held_out.split("+"))
        assert len(held) == 2


def test_mismatched_lengths_are_rejected(setting):
    groups, datasets = setting
    with pytest.raises(ValueError, match="rows"):
        lab_identity_auc(np.zeros((5, 2)), groups, datasets)


def test_summary_line_reports_spread(setting):
    groups, datasets = setting
    rng = np.random.default_rng(3)
    result = lab_identity_auc(rng.normal(size=(len(groups), 2)), groups, datasets)
    assert "area under the curve" in result.summary_line("test")
    assert np.isfinite(result.auc_sd)
