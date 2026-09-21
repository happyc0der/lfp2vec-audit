"""Tests for the signal-over-position study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit.eval.folds import leave_one_group_out
from lfpaudit.eval.position_study import (
    StudyInputs,
    boundary_distance_um,
    boundary_table,
    depth_uncertainty_study,
    fusion_study,
    jitter_depth,
    product_of_experts,
)
from lfpaudit.features.bandpower import band_power
from lfpaudit.features.geometry import position_features


def _probe(regions, spacing=20.0, group="p", gap_after=None):
    rows, depth = [], 0.0
    for i, region in enumerate(regions):
        rows.append({"group": group, "channel": i, "depth_um": depth, "region": region})
        depth += spacing * (5 if gap_after == i else 1)
    return pd.DataFrame(rows)


def test_boundary_distance_is_zero_side_at_a_label_change():
    index = _probe(["DG"] * 4 + ["CA1"] * 4)
    distance = boundary_distance_um(index)
    # The boundary sits midway between channels 3 and 4, ten micrometres from each.
    assert distance[3] == pytest.approx(10.0)
    assert distance[4] == pytest.approx(10.0)
    assert distance[1] == pytest.approx(30.0)  # nearer the end of the kept span than the change


def test_a_gap_in_kept_channels_counts_as_a_boundary():
    """Tissue outside the five regions was dropped, so a gap is a boundary to somewhere else."""
    index = _probe(["CA1"] * 3 + ["VIS"] * 3, gap_after=2)
    distance = boundary_distance_um(index)
    assert distance[2] == pytest.approx(10.0)  # edge of the gap
    assert distance[3] == pytest.approx(10.0)


def test_span_ends_count_as_boundaries():
    distance = boundary_distance_um(_probe(["VIS"] * 9))
    assert distance[0] == pytest.approx(10.0)
    assert distance[4] == pytest.approx(90.0)


def test_boundaries_are_found_per_probe():
    index = pd.concat([_probe(["DG"] * 3 + ["CA1"] * 3, group="a"), _probe(["VIS"] * 6, group="b")])
    distance = boundary_distance_um(index.reset_index(drop=True))
    assert distance[2] == pytest.approx(10.0)
    assert distance[8] == pytest.approx(50.0)  # probe b has no internal boundary


def test_boundary_distance_requires_its_columns():
    with pytest.raises(ValueError, match="depth_um"):
        boundary_distance_um(_probe(["DG"] * 3).drop(columns=["depth_um"]))


def test_product_of_experts_matches_bayes_by_hand():
    prior = np.log(np.array([0.5, 0.5]))
    a = np.log(np.array([[0.8, 0.2]]))
    b = np.log(np.array([[0.6, 0.4]]))
    fused = product_of_experts([a, b], prior)
    expected = np.array([0.8 * 0.6 / 0.5, 0.2 * 0.4 / 0.5])
    np.testing.assert_allclose(fused[0], expected / expected.sum())


def test_one_expert_is_returned_unchanged():
    probs = np.array([[0.7, 0.2, 0.1]])
    fused = product_of_experts([np.log(probs)], np.log(np.ones(3) / 3))
    np.testing.assert_allclose(fused, probs)


def test_an_uninformative_expert_changes_nothing():
    prior = np.log(np.ones(3) / 3)
    informative = np.log(np.array([[0.7, 0.2, 0.1]]))
    flat = np.log(np.ones((1, 3)) / 3)
    np.testing.assert_allclose(
        product_of_experts([informative, flat], prior), product_of_experts([informative], prior)
    )


def test_product_of_experts_needs_a_source():
    with pytest.raises(ValueError, match="at least one"):
        product_of_experts([], np.zeros(2))


def test_jitter_moves_a_whole_probe_together_and_only_in_depth():
    position = np.array([[0.0, 11.0], [20.0, 27.0], [0.0, 11.0], [20.0, 27.0]])
    groups = np.array(["a", "a", "b", "b"])
    shifted = jitter_depth(position, groups, 500.0, np.random.default_rng(0))
    np.testing.assert_allclose(shifted[:, 1], position[:, 1])
    assert shifted[1, 0] - shifted[0, 0] == pytest.approx(20.0)  # spacing within a probe kept
    assert shifted[0, 0] - position[0, 0] != pytest.approx(shifted[2, 0] - position[2, 0])


def test_zero_jitter_is_the_identity():
    position = np.array([[5.0, 1.0], [25.0, 1.0]])
    out = jitter_depth(position, np.array(["a", "a"]), 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(out, position)
    assert out is not position


@pytest.fixture
def study(synthetic_store):
    index = synthetic_store.index
    signal = band_power(synthetic_store.take(np.arange(len(index))), fs=synthetic_store.fs)
    inputs = StudyInputs(
        index=index,
        position=position_features(index),
        signals={"bandpower": signal},
        view="5class",
        per_class=200,
    )
    return inputs, {"loso": leave_one_group_out(index)}


def test_fusion_study_scores_every_source_and_the_fusion(study):
    inputs, schemes = study
    scores, chunks = fusion_study(schemes, inputs, verbose=False)
    assert set(scores["source"]) == {"position", "bandpower", "position+bandpower"}
    assert len(scores) == 3 * len(schemes["loso"])
    assert {"correct::position", "correct::bandpower", "correct::position+bandpower"} <= set(
        chunks.columns
    )


def test_fusion_is_not_worse_than_both_of_its_parts_on_average(study):
    """A product of two informative experts should not lose to each of them."""
    inputs, schemes = study
    scores, _ = fusion_study(schemes, inputs, verbose=False)
    mean = scores.groupby("source")["balanced_accuracy"].mean()
    assert mean["position+bandpower"] >= min(mean["position"], mean["bandpower"])


def test_boundary_table_covers_every_bin_present(study):
    inputs, schemes = study
    _, chunks = fusion_study(schemes, inputs, verbose=False)
    table = boundary_table(chunks, boundary_distance_um(inputs.index))
    assert {"scheme", "bin", "n", "position", "bandpower"} <= set(table.columns)
    assert table["n"].sum() == len(chunks)


def test_depth_uncertainty_hurts_position_and_not_the_signal(study):
    inputs, schemes = study
    table = depth_uncertainty_study(
        schemes,
        inputs,
        signal="bandpower",
        half_widths_um=(0.0, 5000.0),
        train_copies=2,
        test_draws=3,
        verbose=False,
    )
    mean = table.groupby("half_width_um")[["position", "bandpower"]].mean()
    assert mean.loc[5000.0, "position"] < mean.loc[0.0, "position"]
    assert mean.loc[5000.0, "bandpower"] == pytest.approx(mean.loc[0.0, "bandpower"])


def test_inputs_reject_misaligned_tables(synthetic_store):
    index = synthetic_store.index
    with pytest.raises(ValueError, match="rows for an index"):
        StudyInputs(index=index, position=np.zeros((3, 2)), signals={})
