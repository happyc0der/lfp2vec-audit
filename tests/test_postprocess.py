"""Tests for the paper's post-processing pipeline.

The gate that matters: on a synthetic session with their noise structure, correct-but-noisy
per-chunk predictions, the temporal step must improve accuracy and the spatial step must improve
it again. If the implementation cannot rescue a case built to be rescuable, it is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit.eval.postprocess import (
    apply_paper_pipeline,
    channel_level_scores,
    spatial_vote,
    temporal_aggregate,
)


def _noisy_session(seed: int = 0, chunks_per_channel: int = 30, flip: float = 0.4):
    """Two shanks, contiguous regions along depth, and per-chunk labels that are often wrong.

    Each chunk's logits favour the true region with probability ``1 - flip`` and a random wrong
    region otherwise, so raw per-chunk accuracy is around 0.6, while the per-channel majority is
    right almost always, and a rare fully-wrong channel is surrounded by correct neighbours.
    """
    rng = np.random.default_rng(seed)
    rows, logits = [], []
    for shank in ("probeA", "probeB"):
        # Regions in contiguous blocks of eight channels along depth: 0, 3, 4 then 2.
        truth = np.repeat([0, 3, 4, 2], 8)
        for channel, region in enumerate(truth):
            for _chunk in range(chunks_per_channel):
                predicted = (
                    region
                    if rng.random() > flip
                    else rng.choice([c for c in range(5) if c != region])
                )
                vector = rng.normal(0.0, 0.3, 5)
                vector[predicted] += 2.0
                rows.append(
                    {
                        "group": shank,
                        "channel": channel,
                        "depth_um": 20.0 * channel,
                        "label": region,
                    }
                )
                logits.append(vector)
    return pd.DataFrame(rows), np.array(logits)


def test_temporal_aggregate_gives_one_label_per_channel():
    frame, logits = _noisy_session()
    keys = np.array([f"{g}::{c}" for g, c in zip(frame["group"], frame["channel"], strict=True)])
    per_channel, broadcast = temporal_aggregate(logits, keys)
    assert len(per_channel) == frame[["group", "channel"]].drop_duplicates().shape[0]
    # Every chunk of a channel carries the same label after aggregation.
    for key in per_channel:
        assert len(set(broadcast[keys == key])) == 1


def test_temporal_aggregate_averages_logits_not_probabilities():
    """Their notebook averages raw logits; a confident wrong chunk should be able to outvote."""
    logits = np.array([[10.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    per_channel, _ = temporal_aggregate(logits, np.array(["k", "k", "k"]))
    # Mean logit is [3.33, 0.67] -> class 0, even though two of three chunks argmax to class 1.
    assert per_channel["k"] == 0


def test_temporal_prior_reweights():
    logits = np.array([[1.0, 0.9], [1.0, 0.9]])
    without, _ = temporal_aggregate(logits, np.array(["k", "k"]))
    with_prior, _ = temporal_aggregate(logits, np.array(["k", "k"]), prior=np.array([1.0, 2.0]))
    assert without["k"] == 0
    assert with_prior["k"] == 1


def test_temporal_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="channel keys"):
        temporal_aggregate(np.zeros((3, 2)), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="prior has shape"):
        temporal_aggregate(np.zeros((2, 2)), np.array(["a", "a"]), prior=np.ones(3))


def test_spatial_vote_fixes_an_isolated_error():
    labels = {f"c{i}": 1 for i in range(9)}
    labels["c4"] = 3  # one wrong channel in the middle of a run of ones
    smoothed = spatial_vote(labels, {"shank": [f"c{i}" for i in range(9)]}, window=2)
    assert smoothed["c4"] == 1
    assert all(v == 1 for v in smoothed.values())


def test_spatial_vote_preserves_a_real_boundary():
    """A genuine region change is a run, not an isolated error, and must survive."""
    labels = {f"c{i}": (0 if i < 5 else 3) for i in range(10)}
    smoothed = spatial_vote(labels, {"shank": [f"c{i}" for i in range(10)]}, window=2)
    assert [smoothed[f"c{i}"] for i in range(10)] == [0, 0, 0, 0, 0, 3, 3, 3, 3, 3]


def test_spatial_vote_uses_fewer_neighbours_at_shank_ends():
    labels = {"c0": 2, "c1": 2, "c2": 0, "c3": 0, "c4": 0}
    smoothed = spatial_vote(labels, {"shank": ["c0", "c1", "c2", "c3", "c4"]}, window=2)
    # c0 sees c0..c2 = [2, 2, 0] -> 2; c2 sees all five = [2,2,0,0,0] -> 0
    assert smoothed["c0"] == 2
    assert smoothed["c2"] == 0


def test_spatial_vote_ties_resolve_to_the_smallest_label():
    """What scipy.stats.mode does, and therefore what their notebook did."""
    labels = {"c0": 4, "c1": 4, "c2": 1, "c3": 1}
    smoothed = spatial_vote(labels, {"shank": ["c0", "c1", "c2", "c3"]}, window=1)
    # c1 sees [4, 4, 1] -> 4; c2 sees [4, 1, 1] -> 1; c0 sees [4, 4] -> 4; c3 sees [1, 1] -> 1
    assert smoothed["c1"] == 4 and smoothed["c2"] == 1


def test_spatial_vote_does_not_cross_shanks():
    labels = {"a0": 0, "a1": 0, "b0": 3, "b1": 3}
    smoothed = spatial_vote(labels, {"A": ["a0", "a1"], "B": ["b0", "b1"]}, window=2)
    assert smoothed["a1"] == 0 and smoothed["b0"] == 3


def test_pipeline_rescues_a_noisy_but_correct_session():
    """The gate: their own example went 0.716 -> 0.793 -> 0.836. Direction and size must hold."""
    frame, logits = _noisy_session(flip=0.4)
    result = apply_paper_pipeline(frame, logits)
    truth = frame["label"].to_numpy()
    raw = (result.raw == truth).mean()
    temporal = (result.temporal == truth).mean()
    spatial = (result.spatial == truth).mean()
    assert 0.5 < raw < 0.7, f"synthetic noise level is off: raw {raw:.3f}"
    assert temporal > raw + 0.2
    assert spatial >= temporal


def test_pipeline_is_the_identity_when_predictions_are_already_perfect():
    frame, logits = _noisy_session(flip=0.0)
    result = apply_paper_pipeline(frame, logits)
    truth = frame["label"].to_numpy()
    for stage in result.stages().values():
        np.testing.assert_array_equal(stage, truth)


def test_pipeline_cannot_rescue_a_systematic_collapse():
    """The prediction for the real cross-lab case: chunks that all agree have nothing to average."""
    frame, _ = _noisy_session()
    collapsed = np.zeros((len(frame), 5))
    collapsed[:, 4] = 5.0  # every chunk says class 4, confidently
    result = apply_paper_pipeline(frame, collapsed)
    truth = frame["label"].to_numpy()
    base = (truth == 4).mean()
    for stage in result.stages().values():
        assert (stage == truth).mean() == pytest.approx(base)


def test_channel_level_scores_count_channels_not_chunks():
    frame, logits = _noisy_session(chunks_per_channel=10)
    result = apply_paper_pipeline(frame, logits)
    scores = channel_level_scores(result.channels, "spatial")
    assert scores["n_channels"] == 64
    assert 0 < scores["balanced_accuracy"] <= 1
    assert scores["chance"] == pytest.approx(0.25)


def test_pipeline_requires_geometry_columns():
    frame, logits = _noisy_session()
    with pytest.raises(ValueError, match="depth_um"):
        apply_paper_pipeline(frame.drop(columns=["depth_um"]), logits)
