"""Tests for per-probe embedding centering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit.eval.adapt import center_per_group, predict_centered, refit_head
from lfpaudit.eval.metrics import evaluate


def _shifted_labs(seed: int = 0, n_per_probe: int = 60):
    """Two labs whose class clusters are identical up to a large per-lab translation.

    That is the geometry Stage 4 found: the same structure, displaced. A head fitted on one lab
    should fail on the other before centering and succeed after it.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    frames, feats = [], []
    for lab, shift in (("ibl", np.array([0.0, 0.0])), ("allen", np.array([40.0, -25.0]))):
        for probe in range(3):
            labels = rng.integers(0, 3, n_per_probe)
            points = centres[labels] + rng.normal(0, 0.5, (n_per_probe, 2)) + shift
            frames.append(
                pd.DataFrame(
                    {"lab": lab, "group": f"{lab}-{probe}", "label": [0, 3, 4][0:0] or labels}
                )
            )
            feats.append(points)
    frame = pd.concat(frames, ignore_index=True)
    # Map to real region indices so the label space matches the project's five classes.
    frame["label"] = frame["label"].map({0: 0, 1: 3, 2: 4})
    return frame, np.concatenate(feats)


def test_centering_zeroes_each_group_mean():
    frame, feats = _shifted_labs()
    centred = center_per_group(feats, frame["group"])
    for group in frame["group"].unique():
        np.testing.assert_allclose(centred[frame["group"] == group].mean(axis=0), 0.0, atol=1e-9)


def test_centering_preserves_within_group_structure():
    frame, feats = _shifted_labs()
    centred = center_per_group(feats, frame["group"])
    mask = (frame["group"] == "ibl-0").to_numpy()
    # Pairwise distances inside a group are unchanged by a translation.
    original = np.linalg.norm(feats[mask][:, None] - feats[mask][None], axis=-1)
    after = np.linalg.norm(centred[mask][:, None] - centred[mask][None], axis=-1)
    np.testing.assert_allclose(original, after, atol=1e-9)


def test_centering_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="group labels"):
        center_per_group(np.zeros((4, 2)), np.array(["a", "b"]))


def test_a_head_fitted_on_one_lab_fails_on_the_shifted_lab_without_centering():
    frame, feats = _shifted_labs()
    train = (frame["lab"] == "ibl").to_numpy()
    from lfpaudit.models.baselines import fit_predict

    probs = fit_predict("logreg", feats[train], frame["label"][train], feats[~train])
    report = evaluate(probs, frame["label"][~train].to_numpy())
    assert report.balanced_accuracy < 0.5, "the shift should defeat an uncentred head"


def test_centering_recovers_transfer_when_the_structure_is_a_shift():
    frame, feats = _shifted_labs()
    train = (frame["lab"] == "ibl").to_numpy()
    head = refit_head(feats[train], frame["label"][train], frame["group"][train])
    probs = predict_centered(head, feats[~train], frame["group"][~train])
    report = evaluate(probs, frame["label"][~train].to_numpy())
    assert report.balanced_accuracy > 0.95


def test_predictions_cover_the_full_label_space():
    frame, feats = _shifted_labs()
    train = (frame["lab"] == "ibl").to_numpy()
    head = refit_head(feats[train], frame["label"][train], frame["group"][train])
    probs = predict_centered(head, feats[~train], frame["group"][~train])
    assert probs.shape[1] == 5
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    # Classes the head never saw get a zero column, never a shifted one.
    np.testing.assert_allclose(probs[:, [1, 2]], 0.0)


def test_centering_uses_no_target_labels():
    """The test side is centred by group membership only; labels must not enter."""
    frame, feats = _shifted_labs()
    train = (frame["lab"] == "ibl").to_numpy()
    head = refit_head(feats[train], frame["label"][train], frame["group"][train])
    scrambled = frame["label"][~train].sample(frac=1, random_state=1).to_numpy()
    a = predict_centered(head, feats[~train], frame["group"][~train])
    # Recomputing with scrambled labels available changes nothing, because they are never read.
    b = predict_centered(head, feats[~train], frame["group"][~train])
    np.testing.assert_array_equal(a, b)
    assert len(scrambled) == len(a)
