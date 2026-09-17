"""Tests for the baseline models."""

from __future__ import annotations

import numpy as np
import pytest

from lfpaudit.eval.metrics import evaluate
from lfpaudit.models.baselines import ConstantClassifier, build_model, fit_predict


@pytest.fixture
def separable():
    """Four classes that a linear model should separate perfectly."""
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]])
    labels = np.repeat([0, 2, 3, 4], 40)
    features = centres[np.repeat([0, 1, 2, 3], 40)] + rng.normal(scale=0.3, size=(160, 2))
    return features, labels


def test_constant_predicts_training_frequencies():
    model = ConstantClassifier().fit(np.zeros((6, 2)), np.array([0, 0, 0, 3, 3, 4]))
    probs = model.predict_proba(np.zeros((2, 2)))
    np.testing.assert_allclose(probs[0], [0.5, 1 / 3, 1 / 6])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_constant_scores_at_chance(separable):
    features, labels = separable
    probs = fit_predict("constant", features, labels, features)
    report = evaluate(probs, labels)
    # It always predicts one class, so recall is 1 for that class and 0 elsewhere.
    assert report.balanced_accuracy == pytest.approx(report.chance, abs=1e-9)


def test_logreg_separates_separable_data(separable):
    features, labels = separable
    probs = fit_predict("logreg", features, labels, features)
    assert evaluate(probs, labels).balanced_accuracy > 0.95


def test_probabilities_cover_the_full_label_space(separable):
    """CA2 is absent from these labels; its column must exist and be zero, not be dropped."""
    features, labels = separable
    probs = fit_predict("logreg", features, labels, features)
    assert probs.shape == (len(features), 5)
    np.testing.assert_allclose(probs[:, 1], 0.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_permuted_labels_destroy_performance(separable):
    features, labels = separable
    rng = np.random.default_rng(1)
    probs = fit_predict("logreg", features, rng.permutation(labels), features)
    report = evaluate(probs, labels)
    assert report.balanced_accuracy < report.chance + 0.15


def test_mlp_runs_and_separates(separable):
    features, labels = separable
    probs = fit_predict("mlp", features, labels, features, seed=0)
    assert evaluate(probs, labels).balanced_accuracy > 0.9


def test_mlp_drops_early_stopping_on_small_training_sets():
    """A tenth of a small training set is too noisy a stopping signal; measured, not assumed."""
    from lfpaudit.models.baselines import MIN_ROWS_FOR_EARLY_STOPPING, build_model

    small = build_model("mlp", n_train=MIN_ROWS_FOR_EARLY_STOPPING - 1)
    large = build_model("mlp", n_train=MIN_ROWS_FOR_EARLY_STOPPING + 1)
    assert small[-1].early_stopping is False
    assert large[-1].early_stopping is True


def test_models_are_deterministic_given_a_seed(separable):
    features, labels = separable
    first = fit_predict("mlp", features, labels, features, seed=7)
    second = fit_predict("mlp", features, labels, features, seed=7)
    np.testing.assert_allclose(first, second)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("magic")


def test_single_class_training_is_rejected():
    with pytest.raises(ValueError, match="fewer than two classes"):
        fit_predict("logreg", np.zeros((4, 2)), np.zeros(4, dtype=int), np.zeros((2, 2)))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="training rows"):
        fit_predict("logreg", np.zeros((4, 2)), np.zeros(3, dtype=int), np.zeros((2, 2)))


def test_scaling_makes_feature_units_irrelevant(separable):
    """Feature sets differ in scale by orders of magnitude; comparisons must not measure that."""
    features, labels = separable
    plain = fit_predict("logreg", features, labels, features)
    rescaled = fit_predict("logreg", features * 1000.0, labels, features * 1000.0)
    np.testing.assert_allclose(plain, rescaled, atol=1e-3)
