import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from lfpaudit.eval.metrics import (
    brier_score,
    chance_level,
    chance_level_band,
    classes_present,
    evaluate,
    expand_probabilities,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    softmax,
)


def _onehot(labels, n_classes=5):
    out = np.full((len(labels), n_classes), 1e-9)
    out[np.arange(len(labels)), labels] = 1.0
    return out / out.sum(axis=1, keepdims=True)


def test_agrees_with_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=200)
    probs = softmax(rng.normal(size=(200, 5)))
    report = evaluate(probs, labels)
    predicted = probs.argmax(axis=1)
    assert report.balanced_accuracy == pytest.approx(balanced_accuracy_score(labels, predicted))
    assert report.macro_f1 == pytest.approx(f1_score(labels, predicted, average="macro"))
    assert report.accuracy == pytest.approx((predicted == labels).mean())


def test_perfect_confident_predictions_are_calibrated():
    labels = np.array([0, 1, 2, 3, 4, 0, 1])
    probs = _onehot(labels)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-6)
    assert negative_log_likelihood(probs, labels) == pytest.approx(0.0, abs=1e-6)
    assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_ece_of_a_known_overconfident_case():
    # Ten predictions, all at confidence 0.9, of which half are correct.
    # Expected calibration error is |0.5 - 0.9| = 0.4 exactly.
    n = 10
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    assert expected_calibration_error(probs, labels, n_bins=15) == pytest.approx(0.4)


def test_uniform_predictions_have_chance_nll():
    labels = np.zeros(20, dtype=int)
    probs = np.full((20, 5), 0.2)
    assert negative_log_likelihood(probs, labels) == pytest.approx(np.log(5))
    assert brier_score(probs, labels) == pytest.approx(0.8**2 + 4 * 0.2**2)


def test_temperature_flattens_confidence():
    logits = np.array([[3.0, 0.0, 0.0]])
    assert softmax(logits, temperature=1.0).max() > softmax(logits, temperature=5.0).max()
    np.testing.assert_allclose(softmax(logits, temperature=1e6), np.full((1, 3), 1 / 3), atol=1e-4)


def test_rejects_logits_passed_as_probabilities():
    with pytest.raises(ValueError, match="must sum to 1"):
        evaluate(np.array([[2.0, 3.0]]), np.array([1]))


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="disagree"):
        evaluate(np.full((3, 2), 0.5), np.array([0, 1]))


def test_reliability_curve_bins_cover_everything():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=100)
    probs = softmax(rng.normal(size=(100, 3)))
    curve = reliability_curve(probs, labels, n_bins=10)
    assert sum(curve["count"]) == 100
    assert len(curve["bin_lower"]) == 10


def test_confusion_matrix_is_square_and_totals_match():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, size=50)
    report = evaluate(softmax(rng.normal(size=(50, 5))), labels)
    matrix = np.array(report.confusion)
    assert matrix.shape == (5, 5)
    assert matrix.sum() == 50
    assert set(report.per_class_recall) == set(report.class_names)


def test_noise_band_shrinks_as_the_test_set_grows():
    """A fixed threshold cannot serve both a tiny and a large test set; this scales."""
    small = chance_level_band(np.repeat(np.arange(5), 8), n_classes=5)
    large = chance_level_band(np.repeat(np.arange(5), 8000), n_classes=5)
    assert small > large
    assert large < 0.01


def test_noise_band_is_dominated_by_the_rarest_class():
    """CA2 appears on a handful of channels, so it sets how noisy the average recall is."""
    balanced = chance_level_band(np.repeat(np.arange(5), 400), n_classes=5)
    lopsided = chance_level_band(
        np.concatenate([np.repeat(np.arange(4), 400), np.zeros(3) + 4]).astype(int), n_classes=5
    )
    assert lopsided > 5 * balanced


def test_noise_band_matches_the_closed_form():
    labels = np.repeat(np.arange(4), 100)
    expected = 3.0 * float(np.sqrt(4 * (0.25 / 100)) / 4)
    assert chance_level_band(labels, n_classes=4) == pytest.approx(expected)


def test_noise_band_ignores_absent_classes():
    """A class with no test samples contributes no recall, so it must not widen the band."""
    labels = np.repeat(np.arange(3), 50)
    assert chance_level_band(labels, n_classes=5) == pytest.approx(
        chance_level_band(labels, n_classes=3)
    )


def test_noise_band_rejects_empty_labels():
    with pytest.raises(ValueError, match="no labelled samples"):
        chance_level_band(np.array([], dtype=int), n_classes=5)


def test_chance_level_follows_the_classes_actually_present():
    """A held-out insertion need not contain every region, and chance moves when it does not."""
    five = np.repeat(np.arange(5), 10)
    three = np.repeat([0, 3, 4], 10)
    assert chance_level(five, n_classes=5) == pytest.approx(0.2)
    assert chance_level(three, n_classes=5) == pytest.approx(1 / 3)


def test_classes_present_ignores_the_label_space_size():
    np.testing.assert_array_equal(classes_present(np.array([0, 4, 4]), 5), [0, 4])


def test_report_carries_the_chance_level_it_was_scored_against():
    labels = np.repeat([0, 3, 4], 20)
    probs = _onehot(labels)
    report = evaluate(probs, labels)
    assert report.chance == pytest.approx(1 / 3)
    assert report.classes_present == ["CA1", "DG", "VIS"]
    # Absent classes contribute no recall entry rather than a NaN.
    assert set(report.per_class_recall) == {"CA1", "DG", "VIS"}


def test_macro_f1_ignores_classes_absent_from_the_test_set():
    """Including a class with no test samples would understate the model for no good reason."""
    labels = np.repeat([0, 3, 4], 20)
    report = evaluate(_onehot(labels), labels)
    assert report.macro_f1 == pytest.approx(1.0)


def test_expand_probabilities_places_columns_at_their_global_index():
    """A model trained without CA2 returns four columns; they must not shift the others."""
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    expanded = expand_probabilities(probs, classes=np.array([0, 2, 3, 4]), n_classes=5)
    assert expanded.shape == (2, 5)
    np.testing.assert_allclose(expanded[:, 1], 0.0)
    np.testing.assert_allclose(expanded[:, [0, 2, 3, 4]], probs)
    np.testing.assert_allclose(expanded.sum(axis=1), 1.0)


def test_expand_probabilities_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="probability columns"):
        expand_probabilities(np.zeros((2, 3)), classes=np.array([0, 1]), n_classes=5)
    with pytest.raises(ValueError, match="exceeds"):
        expand_probabilities(np.zeros((2, 2)), classes=np.array([0, 9]), n_classes=5)


def test_expanded_probabilities_score_an_unseen_class_as_zero_recall():
    """The honest outcome for a class the model never saw, rather than a crash."""
    labels = np.array([0, 1, 1, 4])
    probs = expand_probabilities(
        np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]]),
        classes=np.array([0, 4]),
        n_classes=5,
    )
    report = evaluate(probs, labels)
    assert report.per_class_recall["CA2"] == 0.0
    assert report.per_class_recall["CA1"] == 1.0
