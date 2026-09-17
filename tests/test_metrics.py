import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from lfpaudit.eval.metrics import (
    brier_score,
    evaluate,
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
