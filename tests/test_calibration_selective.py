"""Tests for temperature scaling and abstention."""

from __future__ import annotations

import numpy as np
import pytest

from lfpaudit.eval.calibration import calibration_report, fit_temperature
from lfpaudit.eval.metrics import softmax
from lfpaudit.eval.selective import (
    MahalanobisScorer,
    confidence_scores,
    risk_coverage,
    separation_auc,
)


def _logits(n=3000, n_classes=4, separation=2.5, seed=0):
    """Logits from a model that is right most of the time, with a known true class."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    values = rng.normal(size=(n, n_classes))
    values[np.arange(n), labels] += separation
    return values, labels


class TestTemperature:
    @pytest.mark.parametrize("factor", [0.4, 2.0])
    def test_recovers_a_planted_temperature(self, factor):
        """Dividing logits by c must scale the optimal temperature by exactly 1/c.

        Verified against the implementation to four decimals; the first version of this test had
        the relationship inverted and expected ``base * c``.
        """
        values, labels = _logits(seed=1)
        base = fit_temperature(values, labels)
        assert fit_temperature(values / factor, labels) == pytest.approx(base / factor, rel=0.02)

    def test_overconfident_logits_need_a_temperature_above_one(self):
        values, labels = _logits(seed=2)
        assert fit_temperature(values * 4.0, labels) > 1.5

    def test_fitting_reduces_calibration_error_in_distribution(self):
        values, labels = _logits(seed=3)
        overconfident = values * 4.0
        temperature = fit_temperature(overconfident, labels)
        report = calibration_report(overconfident, labels, temperature)
        assert report.ece_after < report.ece_before
        assert report.nll_after <= report.nll_before

    def test_scaling_never_changes_the_prediction(self):
        """Temperature moves confidence, not the argmax, so accuracy must be untouched."""
        values, labels = _logits(seed=4)
        report = calibration_report(values, labels, temperature=3.7)
        assert report.accuracy == pytest.approx((values.argmax(axis=1) == labels).mean())

    def test_higher_temperature_lowers_confidence(self):
        values, labels = _logits(seed=5)
        report = calibration_report(values, labels, temperature=5.0)
        assert report.mean_confidence_after < report.mean_confidence_before

    def test_report_carries_both_reliability_curves(self):
        values, labels = _logits(seed=6)
        report = calibration_report(values, labels, temperature=2.0)
        assert sum(report.reliability_before["count"]) == len(labels)
        assert sum(report.reliability_after["count"]) == len(labels)
        assert "T=" in report.summary_line("demo")

    def test_rejects_mismatched_or_degenerate_input(self):
        values, labels = _logits(seed=7)
        with pytest.raises(ValueError, match="logit rows"):
            fit_temperature(values, labels[:10])
        with pytest.raises(ValueError, match="at least two classes"):
            fit_temperature(values[:50], np.zeros(50, dtype=int))


class TestConfidenceScores:
    def test_both_scores_increase_with_uncertainty(self):
        confident = np.array([[10.0, 0.0, 0.0, 0.0]])
        unsure = np.array([[0.1, 0.0, 0.0, 0.0]])
        scores = confidence_scores(np.vstack([confident, unsure]))
        for name in ("max_softmax", "entropy"):
            assert scores[name][1] > scores[name][0], name

    def test_entropy_is_maximal_for_a_uniform_prediction(self):
        uniform = confidence_scores(np.zeros((1, 4)))["entropy"][0]
        assert uniform == pytest.approx(np.log(4), rel=1e-6)


class TestRiskCoverage:
    def test_a_perfect_score_removes_every_error_first(self):
        """A score ranking every error above every correct answer keeps risk at zero until the
        errors are all that is left.

        Its area is not zero: past the point where only errors remain the curve must rise to the
        base error rate, which at 20% errors contributes about 0.021. What matters is that it is
        an order of magnitude below the base rate.
        """
        correct = np.array([True] * 80 + [False] * 20)
        uncertainty = np.concatenate([np.zeros(80), np.ones(20)])
        curve = risk_coverage(uncertainty, correct, "oracle")
        assert curve.risk_at_half == pytest.approx(0.0)
        assert curve.risk_at_fifth == pytest.approx(0.0)
        assert curve.risk_at_full == pytest.approx(0.2)
        assert curve.area < 0.2 / 4

    def test_an_uninformative_score_stays_at_the_base_error_rate(self):
        rng = np.random.default_rng(0)
        correct = rng.random(4000) > 0.3
        curve = risk_coverage(rng.random(4000), correct, "random")
        assert curve.area == pytest.approx(1 - correct.mean(), abs=0.03)

    def test_risk_at_full_coverage_is_the_overall_error_rate(self):
        correct = np.array([True, False, True, True])
        curve = risk_coverage(np.arange(4.0), correct, "any")
        assert curve.risk_at_full == pytest.approx(0.25)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="scores for"):
            risk_coverage(np.zeros(5), np.ones(3, dtype=bool), "bad")

    def test_summary_line_mentions_the_area(self):
        curve = risk_coverage(np.arange(10.0), np.ones(10, dtype=bool), "s")
        assert "AURC" in curve.summary_line()


class TestMahalanobis:
    def test_separates_a_shifted_cluster(self):
        """The whole premise: distance from training data should flag shifted inputs."""
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(2000, 16))
        shifted = rng.normal(size=(2000, 16)) + 4.0
        scorer = MahalanobisScorer().fit(reference)
        assert separation_auc(scorer.score(reference), scorer.score(shifted)) > 0.99

    def test_scores_an_identical_distribution_at_chance(self):
        rng = np.random.default_rng(1)
        scorer = MahalanobisScorer().fit(rng.normal(size=(2000, 16)))
        first = scorer.score(rng.normal(size=(1000, 16)))
        second = scorer.score(rng.normal(size=(1000, 16)))
        assert separation_auc(first, second) == pytest.approx(0.5, abs=0.05)

    def test_distances_are_non_negative_and_finite(self):
        rng = np.random.default_rng(2)
        scorer = MahalanobisScorer().fit(rng.normal(size=(500, 8)))
        scores = scorer.score(rng.normal(size=(100, 8)))
        assert (scores >= 0).all()
        assert np.isfinite(scores).all()

    def test_scoring_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            MahalanobisScorer().score(np.zeros((3, 4)))

    def test_too_small_a_reference_is_rejected(self):
        with pytest.raises(ValueError, match="too small"):
            MahalanobisScorer().fit(np.zeros((5, 256)))


def test_separation_auc_is_symmetric_around_a_half():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=500), rng.normal(size=500) + 2
    assert separation_auc(a, b) == pytest.approx(1 - separation_auc(b, a), abs=1e-9)


def test_softmax_temperature_matches_calibration_path():
    """Guard that the two places temperature is applied agree."""
    values, _ = _logits(seed=8)
    np.testing.assert_allclose(softmax(values, temperature=2.0), softmax(values / 2.0))
