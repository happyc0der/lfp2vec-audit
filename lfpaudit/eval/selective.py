"""Abstention: can the model tell when not to trust itself?

For a tool meant to guide electrode placement, knowing when a prediction is unreliable may matter
more than a small gain in average accuracy. Stage 3 showed the obvious signal is useless: across
labs the model is at chance while reporting 0.98 confidence, so any refusal rule built on
confidence refuses nothing. It also showed the model's own embeddings identify the source lab at
0.997, which suggests a different signal: distance from what the model was trained on.

Three scores are compared here on identical footing. Maximum softmax and predictive entropy read
the output. Mahalanobis distance reads the representation, measured against the in-lab validation
embeddings that the model saw the distribution of but never trained on. Each is evaluated two
ways: how well ranking by it removes errors, as a risk–coverage curve, and how well it separates
in-lab from cross-lab inputs at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from lfpaudit.eval.metrics import softmax


def confidence_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Output-based uncertainty scores, oriented so that higher means more uncertain."""
    probs = softmax(np.asarray(logits, dtype=np.float64))
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return {"max_softmax": 1.0 - probs.max(axis=1), "entropy": entropy}


class MahalanobisScorer:
    """Distance from the in-distribution embedding cloud, with a shrinkage covariance.

    Fit on embeddings the model has seen the distribution of but not trained on, so the reference
    describes what in-lab inputs look like to this model rather than what it memorised.
    Ledoit–Wolf shrinkage keeps the covariance invertible and stable at 256 dimensions with a few
    thousand reference points.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> MahalanobisScorer:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or len(reference) < reference.shape[1] // 4:
            raise ValueError(
                f"reference {reference.shape} is too small for a {reference.shape[-1]}-d covariance"
            )
        estimator = LedoitWolf().fit(reference)
        self.mean = estimator.location_
        self.precision = estimator.precision_
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("scorer has not been fitted")
        delta = np.asarray(embeddings, dtype=np.float64) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, self.precision, delta))


@dataclass
class RiskCoverage:
    """One score's risk–coverage curve and its summary numbers."""

    score_name: str
    coverage: list[float]
    risk: list[float]
    area: float
    risk_at_full: float
    risk_at_half: float
    risk_at_fifth: float

    def summary_line(self) -> str:
        return (
            f"{self.score_name}: AURC {self.area:.3f}, error {self.risk_at_full:.3f} at full "
            f"coverage, {self.risk_at_half:.3f} at half, {self.risk_at_fifth:.3f} at a fifth"
        )


def risk_coverage(
    uncertainty: np.ndarray, correct: np.ndarray, score_name: str, n_points: int = 100
) -> RiskCoverage:
    """Error rate of the retained set as more uncertain predictions are dropped first.

    Coverage runs from keeping everything down to keeping the most confident few. The area under
    the curve is the usual single-number summary: zero for a score that ranks every error above
    every correct prediction, the overall error rate for a score that carries no information.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if len(uncertainty) != len(correct):
        raise ValueError(f"{len(uncertainty)} scores for {len(correct)} outcomes")
    n = len(correct)
    order = np.argsort(uncertainty, kind="stable")  # most certain first
    errors = ~correct[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, n + 1)

    fractions = np.linspace(1.0 / n, 1.0, n_points)
    kept = np.maximum(1, np.round(fractions * n).astype(int))
    risk_curve = cumulative_risk[kept - 1]

    def risk_at(fraction: float) -> float:
        return float(cumulative_risk[max(1, int(round(fraction * n))) - 1])

    return RiskCoverage(
        score_name=score_name,
        coverage=fractions.tolist(),
        risk=risk_curve.tolist(),
        area=float(np.trapezoid(risk_curve, fractions)),
        risk_at_full=risk_at(1.0),
        risk_at_half=risk_at(0.5),
        risk_at_fifth=risk_at(0.2),
    )


def separation_auc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """How well a score separates in-distribution from shifted inputs; 0.5 is no separation."""
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([np.asarray(in_scores), np.asarray(out_scores)])
    return float(roc_auc_score(labels, scores))
