"""Temperature scaling: the standard repair for overconfidence, and a test of whether it reaches.

The paper's Broader Impact section says clinical use of this kind of model should include
calibrated uncertainty estimates. Temperature scaling is the usual way to provide them: divide
the logits by one scalar, fitted on held-out in-distribution data, so that confidence matches
accuracy. It works when a model is systematically too sure of predictions that are otherwise
reasonable.

Stage 3 found something different. Across labs the model is at chance while reporting 0.98
confidence, because the inputs fall in a region of representation space its decision boundary
never saw. A scalar cannot move a boundary. The prediction this module tests is that a temperature
fitted in one lab repairs that lab and does nothing for the other, which would mean the proposed
remedy does not reach the stated risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from lfpaudit.eval.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    softmax,
)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """The scalar that minimises negative log-likelihood on held-out logits.

    Searched on a log scale between 0.05 and 20, which comfortably covers anything a classifier
    of this size produces: temperatures near 1 mean the model was already calibrated, well above
    1 that it was overconfident, below 1 that it was hedging.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logit rows for {len(labels)} labels")
    if len(np.unique(labels)) < 2:
        raise ValueError("need at least two classes to fit a temperature")

    def objective(log_t: float) -> float:
        return negative_log_likelihood(softmax(logits, temperature=np.exp(log_t)), labels)

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


@dataclass
class CalibrationReport:
    """Calibration before and after one temperature, on one set of logits."""

    n_samples: int
    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    accuracy: float
    reliability_before: dict[str, list[float]]
    reliability_after: dict[str, list[float]]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, label: str) -> str:
        return (
            f"{label}: T={self.temperature:.2f} "
            f"ECE {self.ece_before:.3f} -> {self.ece_after:.3f}, "
            f"NLL {self.nll_before:.3f} -> {self.nll_after:.3f}, "
            f"confidence {self.mean_confidence_before:.3f} -> {self.mean_confidence_after:.3f} "
            f"at accuracy {self.accuracy:.3f}"
        )


def calibration_report(
    logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int = 15
) -> CalibrationReport:
    """Score one set of logits before and after dividing by ``temperature``."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    before = softmax(logits)
    after = softmax(logits, temperature=temperature)
    return CalibrationReport(
        n_samples=int(len(labels)),
        temperature=float(temperature),
        ece_before=expected_calibration_error(before, labels, n_bins=n_bins),
        ece_after=expected_calibration_error(after, labels, n_bins=n_bins),
        nll_before=negative_log_likelihood(before, labels),
        nll_after=negative_log_likelihood(after, labels),
        brier_before=brier_score(before, labels),
        brier_after=brier_score(after, labels),
        mean_confidence_before=float(before.max(axis=1).mean()),
        mean_confidence_after=float(after.max(axis=1).mean()),
        accuracy=float((before.argmax(axis=1) == labels).mean()),
        reliability_before=reliability_curve(before, labels, n_bins=n_bins),
        reliability_after=reliability_curve(after, labels, n_bins=n_bins),
    )
