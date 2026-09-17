"""Classification and calibration metrics.

Accuracy alone cannot answer the question this repository asks. A model can keep most of its
accuracy under a distribution shift while its confidence becomes meaningless, and for a tool
meant to guide electrode placement the second failure is the dangerous one. Every evaluation
therefore reports discrimination (balanced accuracy, macro-F1) alongside calibration
(negative log-likelihood, Brier score, expected calibration error).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from lfpaudit import REGIONS


def _as_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"expected (n_samples, n_classes) probabilities, got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("probability rows must sum to 1; pass softmax outputs, not logits")
    return probs


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with an optional temperature."""
    logits = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of the true class. Lower is better; sensitive to confident mistakes."""
    probs = _as_probabilities(probs)
    labels = np.asarray(labels, dtype=np.int64)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error against the one-hot target."""
    probs = _as_probabilities(probs)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    ``adaptive=True`` uses equal-mass bins instead of equal-width ones, which avoids the
    well-known sensitivity of fixed-width ECE to a pile-up of predictions near confidence 1.
    """
    probs = _as_probabilities(probs)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels

    if adaptive:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(confidence, quantiles))
        if len(edges) < 2:
            return float(abs(confidence.mean() - correct.mean()))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        # Include the upper edge in the final bin so confidence exactly 1.0 is never dropped.
        in_bin = (
            (confidence > low) & (confidence <= high)
            if low > edges[0]
            else (confidence >= low) & (confidence <= high)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        total += (count / len(confidence)) * gap
    return float(total)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, list[float]]:
    """Per-bin confidence, accuracy and count, for plotting a reliability diagram from results."""
    probs = _as_probabilities(probs)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    out: dict[str, list[float]] = {
        "bin_lower": [],
        "bin_upper": [],
        "confidence": [],
        "accuracy": [],
        "count": [],
    }
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (
            (confidence > low) & (confidence <= high)
            if low > 0
            else (confidence >= low) & (confidence <= high)
        )
        count = int(in_bin.sum())
        out["bin_lower"].append(float(low))
        out["bin_upper"].append(float(high))
        out["count"].append(float(count))
        out["confidence"].append(float(confidence[in_bin].mean()) if count else float("nan"))
        out["accuracy"].append(float(correct[in_bin].mean()) if count else float("nan"))
    return out


@dataclass
class ClassificationReport:
    """Everything reported for one (model, split) pair."""

    n_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    ece_adaptive: float
    confusion: list[list[int]]
    class_names: list[str]
    per_class_recall: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"n={self.n_samples} bal_acc={self.balanced_accuracy:.3f} "
            f"macro_f1={self.macro_f1:.3f} ece={self.ece:.3f} nll={self.nll:.3f}"
        )


def evaluate(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 15,
) -> ClassificationReport:
    """Compute the full report from predicted probabilities and integer labels."""
    probs = _as_probabilities(probs)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probs) != len(labels):
        raise ValueError(f"probs and labels disagree: {len(probs)} vs {len(labels)}")
    names = list(class_names or REGIONS[: probs.shape[1]])
    predicted = probs.argmax(axis=1)
    present = np.arange(len(names))

    matrix = confusion_matrix(labels, predicted, labels=present)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(matrix) / matrix.sum(axis=1)

    return ClassificationReport(
        n_samples=int(len(labels)),
        accuracy=float((predicted == labels).mean()),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        macro_f1=float(
            f1_score(labels, predicted, average="macro", labels=present, zero_division=0)
        ),
        nll=negative_log_likelihood(probs, labels),
        brier=brier_score(probs, labels),
        ece=expected_calibration_error(probs, labels, n_bins=n_bins),
        ece_adaptive=expected_calibration_error(probs, labels, n_bins=n_bins, adaptive=True),
        confusion=matrix.astype(int).tolist(),
        class_names=names,
        per_class_recall={
            name: (float(r) if np.isfinite(r) else float("nan"))
            for name, r in zip(names, recall, strict=True)
        },
    )
