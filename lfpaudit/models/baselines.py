"""The models that decide whether a later result means anything.

None of these is meant to win. They exist to make a number from a large model interpretable by
bracketing it. A constant predictor says what the task gives away for free. Position and
amplitude say how much is available without looking at the signal. Band power says what six
interpretable numbers per chunk can do, which is the claim LFP-LOC makes against LFP2Vec. Frozen
audio embeddings say what the pretrained representation supplies before any brain data is seen.

Every model returns probabilities in the full five-class space, not in whatever subset it was
trained on, because the IBL recordings contain no CA2 and a classifier trained on them would
otherwise return columns that silently mean the wrong classes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lfpaudit import REGIONS
from lfpaudit.eval.metrics import expand_probabilities

MODEL_NAMES: tuple[str, ...] = ("constant", "logreg", "mlp")


class ConstantClassifier:
    """Predicts the training class distribution, ignoring the features entirely.

    The honest floor. Its balanced accuracy sits at chance by construction, but its calibration
    does not have to be bad: a model that reports the base rates is perfectly calibrated in the
    aggregate while being useless. That contrast is worth having in the table, because it shows
    that low calibration error alone is not evidence of a good model.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self._frequencies: np.ndarray = np.array([])

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ConstantClassifier:
        labels = np.asarray(labels, dtype=np.int64)
        self.classes_, counts = np.unique(labels, return_counts=True)
        self._frequencies = counts / counts.sum()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(self._frequencies, (len(features), 1))


#: Below this many training rows the neural model stops holding out a validation split. Measured
#: on separable synthetic data: with early stopping enabled it scores 0.74 at 400 rows and 1.00 at
#: 1600, because a tenth of a small training set is too noisy a stopping signal. Folds do vary
#: this much in practice once a class view is applied, and a model that underperforms for
#: optimisation reasons would be indistinguishable in the results table from one that lacks signal.
MIN_ROWS_FOR_EARLY_STOPPING = 2000


def build_model(name: str, seed: int = 0, n_train: int | None = None):
    """Construct one of :data:`MODEL_NAMES`.

    The linear and neural models are wrapped in a standardiser because the feature sets differ by
    orders of magnitude in scale: depth is in micrometres, log band power is a small negative
    number, and wav2vec2 activations are something else again. Without scaling, comparisons
    between feature sets would partly measure their units.
    """
    if name == "constant":
        return ConstantClassifier()
    if name == "logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=seed)),
            ]
        )
    if name == "mlp":
        early = n_train is None or n_train >= MIN_ROWS_FOR_EARLY_STOPPING
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=400 if early else 1000,
                        early_stopping=early,
                        n_iter_no_change=10,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def _classes_of(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=np.int64)
    return np.asarray(model[-1].classes_, dtype=np.int64)


def fit_predict(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 0,
    n_classes: int = len(REGIONS),
) -> np.ndarray:
    """Fit one model and return test probabilities in the full label space.

    A class absent from the training labels gets a zero column rather than being omitted, so the
    caller can score every model against the same label indices regardless of what each one
    happened to see.
    """
    if len(train_x) != len(train_y):
        raise ValueError(f"{len(train_x)} training rows for {len(train_y)} labels")
    if len(np.unique(train_y)) < 2:
        raise ValueError("training labels contain fewer than two classes")

    model = build_model(name, seed=seed, n_train=len(train_x))
    model.fit(train_x, np.asarray(train_y, dtype=np.int64))
    return expand_probabilities(model.predict_proba(test_x), _classes_of(model), n_classes)
