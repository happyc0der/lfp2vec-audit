"""What the signal adds to knowing where the electrode is.

Electrode position is the obvious competitor to any model that decodes anatomy from voltage, and
the obvious objection to using it as a baseline is that in practice position is only roughly
known. This module asks the question in the form that is useful to someone building such a
model, in three parts.

**Fusion.** How much does the signal add on top of honest position, fold by fold? The two are
combined as a product of experts: each source gets its own classifier, and their class
log-probabilities are added with the class prior subtracted once. Nothing is fitted on top, so
there is no stacking step to overfit and no hyper-parameter to tune.

**Where.** Along a probe, position should fail first at the boundaries between structures, since
the depth of a boundary varies from animal to animal while the middle of a large structure does
not move much. Accuracy is therefore reported against each channel's distance to the nearest
boundary on its own probe.

**Depth uncertainty.** Position features assume the depth of every contact is known in the frame
the training probes share. If the whole probe may sit deeper or shallower than believed by up to
some amount, how fast does position degrade, and at what uncertainty does the signal become the
better source? The position expert is trained with the same uncertainty it will face, so the
comparison is against the best a position-only model can do, not against a naive one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lfpaudit import REGION_TO_INDEX, REGIONS
from lfpaudit.eval.folds import Fold
from lfpaudit.eval.metrics import evaluate, expand_probabilities
from lfpaudit.eval.runner import CLASS_VIEWS
from lfpaudit.models.baselines import build_model

#: Edges, in micrometres, of the distance-to-boundary bins.
BOUNDARY_BINS: tuple[float, ...] = (0.0, 50.0, 100.0, 200.0, 400.0, np.inf)


def boundary_distance_um(index: pd.DataFrame) -> np.ndarray:
    """Distance from each chunk's channel to the nearest anatomical boundary on its probe.

    A boundary is any of: a change of region label between depth-adjacent kept channels; a gap in
    the kept channels wider than the probe's contact spacing, which marks tissue outside the five
    target regions; or either end of the kept span, beyond which the labels are out of scope by
    definition. Distances are measured along the shank.
    """
    for column in ("group", "channel", "depth_um", "region"):
        if column not in index.columns:
            raise ValueError(f"chunk index is missing {column!r}")

    distance = np.full(len(index), np.nan)
    groups = index["group"].to_numpy()
    depths_all = index["depth_um"].to_numpy(dtype=np.float64)

    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        channels = (
            index.iloc[rows][["depth_um", "region"]]
            .drop_duplicates()
            .groupby("depth_um", as_index=False)["region"]
            .first()
            .sort_values("depth_um")
        )
        depth = channels["depth_um"].to_numpy(dtype=np.float64)
        region = channels["region"].to_numpy()
        steps = np.diff(depth)
        spacing = float(np.median(steps[steps > 0])) if (steps > 0).any() else 20.0

        boundaries = [depth[0] - spacing / 2, depth[-1] + spacing / 2]
        for i in range(len(depth) - 1):
            if region[i] != region[i + 1] and steps[i] <= 1.5 * spacing:
                boundaries.append((depth[i] + depth[i + 1]) / 2)
            elif steps[i] > 1.5 * spacing:
                boundaries += [depth[i] + spacing / 2, depth[i + 1] - spacing / 2]
        boundaries = np.asarray(boundaries)

        distance[rows] = np.abs(depths_all[rows][:, None] - boundaries[None, :]).min(axis=1)
    return distance


def product_of_experts(log_probs: list[np.ndarray], log_prior: np.ndarray) -> np.ndarray:
    """Combine independent per-source class posteriors into one, as normalised probabilities.

    With sources assumed conditionally independent given the class, the fused posterior is the
    product of the individual posteriors divided by the prior once for every source beyond the
    first. Returned rows sum to one.
    """
    if not log_probs:
        raise ValueError("need at least one source")
    fused = np.sum(log_probs, axis=0) - (len(log_probs) - 1) * log_prior[None, :]
    fused -= fused.max(axis=1, keepdims=True)
    probs = np.exp(fused)
    return probs / probs.sum(axis=1, keepdims=True)


def _balanced_cap(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for value in np.unique(labels):
        candidates = np.flatnonzero(labels == value)
        chosen.append(rng.choice(candidates, size=min(len(candidates), per_class), replace=False))
    return np.sort(np.concatenate(chosen))


def _log(probs: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(probs, 1e-12))


def _fit_predict_log(train_x, train_y, test_x, seed: int) -> np.ndarray:
    model = build_model("logreg", seed=seed, n_train=len(train_x))
    model.fit(train_x, train_y)
    classes = np.asarray(model[-1].classes_, dtype=np.int64)
    return _log(expand_probabilities(model.predict_proba(test_x), classes, len(REGIONS)))


def jitter_depth(
    position: np.ndarray, groups: np.ndarray, half_width_um: float, rng: np.random.Generator
) -> np.ndarray:
    """Shift every probe rigidly along its shank by an independent uniform offset.

    Only depth moves, and every channel of a probe moves together, which is what an error in
    reading the insertion depth does. Lateral offset is untouched.
    """
    shifted = np.array(position, dtype=np.float64, copy=True)
    if half_width_um <= 0:
        return shifted
    for group in pd.unique(groups):
        shifted[groups == group, 0] += rng.uniform(-half_width_um, half_width_um)
    return shifted


@dataclass
class StudyInputs:
    """Everything the study reads, aligned row for row with the corpus index."""

    index: pd.DataFrame
    position: np.ndarray
    signals: dict[str, np.ndarray]
    view: str = "4class"
    per_class: int = 7500
    seed: int = 0
    labels: np.ndarray = field(init=False)
    row_of: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.labels = self.index["region"].map(REGION_TO_INDEX).to_numpy()
        self.row_of = {int(c): i for i, c in enumerate(self.index["chunk_id"])}
        for name, table in {"position": self.position, **self.signals}.items():
            if len(table) != len(self.index):
                raise ValueError(f"{name}: {len(table)} rows for an index of {len(self.index)}")

    def rows(self, chunk_ids: list[int]) -> np.ndarray:
        rows = np.array([self.row_of[i] for i in chunk_ids], dtype=np.int64)
        allowed = [REGION_TO_INDEX[name] for name in CLASS_VIEWS[self.view]]
        return rows[np.isin(self.labels[rows], allowed)]


def _train_key(rows: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(rows).tobytes()).hexdigest()


def _score(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    report = evaluate(probs, truth)
    counts = np.bincount(truth, minlength=len(REGIONS))
    return {
        "balanced_accuracy": report.balanced_accuracy,
        "raw_accuracy": report.accuracy,
        "majority": float(counts.max() / counts.sum()),
        "chance": report.chance,
        "ece": report.ece,
    }


def fusion_study(
    schemes: dict[str, list[Fold]], inputs: StudyInputs, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Position, each signal, and each signal fused with position, on every fold.

    Returns the per-fold score table and a per-chunk table of which sources were right, which is
    what the boundary analysis reads. Folds that share a training set, as every fold of a
    cross-lab scheme does, share fitted experts.
    """
    rng = np.random.default_rng(inputs.seed)
    score_rows, chunk_rows = [], []
    cache: dict[tuple[str, str], tuple] = {}

    for scheme, folds in schemes.items():
        for fold in folds:
            train = inputs.rows(fold.split.train)
            test = inputs.rows(fold.split.test)
            if len(test) == 0 or len(np.unique(inputs.labels[train])) < 2:
                continue
            key = _train_key(train)
            if (key, "__sample__") not in cache:
                cache[(key, "__sample__")] = (
                    train[_balanced_cap(inputs.labels[train], inputs.per_class, rng)],
                )
            sample = cache[(key, "__sample__")][0]
            train_y, truth = inputs.labels[sample], inputs.labels[test]
            counts = np.bincount(train_y, minlength=len(REGIONS)).astype(np.float64)
            log_prior = _log(counts / counts.sum())

            sources = {"position": inputs.position, **inputs.signals}
            log_p: dict[str, np.ndarray] = {}
            for name, table in sources.items():
                if (key, name) not in cache:
                    model = build_model("logreg", seed=inputs.seed, n_train=len(sample))
                    model.fit(table[sample], train_y)
                    cache[(key, name)] = (model,)
                model = cache[(key, name)][0]
                classes = np.asarray(model[-1].classes_, dtype=np.int64)
                log_p[name] = _log(
                    expand_probabilities(model.predict_proba(table[test]), classes, len(REGIONS))
                )

            combos = {name: [name] for name in sources}
            combos.update({f"position+{name}": ["position", name] for name in inputs.signals})
            correct = {}
            for combo, members in combos.items():
                probs = product_of_experts([log_p[m] for m in members], log_prior)
                score_rows.append(
                    {
                        "scheme": scheme,
                        "fold": fold.name,
                        "source": combo,
                        "n_test": len(test),
                        **_score(probs, truth),
                    }
                )
                correct[combo] = probs.argmax(axis=1) == truth
            frame = pd.DataFrame({"scheme": scheme, "fold": fold.name, "row": test})
            for combo, hits in correct.items():
                frame[f"correct::{combo}"] = hits
            chunk_rows.append(frame)
            if verbose:
                print(f"  {scheme} / {fold.name}: done", flush=True)

    return pd.DataFrame(score_rows), pd.concat(chunk_rows, ignore_index=True)


def boundary_table(chunks: pd.DataFrame, distance: np.ndarray) -> pd.DataFrame:
    """Accuracy of every source against distance to the nearest boundary, per scheme."""
    chunks = chunks.assign(distance=distance[chunks["row"].to_numpy()])
    labels = [
        f"{int(lo)}–{int(hi)} µm" if np.isfinite(hi) else f"≥{int(lo)} µm"
        for lo, hi in zip(BOUNDARY_BINS[:-1], BOUNDARY_BINS[1:], strict=True)
    ]
    chunks["bin"] = pd.cut(chunks["distance"], bins=list(BOUNDARY_BINS), labels=labels, right=False)
    sources = [c for c in chunks.columns if c.startswith("correct::")]
    rows = []
    for (scheme, bin_label), part in chunks.groupby(["scheme", "bin"], observed=True):
        record = {"scheme": scheme, "bin": bin_label, "n": len(part)}
        record.update({s.split("::", 1)[1]: float(part[s].mean()) for s in sources})
        rows.append(record)
    return pd.DataFrame(rows)


def depth_uncertainty_study(
    schemes: dict[str, list[Fold]],
    inputs: StudyInputs,
    signal: str,
    half_widths_um: tuple[float, ...] = (0.0, 100.0, 200.0, 400.0, 800.0, 1600.0),
    train_copies: int = 4,
    test_draws: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Balanced accuracy of position, one signal, and their fusion as depth becomes uncertain.

    For each half-width the position expert is refit on training probes that were themselves
    shifted by offsets from the same distribution, several copies each, so it has learned how
    much to trust depth. The held-out probe is then shifted ``test_draws`` times and the scores
    averaged. The signal expert never sees depth, so it is fitted once per fold.
    """
    rng = np.random.default_rng(inputs.seed)
    groups_all = inputs.index["group"].to_numpy()
    rows_out = []

    for scheme, folds in schemes.items():
        for fold in folds:
            train = inputs.rows(fold.split.train)
            test = inputs.rows(fold.split.test)
            if len(test) == 0 or len(np.unique(inputs.labels[train])) < 2:
                continue
            sample = train[_balanced_cap(inputs.labels[train], inputs.per_class, rng)]
            train_y, truth = inputs.labels[sample], inputs.labels[test]
            counts = np.bincount(train_y, minlength=len(REGIONS)).astype(np.float64)
            log_prior = _log(counts / counts.sum())
            table = inputs.signals[signal]
            log_signal = _fit_predict_log(table[sample], train_y, table[test], inputs.seed)
            signal_score = _score(product_of_experts([log_signal], log_prior), truth)

            for width in half_widths_um:
                copies = 1 if width <= 0 else train_copies
                train_x = np.concatenate(
                    [
                        jitter_depth(inputs.position[sample], groups_all[sample], width, rng)
                        for _ in range(copies)
                    ]
                )
                model = build_model("logreg", seed=inputs.seed, n_train=len(train_x))
                model.fit(train_x, np.tile(train_y, copies))
                classes = np.asarray(model[-1].classes_, dtype=np.int64)

                draws = 1 if width <= 0 else test_draws
                position_scores, fused_scores = [], []
                for _ in range(draws):
                    test_x = jitter_depth(inputs.position[test], groups_all[test], width, rng)
                    log_position = _log(
                        expand_probabilities(model.predict_proba(test_x), classes, len(REGIONS))
                    )
                    position_scores.append(
                        _score(product_of_experts([log_position], log_prior), truth)[
                            "balanced_accuracy"
                        ]
                    )
                    fused_scores.append(
                        _score(product_of_experts([log_position, log_signal], log_prior), truth)[
                            "balanced_accuracy"
                        ]
                    )
                rows_out.append(
                    {
                        "scheme": scheme,
                        "fold": fold.name,
                        "half_width_um": width,
                        "position": float(np.mean(position_scores)),
                        signal: signal_score["balanced_accuracy"],
                        f"position+{signal}": float(np.mean(fused_scores)),
                        "chance": signal_score["chance"],
                    }
                )
            if verbose:
                print(f"  depth uncertainty: {scheme} / {fold.name} done", flush=True)
    return pd.DataFrame(rows_out)
