"""The experiment loop: every feature set, every model, every fold, with its own control.

Two design choices here are load-bearing.

First, each configuration is run twice, once normally and once with the training labels permuted.
A control computed on some other fold, or once for the whole sweep, would not be a control: folds
differ enormously in class balance and size, and what counts as chance moves with them. Running
the permutation alongside means every row in the results table carries its own null.

Second, nothing is aggregated in memory. Each configuration appends a row to a table and, where
a later stage needs them, writes its probabilities to disk. Every number in the repository is
therefore reproducible from files rather than from a session that has since ended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lfpaudit import REGION_TO_INDEX, REGIONS
from lfpaudit.eval.folds import Fold
from lfpaudit.eval.metrics import chance_level_band, evaluate
from lfpaudit.models.baselines import fit_predict

#: Views on the label space. CA2 exists on four channels in two probes of one dataset, so a
#: macro-averaged score including it is dominated by a class almost nothing can learn.
CLASS_VIEWS: dict[str, tuple[str, ...]] = {
    "4class": ("CA1", "CA3", "DG", "VIS"),
    "5class": REGIONS,
}


@dataclass
class ExperimentSpec:
    """One sweep: which features, models, schemes and views to cross."""

    feature_sets: list[str]
    models: list[str] = field(default_factory=lambda: ["constant", "logreg"])
    views: list[str] = field(default_factory=lambda: ["4class", "5class"])
    seed: int = 0
    save_predictions_for: list[str] = field(default_factory=list)


def _view_mask(labels: np.ndarray, view: str) -> np.ndarray:
    allowed = {REGION_TO_INDEX[name] for name in CLASS_VIEWS[view]}
    return np.isin(labels, list(allowed))


def run_fold(
    fold: Fold,
    features: np.ndarray,
    labels: np.ndarray,
    positions: dict[int, int],
    feature_set: str,
    model_name: str,
    view: str,
    seed: int = 0,
    predictions_dir: Path | None = None,
) -> list[dict]:
    """Fit and score one configuration on one fold, with its permutation control.

    Returns one row per run, the model and its control. Rows are plain dictionaries so the caller
    can build a table without this module knowing what the table looks like.
    """
    rows: list[dict] = []
    train_rows = np.array([positions[i] for i in fold.split.train], dtype=np.int64)
    test_rows = np.array([positions[i] for i in fold.split.test], dtype=np.int64)

    keep_train = _view_mask(labels[train_rows], view)
    keep_test = _view_mask(labels[test_rows], view)
    train_rows, test_rows = train_rows[keep_train], test_rows[keep_test]
    if len(test_rows) == 0 or len(np.unique(labels[train_rows])) < 2:
        # A probe that contributes no class in this view, or a training set with a single class,
        # is skipped rather than scored: there is nothing to measure, and a fabricated zero would
        # be averaged into the scheme's mean as though it were a failure.
        return rows

    train_x, train_y = features[train_rows], labels[train_rows]
    test_x, test_y = features[test_rows], labels[test_rows]
    rng = np.random.default_rng(seed)

    for control, fit_labels in (("model", train_y), ("permuted", rng.permutation(train_y))):
        probs = fit_predict(model_name, train_x, fit_labels, test_x, seed=seed)
        report = evaluate(probs, test_y)
        band = chance_level_band(test_y)
        rows.append(
            {
                "scheme": fold.scheme,
                "fold": fold.name,
                "view": view,
                "features": feature_set,
                "model": model_name,
                "control": control,
                "seed": seed,
                "n_train": int(len(train_rows)),
                "n_test": int(len(test_rows)),
                "chance": report.chance,
                "noise_band": band,
                "balanced_accuracy": report.balanced_accuracy,
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
                "nll": report.nll,
                "brier": report.brier,
                "ece": report.ece,
                "ece_adaptive": report.ece_adaptive,
                "classes_present": "|".join(report.classes_present),
                "above_chance": report.balanced_accuracy - report.chance,
                **{f"recall_{k}": v for k, v in report.per_class_recall.items()},
            }
        )

        # Probabilities are kept only for the real models, not the controls: Stage 4 recalibrates
        # and ablates actual predictions, and a permuted-label model has nothing to recalibrate.
        if predictions_dir is not None and control == "model":
            predictions_dir.mkdir(parents=True, exist_ok=True)
            name = f"{fold.scheme}__{view}__{feature_set}__{model_name}__{fold.name}.parquet"
            frame = pd.DataFrame(
                {"chunk_id": np.asarray(fold.split.test)[keep_test], "label": test_y}
            )
            for i, region in enumerate(REGIONS):
                frame[f"p_{region}"] = probs[:, i].astype(np.float16)
            frame.to_parquet(predictions_dir / name, index=False)

    return rows


def run_sweep(
    schemes: dict[str, list[Fold]],
    feature_tables: dict[str, np.ndarray],
    labels: np.ndarray,
    positions: dict[int, int],
    spec: ExperimentSpec,
    predictions_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every combination in ``spec`` across every fold of every scheme."""
    rows: list[dict] = []
    for scheme_name, folds in schemes.items():
        for feature_set in spec.feature_sets:
            features = feature_tables[feature_set]
            for model_name in spec.models:
                for view in spec.views:
                    save_to = (
                        predictions_dir
                        if predictions_dir is not None and scheme_name in spec.save_predictions_for
                        else None
                    )
                    for fold in folds:
                        rows.extend(
                            run_fold(
                                fold,
                                features,
                                labels,
                                positions,
                                feature_set=feature_set,
                                model_name=model_name,
                                view=view,
                                seed=spec.seed,
                                predictions_dir=save_to,
                            )
                        )
                    if verbose:
                        print(
                            f"  {scheme_name} / {feature_set} / {model_name} / {view}: "
                            f"{len(folds)} folds",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across folds for each configuration."""
    keys = ["scheme", "view", "features", "model", "control"]
    grouped = table.groupby(keys)
    summary = grouped.agg(
        folds=("fold", "count"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        above_chance=("above_chance", "mean"),
        macro_f1=("macro_f1", "mean"),
        ece=("ece", "mean"),
        nll=("nll", "mean"),
        chance=("chance", "mean"),
        n_test=("n_test", "sum"),
    ).reset_index()
    return summary.sort_values(
        ["scheme", "view", "balanced_accuracy"], ascending=[True, True, False]
    )


def paired_comparison(
    table: pd.DataFrame, scheme: str, view: str, model: str = "logreg"
) -> pd.DataFrame:
    """Compare feature sets against each other on the folds they share.

    Folds differ so much in difficulty that an unpaired comparison of means is close to
    meaningless: a feature set can look worse purely by being averaged over harder sessions. The
    Wilcoxon signed-rank test asks the question that matters instead, which is how often one
    feature set beats another on the same held-out session.
    """
    from scipy.stats import wilcoxon

    subset = table[
        (table["scheme"] == scheme)
        & (table["view"] == view)
        & (table["model"] == model)
        & (table["control"] == "model")
    ]
    wide = subset.pivot_table(index="fold", columns="features", values="balanced_accuracy")
    names = sorted(wide.columns)

    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            pair = wide[[first, second]].dropna()
            if len(pair) < 3:
                continue
            difference = pair[first] - pair[second]
            if np.allclose(difference, 0):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = wilcoxon(pair[first], pair[second])
            rows.append(
                {
                    "scheme": scheme,
                    "view": view,
                    "a": first,
                    "b": second,
                    "folds": len(pair),
                    "median_difference": float(difference.median()),
                    "a_wins": int((difference > 0).sum()),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def write_results(table: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Write the per-fold table and its summary as CSV.

    CSV rather than parquet on purpose: these are the numbers the repository stands on, they are
    small, and ``.gitignore`` excludes parquet from results so a parquet summary would never be
    committed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": out_dir / "folds.csv",
        "summary": out_dir / "summary.csv",
    }
    table.to_csv(paths["folds"], index=False)
    summarise(table).to_csv(paths["summary"], index=False)
    (out_dir / "views.json").write_text(json.dumps({k: list(v) for k, v in CLASS_VIEWS.items()}))
    return paths
