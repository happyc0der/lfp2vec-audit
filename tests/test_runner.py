"""Tests for the experiment loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfpaudit import REGION_TO_INDEX
from lfpaudit.eval.folds import build_schemes, leave_one_group_out
from lfpaudit.eval.runner import (
    CLASS_VIEWS,
    ExperimentSpec,
    paired_comparison,
    run_fold,
    run_sweep,
    summarise,
    write_results,
)
from lfpaudit.features.bandpower import band_power


@pytest.fixture
def corpus_pieces(synthetic_store):
    """Band-power features, labels and positions for the synthetic store."""
    index = synthetic_store.index
    features = band_power(synthetic_store.take(np.arange(len(index))), fs=synthetic_store.fs)
    labels = index["region"].map(REGION_TO_INDEX).to_numpy()
    positions = {int(cid): i for i, cid in enumerate(index["chunk_id"])}
    return index, features, labels, positions


def test_run_fold_returns_a_model_and_its_control(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    rows = run_fold(fold, features, labels, positions, "bandpower", "logreg", "5class")
    assert [r["control"] for r in rows] == ["model", "permuted"]
    assert rows[0]["fold"] == fold.name


def test_the_control_is_computed_on_the_same_fold(corpus_pieces):
    """A control from a different fold is not a control; folds differ too much."""
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    rows = run_fold(fold, features, labels, positions, "bandpower", "logreg", "5class")
    assert rows[0]["n_test"] == rows[1]["n_test"]
    assert rows[0]["chance"] == rows[1]["chance"]


def test_model_beats_its_permutation_control(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    model, control = run_fold(fold, features, labels, positions, "bandpower", "logreg", "5class")
    assert model["balanced_accuracy"] > control["balanced_accuracy"]
    assert control["balanced_accuracy"] <= control["chance"] + control["noise_band"]


def test_four_class_view_excludes_ca2(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    rows = run_fold(fold, features, labels, positions, "bandpower", "logreg", "4class")
    assert "CA2" not in rows[0]["classes_present"]
    assert "CA2" in CLASS_VIEWS["5class"]


def test_four_class_view_has_fewer_rows_than_five(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    four = run_fold(fold, features, labels, positions, "bandpower", "logreg", "4class")[0]
    five = run_fold(fold, features, labels, positions, "bandpower", "logreg", "5class")[0]
    assert four["n_test"] < five["n_test"]


def test_sweep_covers_every_combination(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    schemes = build_schemes(index)
    spec = ExperimentSpec(feature_sets=["bandpower"], models=["constant", "logreg"])
    table = run_sweep(schemes, {"bandpower": features}, labels, positions, spec, verbose=False)

    expected_folds = sum(len(f) for f in schemes.values())
    # two models x two views x two controls per fold
    assert len(table) == expected_folds * 2 * 2 * 2
    assert set(table["scheme"]) == set(schemes)


def test_constant_model_sits_at_chance(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    rows = run_fold(fold, features, labels, positions, "bandpower", "constant", "5class")
    assert rows[0]["balanced_accuracy"] == pytest.approx(rows[0]["chance"], abs=1e-9)


def test_summarise_groups_by_configuration(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    schemes = {"loso": leave_one_group_out(index)}
    spec = ExperimentSpec(feature_sets=["bandpower"], models=["logreg"], views=["5class"])
    table = run_sweep(schemes, {"bandpower": features}, labels, positions, spec, verbose=False)

    summary = summarise(table)
    assert set(summary["control"]) == {"model", "permuted"}
    assert summary["folds"].max() == len(schemes["loso"])


def test_results_are_written_as_csv(corpus_pieces, tmp_path):
    """CSV because .gitignore excludes parquet from results, and these numbers must be committed."""
    index, features, labels, positions = corpus_pieces
    schemes = {"loso": leave_one_group_out(index)}
    spec = ExperimentSpec(feature_sets=["bandpower"], models=["logreg"], views=["5class"])
    table = run_sweep(schemes, {"bandpower": features}, labels, positions, spec, verbose=False)

    paths = write_results(table, tmp_path)
    assert paths["folds"].suffix == ".csv"
    assert len(pd.read_csv(paths["folds"])) == len(table)
    assert paths["summary"].exists()


def test_predictions_are_saved_for_real_models_only(corpus_pieces, tmp_path):
    index, features, labels, positions = corpus_pieces
    fold = leave_one_group_out(index)[0]
    run_fold(
        fold,
        features,
        labels,
        positions,
        "bandpower",
        "logreg",
        "5class",
        predictions_dir=tmp_path,
    )
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1, "one file per configuration, the control is not saved"
    frame = pd.read_parquet(files[0])
    assert {"chunk_id", "label"} <= set(frame.columns)
    assert len([c for c in frame.columns if c.startswith("p_")]) == 5


def test_paired_comparison_pairs_on_folds(corpus_pieces):
    index, features, labels, positions = corpus_pieces
    schemes = {"loso": leave_one_group_out(index)}
    tables = {
        "bandpower": features,
        "shuffled": features[np.random.default_rng(0).permutation(len(features))],
    }
    spec = ExperimentSpec(
        feature_sets=["bandpower", "shuffled"], models=["logreg"], views=["5class"]
    )
    table = run_sweep(schemes, tables, labels, positions, spec, verbose=False)

    comparison = paired_comparison(table, "loso", "5class")
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["folds"] == len(schemes["loso"])
    # Real features should beat features whose rows no longer match their labels.
    assert row["median_difference"] > 0
