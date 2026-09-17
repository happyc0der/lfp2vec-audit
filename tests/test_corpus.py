"""Tests for presenting several stores as one dataset."""

from __future__ import annotations

import numpy as np
import pytest

from lfpaudit.data.corpus import ID_STRIDE, Corpus
from lfpaudit.data.splits import make_split, verify_no_leakage
from lfpaudit.data.synthetic import SyntheticSpec, build_synthetic_store


@pytest.fixture
def two_stores(tmp_path):
    a = build_synthetic_store(
        str(tmp_path / "a"), SyntheticSpec(datasets=("synthA",), sessions_per_dataset=3, seed=1)
    )
    b = build_synthetic_store(
        str(tmp_path / "b"), SyntheticSpec(datasets=("synthB",), sessions_per_dataset=3, seed=2)
    )
    return a, b, Corpus.load({"ibl": tmp_path / "a", "allen": tmp_path / "b"})


def test_index_concatenates_and_labels_the_source(two_stores):
    a, b, corpus = two_stores
    assert len(corpus.index) == len(a.index) + len(b.index)
    assert set(corpus.index["store"]) == {"ibl", "allen"}


def test_chunk_ids_are_globally_unique(two_stores):
    _, _, corpus = two_stores
    ids = corpus.index["chunk_id"]
    assert ids.is_unique
    # Stores are offset from one another so ids never collide after concatenation.
    assert ids.max() >= ID_STRIDE


def test_take_returns_the_right_waveforms_in_the_requested_order(two_stores):
    a, b, corpus = two_stores
    # One chunk from each store, deliberately out of index order.
    allen_id = int(corpus.index[corpus.index["store"] == "allen"]["chunk_id"].iloc[3])
    ibl_id = int(corpus.index[corpus.index["store"] == "ibl"]["chunk_id"].iloc[5])

    got = corpus.take([allen_id, ibl_id])
    np.testing.assert_allclose(got[0], b.take([3])[0])
    np.testing.assert_allclose(got[1], a.take([5])[0])


def test_take_rejects_unknown_ids(two_stores):
    _, _, corpus = two_stores
    with pytest.raises(KeyError, match="not present"):
        corpus.take([999_999_999])


def test_positions_locate_rows_in_the_merged_index(two_stores):
    _, _, corpus = two_stores
    ids = corpus.index["chunk_id"].to_numpy()[[0, 7, 100]]
    rows = corpus.positions(ids)
    np.testing.assert_array_equal(corpus.index.iloc[rows]["chunk_id"].to_numpy(), ids)


def test_cross_lab_split_over_a_corpus_is_clean(two_stores):
    _, _, corpus = two_stores
    split = make_split(
        corpus.index, kind="cross_lab", train_datasets=["synthA"], test_datasets=["synthB"]
    )
    verify_no_leakage(split, corpus.index)
    lookup = corpus.index.set_index("chunk_id")
    assert set(lookup.loc[split.test, "dataset"]) == {"synthB"}


def test_mismatched_chunk_length_is_rejected(tmp_path):
    build_synthetic_store(str(tmp_path / "a"), SyntheticSpec(window_s=3.0, seed=1))
    build_synthetic_store(str(tmp_path / "b"), SyntheticSpec(window_s=2.0, seed=2))
    with pytest.raises(ValueError, match="chunk length"):
        Corpus.load({"a": tmp_path / "a", "b": tmp_path / "b"})


def test_mismatched_sampling_rate_is_rejected(tmp_path):
    build_synthetic_store(str(tmp_path / "a"), SyntheticSpec(fs=1250.0, window_s=3.0, seed=1))
    build_synthetic_store(str(tmp_path / "b"), SyntheticSpec(fs=2500.0, window_s=1.5, seed=2))
    with pytest.raises(ValueError, match="sampling rate"):
        Corpus.load({"a": tmp_path / "a", "b": tmp_path / "b"})


def test_close_sampling_rates_are_accepted(tmp_path):
    """The two real datasets land on 1250.012 and 1249.999 Hz; that must not be an error."""
    build_synthetic_store(str(tmp_path / "a"), SyntheticSpec(fs=1250.012, window_s=3.0, seed=1))
    build_synthetic_store(str(tmp_path / "b"), SyntheticSpec(fs=1249.999, window_s=3.0, seed=2))
    corpus = Corpus.load({"a": tmp_path / "a", "b": tmp_path / "b"})
    assert 1249.9 < corpus.fs < 1250.1


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="at least one store"):
        Corpus.load({})
