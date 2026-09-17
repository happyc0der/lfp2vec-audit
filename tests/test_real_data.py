"""Checks that run against the real stores, when they have been built.

Marked slow and skipped when the stores are absent, so continuous integration stays offline.
These assert the properties a silently broken build would violate but a unit test cannot see:
that the sampling rates agree, that no chunk is flat or non-finite, that the index matches the
dataset card, and that every committed split is still clean against the data it describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lfpaudit import REGIONS
from lfpaudit.data.card import DatasetCard
from lfpaudit.data.chunk import ChunkStore
from lfpaudit.data.corpus import Corpus
from lfpaudit.data.splits import Split, verify_no_leakage

IBL_STORE = Path("data/stores/ibl")
ALLEN_STORE = Path("data/stores/allen")
SPLIT_DIR = Path("experiments/splits")

pytestmark = pytest.mark.slow


def _store_or_skip(path: Path) -> ChunkStore:
    if not (path / "index.parquet").exists():
        pytest.skip(f"{path} has not been built")
    return ChunkStore.open(path)


@pytest.fixture
def ibl() -> ChunkStore:
    return _store_or_skip(IBL_STORE)


@pytest.fixture
def allen() -> ChunkStore:
    return _store_or_skip(ALLEN_STORE)


@pytest.mark.parametrize("path", [IBL_STORE, ALLEN_STORE])
def test_sampling_rate_and_chunk_length(path):
    store = _store_or_skip(path)
    assert store.fs == pytest.approx(1250.0, abs=0.1)
    assert store.n_samples == pytest.approx(round(3.0 * store.fs), abs=1)


@pytest.mark.parametrize("path", [IBL_STORE, ALLEN_STORE])
def test_no_chunk_is_flat_or_non_finite(path):
    store = _store_or_skip(path)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(store.index), size=min(len(store.index), 2000), replace=False)
    payload = store.take(rows)
    assert np.isfinite(payload).all()
    assert (payload.std(axis=1) > 0).all()
    assert (store.index["scale_std_uv"] > 0).all()


@pytest.mark.parametrize("path", [IBL_STORE, ALLEN_STORE])
def test_every_region_label_is_known(path):
    store = _store_or_skip(path)
    assert set(store.index["region"]) <= set(REGIONS)


@pytest.mark.parametrize("path", [IBL_STORE, ALLEN_STORE])
def test_card_matches_the_index(path):
    store = _store_or_skip(path)
    card = DatasetCard.read(path / "card.json")
    assert card.n_chunks == len(store.index)
    counts = store.index["region"].value_counts().to_dict()
    for region, expected in card.region_chunks.items():
        assert int(counts.get(region, 0)) == expected
    assert sum(s.chunks for s in card.sources) == len(store.index)


@pytest.mark.parametrize("path", [IBL_STORE, ALLEN_STORE])
def test_chunks_per_channel_is_uniform(path):
    """Every kept channel should contribute the same number of windows."""
    store = _store_or_skip(path)
    per_channel = store.index.groupby(["group", "channel"]).size()
    assert per_channel.nunique() == 1, f"uneven chunk counts: {per_channel.value_counts().head()}"


def test_amplitudes_are_physiological(ibl, allen):
    """Microvolt scales should look like extracellular field potential, not volts or bits."""
    for store in (ibl, allen):
        scale = store.index["scale_std_uv"]
        assert 1.0 < scale.median() < 2000.0, f"{store.path}: median scale {scale.median()}"


def test_corpus_merges_the_two_stores(ibl, allen):
    corpus = Corpus.load({"ibl": IBL_STORE, "allen": ALLEN_STORE})
    assert len(corpus.index) == len(ibl.index) + len(allen.index)
    assert corpus.index["chunk_id"].is_unique
    assert set(corpus.index["dataset"]) == {"ibl", "allen"}


def test_committed_splits_are_still_clean(ibl, allen):
    if not SPLIT_DIR.exists():
        pytest.skip("splits have not been written")
    corpus = Corpus.load({"ibl": IBL_STORE, "allen": ALLEN_STORE})
    files = sorted(SPLIT_DIR.glob("*.json"))
    assert files, "no splits found"
    for path in files:
        verify_no_leakage(Split.from_json(path), corpus.index)
