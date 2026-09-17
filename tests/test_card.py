"""Tests for dataset cards."""

from __future__ import annotations

import pandas as pd

from lfpaudit import REGIONS, UNKNOWN
from lfpaudit.data.card import DatasetCard, SourceRecord, summarise_channels


def test_summarise_channels_lists_every_region_and_unknown():
    table = pd.DataFrame({"region": ["CA1", "CA1", "DG", "UNK"]})
    summary = summarise_channels(table)
    assert set(summary) == set(REGIONS) | {UNKNOWN}
    assert summary["CA1"] == 2
    assert summary["CA2"] == 0
    assert summary[UNKNOWN] == 1


def test_card_from_store_counts_chunks_per_region(synthetic_store):
    card = DatasetCard.from_store(synthetic_store)
    assert card.n_chunks == len(synthetic_store.index)
    assert sum(card.region_chunks.values()) == card.n_chunks
    assert set(card.region_chunks) == set(REGIONS)
    assert card.window_s == synthetic_store.n_samples / synthetic_store.fs


def test_card_roundtrips_through_json(synthetic_store, tmp_path):
    source = SourceRecord(
        dataset="ibl",
        session="s0",
        probe="probe00",
        group="s0",
        channels_total=384,
        channels_kept=100,
        channels_unlabelled=280,
        channels_dropped_quality=4,
        chunks=10000,
        region_channels={"CA1": 50, "DG": 50},
    )
    card = DatasetCard.from_store(synthetic_store, sources=[source])
    path = card.write(tmp_path / "card.json")

    reloaded = DatasetCard.read(path)
    assert reloaded.n_chunks == card.n_chunks
    assert reloaded.sources[0].channels_kept == 100
    assert reloaded.sources[0].region_channels["CA1"] == 50


def test_markdown_mentions_every_region_and_skipped_source(synthetic_store):
    card = DatasetCard.from_store(
        synthetic_store,
        sources=[
            SourceRecord("allen", "719161530", "probeA", "719161530_probeA", 94, 29, 65, 0, 2900)
        ],
        skipped=[{"key": "719161530_probeD", "reason": "file is only 72 MB, verified all zeros"}],
    )
    text = card.to_markdown()
    for region in REGIONS:
        assert region in text
    assert "719161530_probeD" in text
    assert "all zeros" in text
