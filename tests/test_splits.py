import dataclasses

import pytest

from lfpaudit.data.splits import Split, make_split, verify_no_leakage


def test_cross_session_has_no_group_leakage(synthetic_store):
    split = make_split(synthetic_store.index, kind="cross_session", seed=0)
    sizes = verify_no_leakage(split, synthetic_store.index)
    assert all(v > 0 for v in sizes.values())
    assert not set(split.train_groups) & set(split.test_groups)


def test_cross_lab_holds_out_a_whole_dataset(synthetic_store):
    split = make_split(
        synthetic_store.index,
        kind="cross_lab",
        seed=0,
        train_datasets=["synthA"],
        test_datasets=["synthB"],
    )
    verify_no_leakage(split, synthetic_store.index)
    lookup = synthetic_store.index.set_index("chunk_id")
    assert set(lookup.loc[split.train, "dataset"]) == {"synthA"}
    assert set(lookup.loc[split.val, "dataset"]) == {"synthA"}
    assert set(lookup.loc[split.test, "dataset"]) == {"synthB"}


def test_cross_lab_rejects_overlapping_datasets(synthetic_store):
    with pytest.raises(ValueError, match="disjoint"):
        make_split(
            synthetic_store.index,
            kind="cross_lab",
            train_datasets=["synthA"],
            test_datasets=["synthA"],
        )


def test_in_session_is_chunk_level_and_shares_groups(synthetic_store):
    split = make_split(synthetic_store.index, kind="in_session", seed=0)
    # Chunk ids must still be disjoint, but groups deliberately are not; this split is the
    # leaky upper bound and the verifier must not object to that.
    verify_no_leakage(split, synthetic_store.index)
    assert set(split.train_groups) == set(split.test_groups)


def test_splits_are_deterministic_given_a_seed(synthetic_store):
    a = make_split(synthetic_store.index, kind="cross_session", seed=7)
    b = make_split(synthetic_store.index, kind="cross_session", seed=7)
    assert a == b
    c = make_split(synthetic_store.index, kind="cross_session", seed=8)
    assert a.test_groups != c.test_groups or a.test != c.test


def test_verifier_catches_a_corrupted_split(synthetic_store):
    split = make_split(synthetic_store.index, kind="cross_session", seed=0)
    corrupted = dataclasses.replace(split, test=split.test + split.train[:1])
    with pytest.raises(AssertionError, match="appear in both"):
        verify_no_leakage(corrupted, synthetic_store.index)


def test_verifier_catches_group_leakage(synthetic_store):
    split = make_split(synthetic_store.index, kind="cross_session", seed=0)
    lookup = synthetic_store.index.set_index("chunk_id")
    stolen_group = split.train_groups[0]
    ids = [
        int(i)
        for i in synthetic_store.index.loc[
            synthetic_store.index["group"] == stolen_group, "chunk_id"
        ]
    ]
    # Move one training group's chunks into test without touching the train partition.
    corrupted = dataclasses.replace(split, test=sorted(set(split.test) | set(ids[:2])))
    corrupted = dataclasses.replace(corrupted, train=[i for i in split.train if i not in ids[:2]])
    del lookup
    with pytest.raises(AssertionError, match="groups leak"):
        verify_no_leakage(corrupted, synthetic_store.index)


def test_verifier_rejects_empty_partition(synthetic_store):
    split = make_split(synthetic_store.index, kind="cross_session", seed=0)
    with pytest.raises(AssertionError, match="is empty"):
        verify_no_leakage(dataclasses.replace(split, val=[]), synthetic_store.index)


def test_split_json_roundtrip(synthetic_store, tmp_path):
    split = make_split(synthetic_store.index, kind="cross_session", seed=3)
    path = tmp_path / "split.json"
    split.to_json(path)
    assert Split.from_json(path) == split
