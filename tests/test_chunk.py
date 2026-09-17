import numpy as np
import pytest

from lfpaudit.data.chunk import (
    ChunkStore,
    ChunkWriter,
    chunk_signal,
    resample_to,
    zscore_chunks,
)


def test_chunk_count_and_length():
    fs, window_s = 1250.0, 3.0
    signal = np.arange(int(fs * 60))
    chunks, t0 = chunk_signal(signal, fs=fs, window_s=window_s, start_s=10.0, n_chunks=5)
    assert chunks.shape == (5, int(fs * window_s))
    assert t0[0] == pytest.approx(10.0)
    assert t0[1] - t0[0] == pytest.approx(window_s)


def test_chunks_are_contiguous_slices_of_the_signal():
    fs = 100.0
    signal = np.arange(1000, dtype=float)
    chunks, _ = chunk_signal(signal, fs=fs, window_s=1.0, start_s=2.0, n_chunks=3)
    assert chunks[0][0] == 200.0
    assert chunks[1][0] == 300.0
    np.testing.assert_allclose(chunks[2], np.arange(400, 500, dtype=float))


def test_chunking_never_runs_past_the_end():
    fs = 100.0
    signal = np.zeros(450)
    chunks, _ = chunk_signal(signal, fs=fs, window_s=1.0, start_s=0.0, n_chunks=10)
    assert len(chunks) == 4


def test_zscore_gives_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    chunks = rng.normal(loc=17.0, scale=4.0, size=(6, 500))
    normalised = zscore_chunks(chunks)
    np.testing.assert_allclose(normalised.mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(normalised.std(axis=1), 1.0, atol=1e-6)


def test_resample_to_16k_gives_paper_length():
    fs, window_s = 1250.0, 3.0
    chunks = np.zeros((2, int(fs * window_s)))
    assert resample_to(chunks, fs=fs, target_fs=16000).shape == (2, 48000)


def test_resample_is_a_noop_at_matching_rate():
    chunks = np.arange(20, dtype=float).reshape(2, 10)
    np.testing.assert_array_equal(resample_to(chunks, fs=1250.0, target_fs=1250), chunks)


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    payload = rng.normal(size=(4, 32))
    meta = {
        "dataset": "synthA",
        "session": "s0",
        "probe": "s0-p0",
        "channel": 3,
        "depth_um": 60.0,
        "acronym": "CA1",
        "region": "CA1",
        "group": "s0",
    }
    with ChunkWriter(tmp_path / "store", n_samples=32, fs=1250.0) as writer:
        writer.append(payload, meta, t0_s=np.arange(4, dtype=float))

    store = ChunkStore.open(tmp_path / "store")
    assert len(store.index) == 4
    assert store.index["region"].unique().tolist() == ["CA1"]
    # Chunks are stored normalised, so the raw waveform comes back via the recorded scale.
    # float16 storage is lossy by design; the tolerance reflects that, not a bug.
    np.testing.assert_allclose(store.take([0, 1, 2, 3], microvolts=True), payload, atol=1e-2)


def test_writer_rejects_non_finite_values(tmp_path):
    with ChunkWriter(tmp_path / "store", n_samples=4, fs=100.0) as writer:
        bad = np.array([[1.0, 2.0, np.nan, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            writer.append(bad, {"dataset": "d"}, t0_s=[0.0])


def test_writer_rejects_wrong_width(tmp_path):
    with ChunkWriter(tmp_path / "store", n_samples=4, fs=100.0) as writer:
        with pytest.raises(ValueError, match="expected chunks of shape"):
            writer.append(np.zeros((1, 5)), {"dataset": "d"}, t0_s=[0.0])


def test_normalisation_is_reversible_from_the_index(tmp_path):
    """Storing the removed mean and scale is what lets the amplitude ablation exist."""
    rng = np.random.default_rng(3)
    payload = rng.normal(loc=120.0, scale=35.0, size=(4, 32))
    meta = {
        "dataset": "ibl",
        "session": "s0",
        "probe": "probe00",
        "channel": 1,
        "depth_um": 40.0,
        "acronym": "CA1",
        "region": "CA1",
        "group": "s0",
    }
    with ChunkWriter(tmp_path / "store", n_samples=32, fs=1250.0) as writer:
        writer.append(payload, meta, t0_s=np.arange(4, dtype=float))

    store = ChunkStore.open(tmp_path / "store")
    stored = store.take([0, 1, 2, 3])
    np.testing.assert_allclose(stored.mean(axis=1), 0.0, atol=1e-2)
    np.testing.assert_allclose(stored.std(axis=1), 1.0, atol=1e-2)

    restored = store.take([0, 1, 2, 3], microvolts=True)
    np.testing.assert_allclose(restored, payload, rtol=2e-3)
    np.testing.assert_allclose(store.index["scale_mean_uv"], payload.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(store.index["scale_std_uv"], payload.std(axis=1), rtol=1e-6)


def test_flat_chunks_are_dropped_not_stored(tmp_path):
    """Released Allen probe files can be entirely zeros; those must not enter a store."""
    meta = {
        "dataset": "allen",
        "session": "719161530",
        "probe": "probeD",
        "channel": 0,
        "depth_um": 40.0,
        "acronym": "CA1",
        "region": "CA1",
        "group": "719161530_probeD",
    }
    payload = np.vstack([np.zeros(16), np.arange(16, dtype=float), np.zeros(16)])
    with ChunkWriter(tmp_path / "store", n_samples=16, fs=1250.0) as writer:
        written = writer.append(payload, meta, t0_s=[0.0, 3.0, 6.0])

    assert written == 1
    store = ChunkStore.open(tmp_path / "store")
    assert len(store.index) == 1
    assert store.index["t0_s"].iloc[0] == 3.0


def test_optional_geometry_columns_are_kept(tmp_path):
    meta = {
        "dataset": "ibl",
        "session": "s0",
        "probe": "probe00",
        "channel": 0,
        "depth_um": 20.0,
        "acronym": "DG",
        "region": "DG",
        "group": "s0",
        "lateral_um": 43.0,
        "ccf_ap_um": -2175.0,
        "ccf_dv_um": -4148.0,
        "ccf_lr_um": -1464.0,
    }
    rng = np.random.default_rng(4)
    with ChunkWriter(tmp_path / "store", n_samples=16, fs=1250.0) as writer:
        writer.append(rng.normal(size=(2, 16)), meta, t0_s=[0.0, 3.0])

    index = ChunkStore.open(tmp_path / "store").index
    for column in ("lateral_um", "ccf_ap_um", "ccf_dv_um", "ccf_lr_um"):
        assert column in index.columns
    assert index["ccf_ap_um"].iloc[0] == -2175.0
