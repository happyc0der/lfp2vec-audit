"""Tests for the Allen reader, against a miniature file with the same internal layout.

Building a small HDF5 that mirrors the released structure means the reader's real logic (path
names, the electrode-table join, byte-string decoding, empty acronyms, the dead-probe screen) is
exercised offline, and a change in our assumptions about that layout fails here rather than
halfway through a download.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from lfpaudit.data.allen import (
    DEAD_PROBE_MAX_BYTES,
    AllenProbe,
    open_probe,
    probe_is_empty,
    read_electrodes,
    read_sampling_rate,
    read_window,
)

PROBE_ID = 12345
FS = 1249.9986
N_SAMPLES = 20000
LOCATIONS = ["CA1", "CA1", "DG", "VISp2/3", "APN", "", "CA3", "grey"]


def _write_probe_file(path, locations=LOCATIONS, zeros=False, n_samples=N_SAMPLES):
    """Write a miniature file with the same paths and column names as a released one."""
    n_channels = len(locations)
    rng = np.random.default_rng(0)
    data = (
        np.zeros((n_samples, n_channels), dtype=np.float32)
        if zeros
        else rng.normal(scale=1e-4, size=(n_samples, n_channels)).astype(np.float32)
    )
    with h5py.File(path, "w") as h5:
        group = h5.create_group(f"/acquisition/probe_{PROBE_ID}_lfp/probe_{PROBE_ID}_lfp_data")
        group.create_dataset(
            "data", data=data, chunks=(min(4096, n_samples), 1), compression="gzip"
        )
        group.create_dataset("timestamps", data=np.arange(n_samples) / FS + 0.62)
        group.create_dataset("electrodes", data=np.arange(n_channels, dtype=np.int64))

        table = h5.create_group("/general/extracellular_ephys/electrodes")
        table.create_dataset("id", data=np.arange(100, 100 + n_channels, dtype=np.int64))
        table.create_dataset("local_index", data=np.arange(2, 2 + 4 * n_channels, 4))
        # Released files store the acronym as a variable-length byte string.
        table.create_dataset(
            "location", data=np.array(locations, dtype=object), dtype=h5py.special_dtype(vlen=bytes)
        )
        table.create_dataset("probe_vertical_position", data=np.arange(n_channels) * 40.0 + 40.0)
        table.create_dataset("probe_horizontal_position", data=np.full(n_channels, 27.0))
        for axis in "xyz":
            table.create_dataset(axis, data=np.arange(n_channels, dtype=np.float64) * 10)
        table.create_dataset("valid_data", data=np.array([True] * (n_channels - 1) + [False]))
    return path


@pytest.fixture
def probe_file(tmp_path):
    path = _write_probe_file(tmp_path / "probe.nwb")
    probe = AllenProbe(999, PROBE_ID, "probeA", path.stat().st_size)
    return probe, path


def test_electrode_table_columns_and_labels(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        table = read_electrodes(h5, probe)

    assert len(table) == len(LOCATIONS)
    assert table["acronym"].tolist() == LOCATIONS
    # The five-way mapping, including the empty acronym released files use for "unlabelled".
    assert table["region"].tolist() == ["CA1", "CA1", "DG", "VIS", "UNK", "UNK", "CA3", "UNK"]
    assert table["channel"].tolist() == list(range(len(LOCATIONS)))
    assert not bool(table["valid_data"].iloc[-1])


def test_sampling_rate_measured_from_timestamps(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        assert read_sampling_rate(h5, probe) == pytest.approx(FS, rel=1e-6)


def test_read_window_shape_orientation_and_units(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        fs = read_sampling_rate(h5, probe)
        window = read_window(h5, probe, 1.0, 3.0, fs)
        raw = h5[f"/acquisition/probe_{PROBE_ID}_lfp/probe_{PROBE_ID}_lfp_data/data"][:]

    # (channels, samples), not the file's (samples, channels).
    assert window.shape == (len(LOCATIONS), int(round(2 * fs)))
    first, last = int(round(1.0 * fs)), int(round(3.0 * fs))
    # Volts in the file, microvolts out.
    np.testing.assert_allclose(window, raw[first:last, :].T * 1e6, rtol=1e-6)


def test_read_window_clips_at_the_end_of_the_recording(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        fs = read_sampling_rate(h5, probe)
        window = read_window(h5, probe, 0.0, 10_000.0, fs)
    assert window.shape[1] == N_SAMPLES


def test_read_window_rejects_an_empty_window(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        with pytest.raises(ValueError, match="is empty"):
            read_window(h5, probe, 5.0, 5.0, FS)


def test_electrode_mismatch_is_caught(tmp_path):
    """A table that does not match the data columns would mislabel every channel."""
    path = _write_probe_file(tmp_path / "bad.nwb")
    with h5py.File(path, "a") as h5:
        table = h5["/general/extracellular_ephys/electrodes"]
        for name in list(table.keys()):
            values = table[name][:]
            del table[name]
            table.create_dataset(name, data=np.concatenate([values, values[-1:]]))
    probe = AllenProbe(999, PROBE_ID, "probeA", path.stat().st_size)
    with open_probe(probe, local_path=path) as h5:
        with pytest.raises(ValueError, match="electrode table"):
            read_electrodes(h5, probe)


def test_empty_probe_detected(tmp_path):
    path = _write_probe_file(tmp_path / "zeros.nwb", zeros=True)
    probe = AllenProbe(999, PROBE_ID, "probeA", path.stat().st_size)
    with open_probe(probe, local_path=path) as h5:
        assert probe_is_empty(h5, probe)


def test_probe_with_data_is_not_reported_empty(probe_file):
    probe, path = probe_file
    with open_probe(probe, local_path=path) as h5:
        assert not probe_is_empty(h5, probe)


def test_dead_probe_size_screen():
    live = AllenProbe(1, 2, "probeA", 2_400_000_000)
    dead = AllenProbe(1, 3, "probeD", 72_000_000)
    assert not live.looks_dead
    assert dead.looks_dead
    assert dead.size_bytes < DEAD_PROBE_MAX_BYTES < live.size_bytes


def test_probe_url_and_key():
    probe = AllenProbe(719161530, 729445648, "probeA", 1)
    assert probe.url.endswith("session_719161530/probe_729445648_lfp.nwb")
    assert probe.key == "719161530_probeA"
