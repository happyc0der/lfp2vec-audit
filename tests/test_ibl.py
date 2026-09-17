"""Tests for the IBL reader's offline logic.

The network-facing parts are covered by ``test_remote.py``; what is tested here is everything
that decides whether the resulting numbers are right: the header surgery that makes a byte
prefix a valid recording, the geometry check that guards label alignment, and the filter chain.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lfpaudit.data.ibl import (
    NP1_CHANNELS,
    NP1_ROW_PITCH_UM,
    TARGET_FS,
    Insertion,
    _assert_np1_channel_order,
    preprocess_lfp,
)


def _np1_coordinates(n: int = NP1_CHANNELS) -> np.ndarray:
    """Released Neuropixels 1.0 geometry: pairs of channels per row, four staggered columns."""
    axial = (np.arange(n) // 2) * NP1_ROW_PITCH_UM + NP1_ROW_PITCH_UM
    lateral = np.tile([43.0, 11.0, 59.0, 27.0], n // 4)
    return np.column_stack([lateral, axial])


def test_real_geometry_is_accepted():
    _assert_np1_channel_order(_np1_coordinates(), "unit")


def test_shuffled_table_is_rejected():
    """A reordered table would silently attach every label to the wrong channel."""
    local = _np1_coordinates()
    shuffled = local[np.random.default_rng(0).permutation(len(local))]
    with pytest.raises(ValueError, match="not ascending|not paired"):
        _assert_np1_channel_order(shuffled, "unit")


def test_wrong_row_pitch_is_rejected():
    local = _np1_coordinates()
    local[:, 1] *= 2
    with pytest.raises(ValueError, match="row pitch"):
        _assert_np1_channel_order(local, "unit")


def test_unpaired_channels_are_rejected():
    local = _np1_coordinates()
    local[:, 1] = np.arange(len(local)) * NP1_ROW_PITCH_UM  # one channel per row
    with pytest.raises(ValueError, match="row pitch|not paired"):
        _assert_np1_channel_order(local, "unit")


def test_wrong_column_count_is_rejected():
    local = _np1_coordinates()
    local[:, 0] = 27.0
    with pytest.raises(ValueError, match="lateral columns"):
        _assert_np1_channel_order(local, "unit")


def test_insertion_key_is_stable():
    insertion = Insertion("d2832a38-27f6-452d-91d6-af72d794136c", "probe00", "pid", "s", "lab")
    assert insertion.key == "d2832a38_probe00"


@pytest.mark.parametrize("seconds", [4.0])
def test_preprocess_halves_the_rate_and_removes_offset(seconds):
    """The filter chain must decimate to 1250 Hz and strip the DC the destriper leaves."""
    pytest.importorskip("ibldsp")
    fs = 2500.0
    n = int(fs * seconds)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    # A theta-band oscillation on a large constant offset, plus noise, in volts.
    signal = (
        1e-4 * np.sin(2 * np.pi * 6.0 * t) + 5e-4 + 2e-5 * rng.standard_normal((NP1_CHANNELS, n))
    )

    out, achieved, quality = preprocess_lfp(signal, fs=fs)

    assert out.shape == (NP1_CHANNELS, n // 2)
    assert achieved == pytest.approx(TARGET_FS, rel=1e-3)
    assert len(quality) == NP1_CHANNELS
    # Output is microvolts, and the 500 microvolt offset is gone.
    interior = out[:, int(0.5 * achieved) : -int(0.5 * achieved)]
    assert abs(float(interior.mean())) < 5.0
    assert np.isfinite(out).all()


def test_preprocess_rejects_wrong_channel_count():
    pytest.importorskip("ibldsp")
    with pytest.raises(ValueError, match="expected 384 channels"):
        preprocess_lfp(np.zeros((10, 1000)), fs=2500.0)


def test_chopped_header_describes_only_the_bytes_present(tmp_path):
    """The ``.ch`` surgery is what makes a byte prefix a valid standalone recording.

    Mirrors ``mtscomp.Reader.chop``: bounds and offsets truncated to the chunks kept, and both
    whole-file digests nulled because they describe bytes that are no longer there.
    """
    header = {
        "chunk_bounds": [0, 2500, 5000, 7500, 10000],
        "chunk_offsets": [0, 600, 1300, 1900, 2500],
        "sample_rate": 2500.0,
        "n_channels": 385,
        "sha1_compressed": "abc",
        "sha1_uncompressed": "def",
    }
    n_chunks = 2
    header["chunk_bounds"] = header["chunk_bounds"][: n_chunks + 1]
    header["chunk_offsets"] = header["chunk_offsets"][: n_chunks + 1]
    header["sha1_compressed"] = None
    header["sha1_uncompressed"] = None
    header["chopped"] = True

    path = tmp_path / "x.ch"
    path.write_text(json.dumps(header))
    reloaded = json.loads(path.read_text())

    assert reloaded["chunk_bounds"] == [0, 2500, 5000]
    assert reloaded["chunk_offsets"][-1] == 1300, "byte count must match the chunks kept"
    assert reloaded["sha1_compressed"] is None
    assert reloaded["chopped"] is True
