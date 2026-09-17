"""Allen Brain Observatory Visual Coding Neuropixels LFP.

The upstream pipeline reaches this data through AllenSDK, which pins an old scientific Python
stack that cannot coexist with a current torch install. Nothing about the access actually needs
it: the released files are ordinary HDF5, served anonymously over HTTPS with byte-range support,
and the two things we want from them are a dataset of LFP samples and a table of per-channel
brain-region labels. This module reads them directly with h5py.

Reading remotely is worth the small amount of machinery because of how the files are laid out.
Each is 1-2.5 GB, chunked one channel at a time along 37.8-second blocks, and the analysis window
is 300 seconds. Pulling whole files would move roughly twenty times more data than is read.

One caveat is load-bearing and is why :func:`is_dead_probe` exists: some released probe files
contain nothing but zeros while still advertising that they hold LFP data.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lfpaudit import UNKNOWN
from lfpaudit.data.card import DatasetCard, SourceRecord, summarise_channels
from lfpaudit.data.chunk import ChunkStore, ChunkWriter, chunk_signal
from lfpaudit.data.labels import map_acronym
from lfpaudit.data.remote import HttpRangeFile, content_length

BUCKET_URL = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com"
CACHE_PREFIX = "visual-coding-neuropixels/ecephys-cache"

#: Sessions named by the upstream training script, in its order.
PAPER_SESSIONS: tuple[int, ...] = (
    719161530,
    794812542,
    778998620,
    798911424,
    771990200,
    771160300,
    768515987,
)

#: A live probe file is gigabytes of compressed float; an all-zero one compresses to a fraction
#: of that. Anything under this threshold is treated as empty and is verified by sampling before
#: being skipped. Observed dead files are ~72 MB and live ones 1.27-2.44 GB.
DEAD_PROBE_MAX_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class AllenProbe:
    """One probe's LFP file within a session."""

    session_id: int
    probe_id: int
    name: str
    size_bytes: int

    @property
    def url(self) -> str:
        return (
            f"{BUCKET_URL}/{CACHE_PREFIX}/session_{self.session_id}/probe_{self.probe_id}_lfp.nwb"
        )

    @property
    def key(self) -> str:
        """Grouping unit for splits: one insertion, matching the IBL convention."""
        return f"{self.session_id}_{self.name}"

    @property
    def looks_dead(self) -> bool:
        return self.size_bytes < DEAD_PROBE_MAX_BYTES


def _read_csv(name: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(f"{BUCKET_URL}/{CACHE_PREFIX}/{name}", timeout=60) as response:
        text = response.read().decode()
    return list(csv.DictReader(io.StringIO(text)))


def list_probes(session_id: int, check_sizes: bool = True) -> list[AllenProbe]:
    """Probes for a session that claim to have LFP, annotated with their file sizes."""
    probes = []
    for row in _read_csv("probes.csv"):
        if int(row["ecephys_session_id"]) != session_id or row["has_lfp_data"] != "True":
            continue
        probe_id = int(row["id"])
        url = f"{BUCKET_URL}/{CACHE_PREFIX}/session_{session_id}/probe_{probe_id}_lfp.nwb"
        size = content_length(url) if check_sizes else -1
        probes.append(AllenProbe(session_id, probe_id, row["name"], size))
    return sorted(probes, key=lambda p: p.name)


@contextmanager
def open_probe(probe: AllenProbe, local_path: str | Path | None = None):
    """Open a probe's HDF5 file, remotely by default or from a local copy if given.

    Always closes explicitly: h5py holding a Python file object can crash the interpreter if
    teardown order goes wrong.
    """
    import h5py

    handle = None
    source: HttpRangeFile | str
    if local_path is not None and Path(local_path).exists():
        source = str(local_path)
    else:
        source = HttpRangeFile(probe.url, size=probe.size_bytes if probe.size_bytes > 0 else None)
    try:
        handle = h5py.File(source, "r")
        yield handle
    finally:
        if handle is not None:
            handle.close()
        if isinstance(source, HttpRangeFile):
            source.close()


def _lfp_group(h5, probe_id: int):
    series = h5[f"/acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data"]
    return series["data"], series["timestamps"]


def read_electrodes(h5, probe: AllenProbe) -> pd.DataFrame:
    """Per-channel anatomy and geometry from the file's electrode table.

    The electrode table in a probe file lists exactly the channels present in its LFP data, in
    the same order, so row position is the data column. Unlabelled channels carry an empty
    acronym rather than a missing value.
    """
    table = h5["/general/extracellular_ephys/electrodes"]
    locations = [
        value.decode() if isinstance(value, bytes) else str(value) for value in table["location"][:]
    ]
    n = len(locations)

    data, _ = _lfp_group(h5, probe.probe_id)
    if data.shape[1] != n:
        raise ValueError(
            f"{probe.key}: LFP has {data.shape[1]} columns but the electrode table has {n} rows"
        )

    return pd.DataFrame(
        {
            "channel": np.arange(n, dtype=np.int64),
            "channel_id": table["id"][:].astype(np.int64),
            "local_index": table["local_index"][:].astype(np.int64),
            "acronym": locations,
            "region": [map_acronym(a) for a in locations],
            "depth_um": table["probe_vertical_position"][:].astype(np.float64),
            "lateral_um": table["probe_horizontal_position"][:].astype(np.float64),
            "ccf_ap_um": table["x"][:].astype(np.float64),
            "ccf_dv_um": table["y"][:].astype(np.float64),
            "ccf_lr_um": table["z"][:].astype(np.float64),
            "valid_data": table["valid_data"][:].astype(bool),
        }
    )


def read_sampling_rate(h5, probe: AllenProbe, n_probe: int = 10000) -> float:
    """Sampling rate measured from the timestamps.

    The rate recorded in the file's own metadata attribute is wrong in the released files (it
    reads half the true value), so it is measured from sample times instead.
    """
    _, timestamps = _lfp_group(h5, probe.probe_id)
    sample = timestamps[: min(n_probe, len(timestamps))]
    step = float(np.median(np.diff(sample)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"{probe.key}: cannot determine a sampling rate from timestamps")
    return 1.0 / step


def read_window(h5, probe: AllenProbe, t0_s: float, t1_s: float, fs: float) -> np.ndarray:
    """Every channel's samples over a time window, as ``(n_channels, n_samples)`` in microvolts.

    All channels are read in one slab rather than a column at a time, which is worth a comment
    because the file's chunk shape of one channel by 37.8 seconds suggests the opposite. Chunks
    are laid out with neighbouring channels of the same time block adjacent on disk, so a time
    slab is a near-contiguous run while a single channel is a scatter of distant reads. Measured
    over a remote 30-second window: 5 seconds for all 94 channels together, against 12 seconds
    for one channel alone, so looping channels would cost roughly 230 times more.
    """
    data, _ = _lfp_group(h5, probe.probe_id)
    first = int(round(t0_s * fs))
    last = min(int(round(t1_s * fs)), data.shape[0])
    if first >= last:
        raise ValueError(f"{probe.key}: window {t0_s}-{t1_s}s is empty for {data.shape[0]} samples")
    slab = np.asarray(data[first:last, :], dtype=np.float64).T
    return slab * 1e6


def probe_is_empty(h5, probe: AllenProbe, n_samples: int = 200000) -> bool:
    """Whether a probe file holds nothing but zeros, checked on a sample of its data.

    Some released files advertise LFP data and contain only zeros. A store built from them would
    be full of chunks that look perfectly well formed and carry no signal, so they are detected
    at the boundary instead.
    """
    data, _ = _lfp_group(h5, probe.probe_id)
    stride = max(1, data.shape[0] // 3)
    for start in (0, stride, 2 * stride):
        block = data[start : min(start + n_samples // 3, data.shape[0]), :]
        if np.any(block):
            return False
    return True


def build_allen_store(
    out_path: str | Path,
    sessions: Sequence[int] = (719161530, 798911424),
    cache_dir: str | Path = "data/allen-cache",
    window_s: float = 3.0,
    start_s: float = 200.0,
    chunks_per_channel: int = 100,
    verify_dead: bool = True,
) -> tuple[ChunkStore, DatasetCard]:
    """Read the requested sessions' live probes into a single chunk store.

    Probes whose file size marks them as empty are checked against their contents before being
    skipped, so the screen can never silently discard real data. No filtering is applied: the
    released files are already band-limited and downsampled, and the upstream pipeline adds
    nothing further.
    """
    writer: ChunkWriter | None = None
    sources: list[SourceRecord] = []
    skipped: list[dict] = []
    end_s = start_s + window_s * chunks_per_channel

    for session_id in sessions:
        for probe in list_probes(session_id):
            if probe.looks_dead:
                reason = f"file is only {probe.size_bytes / 1e6:.0f} MB"
                if verify_dead:
                    with open_probe(probe) as h5:
                        if not probe_is_empty(h5, probe):
                            skipped.append(
                                {"key": probe.key, "reason": f"{reason} but holds data; inspect it"}
                            )
                            print(f"{probe.key}: SMALL BUT NOT EMPTY, skipped for review")
                            continue
                    reason += ", verified all zeros"
                skipped.append({"key": probe.key, "reason": reason})
                print(f"{probe.key}: skipped ({reason})", flush=True)
                continue

            try:
                with open_probe(probe, local_path=Path(cache_dir) / f"{probe.probe_id}.nwb") as h5:
                    fs = read_sampling_rate(h5, probe)
                    table = read_electrodes(h5, probe)
                    signal = read_window(h5, probe, 0.0, end_s + 2.0, fs)
            except Exception as error:  # noqa: BLE001 - one bad probe must not lose the rest
                skipped.append({"key": probe.key, "reason": f"{type(error).__name__}: {error}"})
                print(f"{probe.key}: SKIPPED ({error})", flush=True)
                continue

            n_samples = int(round(window_s * fs))
            if writer is None:
                writer = ChunkWriter(out_path, n_samples=n_samples, fs=fs)
            elif n_samples != writer.n_samples:
                skipped.append(
                    {"key": probe.key, "reason": f"chunk length {n_samples} != {writer.n_samples}"}
                )
                continue

            usable = table[(table["region"] != UNKNOWN) & table["valid_data"]]
            written = 0
            for row in usable.itertuples():
                chunks, t0 = chunk_signal(
                    signal[row.channel],
                    fs=fs,
                    window_s=window_s,
                    start_s=start_s,
                    n_chunks=chunks_per_channel,
                )
                if not len(chunks):
                    continue
                written += writer.append(
                    chunks,
                    metadata={
                        "dataset": "allen",
                        "session": str(session_id),
                        "probe": probe.name,
                        "channel": int(row.channel),
                        "depth_um": float(row.depth_um),
                        "acronym": str(row.acronym),
                        "region": str(row.region),
                        "group": probe.key,
                        "lateral_um": float(row.lateral_um),
                        "ccf_ap_um": float(row.ccf_ap_um),
                        "ccf_dv_um": float(row.ccf_dv_um),
                        "ccf_lr_um": float(row.ccf_lr_um),
                    },
                    t0_s=t0,
                )

            labelled = table[table["region"] != UNKNOWN]
            sources.append(
                SourceRecord(
                    dataset="allen",
                    session=str(session_id),
                    probe=probe.name,
                    group=probe.key,
                    channels_total=len(table),
                    channels_kept=len(usable),
                    channels_unlabelled=int((table["region"] == UNKNOWN).sum()),
                    channels_dropped_quality=len(labelled) - len(usable),
                    chunks=written,
                    region_channels=summarise_channels(usable),
                    notes=f"probe {probe.probe_id}, fs={fs:.4f} Hz",
                )
            )
            print(f"{probe.key}: {len(usable)} channels, {written} chunks", flush=True)
            del signal

    if writer is None:
        raise RuntimeError("no probe produced any data")
    store = writer.close()
    card = DatasetCard.from_store(store, sources=sources, skipped=skipped)
    card.write()
    return store, card
