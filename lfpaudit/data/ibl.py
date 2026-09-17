"""International Brain Laboratory Neuropixels LFP.

The upstream LFP2Vec pipeline downloads each insertion's whole ``.lf.cbin`` (2.4-3.8 GB) and
reads the first 500 seconds of it. That is a factor of ten more bytes than the analysis uses.
Here the same 500 seconds is fetched as a byte prefix instead.

This is safe because of how mtscomp lays out a compressed recording: fixed one-second chunks,
stored in order, with a sidecar ``.ch`` listing each chunk's byte offset. The first N seconds are
therefore exactly the first ``chunk_offsets[N]`` bytes of the file, and a truncated ``.ch``
describing only those chunks makes the prefix a valid standalone recording. ``mtscomp.Reader``
already implements this as :meth:`~mtscomp.Reader.chop` for local files; the same header surgery
is done here against a remote byte range, so nothing is downloaded that is not read.

Preprocessing follows the upstream code rather than the paper where the two differ; see
``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfpaudit import UNKNOWN
from lfpaudit.data.card import DatasetCard, SourceRecord, summarise_channels
from lfpaudit.data.chunk import ChunkStore, ChunkWriter, chunk_signal
from lfpaudit.data.labels import map_acronym
from lfpaudit.data.remote import download

#: Native sampling rate of the LF band. The exact value per file is read from its ``.ch``.
NOMINAL_LF_FS = 2500.0

#: Sampling rate every downstream stage assumes, reached by decimating the LF band by two.
TARGET_FS = 1250.0

#: Neuropixels 1.0 geometry: 384 recording channels plus one sync channel, two channels per
#: 20 micrometre row in a staggered pattern.
NP1_CHANNELS = 384
NP1_ROW_PITCH_UM = 20.0


@dataclass(frozen=True)
class Insertion:
    """One probe insertion, identified as the upstream code identifies it (session + probe)."""

    eid: str
    probe: str
    pid: str
    subject: str
    lab: str

    @property
    def key(self) -> str:
        """Stable name used for the session column and as the split grouping unit."""
        return f"{self.eid[:8]}_{self.probe}"


#: The seven insertions the LFP2Vec training script names for IBL. Probe insertion ids and
#: metadata were resolved against the public Alyx database on 2026-09-17; all seven are
#: Neuropixels 1.0 with resolved histological alignment.
PAPER_INSERTIONS: tuple[Insertion, ...] = (
    Insertion(
        "d2832a38-27f6-452d-91d6-af72d794136c",
        "probe00",
        "b25799a5-09e8-4656-9c1b-44bc9cbb5279",
        "ibl_witten_29",
        "wittenlab",
    ),
    Insertion(
        "0802ced5-33a3-405e-8336-b65ebc5cb07c",
        "probe00",
        "7d999a68-0215-4e45-8e6c-879c6ca2b771",
        "ZFM-02373",
        "mainenlab",
    ),
    Insertion(
        "0802ced5-33a3-405e-8336-b65ebc5cb07c",
        "probe01",
        "3eb6e6e0-8a57-49d6-b7c9-f39d5834e682",
        "ZFM-02373",
        "mainenlab",
    ),
    Insertion(
        "0a018f12-ee06-4b11-97aa-bbbff5448e9f",
        "probe00",
        "16799c7a-e395-435d-a4c4-a678007e1550",
        "KS051",
        "cortexlab",
    ),
    Insertion(
        "3638d102-e8b6-4230-8742-e548cd87a949",
        "probe01",
        "143dd7cf-6a47-47a1-906d-927ad7fe9117",
        "SWC_058",
        "mrsicflogellab",
    ),
    Insertion(
        "5dcee0eb-b34d-4652-acc3-d10afc6eae68",
        "probe00",
        "9e44ddb5-7c7c-48f1-954a-6cec2ad26088",
        "KS075",
        "cortexlab",
    ),
    Insertion(
        "54238fd6-d2d0-4408-b1a9-d19d24fd29ce",
        "probe00",
        "f26a6ab1-7e37-4f8d-bb50-295c056e1062",
        "DY_018",
        "danlab",
    ),
)


def connect(cache_dir: str | Path | None = None):
    """Open a connection to the public IBL database.

    Credentials are the documented public ones. The returned object caches its authentication
    token, so only the first call in a session touches the login endpoint.
    """
    from one.api import ONE

    kwargs: dict[str, Any] = {
        "base_url": "https://openalyx.internationalbrainlab.org",
        "password": "international",
        "silent": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return ONE(**kwargs)


def dataset_url(one, eid: str, collection: str, suffix: str) -> str:
    """Resolve one dataset to a public HTTPS URL without downloading it.

    Datasets are usually mirrored on both S3 and the Flatiron server. S3 is preferred because it
    has been the more reliable of the two for ranged requests.
    """
    records = one.alyx.rest(
        "datasets", "list", session=eid, django=f"name__endswith,{suffix}", no_cache=True
    )
    matching = [r for r in records if r["collection"] == collection]
    if not matching:
        raise FileNotFoundError(f"no dataset ending in {suffix!r} in {collection!r} for {eid}")

    urls = [
        url
        for record in matching
        for file_record in record.get("file_records", [])
        if file_record.get("exists", False) and (url := file_record.get("data_url"))
    ]
    if not urls:
        raise FileNotFoundError(f"dataset {suffix!r} in {collection!r} has no accessible mirror")
    return next((u for u in urls if "s3" in u), urls[0])


def fetch_prefix(
    one,
    insertion: Insertion,
    seconds: float,
    cache_dir: str | Path,
) -> Path:
    """Download the first ``seconds`` of an insertion's LF band as a standalone recording.

    Returns the directory holding a ``.cbin``/``.ch``/``.meta`` triple that
    ``spikeglx.Reader`` opens like any other file. The ``.ch`` is rewritten to describe only the
    chunks present, exactly as :meth:`mtscomp.Reader.chop` does for local files: bounds and
    offsets truncated, both whole-file SHA-1 digests set to null (they describe bytes that are no
    longer there), and a ``chopped`` flag recorded.
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / "prefix" / insertion.key
    out_dir.mkdir(parents=True, exist_ok=True)
    collection = f"raw_ephys_data/{insertion.probe}"

    # The .ch and .meta are small; let ONE fetch and cache them normally.
    source_ch = one.load_dataset(
        insertion.eid, "*.lf.ch", collection=collection, download_only=True
    )
    source_meta = one.load_dataset(
        insertion.eid, "*.lf.meta", collection=collection, download_only=True
    )
    header = json.loads(Path(source_ch).read_text())

    fs = float(header["sample_rate"])
    bounds = header["chunk_bounds"]
    offsets = header["chunk_offsets"]
    wanted_samples = int(round(seconds * fs))
    n_chunks = int(np.searchsorted(bounds, wanted_samples, side="left"))
    n_chunks = max(1, min(n_chunks, len(bounds) - 1))
    prefix_bytes = int(offsets[n_chunks])

    stem = Path(source_ch).stem  # e.g. _spikeglx_ephysData_g0_t0.imec0.lf
    cbin_path = out_dir / f"{stem}.cbin"
    url = dataset_url(one, insertion.eid, collection, "lf.cbin")
    print(
        f"{insertion.key}: fetching {prefix_bytes / 1e6:.0f} MB "
        f"({n_chunks} chunks, {bounds[n_chunks] / fs:.1f} s of {bounds[-1] / fs:.0f} s)",
        flush=True,
    )
    download(url, cbin_path, expected_bytes=prefix_bytes)

    header["chunk_bounds"] = bounds[: n_chunks + 1]
    header["chunk_offsets"] = offsets[: n_chunks + 1]
    header["sha1_compressed"] = None
    header["sha1_uncompressed"] = None
    header["chopped"] = True
    (out_dir / f"{stem}.ch").write_text(json.dumps(header, indent=2, sort_keys=True))
    shutil.copyfile(source_meta, out_dir / f"{stem}.meta")
    return cbin_path


def read_window(cbin_path: str | Path, seconds: float | None = None) -> tuple[np.ndarray, float]:
    """Read a prefix recording into memory as ``(n_channels, n_samples)`` in volts.

    The sync channel is dropped, matching the upstream ``[:, :-sr.nsync]``.
    """
    import spikeglx

    # The .meta still describes the full recording, so the reader would warn that its duration
    # disagrees with the chunks present and then correct itself. That mismatch is the entire
    # point of a prefix, so the warning is declined rather than printed on every build.
    reader = spikeglx.Reader(str(cbin_path), ignore_warnings=True)
    try:
        fs = float(reader.fs)
        available = reader.ns
        n_samples = available if seconds is None else min(available, int(round(seconds * fs)))
        data = reader[0:n_samples, : -reader.nsync]
        return np.ascontiguousarray(data.T, dtype=np.float64), fs
    finally:
        reader.close()


def _assert_np1_channel_order(local: np.ndarray, key: str) -> None:
    """Fail loudly if an electrode table is not in raw channel order.

    Anatomy is joined to data columns by position, so a reordered table would mislabel every
    channel while looking perfectly healthy downstream. A Neuropixels 1.0 shank has a known
    geometry that any correctly ordered table must satisfy: channels ascend the shank two at a
    time, one row every 20 micrometres, with lateral positions drawn from four staggered
    columns. Checking the pattern rather than exact values keeps this robust to the base offset,
    which is 20 micrometres in the released data rather than the zero one might assume.
    """
    axial, lateral = local[:, 1], local[:, 0]

    rows = np.unique(axial)
    pitch = np.unique(np.round(np.diff(rows), 6))
    if pitch.size != 1 or not np.isclose(pitch[0], NP1_ROW_PITCH_UM):
        raise ValueError(f"{key}: axial row pitch is {pitch.tolist()}, expected {NP1_ROW_PITCH_UM}")
    if np.any(np.diff(axial) < 0):
        raise ValueError(f"{key}: axial coordinates are not ascending; table has been reordered")
    if not np.allclose(axial[0::2], axial[1::2]):
        raise ValueError(f"{key}: channels are not paired two per row; table has been reordered")
    if len(np.unique(lateral)) != 4:
        raise ValueError(
            f"{key}: expected 4 staggered lateral columns, found {len(np.unique(lateral))}"
        )


def load_channel_table(one, insertion: Insertion) -> pd.DataFrame:
    """Per-channel anatomy and geometry for an insertion.

    Reads the ``electrodeSites`` objects, three small arrays produced by histological alignment,
    rather than going through the full spike-sorting loader. That avoids pulling in the heavy
    analysis stack and downloading an atlas volume, for the same numbers.
    """
    from iblatlas.atlas import BrainRegions

    sites = one.load_object(
        insertion.eid, "electrodeSites", collection=f"alf/{insertion.probe}", download_only=False
    )
    atlas_ids = np.asarray(sites["brainLocationIds_ccf_2017"]).ravel()
    local = np.asarray(sites["localCoordinates"], dtype=np.float64)
    mlapdv = np.asarray(sites["mlapdv"], dtype=np.float64)

    if len(atlas_ids) != NP1_CHANNELS:
        raise ValueError(f"{insertion.key}: expected {NP1_CHANNELS} sites, got {len(atlas_ids)}")

    _assert_np1_channel_order(local, insertion.key)

    acronyms = BrainRegions().get(atlas_ids)["acronym"]
    return pd.DataFrame(
        {
            "channel": np.arange(NP1_CHANNELS, dtype=np.int64),
            "acronym": [str(a) for a in acronyms],
            "region": [map_acronym(a) for a in acronyms],
            "depth_um": local[:, 1],
            "lateral_um": local[:, 0],
            "ccf_ap_um": mlapdv[:, 1],
            "ccf_dv_um": mlapdv[:, 2],
            "ccf_lr_um": mlapdv[:, 0],
        }
    )


def preprocess_lfp(
    x_volts: np.ndarray, fs: float, target_fs: float = TARGET_FS
) -> tuple[np.ndarray, float, np.ndarray]:
    """Destripe, decimate and high-pass a raw LF window, following the upstream recipe.

    Returns the signal in microvolts at ``target_fs``, the achieved rate, and the per-channel
    quality labels from bad-channel detection (0 good, 1 dead, 2 noisy, 3 outside the brain).
    Those labels are kept so the dataset card can report how many channels were interpolated
    rather than silently measured.

    Note the library's destriping default is a 0.5 Hz high-pass corner, while the paper's text
    says 2 Hz. The code is followed here and the discrepancy is recorded in ``docs/DEVIATIONS.md``.
    """
    from ibldsp.voltage import destripe_lfp, detect_bad_channels
    from scipy.signal import butter, sosfiltfilt

    x_volts = np.asarray(x_volts, dtype=np.float64)
    if x_volts.shape[0] != NP1_CHANNELS:
        raise ValueError(f"expected {NP1_CHANNELS} channels, got {x_volts.shape[0]}")

    channel_labels, _ = detect_bad_channels(x_volts, fs=fs)
    destriped = destripe_lfp(x_volts, fs=fs, channel_labels=channel_labels)

    factor = int(round(fs / target_fs))
    if factor > 1:
        # Anti-alias below the new Nyquist before throwing samples away.
        sos = butter(4, (0.5 * target_fs) / (0.5 * fs), btype="low", output="sos")
        destriped = sosfiltfilt(sos, destriped, axis=1)
        destriped = destriped[:, ::factor]
    achieved = fs / factor

    # Remove the residual slow drift the band-pass leaves behind.
    sos = butter(2, 0.1 / (0.5 * achieved), btype="highpass", output="sos")
    destriped = sosfiltfilt(sos, destriped, axis=1)

    return destriped * 1e6, achieved, channel_labels


def build_ibl_store(
    out_path: str | Path,
    insertions: Sequence[Insertion] | None = None,
    cache_dir: str | Path = "data/ibl-cache",
    window_s: float = 3.0,
    start_s: float = 200.0,
    chunks_per_channel: int = 100,
    one=None,
) -> tuple[ChunkStore, DatasetCard]:
    """Fetch, preprocess and chunk the IBL insertions into a single store.

    The fetched window runs from zero to just past the last chunk, because destriping and
    decimation are applied to a whole window at once and doing that only over the part that is
    kept would introduce filter edge effects inside the data.
    """
    insertions = list(insertions if insertions is not None else PAPER_INSERTIONS)
    one = one or connect(cache_dir)
    needed_s = start_s + window_s * chunks_per_channel
    # A margin so the final chunk is complete after resampling rounds the sample count down.
    fetch_s = needed_s + 2.0

    writer: ChunkWriter | None = None
    sources: list[SourceRecord] = []
    skipped: list[dict] = []

    for insertion in insertions:
        try:
            table = load_channel_table(one, insertion)
            cbin = fetch_prefix(one, insertion, seconds=fetch_s, cache_dir=cache_dir)
            raw, fs_in = read_window(cbin)
            signal, fs, quality = preprocess_lfp(raw, fs_in)
            del raw
        except Exception as error:  # noqa: BLE001 - one bad insertion must not lose the rest
            skipped.append({"key": insertion.key, "reason": f"{type(error).__name__}: {error}"})
            print(f"{insertion.key}: SKIPPED ({error})", flush=True)
            continue

        n_samples = int(round(window_s * fs))
        if writer is None:
            writer = ChunkWriter(out_path, n_samples=n_samples, fs=fs)
        elif n_samples != writer.n_samples:
            skipped.append(
                {"key": insertion.key, "reason": f"chunk length {n_samples} != {writer.n_samples}"}
            )
            continue

        labelled = table[table["region"] != UNKNOWN]
        # Quality label 0 is a healthy channel; anything else was dead, noisy or outside the
        # brain and was reconstructed by interpolation from its neighbours during destriping.
        healthy = labelled[quality[labelled["channel"].to_numpy()] == 0]

        written = 0
        for row in healthy.itertuples():
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
                    "dataset": "ibl",
                    "session": insertion.key,
                    "probe": insertion.probe,
                    "channel": int(row.channel),
                    "depth_um": float(row.depth_um),
                    "acronym": str(row.acronym),
                    "region": str(row.region),
                    "group": insertion.key,
                    "lateral_um": float(row.lateral_um),
                    "ccf_ap_um": float(row.ccf_ap_um),
                    "ccf_dv_um": float(row.ccf_dv_um),
                    "ccf_lr_um": float(row.ccf_lr_um),
                },
                t0_s=t0,
            )

        sources.append(
            SourceRecord(
                dataset="ibl",
                session=insertion.key,
                probe=insertion.probe,
                group=insertion.key,
                channels_total=len(table),
                channels_kept=len(healthy),
                channels_unlabelled=int((table["region"] == UNKNOWN).sum()),
                channels_dropped_quality=len(labelled) - len(healthy),
                chunks=written,
                region_channels=summarise_channels(healthy),
                notes=f"{insertion.subject} ({insertion.lab}), fs={fs:.4f} Hz",
            )
        )
        print(f"{insertion.key}: {len(healthy)} channels, {written} chunks", flush=True)
        del signal

    if writer is None:
        raise RuntimeError("no insertion produced any data")
    store = writer.close()
    card = DatasetCard.from_store(store, sources=sources, skipped=skipped)
    card.write()
    return store, card
