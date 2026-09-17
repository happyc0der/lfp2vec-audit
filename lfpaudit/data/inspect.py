"""Looking at the data before trusting it.

Every check elsewhere in this package is one a broken pipeline could pass. Labels can be shifted
by one channel, a filter can invert a spectrum, a unit conversion can be off by a million, and
none of that shows up as an exception or as chance-level accuracy. The cheapest defence is to
plot a few real chunks per region and see whether they look like the tissue they claim to come
from: ripple-band bursts on hippocampal pyramidal layers, strong theta in dentate gyrus, a
different spectral tilt in cortex.

The figure is produced from the CLI rather than a notebook so it regenerates deterministically
and its inputs are recorded in the lab notebook alongside everything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lfpaudit import REGIONS
from lfpaudit.data.chunk import ChunkStore
from lfpaudit.features.bandpower import power_spectrum


def inspection_figure(
    store: ChunkStore,
    out_path: str | Path,
    n_per_region: int = 4,
    seed: int = 0,
    title: str | None = None,
) -> Path:
    """Plot sampled chunks and their spectra, one row per region.

    Each row shows a few seconds of real signal from different channels and sessions, then that
    region's mean power spectrum against the store-wide mean, so a region with no distinguishing
    spectral content is visible as a flat difference rather than having to be inferred from a
    disappointing accuracy later.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index = store.index
    rng = np.random.default_rng(seed)
    present = [r for r in REGIONS if (index["region"] == r).any()]
    if not present:
        raise ValueError("this store contains no labelled regions")

    freqs, overall = None, None
    fig, axes = plt.subplots(
        len(present),
        n_per_region + 1,
        figsize=(3.0 * (n_per_region + 1), 2.1 * len(present)),
        squeeze=False,
    )

    # A store-wide mean spectrum gives each region's curve something to be compared against.
    sample_rows = rng.choice(len(index), size=min(len(index), 400), replace=False)
    freqs, psd = power_spectrum(store.take(sample_rows), fs=store.fs)
    overall = psd.mean(axis=0)

    for row_i, region in enumerate(present):
        rows = np.flatnonzero((index["region"] == region).to_numpy())
        # Spread the examples over distinct channels so one noisy channel cannot fill a row.
        channels = index.iloc[rows]["channel"].to_numpy()
        pick: list[int] = []
        for channel in rng.permutation(np.unique(channels)):
            candidates = rows[channels == channel]
            pick.append(int(rng.choice(candidates)))
            if len(pick) == n_per_region:
                break

        waveforms = store.take(pick, microvolts=True)
        time = np.arange(store.n_samples) / store.fs
        for col, (position, wave) in enumerate(zip(pick, waveforms, strict=True)):
            meta = index.iloc[position]
            ax = axes[row_i][col]
            ax.plot(time, wave, linewidth=0.4, color="#1f4e79")
            ax.set_title(
                f"{meta['session']} ch{int(meta['channel'])} ({meta['acronym']})", fontsize=7
            )
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{region}\nmicrovolts", fontsize=8)
            if row_i == len(present) - 1:
                ax.set_xlabel("seconds", fontsize=7)

        region_rows = rng.choice(rows, size=min(len(rows), 300), replace=False)
        _, region_psd = power_spectrum(store.take(region_rows), fs=store.fs)
        ax = axes[row_i][-1]
        band = freqs > 0
        ax.loglog(freqs[band], overall[band], color="#999999", linewidth=1.0, label="all regions")
        ax.loglog(
            freqs[band], region_psd.mean(axis=0)[band], color="#c1440e", linewidth=1.2, label=region
        )
        ax.set_title(f"{region} mean spectrum (n={len(rows)})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        if row_i == len(present) - 1:
            ax.set_xlabel("Hz", fontsize=7)

    fig.suptitle(
        title or f"{Path(store.path).name}: sampled chunks by region", fontsize=10, y=0.999
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
