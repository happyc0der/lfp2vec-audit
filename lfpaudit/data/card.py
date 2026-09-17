"""Dataset cards: what a chunk store actually contains.

A store is a memory-mapped array and a parquet index, which is convenient for a model and opaque
to a reader. The card is the human-facing counterpart, recording how many channels each probe
contributed, how many were discarded as unlabelled or empty, and how lopsided the class balance
is. That last point matters here: CA2 appears on a handful of channels at best, so a macro
averaged score is dominated by a class with almost no support, and anybody reading a result table
needs to see that without digging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from lfpaudit import REGIONS, UNKNOWN
from lfpaudit.data.chunk import ChunkStore


@dataclass
class SourceRecord:
    """One probe's contribution, including what it lost on the way in."""

    dataset: str
    session: str
    probe: str
    group: str
    channels_total: int
    channels_kept: int
    channels_unlabelled: int
    channels_dropped_quality: int
    chunks: int
    region_channels: dict[str, int] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DatasetCard:
    """Everything a reader needs to judge what a store is made of."""

    store: str
    fs: float
    window_s: float
    n_chunks: int
    n_samples: int
    sources: list[SourceRecord] = field(default_factory=list)
    region_chunks: dict[str, int] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)

    @classmethod
    def from_store(
        cls,
        store: ChunkStore,
        sources: list[SourceRecord] | None = None,
        skipped: list[dict] | None = None,
    ) -> DatasetCard:
        index = store.index
        counts = index["region"].value_counts().to_dict() if len(index) else {}
        return cls(
            store=str(store.path),
            fs=store.fs,
            window_s=store.n_samples / store.fs,
            n_chunks=len(index),
            n_samples=store.n_samples,
            sources=sources or [],
            region_chunks={r: int(counts.get(r, 0)) for r in REGIONS},
            skipped=skipped or [],
        )

    def write(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else Path(self.store) / "card.json"
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return target

    @classmethod
    def read(cls, path: str | Path) -> DatasetCard:
        payload = json.loads(Path(path).read_text())
        payload["sources"] = [SourceRecord(**s) for s in payload.get("sources", [])]
        return cls(**payload)

    def to_markdown(self) -> str:
        """A table per store, listing each probe's channels by region and what was dropped."""
        lines = [
            f"### `{Path(self.store).name}`",
            "",
            f"{self.n_chunks} chunks of {self.window_s:.1f} s at {self.fs:.3f} Hz "
            f"({self.n_samples} samples each).",
            "",
            "| group | channels kept | "
            + " | ".join(REGIONS)
            + " | unlabelled | dropped | chunks |",
            "|---|---:|" + "---:|" * len(REGIONS) + "---:|---:|---:|",
        ]
        for source in self.sources:
            per_region = " | ".join(str(source.region_channels.get(r, 0)) for r in REGIONS)
            lines.append(
                f"| {source.group} | {source.channels_kept} | {per_region} | "
                f"{source.channels_unlabelled} | {source.channels_dropped_quality} | "
                f"{source.chunks} |"
            )
        total = " | ".join(str(self.region_chunks.get(r, 0)) for r in REGIONS)
        lines += ["", "Chunks per region: " + total.replace(" | ", " / ") + ".", ""]

        if self.skipped:
            lines.append("Skipped sources:")
            lines += [f"- `{s['key']}`: {s['reason']}" for s in self.skipped]
            lines.append("")
        return "\n".join(lines)


def summarise_channels(table: pd.DataFrame) -> dict[str, int]:
    """Channel counts per region for a channel table, including the unlabelled bucket."""
    counts = table["region"].value_counts().to_dict()
    summary = {region: int(counts.get(region, 0)) for region in REGIONS}
    summary[UNKNOWN] = int(counts.get(UNKNOWN, 0))
    return summary
