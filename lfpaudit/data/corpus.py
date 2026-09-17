"""Treating several chunk stores as one dataset.

Each dataset is built into its own store, because they are fetched at different times from
different servers and a failure in one should not invalidate the other. Cross-lab experiments
need them side by side, which is all this module does: concatenate the indices with chunk ids
made globally unique, and route reads back to whichever store a row came from.

Keeping the stores separate rather than merging the arrays on disk means adding a dataset later
costs nothing, and the existing split machinery works unchanged on the merged index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lfpaudit.data.chunk import ChunkStore

#: Chunk ids are offset by this much per store so ids stay unique after concatenation. Stores
#: hold far fewer rows than this, and the gap keeps ids readable: store 1 starts at 10000000.
ID_STRIDE = 10_000_000

#: Sampling rates are compared with this tolerance. The two datasets arrive at 1250 Hz by
#: different routes and land on 1250.012 and 1249.999, a difference of one part in a hundred
#: thousand, which is irrelevant to a 3-second window but would fail an equality check.
FS_TOLERANCE_HZ = 0.1


@dataclass
class Corpus:
    """Several stores presented as a single index plus a read method."""

    stores: dict[str, ChunkStore]
    index: pd.DataFrame

    @classmethod
    def load(cls, paths: dict[str, str | Path]) -> Corpus:
        stores: dict[str, ChunkStore] = {}
        frames: list[pd.DataFrame] = []

        for offset, (name, path) in enumerate(sorted(paths.items())):
            store = ChunkStore.open(path)
            stores[name] = store
            frame = store.index.copy()
            frame["store"] = name
            frame["row"] = np.arange(len(frame), dtype=np.int64)
            frame["fs"] = store.fs
            frame["chunk_id"] = frame["row"] + offset * ID_STRIDE
            frames.append(frame)

        if not frames:
            raise ValueError("a corpus needs at least one store")

        widths = {s.n_samples for s in stores.values()}
        if len(widths) != 1:
            raise ValueError(f"stores disagree on chunk length: {sorted(widths)}")
        rates = [s.fs for s in stores.values()]
        if max(rates) - min(rates) > FS_TOLERANCE_HZ:
            raise ValueError(f"stores disagree on sampling rate: {rates}")

        index = pd.concat(frames, ignore_index=True)
        return cls(stores=stores, index=index)

    @property
    def fs(self) -> float:
        """Nominal rate for the corpus, the mean of its stores' rates."""
        return float(np.mean([s.fs for s in self.stores.values()]))

    @property
    def n_samples(self) -> int:
        return next(iter(self.stores.values())).n_samples

    def take(self, chunk_ids: np.ndarray | list[int], microvolts: bool = False) -> np.ndarray:
        """Load waveforms for the given global chunk ids, in the order requested."""
        wanted = pd.DataFrame({"chunk_id": np.asarray(chunk_ids, dtype=np.int64)})
        wanted["order"] = np.arange(len(wanted), dtype=np.int64)
        merged = wanted.merge(
            self.index[["chunk_id", "store", "row"]], on="chunk_id", how="left", validate="m:1"
        )
        if merged["store"].isna().any():
            missing = merged.loc[merged["store"].isna(), "chunk_id"].tolist()[:5]
            raise KeyError(f"chunk ids not present in this corpus: {missing}")

        out = np.empty((len(merged), self.n_samples), dtype=np.float32)
        for name, part in merged.groupby("store", sort=False):
            payload = self.stores[str(name)].take(part["row"].to_numpy(), microvolts=microvolts)
            out[part["order"].to_numpy()] = payload
        return out

    def positions(self, chunk_ids: np.ndarray | list[int]) -> np.ndarray:
        """Row positions in :attr:`index` for the given chunk ids."""
        lookup = pd.Series(
            np.arange(len(self.index), dtype=np.int64), index=self.index["chunk_id"].to_numpy()
        )
        return lookup.loc[np.asarray(chunk_ids, dtype=np.int64)].to_numpy()
