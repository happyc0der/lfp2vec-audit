"""Caching computed features, with an expiry condition that actually works.

Features are computed once per store and reused across every fold, which is what makes
leave-one-session-out evaluation cost the same as a single split. The risk that introduces is a
stale cache: a store gets rebuilt, the features on disk no longer describe it, and every
subsequent number is quietly wrong in a way no test would catch.

So a cache entry records the fingerprint of the data it was built from, and refuses itself when
that no longer matches. The fingerprint is the store's row count and the hash of its index rather
than of the waveform array, because hashing a gigabyte of float16 on every run would cost more
than recomputing most features.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def index_fingerprint(index: pd.DataFrame) -> str:
    """A short, stable digest of a chunk index.

    Covers the identifying columns rather than the whole frame, so adding an unrelated column
    later does not invalidate every cached feature, while any change to which chunks exist or
    what they are labelled does.
    """
    columns = [c for c in ("chunk_id", "group", "channel", "region", "t0_s") if c in index.columns]
    if not columns:
        raise ValueError("chunk index has none of the identifying columns")
    payload = index[columns].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class FeatureCache:
    """Named feature arrays for one store, keyed on that store's fingerprint."""

    def __init__(self, root: str | Path, index: pd.DataFrame) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.fingerprint = index_fingerprint(index)

    def _paths(self, name: str) -> tuple[Path, Path]:
        return self.root / f"{name}.npy", self.root / f"{name}.json"

    def load(self, name: str) -> np.ndarray | None:
        """Return a cached array, or None when it is absent or no longer describes the store."""
        array_path, meta_path = self._paths(name)
        if not (array_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != self.fingerprint:
            return None
        values = np.load(array_path)
        if len(values) != len(self.index):
            return None
        return values

    def save(self, name: str, values: np.ndarray, columns: list[str] | None = None) -> np.ndarray:
        values = np.asarray(values)
        if len(values) != len(self.index):
            raise ValueError(f"{name}: {len(values)} rows for an index of {len(self.index)}")
        array_path, meta_path = self._paths(name)
        np.save(array_path, values)
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "shape": list(values.shape),
                    "columns": columns or [],
                },
                indent=2,
            )
        )
        return values

    def get_or_build(
        self,
        name: str,
        builder: Callable[[], np.ndarray],
        columns: list[str] | None = None,
        rebuild: bool = False,
    ) -> np.ndarray:
        """Return the cached array for ``name``, building and storing it when necessary."""
        if not rebuild:
            cached = self.load(name)
            if cached is not None:
                return cached
        return self.save(name, builder(), columns=columns)
