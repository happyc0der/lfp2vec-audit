"""Shared fixtures. Everything here is synthetic; no test touches the network."""

from __future__ import annotations

import pytest

from lfpaudit.data.synthetic import SyntheticSpec, build_synthetic_store


@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory):
    """A small two-dataset synthetic store reused across tests."""
    path = tmp_path_factory.mktemp("store")
    spec = SyntheticSpec(
        datasets=("synthA", "synthB"),
        sessions_per_dataset=3,
        channels_per_session=10,
        chunks_per_channel=6,
        seed=0,
    )
    return build_synthetic_store(str(path), spec)
