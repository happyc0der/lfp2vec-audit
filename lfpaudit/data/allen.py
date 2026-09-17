"""Allen Visual Coding Neuropixels LFP. Implemented in Stage 1.

Plan: read the per-probe LFP NWB files directly from the public S3 bucket with h5py rather than
through AllenSDK, whose dependency pins conflict with a modern torch stack. Channel labels come
from the session file's ``ecephys_structure_acronym`` column.
"""

from __future__ import annotations

#: First session in the upstream hard-coded list; a full session is roughly 15 GB.
PAPER_SESSIONS: tuple[int, ...] = (
    719161530,
    794812542,
    778998620,
    798911424,
    771990200,
    771160300,
    768515987,
)


def build_allen_store(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    raise NotImplementedError("Stage 1: Allen loader not implemented yet")
