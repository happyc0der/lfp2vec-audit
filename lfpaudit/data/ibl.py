"""IBL Neuropixels LFP: download, destripe, label. Implemented in Stage 1.

Plan: use ONE-api against ``openalyx.internationalbrainlab.org`` to stream the LF band for the
seven insertions used by LFP2Vec, take the first 500 s as the upstream code does, destripe with
``ibldsp.voltage.destripe_lfp``, and label channels from
``SpikeSortingLoader.load_channels()['acronym']``.
"""

from __future__ import annotations

#: Probe insertions used by the upstream LFP2Vec preprocessing, as (eid, probe) pairs.
PAPER_INSERTIONS: tuple[tuple[str, str], ...] = (
    ("0802ced5-33a3-405e-8336-b65ebc5cb07c", "probe00"),
    ("0802ced5-33a3-405e-8336-b65ebc5cb07c", "probe01"),
    ("0a018f12-ee06-4b11-97aa-bbbff5448e9f", "probe00"),
    ("3638d102-e8b6-4230-8742-e548cd87a949", "probe01"),
    ("5dcee0eb-b34d-4652-acc3-d10afc6eae68", "probe00"),
    ("d2832a38-27f6-452d-91d6-af72d794136c", "probe00"),
    ("54238fd6-d2d0-4408-b1a9-d19d24fd29ce", "probe00"),
)


def build_ibl_store(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    raise NotImplementedError("Stage 1: IBL loader not implemented yet")
