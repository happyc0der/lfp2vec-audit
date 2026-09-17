"""lfpaudit: reproduction and audit of LFP2Vec-style anatomical localisation from raw LFP.

The package deliberately separates three concerns:

* ``lfpaudit.data``    -- fetching, chunking and splitting Neuropixels LFP with CCF labels.
* ``lfpaudit.models``  -- the classifiers under test, from a constant predictor to wav2vec2.
* ``lfpaudit.eval``    -- everything that is computed *after* a model has written its logits.

Every experiment writes its outputs to ``results/<experiment>/<run_id>/`` and every figure is
generated from those files alone, so no number in the repository is unreproducible.
"""

__version__ = "0.1.0"

#: Region classes used throughout, matching the five-way rodent task in the LFP2Vec paper.
REGIONS: tuple[str, ...] = ("CA1", "CA2", "CA3", "DG", "VIS")

#: Label given to channels whose CCF acronym falls outside :data:`REGIONS`. These are dropped
#: from every supervised experiment but counted in the dataset card.
UNKNOWN = "UNK"

REGION_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(REGIONS)}

__all__ = ["REGIONS", "REGION_TO_INDEX", "UNKNOWN", "__version__"]
