"""wav2vec2-base with a five-way region head: the LFP2Vec reproduction. Stage 3.

The self-supervised continuation stage of the original work is out of scope on laptop-class
compute; this module starts from the audio checkpoint and fine-tunes, which the paper's own
ablation shows is the larger part of the benefit.
"""

from __future__ import annotations


def build_model(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    raise NotImplementedError("Stage 3: model not implemented yet")
