"""wav2vec2 with a region head: the reduced LFP2Vec reproduction.

The published method has three stages: start from an audio checkpoint, continue self-supervised
training on unlabelled local field potential, then fine-tune for region classification. Only the
first and third are here. The middle stage needs a day on a datacentre accelerator, and the
paper's own ablation reports that audio initialisation with about six thousand trials matches
random initialisation with over four hundred thousand, which is evidence that the audio prior
carries a large part of the benefit. The omission is recorded as D1 in ``docs/DEVIATIONS.md`` and
every number produced here is a number for the reduced method, not for the published one.

The classification head is the stock Hugging Face one, which already matches the architecture the
paper describes: project the 768-dimensional hidden states to 256, mean-pool over time, classify.
That pooled 256-dimensional vector is also what the lab-identity probe reads, since the question
of whether fine-tuning removes acquisition structure is a question about the representation the
classifier actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfpaudit import REGIONS

DEFAULT_MODEL = "facebook/wav2vec2-base"


@dataclass(frozen=True)
class ModelInfo:
    """What was actually built, recorded in the run manifest."""

    name: str
    num_labels: int
    total_parameters: int
    trainable_parameters: int
    frozen_feature_encoder: bool
    frozen_layers: int

    def describe(self) -> str:
        share = self.trainable_parameters / max(self.total_parameters, 1)
        return (
            f"{self.name}: {self.trainable_parameters / 1e6:.1f}M trainable of "
            f"{self.total_parameters / 1e6:.1f}M ({share:.0%})"
        )


def build_model(
    model_name: str = DEFAULT_MODEL,
    num_labels: int = len(REGIONS),
    freeze_feature_encoder: bool = True,
    freeze_layers: int = 0,
):
    """Load the audio checkpoint with a fresh region head.

    The convolutional feature encoder is frozen by default. That is standard practice for
    wav2vec2 fine-tuning, and it also removes the most expensive part of the backward pass, which
    on laptop-class hardware is the difference between a run that fits in an evening and one that
    does not. ``freeze_layers`` additionally freezes the lowest transformer blocks, available as a
    fallback if measured throughput comes in below what the run budget allows.
    """
    from transformers import Wav2Vec2ForSequenceClassification

    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    for index, layer in enumerate(model.wav2vec2.encoder.layers):
        if index < freeze_layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    return model, describe_model(
        model, model_name, num_labels, freeze_feature_encoder, freeze_layers
    )


def describe_model(
    model,
    model_name: str,
    num_labels: int,
    frozen_feature_encoder: bool,
    frozen_layers: int,
) -> ModelInfo:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelInfo(
        name=model_name,
        num_labels=num_labels,
        total_parameters=int(total),
        trainable_parameters=int(trainable),
        frozen_feature_encoder=frozen_feature_encoder,
        frozen_layers=frozen_layers,
    )


def pooled_embeddings(model, input_values) -> np.ndarray:
    """The representation the classifier sees, for one batch.

    Reproduces the head's own pooling rather than approximating it: project the last hidden state,
    then average over time. Running the lab-identity probe on anything else would answer a
    different question from the one being asked.
    """
    import torch

    with torch.no_grad():
        hidden = model.wav2vec2(input_values).last_hidden_state
        pooled = model.projector(hidden).mean(dim=1)
    return pooled.float().cpu().numpy()
