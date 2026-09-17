"""Typed configuration objects and seed control.

Experiments are described by YAML files in ``experiments/``; each file deserialises into an
:class:`ExperimentConfig` so that a run can be reproduced from the manifest alone.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml

T = TypeVar("T")

#: Canonical band edges in Hz, following the LFP-LOC feature definition (Perna et al., 2026).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
    "ripple": (100.0, 250.0),
}


@dataclass
class DataConfig:
    """How raw LFP becomes fixed-length chunks.

    Defaults mirror the upstream LFP2Vec preprocessing: 3-second windows taken from a 1250 Hz
    signal, 100 chunks per channel beginning 200 s into the recording.
    """

    root: str = "data"
    fs: float = 1250.0
    window_s: float = 3.0
    start_s: float = 200.0
    chunks_per_channel: int = 100
    target_fs: int = 16000
    normalise: str = "per_chunk_zscore"


@dataclass
class SplitConfig:
    """Group-aware split definition. ``kind`` selects the grouping unit."""

    kind: str = "cross_session"
    seed: int = 0
    val_fraction: float = 0.30
    test_fraction: float = 0.15
    train_datasets: list[str] = field(default_factory=lambda: ["ibl"])
    test_datasets: list[str] = field(default_factory=lambda: ["ibl"])


@dataclass
class TrainConfig:
    """Fine-tuning hyper-parameters. Defaults are the paper's supplementary Table 2."""

    model: str = "facebook/wav2vec2-base"
    learning_rate: float = 3e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    warmup_ratio: float = 0.1
    max_train_chunks: int | None = 20000
    freeze_feature_encoder: bool = True
    freeze_transformer_layers: int = 0
    seed: int = 0
    device: str | None = None


@dataclass
class ExperimentConfig:
    """Everything needed to launch one run."""

    name: str = "unnamed"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""


def _from_mapping(cls: type[T], payload: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a plain mapping, rejecting unknown keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        field_type = known[key].type
        if isinstance(value, dict) and is_dataclass(field_type):
            kwargs[key] = _from_mapping(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[return-value]


def load_config(path: str | Path) -> ExperimentConfig:
    """Read an experiment YAML into an :class:`ExperimentConfig`."""
    payload = yaml.safe_load(Path(path).read_text()) or {}
    nested = {"data": DataConfig, "split": SplitConfig, "train": TrainConfig}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in nested:
            kwargs[key] = _from_mapping(nested[key], value or {})
        else:
            kwargs[key] = value
    return _from_mapping(ExperimentConfig, kwargs)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a config back to YAML, so a manifest can be replayed verbatim."""
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch. Called at the top of every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
