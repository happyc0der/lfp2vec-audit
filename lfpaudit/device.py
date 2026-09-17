"""Device and dtype selection shared by the laptop (MPS) and Colab (CUDA) code paths."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch

_PREFERENCE_ORDER = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class DevicePolicy:
    """Resolved device plus the dtype and autocast decisions that follow from it."""

    device: str
    autocast_dtype: torch.dtype | None
    pin_memory: bool
    num_workers: int

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def describe(self) -> str:
        dtype = "fp32" if self.autocast_dtype is None else str(self.autocast_dtype).split(".")[-1]
        return f"{self.device} (autocast={dtype}, workers={self.num_workers})"


def available_devices() -> list[str]:
    """Return the device strings usable on this machine, most capable first."""
    found = ["cpu"]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        found.insert(0, "mps")
    if torch.cuda.is_available():
        found.insert(0, "cuda")
    return found


def pick_device(prefer: str | None = None) -> DevicePolicy:
    """Choose a device, honouring ``prefer`` when that device really is available.

    Falls back to the best available device rather than raising, so the same command runs
    unchanged on the laptop and in Colab. ``LFPAUDIT_DEVICE`` overrides the default but not an
    explicit ``prefer`` argument.
    """
    usable = available_devices()
    requested = prefer or os.environ.get("LFPAUDIT_DEVICE")
    if requested is not None and requested not in usable:
        raise ValueError(f"device {requested!r} is not available; usable devices: {usable}")
    if requested is None:
        requested = next(d for d in _PREFERENCE_ORDER if d in usable)

    if requested == "cuda":
        # bf16 autocast is safe on Ampere and newer, which is what Colab Pro hands out.
        supports_bf16 = torch.cuda.is_bf16_supported()
        return DevicePolicy("cuda", torch.bfloat16 if supports_bf16 else torch.float16, True, 2)
    if requested == "mps":
        # MPS autocast is still unreliable for wav2vec2's conv stack; stay in fp32 on purpose.
        return DevicePolicy("mps", None, False, 0)
    return DevicePolicy("cpu", None, False, 0)


def environment_summary() -> dict[str, str]:
    """Small dictionary of versions recorded in every run manifest."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "devices": ",".join(available_devices()),
    }
