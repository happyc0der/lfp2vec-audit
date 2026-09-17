import pytest

from lfpaudit.device import available_devices, environment_summary, pick_device


def test_cpu_is_always_available():
    assert "cpu" in available_devices()


def test_pick_device_returns_a_usable_device():
    policy = pick_device()
    assert policy.device in available_devices()
    assert policy.describe()


def test_explicit_cpu_is_honoured():
    policy = pick_device("cpu")
    assert policy.device == "cpu"
    assert policy.autocast_dtype is None


def test_unavailable_device_raises():
    with pytest.raises(ValueError, match="not available"):
        pick_device("definitely-not-a-device")


def test_environment_summary_has_versions():
    summary = environment_summary()
    assert {"platform", "python", "torch", "devices"} <= set(summary)
