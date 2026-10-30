import pytest

from proyecto import hardware
from proyecto.hardware import Backend, Device


def test_cpu_fallback(monkeypatch):
    monkeypatch.delenv("PROYECTO_DEVICE", raising=False)
    monkeypatch.setattr(hardware, "list_gpus", lambda: [])
    assert hardware.select_device().backend is Backend.CPU


def test_priority_nvidia_first(monkeypatch):
    monkeypatch.delenv("PROYECTO_DEVICE", raising=False)
    gpus = [Device(Backend.INTEL, "0000:00:02.0"), Device(Backend.NVIDIA, "0000:01:00.0")]
    monkeypatch.setattr(hardware, "list_gpus", lambda: gpus)
    assert hardware.select_device().backend is Backend.NVIDIA


def test_unusable_gpu_falls_back(monkeypatch):
    monkeypatch.delenv("PROYECTO_DEVICE", raising=False)
    monkeypatch.setattr(hardware, "list_gpus", lambda: [Device(Backend.AMD, "x", usable=False)])
    assert hardware.select_device().backend is Backend.CPU


def test_forced_device(monkeypatch):
    monkeypatch.setenv("PROYECTO_DEVICE", "amd")
    assert hardware.select_device().backend is Backend.AMD


def test_invalid_forced(monkeypatch):
    monkeypatch.setenv("PROYECTO_DEVICE", "foo")
    with pytest.raises(SystemExit):
        hardware.select_device()
