"""Detección de GPU sin dependencias externas (solo sysfs y /dev).

Vendor IDs PCI: NVIDIA 0x10de, AMD 0x1002, Intel 0x8086.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PCI_DEVICES = Path("/sys/bus/pci/devices")
DEV = Path("/dev")


class Backend(StrEnum):
    NVIDIA = "nvidia"  # CUDA
    AMD = "amd"  # ROCm
    INTEL = "intel"  # oneAPI / XPU
    CPU = "cpu"


_VENDORS = {"0x10de": Backend.NVIDIA, "0x1002": Backend.AMD, "0x8086": Backend.INTEL}


@dataclass(frozen=True)
class Device:
    backend: Backend
    pci_address: str | None = None
    usable: bool = True  # el driver/nodo de dispositivo necesario está presente

    def __str__(self) -> str:
        where = f" @ {self.pci_address}" if self.pci_address else ""
        return f"{self.backend}{where}"


def _read(path: Path) -> str:
    try:
        return path.read_text().strip().lower()
    except OSError:
        return ""


def _driver_ready(backend: Backend) -> bool:
    match backend:
        case Backend.NVIDIA:
            return Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi") is not None
        case Backend.AMD:
            return (DEV / "kfd").exists()  # ROCm necesita /dev/kfd
        case Backend.INTEL:
            return any(DEV.glob("dri/renderD*"))
        case _:
            return True


def list_gpus() -> list[Device]:
    """Todas las GPU PCI (clase 0x03xxxx) reconocidas."""
    found: list[Device] = []
    if not PCI_DEVICES.exists():
        return found
    for dev in sorted(PCI_DEVICES.iterdir()):
        if not _read(dev / "class").startswith("0x03"):
            continue
        backend = _VENDORS.get(_read(dev / "vendor"))
        if backend:
            found.append(Device(backend, dev.name, _driver_ready(backend)))
    return found


def select_device() -> Device:
    """Elige el mejor dispositivo; CPU si no hay GPU utilizable.

    Se puede forzar con PROYECTO_DEVICE=nvidia|amd|intel|cpu.
    """
    forced = os.environ.get("PROYECTO_DEVICE", "").strip().lower()
    if forced:
        try:
            return Device(Backend(forced))
        except ValueError:
            raise SystemExit(f"PROYECTO_DEVICE inválido: {forced!r}") from None

    # Prioridad: NVIDIA > AMD > Intel
    order = {Backend.NVIDIA: 0, Backend.AMD: 1, Backend.INTEL: 2}
    usable = [g for g in list_gpus() if g.usable]
    if usable:
        return min(usable, key=lambda g: order[g.backend])
    return Device(Backend.CPU)
