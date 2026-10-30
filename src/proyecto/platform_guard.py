"""Garantiza que el proyecto solo corra en Linux y con Python soportado."""

import sys

MIN_PYTHON = (3, 12)


def ensure_supported() -> None:
    if not sys.platform.startswith("linux"):
        raise SystemExit(f"Este proyecto solo funciona en Linux (detectado: {sys.platform}).")
    if sys.version_info < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        raise SystemExit(f"Se requiere Python >= {need} (detectado: {sys.version.split()[0]}).")
