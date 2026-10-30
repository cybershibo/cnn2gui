from .platform_guard import ensure_supported

ensure_supported()

from .hardware import Backend, Device, list_gpus, select_device  # noqa: E402

__all__ = ["Backend", "Device", "list_gpus", "select_device"]
__version__ = "0.1.0"
