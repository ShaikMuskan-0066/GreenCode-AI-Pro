"""
monitor.py — Sample CPU, RAM, and optional GPU utilization for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class SystemSnapshot:
    """One row of system telemetry."""

    cpu_percent: float
    ram_percent: float
    gpu_percent: float | None
    gpu_name: str | None


_nvml_state: str = "unknown"  # unknown | ready | unavailable


def _try_init_nvml() -> bool:
    """
    Initialize NVML once if pynvml is installed and a GPU is present.

    Returns:
        True if NVML is ready for queries.
    """
    global _nvml_state  # noqa: PLW0603
    if _nvml_state == "ready":
        return True
    if _nvml_state == "unavailable":
        return False
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _nvml_state = "ready"
        return True
    except Exception:
        _nvml_state = "unavailable"
        return False


def sample_gpu_utilization() -> tuple[float | None, str | None]:
    """
    Read GPU utilization (%) and device name from the first NVML device.

    Returns:
        (gpu_percent, gpu_name) or (None, None) if unavailable.
    """
    if not _try_init_nvml():
        return None, None
    try:
        import pynvml  # type: ignore

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name_raw, bytes):
            gpu_name = name_raw.decode("utf-8", errors="replace")
        else:
            gpu_name = str(name_raw)
        return float(util.gpu), gpu_name
    except Exception:
        return None, None


def sample_system_metrics(interval: float | None = 0.15) -> SystemSnapshot:
    """
    Take a single snapshot of CPU, RAM, and optional GPU usage.

    Args:
        interval: Seconds for ``cpu_percent`` averaging (psutil); None for non-blocking.

    Returns:
        SystemSnapshot with utilization fields.
    """
    cpu = float(psutil.cpu_percent(interval=interval))
    ram = float(psutil.virtual_memory().percent)
    gpu_pct, gpu_name = sample_gpu_utilization()
    return SystemSnapshot(
        cpu_percent=cpu,
        ram_percent=ram,
        gpu_percent=gpu_pct,
        gpu_name=gpu_name,
    )


def snapshot_to_dict(snapshot: SystemSnapshot) -> dict[str, Any]:
    """
    Convert a SystemSnapshot into a JSON/plot friendly dict with a timestamp key omitted.

    Args:
        snapshot: Telemetry row.

    Returns:
        Dict with cpu_percent, ram_percent, gpu_percent, gpu_name.
    """
    return {
        "cpu_percent": snapshot.cpu_percent,
        "ram_percent": snapshot.ram_percent,
        "gpu_percent": snapshot.gpu_percent,
        "gpu_name": snapshot.gpu_name,
    }
