"""
memory_optimizer.py — Monitor RAM and trigger garbage collection when needed.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import psutil


@dataclass
class MemoryOptimizationResult:
    """Outcome of a memory check / optional GC run."""

    memory_percent: float
    threshold_percent: float
    gc_triggered: bool
    objects_collected: int
    gc_count: tuple[int, int, int]


def get_memory_percent() -> float:
    """
    Return current system RAM utilization percentage.

    Returns:
        RAM used percent (0–100).
    """
    return float(psutil.virtual_memory().percent)


def check_and_optimize_memory(threshold_percent: float = 80.0) -> MemoryOptimizationResult:
    """
    Check RAM usage; run ``gc.collect()`` if above threshold.

    Args:
        threshold_percent: Trigger GC when RAM % exceeds this value.

    Returns:
        MemoryOptimizationResult with GC stats.
    """
    mem = get_memory_percent()
    triggered = False
    collected = 0

    if mem >= threshold_percent:
        collected = int(gc.collect())
        triggered = True

    return MemoryOptimizationResult(
        memory_percent=mem,
        threshold_percent=threshold_percent,
        gc_triggered=triggered,
        objects_collected=collected,
        gc_count=tuple(gc.get_count()),
    )
