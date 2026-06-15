"""
carbon_tracker.py — Estimate energy, CO₂, and cost using CodeCarbon (India grid mix).
"""

from __future__ import annotations

from dataclasses import dataclass

from analyzer import AnalysisResult


@dataclass
class CarbonEstimate:
    """Estimated environmental and cost impact for a training configuration."""

    energy_kwh: float
    co2_kg: float
    cost_inr: float
    method: str
    notes: str


def _heuristic_base_kwh(analysis: AnalysisResult) -> float:
    """
    Map detected issues to a simple baseline kWh for one training hour (teaching model).
    """
    base = 0.35
    codes = {i.code for i in analysis.issues}

    if "LARGE_BATCH" in codes:
        base += 0.25
    if "WORKERS_ZERO" in codes or "LOADER_DEFAULT" in codes or "LOADER_WEAK" in codes:
        base += 0.08
    if "NO_AMP" in codes:
        base += 0.20
    if "FULL_FT" in codes:
        base += 0.30

    return round(base, 3)


def _inr_per_kwh() -> float:
    """
    Approximate retail electricity price in India (INR per kWh).
    """
    return 8.5


def estimate_carbon_footprint(analysis: AnalysisResult, duration_hours: float = 1.0) -> CarbonEstimate:
    """
    Estimate energy (kWh), CO₂ (kg), and electricity cost (INR).

    Args:
        analysis: Output from the analyzer.
        duration_hours: Hours of training assumed for scaling.

    Returns:
        CarbonEstimate with numeric fields and notes.
    """
    kwh_per_hour = _heuristic_base_kwh(analysis)
    energy_kwh = round(max(0.05, kwh_per_hour * duration_hours), 3)
    notes_parts: list[str] = []

    co2_kg: float | None = None
    method = "heuristic"

    try:
        from codecarbon.core.emissions import Emissions  # type: ignore
        from codecarbon.core.units import Energy  # type: ignore
        from codecarbon.external.geography import GeoMetadata  # type: ignore
        from codecarbon.input import DataSource  # type: ignore

        geo = GeoMetadata(country_iso_code="IND")
        energy = Energy.from_energy(kWh=energy_kwh)
        emissions_calc = Emissions(DataSource(), electricitymaps_api_token=None)
        co2_kg = float(emissions_calc.get_country_emissions(energy, geo))
        method = "codecarbon_energy_mix"
        notes_parts.append("CO₂ from CodeCarbon India grid mix + heuristic kWh.")
    except Exception as exc:  # noqa: BLE001
        notes_parts.append(f"CodeCarbon mix lookup failed ({exc!r}). Using fallback factor.")

    if co2_kg is None:
        factor = 0.82
        co2_kg = round(energy_kwh * factor, 3)
        method = "heuristic_emission_factor"
        notes_parts.append("Fallback emission factor ~0.82 kg CO₂e / kWh.")

    cost_inr = round(energy_kwh * _inr_per_kwh(), 2)

    return CarbonEstimate(
        energy_kwh=energy_kwh,
        co2_kg=round(float(co2_kg), 3),
        cost_inr=cost_inr,
        method=method,
        notes=" ".join(notes_parts),
    )


def live_adjusted_estimate(
    base: CarbonEstimate,
    cpu_percent: float,
    ram_percent: float,
    gpu_percent: float | None,
) -> tuple[float, float, float]:
    """
    Scale baseline carbon numbers by current system load for dashboard animation.

    This does not replace lab measurements; it visualizes how busy the machine is.

    Args:
        base: Static estimate from ``estimate_carbon_footprint``.
        cpu_percent: Current CPU utilization (0–100).
        ram_percent: Current RAM utilization (0–100).
        gpu_percent: GPU utilization if available, else None.

    Returns:
        Tuple (energy_kwh, co2_kg, cost_inr) adjusted for display.
    """
    gpu = float(gpu_percent) if gpu_percent is not None else 0.0
    load = min(1.6, 0.55 + 0.45 * ((cpu_percent + gpu * 1.2 + ram_percent * 0.25) / 150.0))
    energy = round(max(0.01, base.energy_kwh * load), 3)
    co2 = round(max(0.001, base.co2_kg * load), 3)
    cost = round(energy * _inr_per_kwh(), 2)
    return energy, co2, cost
