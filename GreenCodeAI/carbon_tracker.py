"""
carbon_tracker.py — Estimate energy, CO₂, and cost using CodeCarbon where possible.

Falls back to heuristic kWh if CodeCarbon cannot run (missing deps / errors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    Convert analysis signals into a simple baseline kWh for one training hour.

    This is a teaching heuristic, not a lab measurement.
    """
    base = 0.35  # modest GPU hour starting point
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
    Rough average retail electricity price in India (INR per kWh).

    Users can tune this constant for their state or provider.
    """
    return 8.5


def estimate_carbon_footprint(analysis: AnalysisResult, duration_hours: float = 1.0) -> CarbonEstimate:
    """
    Estimate energy (kWh), CO₂ (kg), and electricity cost (INR).

    Energy (kWh) is a simple heuristic from detected issues. CO₂ uses CodeCarbon's
    country energy-mix data (India / IND) when the library is available.

    Args:
        analysis: Output from analyze_training_script.
        duration_hours: Scale factor for how long training might run (default 1 hour).

    Returns:
        CarbonEstimate with numeric fields and a short method description.
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
    except Exception as exc:  # noqa: BLE001 — beginner-friendly broad catch
        notes_parts.append(f"CodeCarbon mix lookup failed ({exc!r}). Using fallback factor.")

    if co2_kg is None:
        # Approximate India grid: ~0.82 kg CO2e / kWh (order-of-magnitude teaching value).
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


def try_live_codecarbon_sample_seconds(seconds: float = 2.0) -> dict[str, Any] | None:
    """
    Optional: run a very short live CodeCarbon sample (uses real hardware sensors if available).

    Returns:
        Dict with keys like emissions_kg, energy_kwh if successful, else None.
    """
    try:
        from codecarbon import EmissionsTracker  # type: ignore
        import time

        tracker = EmissionsTracker(
            project_name="GreenCodeAI_live_sample",
            measure_power_secs=1,
            save_to_file=False,
            save_to_api=False,
        )
        tracker.start()
        time.sleep(max(0.5, seconds))
        tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        if isinstance(data, dict):
            return dict(data)
        return None
    except Exception:
        return None
