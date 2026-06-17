"""
sustainability_insights.py — Green software engineering insights for comparison and repos.
"""

from __future__ import annotations

from analyzer import AnalysisIssue, AnalysisResult
from carbon_tracker import CarbonEstimate
from suggestions import build_suggestions, build_suggestions_from_codes
from sustainability_score import score_status_label

LOADER_CODES = frozenset({"LOADER_DEFAULT", "LOADER_WEAK"})

SUSTAINABILITY_CHECKLIST: list[tuple[str, str]] = [
    ("LARGE_BATCH", "Large Batch Size"),
    ("WORKERS_ZERO", "num_workers = 0"),
    ("NO_AMP", "Missing Mixed Precision"),
    ("FULL_FT", "Full Fine-Tuning"),
    ("RESOURCE_HEAVY", "Resource Heavy Patterns"),
    ("LOADER", "Inefficient Data Loading"),
]


def sustainability_grade(score: int) -> str:
    """Map sustainability score to letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _codes_from_issues(issues: list[AnalysisIssue]) -> set[str]:
    return {i.code for i in issues}


def _checklist_detected(code_key: str, codes: set[str]) -> bool:
    if code_key == "LOADER":
        return bool(codes & LOADER_CODES)
    return code_key in codes


def build_issue_checklist(issues: list[AnalysisIssue]) -> list[dict]:
    """
    Build sustainability issue checklist for display.

    Returns:
        List of dicts with label, detected (bool), and detail when found.
    """
    codes = _codes_from_issues(issues)
    by_code = {i.code: i for i in issues}
    checklist: list[dict] = []
    for code_key, label in SUSTAINABILITY_CHECKLIST:
        detected = _checklist_detected(code_key, codes)
        detail = ""
        if detected:
            if code_key == "LOADER":
                hit = next((by_code[c] for c in LOADER_CODES if c in by_code), None)
                detail = hit.detail if hit else "Data loading without efficient worker configuration."
            else:
                hit = by_code.get(code_key)
                detail = hit.detail if hit else "Detected in source."
        checklist.append({"label": label, "detected": detected, "detail": detail})
    return checklist


def build_issue_checklist_from_codes(codes: set[str]) -> list[dict]:
    """Build checklist from a set of issue codes (e.g. repo aggregate)."""
    fake = [AnalysisIssue(code=c, title=c, detail="Aggregated across repository.") for c in codes]
    return build_issue_checklist(fake)


def compute_resource_efficiency_score(issue_count: int, carbon: CarbonEstimate) -> int:
    """
    Estimate resource efficiency from issues and carbon/energy signals.

    Higher is better (more efficient).
    """
    score = 100.0
    score -= min(40.0, issue_count * 8.0)
    score -= min(25.0, carbon.energy_kwh * 18.0)
    score -= min(20.0, carbon.co2_kg * 45.0)
    return int(max(0, min(100, round(score))))


def enrich_sustainability_profile(
    profile: dict,
) -> dict:
    """Attach sustainability-focused fields to a comparison profile dict."""
    analysis: AnalysisResult = profile["analysis"]
    carbon: CarbonEstimate = profile["carbon"]
    profile["resource_efficiency"] = compute_resource_efficiency_score(len(analysis.issues), carbon)
    profile["sustainability_grade"] = sustainability_grade(profile["score"])
    profile["sustainability_status"] = score_status_label(profile["score"])
    profile["issue_checklist"] = build_issue_checklist(analysis.issues)
    profile["issue_details"] = [
        {"title": i.title, "detail": i.detail, "severity": i.severity} for i in analysis.issues
    ]
    profile["green_suggestions"] = build_suggestions(analysis.issues)
    return profile


def sustainability_comparison_table_rows(profile_a: dict, profile_b: dict) -> list[dict]:
    """Side-by-side sustainability comparison table."""
    ca: CarbonEstimate = profile_a["carbon"]
    cb: CarbonEstimate = profile_b["carbon"]

    def row(metric: str, a_val, b_val) -> dict:
        return {"Sustainability Metric": metric, "File A (Baseline)": a_val, "File B (Candidate)": b_val}

    return [
        row("Sustainability Score", f"{profile_a['score']}/100 ({profile_a['sustainability_grade']})", f"{profile_b['score']}/100 ({profile_b['sustainability_grade']})"),
        row("Estimated CO₂ Emissions (kg)", f"{ca.co2_kg:.4f}", f"{cb.co2_kg:.4f}"),
        row("Estimated Energy Usage (kWh)", f"{ca.energy_kwh:.4f}", f"{cb.energy_kwh:.4f}"),
        row("Estimated Electricity Cost", f"₹{ca.cost_inr:.2f}", f"₹{cb.cost_inr:.2f}"),
        row("Resource Efficiency", f"{profile_a['resource_efficiency']}/100", f"{profile_b['resource_efficiency']}/100"),
        row("Issues Found", profile_a["issues"], profile_b["issues"]),
        row("Optimization Suggestions", len(profile_a["green_suggestions"]), len(profile_b["green_suggestions"])),
    ]


def sustainability_chart_values(profile_a: dict, profile_b: dict) -> dict[str, float]:
    """Values for sustainability and carbon comparison charts."""
    ca: CarbonEstimate = profile_a["carbon"]
    cb: CarbonEstimate = profile_b["carbon"]

    def carbon_green_score(co2: float) -> float:
        return max(0.0, min(100.0, 100.0 - co2 * 500.0))

    return {
        "a_sustainability": float(profile_a["score"]),
        "b_sustainability": float(profile_b["score"]),
        "a_resource": float(profile_a["resource_efficiency"]),
        "b_resource": float(profile_b["resource_efficiency"]),
        "a_carbon_green": carbon_green_score(ca.co2_kg),
        "b_carbon_green": carbon_green_score(cb.co2_kg),
        "a_co2": ca.co2_kg,
        "b_co2": cb.co2_kg,
        "a_energy": ca.energy_kwh,
        "b_energy": cb.energy_kwh,
    }


def determine_green_winner(profile_a: dict, profile_b: dict) -> tuple[str, list[str]]:
    """Pick greener file with sustainability-focused reasons."""
    reasons: list[str] = []
    points_a = 0
    points_b = 0

    if profile_b["score"] > profile_a["score"]:
        points_b += 1
        reasons.append("Higher sustainability score — greener training profile")
    elif profile_a["score"] > profile_b["score"]:
        points_a += 1

    if profile_b["carbon"].co2_kg < profile_a["carbon"].co2_kg:
        points_b += 1
        reasons.append("Lower estimated CO₂ emissions")
    elif profile_a["carbon"].co2_kg < profile_b["carbon"].co2_kg:
        points_a += 1

    if profile_b["carbon"].energy_kwh < profile_a["carbon"].energy_kwh:
        points_b += 1
        reasons.append("Lower estimated energy usage")
    elif profile_a["carbon"].energy_kwh < profile_b["carbon"].energy_kwh:
        points_a += 1

    if profile_b["resource_efficiency"] > profile_a["resource_efficiency"]:
        points_b += 1
        reasons.append("Better resource efficiency")
    elif profile_a["resource_efficiency"] > profile_b["resource_efficiency"]:
        points_a += 1

    if profile_b["issues"] < profile_a["issues"]:
        points_b += 1
        reasons.append("Fewer green software practice violations")
    elif profile_a["issues"] < profile_b["issues"]:
        points_a += 1

    if points_b > points_a:
        winner = f"File B ({profile_b['name']})"
        rec = "Adopt File B to reduce carbon impact and improve energy efficiency."
    elif points_a > points_b:
        winner = f"File A ({profile_a['name']})"
        rec = "Baseline is greener — improve File B using the optimization suggestions below."
        reasons = reasons or ["Baseline shows stronger sustainability signals"]
    else:
        winner = "Tie — similar sustainability profile"
        rec = "Both scripts need green optimizations — review issues and suggestions for each file."

    return winner, ([rec] + reasons)[:6]


def aggregate_repo_green_suggestions(issue_codes: set[str]) -> list[str]:
    """Build repo-level optimization suggestions from aggregated issue codes."""
    return build_suggestions_from_codes(issue_codes)
