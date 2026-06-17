"""
metrics_insights.py — Quality scores, complexity estimates, and comparison helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from analyzer import AnalysisResult, analyze_uploaded_source
from carbon_tracker import CarbonEstimate, estimate_carbon_footprint
from code_metrics import CodeMetrics, compute_code_metrics
from suggestions import build_suggestions
from sustainability_insights import enrich_sustainability_profile
from sustainability_score import compute_green_score


@dataclass
class QualityInsights:
    """Extended quality and complexity insights for a source file."""

    cyclomatic_complexity: int
    maintainability_score: int
    readability_score: int
    code_quality_score: int
    maintainability_label: str
    readability_label: str
    code_quality_label: str
    complexity_label: str


def quality_label(score: int) -> str:
    """Map a 0–100 score to a qualitative rating."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def complexity_label(value: int) -> str:
    """Map cyclomatic complexity to a qualitative rating."""
    if value <= 10:
        return "Excellent"
    if value <= 20:
        return "Good"
    if value <= 35:
        return "Average"
    return "Needs Improvement"


def estimate_cyclomatic_complexity(source: str, language: str) -> int:
    """
    Heuristic cyclomatic complexity estimate from control-flow keywords.

    Args:
        source: Full source text.
        language: Detected language label.

    Returns:
        Estimated complexity (minimum 1).
    """
    _ = language
    patterns = [
        r"\bif\b",
        r"\belif\b",
        r"\belse\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\bcase\b",
        r"\bcatch\b",
        r"\?\s*[^:]+:",
        r"&&",
        r"\|\|",
    ]
    total = 1
    for pattern in patterns:
        total += len(re.findall(pattern, source))
    return min(max(total, 1), 500)


def compute_quality_insights(metrics: CodeMetrics, source: str, issues_count: int) -> QualityInsights:
    """
    Derive maintainability, readability, and code-quality scores.

    Args:
        metrics: Structural code metrics.
        source: Full source text.
        issues_count: Number of analyzer issues.

    Returns:
        QualityInsights dataclass.
    """
    complexity = estimate_cyclomatic_complexity(source, metrics.language)
    comment_ratio = metrics.comment_lines / max(metrics.code_lines, 1)
    structure_bonus = min(15, metrics.functions + metrics.classes)

    readability = int(
        min(100, max(0, 45 + comment_ratio * 120 - complexity * 0.35 + structure_bonus * 0.5))
    )
    maintainability = int(
        min(100, max(0, 65 - issues_count * 6 - complexity * 0.25 + comment_ratio * 40))
    )
    code_quality = int(min(100, max(0, (readability + maintainability) / 2 - issues_count * 4)))

    return QualityInsights(
        cyclomatic_complexity=complexity,
        maintainability_score=maintainability,
        readability_score=readability,
        code_quality_score=code_quality,
        maintainability_label=quality_label(maintainability),
        readability_label=quality_label(readability),
        code_quality_label=quality_label(code_quality),
        complexity_label=complexity_label(complexity),
    )


def build_file_comparison_profile(
    display_name: str,
    source: str,
    filename: str,
    duration_hours: float,
    memory_percent: float = 70.0,
) -> dict:
    """
    Build a full comparison profile for one uploaded file.

    Args:
        display_name: Label for UI (File A / File B).
        source: Source text.
        filename: Original filename.
        duration_hours: Training duration for carbon estimate.
        memory_percent: RAM percent for green score.

    Returns:
        Dict with metrics, carbon, quality, and suggestions.
    """
    analysis = analyze_uploaded_source(source, filename=filename)
    metrics = compute_code_metrics(source, filename)
    carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
    quality = compute_quality_insights(metrics, source, len(analysis.issues))
    score = compute_green_score(carbon, len(analysis.issues), metrics, memory_percent)
    suggestions = build_suggestions(analysis.issues)

    profile = {
        "label": display_name,
        "name": filename,
        "language": metrics.language,
        "file_size": len(source.encode("utf-8")),
        "metrics": metrics,
        "quality": quality,
        "analysis": analysis,
        "carbon": carbon,
        "score": score,
        "issues": len(analysis.issues),
        "suggestions": suggestions,
        "optimization_opportunities": len(suggestions),
    }
    return enrich_sustainability_profile(profile)


def comparison_table_rows(profile_a: dict, profile_b: dict) -> list[dict]:
    """Build side-by-side comparison rows for display."""
    ma: CodeMetrics = profile_a["metrics"]
    mb: CodeMetrics = profile_b["metrics"]
    qa: QualityInsights = profile_a["quality"]
    qb: QualityInsights = profile_b["quality"]
    ca: CarbonEstimate = profile_a["carbon"]
    cb: CarbonEstimate = profile_b["carbon"]

    def row(metric: str, a_val, b_val) -> dict:
        return {"Metric": metric, "File A (Baseline)": a_val, "File B (Candidate)": b_val}

    return [
        row("Language", ma.language, mb.language),
        row("File Size (bytes)", profile_a["file_size"], profile_b["file_size"]),
        row("Lines of Code", ma.total_lines, mb.total_lines),
        row("Blank Lines", ma.blank_lines, mb.blank_lines),
        row("Comment Lines", ma.comment_lines, mb.comment_lines),
        row("Functions Count", ma.functions, mb.functions),
        row("Classes Count", ma.classes, mb.classes),
        row("Imports Count", ma.imports, mb.imports),
        row("Estimated Complexity", qa.cyclomatic_complexity, qb.cyclomatic_complexity),
        row("Sustainability Score", profile_a["score"], profile_b["score"]),
        row("Estimated Energy Usage (kWh)", f"{ca.energy_kwh:.4f}", f"{cb.energy_kwh:.4f}"),
        row("Estimated CO₂ Emissions (kg)", f"{ca.co2_kg:.4f}", f"{cb.co2_kg:.4f}"),
        row("Estimated Cost (INR)", f"₹{ca.cost_inr:.2f}", f"₹{cb.cost_inr:.2f}"),
        row("Issues Found", profile_a["issues"], profile_b["issues"]),
        row("Optimization Opportunities", profile_a["optimization_opportunities"], profile_b["optimization_opportunities"]),
        row("Maintainability Score", qa.maintainability_score, qb.maintainability_score),
        row("Readability Score", qa.readability_score, qb.readability_score),
        row("Code Quality Score", qa.code_quality_score, qb.code_quality_score),
    ]


def chart_dimensions(profile_a: dict, profile_b: dict) -> dict[str, float]:
    """
    Normalized 0–100 dimensions for radar/bar charts.

    Returns:
        Dict with keys efficiency, sustainability, complexity, carbon for A and B.
    """
    qa: QualityInsights = profile_a["quality"]
    qb: QualityInsights = profile_b["quality"]
    ca: CarbonEstimate = profile_a["carbon"]
    cb: CarbonEstimate = profile_b["carbon"]

    def carbon_score(co2: float) -> float:
        return max(0.0, min(100.0, 100.0 - co2 * 500.0))

    def efficiency_score(profile: dict) -> float:
        q: QualityInsights = profile["quality"]
        issues = profile["issues"]
        return max(0.0, min(100.0, q.maintainability_score * 0.5 + (100 - min(issues * 8, 80)) * 0.5))

    return {
        "a_efficiency": efficiency_score(profile_a),
        "b_efficiency": efficiency_score(profile_b),
        "a_sustainability": float(profile_a["score"]),
        "b_sustainability": float(profile_b["score"]),
        "a_complexity": max(0.0, min(100.0, 100.0 - qa.cyclomatic_complexity * 2.5)),
        "b_complexity": max(0.0, min(100.0, 100.0 - qb.cyclomatic_complexity * 2.5)),
        "a_carbon": carbon_score(ca.co2_kg),
        "b_carbon": carbon_score(cb.co2_kg),
    }


def determine_comparison_winner(profile_a: dict, profile_b: dict) -> tuple[str, list[str]]:
    """
    Pick File A or File B with human-readable reasons.

    Returns:
        (winner_label, reasons_list)
    """
    reasons: list[str] = []
    score_b = 0
    score_a = 0

    if profile_b["score"] > profile_a["score"]:
        score_b += 1
        reasons.append("Better sustainability score")
    elif profile_a["score"] > profile_b["score"]:
        score_a += 1
        reasons.append("Better sustainability score (baseline)")

    if profile_b["carbon"].co2_kg < profile_a["carbon"].co2_kg:
        score_b += 1
        reasons.append("Lower carbon footprint")
    elif profile_a["carbon"].co2_kg < profile_b["carbon"].co2_kg:
        score_a += 1

    if profile_b["issues"] < profile_a["issues"]:
        score_b += 1
        reasons.append("Fewer inefficiencies detected")
    elif profile_a["issues"] < profile_b["issues"]:
        score_a += 1

    if profile_b["carbon"].energy_kwh < profile_a["carbon"].energy_kwh:
        score_b += 1
        reasons.append("Lower estimated energy usage")
    elif profile_a["carbon"].energy_kwh < profile_b["carbon"].energy_kwh:
        score_a += 1

    qb: QualityInsights = profile_b["quality"]
    qa: QualityInsights = profile_a["quality"]
    if qb.code_quality_score > qa.code_quality_score:
        score_b += 1
        reasons.append("Higher code quality score")
    elif qa.code_quality_score > qb.code_quality_score:
        score_a += 1

    if score_b > score_a:
        winner = f"File B ({profile_b['name']})"
    elif score_a > score_b:
        winner = f"File A ({profile_a['name']})"
        reasons = [r.replace(" (baseline)", "") for r in reasons] or ["Comparable overall performance"]
    else:
        winner = "Tie — comparable performance"
        if not reasons:
            reasons = ["Both files show similar sustainability and quality profiles"]

    return winner, reasons[:6]


def format_quality_insights_report(quality: QualityInsights) -> list[str]:
    """Format quality insights as report lines."""
    return [
        f"Cyclomatic Complexity (est.): {quality.cyclomatic_complexity} ({quality.complexity_label})",
        f"Maintainability Score: {quality.maintainability_score}/100 ({quality.maintainability_label})",
        f"Readability Score: {quality.readability_score}/100 ({quality.readability_label})",
        f"Code Quality Score: {quality.code_quality_score}/100 ({quality.code_quality_label})",
    ]
