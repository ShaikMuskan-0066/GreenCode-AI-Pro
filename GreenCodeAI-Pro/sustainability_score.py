"""
sustainability_score.py — Compute a 0–100 Green Score from analysis signals.
"""

from __future__ import annotations

from code_metrics import CodeMetrics
from carbon_tracker import CarbonEstimate


def score_status_label(score: int) -> str:
    """
    Map a numeric score to a human-readable status band.

    Args:
        score: Green score 0–100.

    Returns:
        Status string (Excellent, Good, Average, Needs Improvement).
    """
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def compute_green_score(
    carbon: CarbonEstimate,
    issue_count: int,
    metrics: CodeMetrics | None,
    memory_percent: float,
) -> int:
    """
    Compute composite sustainability score (0–100).

    Higher is greener. Based on carbon, issues, code quality hints, and memory headroom.

    Args:
        carbon: Carbon estimate from code analysis.
        issue_count: Number of detected issues.
        metrics: Optional code metrics.
        memory_percent: Current or recent RAM utilization.

    Returns:
        Integer score clamped to 0–100.
    """
    score = 100.0

    # Carbon penalty (heuristic caps)
    score -= min(25.0, carbon.co2_kg * 8.0)
    score -= min(15.0, carbon.energy_kwh * 6.0)

    # Issues penalty
    score -= min(30.0, issue_count * 6.0)

    # Code quality bonus/penalty
    if metrics is not None:
        if metrics.total_lines > 0:
            comment_ratio = metrics.comment_lines / max(1, metrics.total_lines)
            if comment_ratio >= 0.08:
                score += 3.0
            elif comment_ratio < 0.02:
                score -= 3.0
        if metrics.functions > 50:
            score -= 4.0

    # Memory efficiency (headroom)
    if memory_percent > 90:
        score -= 12.0
    elif memory_percent > 80:
        score -= 6.0
    elif memory_percent < 60:
        score += 2.0

    return int(max(0, min(100, round(score))))
