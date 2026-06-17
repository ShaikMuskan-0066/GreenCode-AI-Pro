"""
suggestions.py — Map detected issues to green optimization recommendations.
"""

from __future__ import annotations

from analyzer import AnalysisIssue


def build_suggestions(issues: list[AnalysisIssue]) -> list[str]:
    """
    Build a prioritized list of green optimization suggestions from detected issues.

    Args:
        issues: Issues returned by the analyzer.

    Returns:
        Unique suggestion strings in stable order.
    """
    return build_suggestions_from_codes({i.code for i in issues})


def build_suggestions_from_codes(codes: set[str]) -> list[str]:
    """
    Build green optimization suggestions from issue codes.

    Args:
        codes: Set of analyzer issue codes.

    Returns:
        Prioritized sustainability recommendations.
    """
    suggestions: list[str] = []

    def add(text: str) -> None:
        if text not in suggestions:
            suggestions.append(text)

    if "FULL_FT" in codes:
        add("Use LoRA")
    if "NO_AMP" in codes:
        add("Enable Mixed Precision")
    if "LARGE_BATCH" in codes:
        add("Reduce Batch Size")
    if "WORKERS_ZERO" in codes or "LOADER_DEFAULT" in codes or "LOADER_WEAK" in codes:
        add("Improve DataLoader Workers")
    if "LARGE_BATCH" in codes or "FULL_FT" in codes or "NO_AMP" in codes or "RESOURCE_HEAVY" in codes:
        add("Apply Quantization")
    if "RESOURCE_HEAVY" in codes or "FULL_FT" in codes or "LARGE_BATCH" in codes:
        add("Optimize Resource Usage")

    if not suggestions:
        add("No major sustainability issues — continue monitoring energy during training runs")

    return suggestions


def issues_to_display_lines(issues: list[AnalysisIssue]) -> list[str]:
    """
    Convert issues to short titles for lists and reports.

    Args:
        issues: Detected issues.

    Returns:
        One-line titles.
    """
    return [i.title for i in issues]
