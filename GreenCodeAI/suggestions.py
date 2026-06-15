"""
suggestions.py — Map detected issues to human-readable green optimizations.
"""

from __future__ import annotations

from analyzer import AnalysisIssue


def build_suggestions(issues: list[AnalysisIssue]) -> list[str]:
    """
    Build a prioritized list of optimization suggestions from detected issues.

    Args:
        issues: Issues returned by the analyzer.

    Returns:
        Unique suggestion strings, stable order for display.
    """
    codes = {i.code for i in issues}
    suggestions: list[str] = []

    def add(text: str) -> None:
        """Append a suggestion if it is not already present."""
        if text not in suggestions:
            suggestions.append(text)

    if "FULL_FT" in codes:
        add("Use LoRA (PEFT) or adapters instead of full fine-tuning")
    if "NO_AMP" in codes:
        add("Enable mixed precision (torch.cuda.amp / autocast)")
    if "LARGE_BATCH" in codes:
        add("Reduce batch size or use gradient accumulation to save memory and smooth power draw")
    if "WORKERS_ZERO" in codes or "LOADER_DEFAULT" in codes or "LOADER_WEAK" in codes:
        add("Increase DataLoader num_workers and consider pin_memory=True on CUDA")
    if "LARGE_BATCH" in codes or "FULL_FT" in codes or "NO_AMP" in codes:
        add("Use quantization (INT8/4-bit) for inference-heavy or large models where quality allows")

    if not suggestions:
        add("Your script looks relatively efficient for the checks we ran — still profile GPU utilization")

    return suggestions


def issues_to_display_lines(issues: list[AnalysisIssue]) -> list[str]:
    """
    Convert issues to short titles for checklist-style CLI output.

    Args:
        issues: Detected issues.

    Returns:
        One-line titles.
    """
    return [i.title for i in issues]
