"""
language_detector.py — Detect programming language from filename and content hints.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
}

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(SUPPORTED_EXTENSIONS.values())


def detect_language_from_filename(filename: str) -> str | None:
    """
    Guess language from file extension only.

    Args:
        filename: Uploaded or on-disk file name.

    Returns:
        Language label or None if extension is unknown.
    """
    ext = Path(filename).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def detect_language_from_content(source: str) -> str | None:
    """
    Apply lightweight content heuristics when extension is missing or ambiguous.

    Args:
        source: File text.

    Returns:
        Best-guess language label or None.
    """
    sample = source[:4000]
    if "import torch" in sample or "def " in sample and "self" in sample:
        return "Python"
    if "public class " in sample or "import java." in sample:
        return "Java"
    if "interface " in sample and ": " in sample and "export " in sample:
        return "TypeScript"
    if "function " in sample or "const " in sample and "=>" in sample:
        return "JavaScript"
    if "#include" in sample or "std::" in sample:
        return "C++"
    return None


def detect_language(filename: str, source: str | None = None) -> str:
    """
    Detect the programming language for an uploaded file.

    Args:
        filename: Original file name.
        source: Optional source text for content-based fallback.

    Returns:
        One of: Python, Java, JavaScript, TypeScript, C++.
    """
    by_name = detect_language_from_filename(filename)
    if by_name:
        return by_name
    if source:
        by_content = detect_language_from_content(source)
        if by_content:
            return by_content
    return "Python"


def language_badge_html(language: str) -> str:
    """
    Return a small HTML badge for the detected language (Streamlit ``unsafe_allow_html``).

    Args:
        language: Detected language label.

    Returns:
        HTML snippet for a styled badge.
    """
    colors = {
        "Python": "#3572A5",
        "Java": "#b07219",
        "JavaScript": "#f1e05a",
        "TypeScript": "#3178c6",
        "C++": "#f34b7d",
    }
    fg = "#0e1117" if language in {"JavaScript"} else "#f0f2f6"
    bg = colors.get(language, "#00d4aa")
    return (
        f'<span style="background:{bg};color:{fg};padding:0.25rem 0.65rem;'
        f'border-radius:8px;font-weight:600;font-size:0.85rem;">{language}</span>'
    )
