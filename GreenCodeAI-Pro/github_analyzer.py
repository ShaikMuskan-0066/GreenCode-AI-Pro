"""
github_analyzer.py — Clone a GitHub repository and run sustainability scans.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from analyzer import AnalysisResult, analyze_uploaded_source
from carbon_tracker import CarbonEstimate, estimate_carbon_footprint
from code_metrics import CodeMetrics, compute_code_metrics
from language_detector import SUPPORTED_EXTENSIONS, detect_language
from sustainability_score import compute_green_score


SCAN_EXTENSIONS: frozenset[str] = frozenset(SUPPORTED_EXTENSIONS.keys())


@dataclass
class RepoScanResult:
    """Aggregated scan of a cloned repository."""

    repo_url: str
    total_files: int
    total_lines: int
    languages: dict[str, int] = field(default_factory=dict)
    aggregate_issues: int = 0
    repository_score: int = 0
    file_results: list[dict] = field(default_factory=list)
    error: str | None = None


def _normalize_repo_url(url: str) -> str:
    """
    Ensure GitHub URL has a scheme for cloning.

    Args:
        url: User-provided repository URL.

    Returns:
        URL with https scheme if missing.
    """
    u = url.strip()
    if not u.startswith(("http://", "https://", "git@")):
        u = "https://" + u
    return u


def _clone_repository(url: str, dest: Path) -> None:
    """
    Shallow-clone a remote Git repository into ``dest``.

    Args:
        url: Repository URL.
        dest: Local directory for the clone.

    Raises:
        RuntimeError: If GitPython is missing or clone fails.
    """
    try:
        from git import Repo  # type: ignore
    except ImportError as exc:
        raise RuntimeError("GitPython is required. pip install GitPython") from exc

    Repo.clone_from(_normalize_repo_url(url), dest, depth=1)


def _iter_source_files(root: Path) -> list[Path]:
    """
    Walk a repo tree and collect supported source files (skip .git).

    Args:
        root: Repository root path.

    Returns:
        List of file paths to analyze.
    """
    files: list[Path] = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in SCAN_EXTENSIONS:
            files.append(path)
    return files


def analyze_github_repository(url: str, duration_hours: float = 1.0) -> RepoScanResult:
    """
    Clone and scan a GitHub repository for sustainability signals.

    Args:
        url: Public Git repository URL.
        duration_hours: Hours assumed per-file for carbon scaling.

    Returns:
        RepoScanResult with aggregates and per-file summaries.
    """
    parsed = urlparse(_normalize_repo_url(url))
    if not parsed.netloc:
        return RepoScanResult(repo_url=url, total_files=0, total_lines=0, error="Invalid URL")

    tmp = Path(tempfile.mkdtemp(prefix="greencode_repo_"))
    try:
        _clone_repository(url, tmp)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        return RepoScanResult(repo_url=url, total_files=0, total_lines=0, error=str(exc))

    try:
        files = _iter_source_files(tmp)
        if not files:
            return RepoScanResult(
                repo_url=url,
                total_files=0,
                total_lines=0,
                error="No supported source files found in repository.",
            )

        total_lines = 0
        languages: dict[str, int] = {}
        issue_total = 0
        scores: list[int] = []
        file_results: list[dict] = []

        for fpath in files[:80]:  # cap for responsiveness
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = fpath.name
            lang = detect_language(rel, text)
            languages[lang] = languages.get(lang, 0) + 1
            metrics = compute_code_metrics(text, rel)
            total_lines += metrics.total_lines
            analysis = analyze_uploaded_source(text, rel)
            carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
            score = compute_green_score(carbon, len(analysis.issues), metrics, 70.0)
            issue_total += len(analysis.issues)
            scores.append(score)
            file_results.append(
                {
                    "file": str(fpath.relative_to(tmp)),
                    "language": lang,
                    "lines": metrics.total_lines,
                    "issues": len(analysis.issues),
                    "score": score,
                }
            )

        repo_score = int(round(sum(scores) / len(scores))) if scores else 0
        return RepoScanResult(
            repo_url=url,
            total_files=len(file_results),
            total_lines=total_lines,
            languages=languages,
            aggregate_issues=issue_total,
            repository_score=repo_score,
            file_results=file_results,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
