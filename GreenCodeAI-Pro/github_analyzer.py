"""
github_analyzer.py — Clone a GitHub repository and run sustainability scans.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from analyzer import analyze_uploaded_source
from carbon_tracker import estimate_carbon_footprint
from code_metrics import compute_code_metrics
from language_detector import SUPPORTED_EXTENSIONS, detect_language
from sustainability_insights import (
    aggregate_repo_green_suggestions,
    build_issue_checklist_from_codes,
    compute_resource_efficiency_score,
    sustainability_grade,
)
from sustainability_score import compute_green_score, score_status_label


SCAN_EXTENSIONS: frozenset[str] = frozenset(SUPPORTED_EXTENSIONS.keys())


@dataclass
class RepoMetadata:
    """GitHub REST API metadata for a repository."""

    name: str = ""
    owner: str = ""
    description: str = ""
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    size_kb: int = 0
    updated_at: str = ""
    main_language: str = ""


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
    metadata: RepoMetadata = field(default_factory=RepoMetadata)
    largest_files: list[dict] = field(default_factory=list)
    folder_structure: dict[str, int] = field(default_factory=dict)
    carbon_impact_score: int = 0
    energy_efficiency_score: int = 0
    sustainability_score: int = 0
    recommendations: list[str] = field(default_factory=list)
    complexity_distribution: list[dict] = field(default_factory=list)
    duplicate_candidates: list[str] = field(default_factory=list)
    resource_heavy_files: list[str] = field(default_factory=list)
    sustainability_grade: str = "F"
    sustainability_status: str = "Needs Improvement"
    resource_efficiency_score: int = 0
    health_score: int = 0
    issue_checklist: list[dict] = field(default_factory=list)
    green_suggestions: list[str] = field(default_factory=list)
    detected_issue_codes: set[str] = field(default_factory=set)


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


def _parse_owner_repo(url: str) -> tuple[str, str]:
    """Extract owner and repo name from a GitHub URL."""
    parsed = urlparse(_normalize_repo_url(url))
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) >= 2:
        return parts[0], parts[1].replace(".git", "")
    return "", ""


def _fetch_github_metadata(url: str) -> RepoMetadata:
    """
    Fetch repository metadata from the GitHub REST API.

    Args:
        url: Repository URL.

    Returns:
        RepoMetadata (empty fields on failure).
    """
    owner, repo = _parse_owner_repo(url)
    if not owner or not repo:
        return RepoMetadata()

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "GreenCodeAI-Pro/1.0", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return RepoMetadata(
            name=data.get("name", repo),
            owner=data.get("owner", {}).get("login", owner),
            description=data.get("description") or "",
            stars=int(data.get("stargazers_count", 0)),
            forks=int(data.get("forks_count", 0)),
            open_issues=int(data.get("open_issues_count", 0)),
            size_kb=int(data.get("size", 0)),
            updated_at=str(data.get("updated_at", ""))[:10],
            main_language=data.get("language") or "",
        )
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, ValueError):
        return RepoMetadata(name=repo, owner=owner)


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


def _folder_structure_summary(files: list[Path], root: Path) -> dict[str, int]:
    """Count files per top-level folder."""
    counts: Counter[str] = Counter()
    for fpath in files:
        rel = fpath.relative_to(root)
        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        counts[top] += 1
    return dict(counts.most_common(12))


def _find_duplicate_candidates(file_results: list[dict]) -> list[str]:
    """Flag files with identical line counts and languages (heuristic duplication)."""
    buckets: dict[tuple[str, int], list[str]] = {}
    for item in file_results:
        key = (item["language"], item["lines"])
        buckets.setdefault(key, []).append(item["file"])
    dups: list[str] = []
    for paths in buckets.values():
        if len(paths) > 1:
            dups.extend(paths[:3])
    return dups[:8]


def _build_green_recommendations(issue_codes: set[str], resource_heavy: list[str]) -> list[str]:
    """Generate sustainability-first recommendations for a repository."""
    recs = list(aggregate_repo_green_suggestions(issue_codes))
    if resource_heavy and "Optimize Resource Usage" not in recs:
        recs.append("Optimize Resource Usage")
    return recs[:8]


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

    metadata = _fetch_github_metadata(url)
    tmp = Path(tempfile.mkdtemp(prefix="greencode_repo_"))
    try:
        _clone_repository(url, tmp)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        return RepoScanResult(repo_url=url, total_files=0, total_lines=0, error=str(exc), metadata=metadata)

    try:
        all_files = _iter_source_files(tmp)
        if not all_files:
            return RepoScanResult(
                repo_url=url,
                total_files=0,
                total_lines=0,
                error="No supported source files found in repository.",
                metadata=metadata,
            )

        folder_structure = _folder_structure_summary(all_files, tmp)
        total_lines = 0
        languages: dict[str, int] = {}
        issue_total = 0
        scores: list[int] = []
        file_results: list[dict] = []
        energy_scores: list[float] = []
        carbon_scores: list[float] = []
        resource_scores: list[int] = []
        issue_codes: set[str] = set()
        total_energy = 0.0
        total_co2 = 0.0

        for fpath in all_files[:80]:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(fpath.relative_to(tmp))
            lang = detect_language(fpath.name, text)
            languages[lang] = languages.get(lang, 0) + 1
            metrics = compute_code_metrics(text, fpath.name)
            total_lines += metrics.total_lines
            analysis = analyze_uploaded_source(text, rel)
            carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
            score = compute_green_score(carbon, len(analysis.issues), metrics, 70.0)
            res_eff = compute_resource_efficiency_score(len(analysis.issues), carbon)
            file_codes = {issue.code for issue in analysis.issues}
            issue_codes.update(file_codes)
            issue_total += len(analysis.issues)
            scores.append(score)
            resource_scores.append(res_eff)
            energy_scores.append(max(0.0, 100.0 - carbon.energy_kwh * 200.0))
            carbon_scores.append(max(0.0, 100.0 - carbon.co2_kg * 500.0))
            file_results.append(
                {
                    "file": rel,
                    "language": lang,
                    "lines": metrics.total_lines,
                    "issues": len(analysis.issues),
                    "score": score,
                    "resource_efficiency": res_eff,
                    "co2_kg": round(carbon.co2_kg, 4),
                    "energy_kwh": round(carbon.energy_kwh, 4),
                    "issue_codes": sorted(file_codes),
                }
            )

        repo_score = int(round(sum(scores) / len(scores))) if scores else 0
        energy_eff = int(round(sum(energy_scores) / len(energy_scores))) if energy_scores else 0
        carbon_impact = int(round(sum(carbon_scores) / len(carbon_scores))) if carbon_scores else 0
        resource_eff = int(round(sum(resource_scores) / len(resource_scores))) if resource_scores else 0

        largest_files = sorted(file_results, key=lambda x: x["lines"], reverse=True)[:8]
        resource_heavy = [
            f["file"]
            for f in file_results
            if "RESOURCE_HEAVY" in f.get("issue_codes", []) or f["issues"] >= 3
        ][:8]
        duplicate_candidates = _find_duplicate_candidates(file_results)
        green_suggestions = aggregate_repo_green_suggestions(issue_codes)
        recommendations = _build_green_recommendations(issue_codes, resource_heavy)
        issue_checklist = build_issue_checklist_from_codes(issue_codes)
        grade = sustainability_grade(repo_score)
        status = score_status_label(repo_score)

        complexity_distribution: list[dict] = []

        if not metadata.main_language and languages:
            metadata.main_language = max(languages, key=languages.get)

        return RepoScanResult(
            repo_url=url,
            total_files=len(file_results),
            total_lines=total_lines,
            languages=languages,
            aggregate_issues=issue_total,
            repository_score=repo_score,
            file_results=file_results,
            metadata=metadata,
            largest_files=largest_files,
            folder_structure=folder_structure,
            carbon_impact_score=carbon_impact,
            energy_efficiency_score=energy_eff,
            sustainability_score=repo_score,
            recommendations=recommendations,
            complexity_distribution=complexity_distribution,
            duplicate_candidates=duplicate_candidates,
            resource_heavy_files=resource_heavy,
            sustainability_grade=grade,
            sustainability_status=status,
            resource_efficiency_score=resource_eff,
            health_score=repo_score,
            issue_checklist=issue_checklist,
            green_suggestions=green_suggestions,
            detected_issue_codes=issue_codes,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
