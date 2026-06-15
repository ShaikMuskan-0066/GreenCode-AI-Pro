"""
utils.py — Shared helpers for GreenCode AI Pro (paths, reports, styling).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from analyzer import AnalysisResult
from carbon_tracker import CarbonEstimate


def project_root() -> Path:
    """
    Return the directory containing ``app.py`` (project root).

    Returns:
        Absolute Path to GreenCodeAI-Pro/.
    """
    return Path(__file__).resolve().parent


def reports_dir() -> Path:
    """
    Return the ``reports/`` folder, creating it if needed.

    Returns:
        Path to the reports directory.
    """
    path = project_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_report_path() -> Path:
    """
    Default path for the text report file.

    Returns:
        Path to ``reports/report.txt``.
    """
    return reports_dir() / "report.txt"


def history_csv_path() -> Path:
    """
    Path to the legacy global analysis history CSV file.

    Returns:
        Path to ``reports/history.csv``.
    """
    return reports_dir() / "history.csv"


def user_reports_csv_path() -> Path:
    """
    Path to per-user analysis reports CSV.

    Returns:
        Path to ``reports/user_reports.csv``.
    """
    return reports_dir() / "user_reports.csv"


HISTORY_COLUMNS = [
    "Date",
    "Filename",
    "Language",
    "Energy",
    "CO2",
    "Cost",
    "Issues Count",
    "Green Score",
]

USER_REPORT_COLUMNS = [
    "user_id",
    "report_date",
    "filename",
    "language",
    "carbon",
    "energy",
    "score",
    "issues_count",
    "cost",
]


def load_all_user_reports() -> pd.DataFrame:
    """
    Load all rows from ``reports/user_reports.csv``.

    Returns:
        DataFrame with USER_REPORT_COLUMNS.
    """
    path = user_reports_csv_path()
    if path.is_file():
        try:
            df = pd.read_csv(path)
            for col in USER_REPORT_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[USER_REPORT_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=USER_REPORT_COLUMNS)


def load_user_analysis_history(user_id: str) -> pd.DataFrame:
    """
    Load analysis history for one user, shaped like the legacy HISTORY_COLUMNS.

    Args:
        user_id: Authenticated user UUID.

    Returns:
        DataFrame filtered to the user with display-friendly column names.
    """
    df = load_all_user_reports()
    if df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    subset = df[df["user_id"].astype(str) == str(user_id)].copy()
    if subset.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(
        {
            "Date": subset["report_date"],
            "Filename": subset["filename"],
            "Language": subset["language"],
            "Energy": subset["energy"],
            "CO2": subset["carbon"],
            "Cost": subset["cost"],
            "Issues Count": subset["issues_count"],
            "Green Score": subset["score"],
        }
    )


def load_analysis_history(user_id: str | None = None) -> pd.DataFrame:
    """
    Load analysis history (per-user if ``user_id`` is given).

    Args:
        user_id: Optional user UUID filter.

    Returns:
        History DataFrame in display column format.
    """
    if user_id:
        return load_user_analysis_history(user_id)
    path = history_csv_path()
    if path.is_file():
        try:
            df = pd.read_csv(path)
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[HISTORY_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_user_report(
    user_id: str,
    filename: str,
    language: str,
    carbon: CarbonEstimate,
    issues_count: int,
    green_score: int,
) -> Path:
    """
    Append one per-user report row to ``reports/user_reports.csv``.

    Args:
        user_id: Owner user UUID.
        filename: Analyzed file name.
        language: Detected language.
        carbon: Carbon estimate.
        issues_count: Issue count.
        green_score: Sustainability score.

    Returns:
        Path to user_reports.csv.
    """
    path = user_reports_csv_path()
    row = {
        "user_id": user_id,
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "language": language,
        "carbon": carbon.co2_kg,
        "energy": carbon.energy_kwh,
        "score": green_score,
        "issues_count": issues_count,
        "cost": carbon.cost_inr,
    }
    df = load_all_user_reports()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def get_user_account_stats(user_id: str) -> dict[str, float | int]:
    """
    Compute profile statistics from a user's report history.

    Args:
        user_id: User UUID.

    Returns:
        Dict with analyses_completed, average_score, highest_score, carbon metrics.
    """
    df = load_all_user_reports()
    subset = df[df["user_id"].astype(str) == str(user_id)]
    if subset.empty:
        return {
            "analyses_completed": 0,
            "average_score": 0.0,
            "highest_score": 0,
            "total_carbon_tracked": 0.0,
            "total_carbon_saved": 0.0,
        }
    scores = subset["score"].astype(float)
    carbons = subset["carbon"].astype(float)
    baseline = 1.0
    saved = sum(max(0.0, baseline - c) for c in carbons)
    return {
        "analyses_completed": int(len(subset)),
        "average_score": float(scores.mean()),
        "highest_score": int(scores.max()),
        "total_carbon_tracked": float(carbons.sum()),
        "total_carbon_saved": float(saved),
    }


def append_analysis_history(
    filename: str,
    language: str,
    carbon: CarbonEstimate,
    issues_count: int,
    green_score: int,
    user_id: str | None = None,
) -> Path:
    """
    Append one analysis row to legacy history and per-user reports.

    Args:
        filename: Analyzed file name.
        language: Detected language.
        carbon: Carbon estimate.
        issues_count: Number of issues found.
        green_score: Sustainability score 0–100.
        user_id: Optional owner UUID for user_reports.csv.

    Returns:
        Path to the legacy history CSV.
    """
    path = history_csv_path()
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Language": language,
        "Energy": carbon.energy_kwh,
        "CO2": carbon.co2_kg,
        "Cost": carbon.cost_inr,
        "Issues Count": issues_count,
        "Green Score": green_score,
    }
    df = load_analysis_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

    if user_id:
        append_user_report(user_id, filename, language, carbon, issues_count, green_score)
        from auth.auth_utils import increment_user_analyses_count  # noqa: PLC0415

        increment_user_analyses_count(user_id)

    return path


def save_analysis_report(
    analysis: AnalysisResult,
    carbon: CarbonEstimate,
    suggestions: list[str],
    report_path: Path | None = None,
) -> Path:
    """
    Write a plain-text sustainability report to disk.

    Args:
        analysis: Parsed analysis result.
        carbon: Carbon / cost estimate.
        suggestions: List of human-readable suggestions.
        report_path: Output file path (defaults to reports/report.txt).

    Returns:
        Path where the file was written.
    """
    out = report_path or default_report_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    titles = [i.title for i in analysis.issues]
    issue_lines = [f"  - {t}" for t in titles] if titles else ["  (none)"]
    lines = [
        "=" * 52,
        "GreenCode AI Pro — Sustainability Report",
        "=" * 52,
        "",
        f"Source: {analysis.file_path}",
        "",
        "Detected Issues:",
        *issue_lines,
        "",
        f"Estimated Energy: {carbon.energy_kwh} kWh",
        f"Estimated CO₂: {carbon.co2_kg} kg",
        f"Estimated Cost (INR): {carbon.cost_inr}",
        "",
        f"Method: {carbon.method}",
        f"Notes: {carbon.notes}",
        "",
        "Optimization Suggestions:",
        *[f"  - {s}" for s in suggestions],
        "",
        "=" * 52,
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def dark_theme_css() -> str:
    """
    Return minimal custom CSS for a darker Streamlit shell.

    Returns:
        A CSS string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    return """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.75rem; }
      .gc-card {
        background: linear-gradient(145deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid #2d3340;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
      }
      .gc-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
      }
      h1, h2, h3 { letter-spacing: 0.02em; }
    </style>
    """
