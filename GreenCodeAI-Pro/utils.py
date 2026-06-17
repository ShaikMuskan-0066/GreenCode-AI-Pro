"""
utils.py — Shared helpers for GreenCode AI Pro (paths, reports, styling).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from analyzer import AnalysisResult
from carbon_tracker import CarbonEstimate
from code_metrics import CodeMetrics
from metrics_insights import QualityInsights, format_quality_insights_report


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
    metrics: CodeMetrics | None = None,
    quality: QualityInsights | None = None,
) -> Path:
    """
    Write a plain-text sustainability report to disk.

    Args:
        analysis: Parsed analysis result.
        carbon: Carbon / cost estimate.
        suggestions: List of human-readable suggestions.
        report_path: Output file path (defaults to reports/report.txt).
        metrics: Optional structural code metrics.
        quality: Optional quality and complexity insights.

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
    ]

    if metrics is not None:
      lines.extend(
        [
            "Code Metrics:",
            f"  Total Lines: {metrics.total_lines}",
            f"  Code Lines: {metrics.code_lines}",
            f"  Blank Lines: {metrics.blank_lines}",
            f"  Comment Lines: {metrics.comment_lines}",
            f"  Functions: {metrics.functions}",
            f"  Classes: {metrics.classes}",
            f"  Imports: {metrics.imports}",
            f"  Language: {metrics.language}",
            "",
        ]
    )

    try:
        sus_score = getattr(analysis, "sustainability_score", None)
        sus_grade = getattr(analysis, "sustainability_grade", None)
        resource_eff = getattr(analysis, "resource_efficiency", None)

        opt_count = (
            len(getattr(analysis, "green_suggestions", []))
            if getattr(analysis, "green_suggestions", None) is not None
            else None
        )

        if (
            sus_score is not None
            or sus_grade is not None
            or resource_eff is not None
        ):
            lines.extend(["Sustainability Summary:"])

            if sus_score is not None:
                lines.append(f"  Sustainability Score: {sus_score}/100")

            if sus_grade is not None:
                lines.append(f"  Sustainability Grade: {sus_grade}")

            if resource_eff is not None:
                lines.append(f"  Resource Efficiency: {resource_eff}/100")

            if opt_count is not None:
                lines.append(
                    f"  Optimization Opportunities: {opt_count}"
                )

            lines.append("")

    except Exception:
        pass
    if quality is not None:
        lines.extend(["Quality Analysis:", *[f"  {row}" for row in format_quality_insights_report(quality)], ""])

    lines.extend(
        [
            "Optimization Suggestions:",
            *[f"  - {s}" for s in suggestions],
            "",
            "=" * 52,
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def dark_theme_css() -> str:
    """
    Return premium SaaS dark-theme CSS for the Streamlit shell.

    Returns:
        A CSS string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    return """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .block-container { padding-top: 0.35rem; padding-bottom: 1.5rem; max-width: 1140px; }
      .main .block-container { padding-top: 0.35rem; }
      header[data-testid="stHeader"] { background: rgba(3,7,18,0.55); backdrop-filter: blur(8px); }
      [data-testid="stAppViewContainer"] .main { background: transparent; }
      div[data-testid="stVerticalBlock"] > div { gap: 0.35rem; }
      div[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 700; }

      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #020617 100%);
        border-right: 1px solid rgba(52, 211, 153, 0.15);
      }
      section[data-testid="stSidebar"] .gc-sidebar-logo {
        font-size: 1.15rem; font-weight: 800; color: #ecfdf5;
        margin-bottom: 0.5rem; letter-spacing: -0.02em;
      }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px; padding: 0.35rem 0.5rem; transition: all 0.2s ease;
      }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(52, 211, 153, 0.08);
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(52, 211, 153, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.35);
        font-weight: 600;
      }

      .gc-fade-in { animation: gcFadeIn 0.5s ease both; }
      @keyframes gcFadeIn { from { opacity: 0.6; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
      .gc-hero-premium.gc-fade-in,
      .gc-hero-premium .gc-hero-title-gradient,
      .gc-hero-premium .gc-hero-subtitle,
      .gc-hero-premium .gc-hero-description,
      .gc-hero-premium .gc-hero-plant {
        opacity: 1 !important;
        visibility: visible !important;
        transform: none !important;
      }
      @keyframes gcGradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      @keyframes gcFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
      }
      @keyframes gcPulseGlow {
        0%, 100% { opacity: 0.45; }
        50% { opacity: 0.85; }
      }

      .gc-hero-premium {
        position: relative;
        overflow: visible;
        border-radius: 0;
        padding: 0;
        margin: 0 0 0.25rem 0;
        border: none;
        border-bottom: 1px solid rgba(34, 197, 94, 0.15);
        background: linear-gradient(180deg, rgba(3,7,18,0.4) 0%, rgba(2,6,23,0.85) 45%, rgba(7,26,26,0.55) 100%);
        background-size: 220% 220%;
        animation: gcGradientShift 12s ease infinite;
        box-shadow: none;
      }
      .gc-hero-saas {
        width: 100%;
        min-height: 0;
      }
      .gc-hero-fx {
        position: absolute;
        inset: 0;
        overflow: hidden;
        pointer-events: none;
        border-radius: 0;
      }
      .gc-hero-premium::before {
        content: ""; position: absolute; inset: 0;
        background-image:
          radial-gradient(circle at 15% 25%, rgba(34,197,94,0.2) 0%, transparent 38%),
          radial-gradient(circle at 85% 75%, rgba(6,182,212,0.16) 0%, transparent 36%),
          radial-gradient(circle at 50% 50%, rgba(20,184,166,0.08) 0%, transparent 55%),
          linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px);
        background-size: auto, auto, auto, 32px 32px, 32px 32px;
        pointer-events: none;
      }
      .gc-hero-premium::after {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(ellipse at center, transparent 55%, rgba(3,7,18,0.35) 100%);
        pointer-events: none;
      }
      .gc-hero-network {
        position: absolute; inset: 0; opacity: 0.35; pointer-events: none;
        background-image:
          radial-gradient(circle at 10% 80%, rgba(34,197,94,0.5) 1px, transparent 1px),
          radial-gradient(circle at 30% 40%, rgba(6,182,212,0.45) 1px, transparent 1px),
          radial-gradient(circle at 70% 60%, rgba(16,185,129,0.5) 1px, transparent 1px),
          radial-gradient(circle at 90% 20%, rgba(20,184,166,0.45) 1px, transparent 1px);
        background-size: 100% 100%;
      }
      .gc-hero-glow {
        position: absolute; width: 280px; height: 280px; border-radius: 50%;
        right: -40px; top: -60px; background: radial-gradient(circle, rgba(52,211,153,0.35), transparent 70%);
        animation: gcPulseGlow 4s ease-in-out infinite;
      }
      .gc-particles span {
        position: absolute; width: 6px; height: 6px; border-radius: 50%;
        background: rgba(52, 211, 153, 0.7); box-shadow: 0 0 12px rgba(52,211,153,0.8);
        animation: gcFloat 5s ease-in-out infinite;
      }
      .gc-particles span:nth-child(1) { left: 12%; top: 22%; animation-delay: 0s; }
      .gc-particles span:nth-child(2) { left: 28%; top: 68%; animation-delay: 0.8s; }
      .gc-particles span:nth-child(3) { left: 55%; top: 35%; animation-delay: 1.4s; }
      .gc-particles span:nth-child(4) { left: 72%; top: 58%; animation-delay: 2s; }
      .gc-particles span:nth-child(5) { left: 88%; top: 28%; animation-delay: 2.6s; }
      .gc-particles span:nth-child(6) { left: 40%; top: 78%; animation-delay: 3.2s; }
      .gc-particles span:nth-child(7) { left: 62%; top: 18%; animation-delay: 3.8s; }
      .gc-particles span:nth-child(8) { left: 48%; top: 52%; animation-delay: 4.4s; }

      .gc-hero-glow-left {
        left: -60px; top: 20%; right: auto;
        background: radial-gradient(circle, rgba(52,211,153,0.3), transparent 70%);
      }
      .gc-hero-glow-right {
        right: -50px; top: 10%; width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(59,130,246,0.22), transparent 70%);
      }

      .gc-public-navbar-marker { display: none; }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] {
        position: sticky;
        top: 0.25rem;
        z-index: 1000;
        background: rgba(3, 7, 18, 0.94);
        backdrop-filter: blur(20px) saturate(150%);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 14px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        margin: 0 0 0.65rem 0;
        box-shadow: 0 8px 32px rgba(2, 6, 23, 0.45);
        overflow: visible !important;
        min-height: auto !important;
        max-height: none !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        overflow: visible !important;
        min-height: auto !important;
        max-height: none !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton {
        overflow: visible !important;
        width: 100% !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.55rem 0.65rem !important;
        min-height: 2.85rem !important;
        height: auto !important;
        max-height: none !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: rgba(15, 23, 42, 0.55) !important;
        color: #d1d5db !important;
        font-size: 0.84rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, color 0.2s ease !important;
        transform: none !important;
        overflow: visible !important;
        white-space: normal !important;
        line-height: 1.2 !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button p {
        font-size: 0.84rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.2 !important;
        margin: 0 !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: rgba(34, 197, 94, 0.55) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(34, 197, 94, 0.2) !important;
        transform: none !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        text-align: left !important;
        padding: 0.45rem 0.25rem !important;
        white-space: normal !important;
        line-height: 1.25 !important;
        min-height: 2.75rem !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child .stButton > button p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669, #14b8a6) !important;
        border: 1px solid rgba(34, 197, 94, 0.55) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.22) !important;
      }

      .gc-top-nav-shell {
        position: sticky; top: 0; z-index: 1000;
        margin: 0 0 0.35rem 0; padding: 0.45rem 0;
        background: rgba(3, 7, 18, 0.88);
        backdrop-filter: blur(18px) saturate(140%);
        border-bottom: 1px solid rgba(34, 197, 94, 0.22);
        box-shadow: 0 4px 24px rgba(2, 6, 23, 0.45);
      }
      .gc-top-nav-shell .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: rgba(15, 23, 42, 0.55) !important;
        color: #d1d5db !important;
        transition: all 0.22s ease !important;
      }
      .gc-top-nav-shell .stButton > button:hover {
        border-color: rgba(34, 197, 94, 0.55) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2) !important;
        transform: translateY(-1px) !important;
      }
      .gc-top-nav-shell div[data-testid="column"]:first-child .stButton > button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: #ffffff !important; font-size: 1.08rem !important; font-weight: 800 !important;
        letter-spacing: -0.02em !important; text-align: left !important; padding-left: 0 !important;
      }
      .gc-top-nav-shell div[data-testid="column"]:first-child .stButton > button:hover {
        color: #22c55e !important; transform: none !important; box-shadow: none !important;
      }
      .gc-top-nav-shell div[data-testid="column"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669, #14b8a6) !important;
        border: 1px solid rgba(34, 197, 94, 0.55) !important;
        color: #ffffff !important;
        box-shadow: 0 0 24px rgba(34, 197, 94, 0.25) !important;
      }
      .gc-hero-center {
        text-align: center;
        padding: 3.5rem 1.5rem 2.75rem 1.5rem;
        min-height: 0;
        display: block;
      }
      .gc-hero-content-center {
        position: relative;
        z-index: 5;
        width: 100%;
        max-width: 960px;
        margin: 0 auto;
      }
      .gc-hero-title-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.35rem;
        margin-bottom: 0.25rem;
      }
      .gc-hero-plant {
        font-size: clamp(2.5rem, 5vw, 3.75rem);
        line-height: 1;
        filter: drop-shadow(0 0 20px rgba(34, 197, 94, 0.55));
      }
      .gc-hero-title-gradient {
        margin: 0;
        font-size: clamp(3.75rem, 10vw, 6.5rem);
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.02;
        color: #4ade80;
        background: linear-gradient(135deg, #ffffff 0%, #86efac 25%, #22c55e 55%, #14b8a6 80%, #06b6d4 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: #4ade80;
        text-shadow: 0 0 40px rgba(34, 197, 94, 0.55), 0 0 80px rgba(34, 197, 94, 0.2);
        animation: gcTitleGlow 4s ease-in-out infinite;
      }
      @keyframes gcTitleGlow {
        0%, 100% { filter: drop-shadow(0 0 28px rgba(34, 197, 94, 0.45)); }
        50% { filter: drop-shadow(0 0 42px rgba(6, 182, 212, 0.4)); }
      }
      .gc-hero-subtitle {
        margin: 1.1rem auto 0 auto;
        color: #e5e7eb !important;
        font-size: clamp(1.15rem, 2.4vw, 1.55rem);
        max-width: 820px;
        line-height: 1.55;
        font-weight: 500;
        opacity: 1 !important;
      }
      .gc-hero-description {
        margin: 0.9rem auto 0 auto;
        color: #9ca3af !important;
        font-size: clamp(0.95rem, 1.6vw, 1.1rem);
        max-width: 720px;
        line-height: 1.7;
        opacity: 1 !important;
      }
      .gc-nav-divider {
        border: none;
        border-top: 1px solid rgba(34, 197, 94, 0.12);
        margin: 0 0 0.35rem 0;
      }
      .gc-public-landing .block-container { padding-top: 1rem !important; padding-bottom: 1.25rem !important; overflow: visible !important; }
      .gc-public-landing [data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
      .gc-hero-cta-marker { display: none; }
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"],
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] {
        max-width: 440px;
        margin-left: auto !important;
        margin-right: auto !important;
        overflow: visible !important;
      }
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] .stButton > button,
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.25rem !important;
        min-height: 2.75rem !important;
        border: 1px solid rgba(34, 197, 94, 0.5) !important;
        background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(6,182,212,0.14)) !important;
        color: #ecfdf5 !important;
        box-shadow: 0 0 22px rgba(34, 197, 94, 0.18), inset 0 0 12px rgba(34, 197, 94, 0.06) !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
        transform: none !important;
      }
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] .stButton > button:hover,
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #22c55e !important;
        box-shadow: 0 0 32px rgba(34, 197, 94, 0.35), 0 0 48px rgba(6, 182, 212, 0.15) !important;
        transform: none !important;
      }
      .gc-workflow-compact { max-width: 520px; margin: 0 auto; text-align: center; }
      .gc-workflow-compact-step {
        background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(7,26,26,0.82));
        border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 12px;
        padding: 0.75rem 1rem; color: #f8fafc; font-weight: 600; font-size: 0.95rem;
        transition: all 0.2s ease;
      }
      .gc-workflow-compact-step:hover {
        border-color: rgba(6, 182, 212, 0.5);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.12);
        transform: scale(1.02);
      }
      .gc-workflow-compact-arrow { color: #22c55e; font-size: 1.1rem; padding: 0.2rem 0; text-shadow: 0 0 10px rgba(34,197,94,0.5); }
      .gc-info-panel {
        background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(7,26,26,0.82));
        border: 1px solid rgba(34, 197, 94, 0.22);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 0.5rem;
        text-align: center;
      }
      .gc-info-panel h2 {
        margin: 0 0 0.75rem 0;
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
      }
      .gc-info-panel p {
        margin: 0;
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 880px;
        margin-left: auto;
        margin-right: auto;
      }
      .gc-step-card {
        background: linear-gradient(160deg, rgba(15,23,42,0.94), rgba(7,26,26,0.86));
        border: 1px solid rgba(34, 197, 94, 0.22);
        border-left: 3px solid #22c55e;
        border-radius: 14px;
        padding: 1.1rem 1.15rem;
        min-height: 140px;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }
      .gc-step-card:hover {
        border-color: rgba(6, 182, 212, 0.45);
        box-shadow: 0 8px 28px rgba(2, 6, 23, 0.4), 0 0 18px rgba(34, 197, 94, 0.1);
      }
      .gc-step-card-head {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.45rem;
      }
      .gc-step-badge {
        font-size: 0.68rem;
        font-weight: 700;
        color: #22c55e;
        letter-spacing: 0.08em;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.28);
        border-radius: 6px;
        padding: 0.12rem 0.4rem;
      }
      .gc-step-icon { font-size: 1.3rem; }
      .gc-step-card h4 { margin: 0; color: #f8fafc; font-size: 1rem; font-weight: 700; }
      .gc-step-card p { margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.86rem; line-height: 1.5; }
      .gc-step-card ul { margin: 0.4rem 0 0 0; padding-left: 1.1rem; color: #cbd5e1; font-size: 0.84rem; line-height: 1.55; }
      .gc-feature-grid-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(7,26,26,0.8));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        min-height: 110px;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
      }
      .gc-feature-grid-card:hover {
        border-color: rgba(34, 197, 94, 0.4);
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
        transform: translateY(-2px);
      }
      .gc-feature-grid-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
      .gc-feature-grid-card h4 { margin: 0; color: #f1f5f9; font-size: 0.9rem; font-weight: 700; }
      .gc-tech-home-card {
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.65rem;
        transition: all 0.2s ease;
      }
      .gc-tech-home-card:hover { border-color: #22c55e; box-shadow: 0 0 16px rgba(34,197,94,0.15); }
      .gc-feature-row-card {
        background: linear-gradient(160deg, rgba(15,23,42,0.88), rgba(7,26,26,0.75));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px; padding: 1.15rem 0.9rem; min-height: 175px;
        text-align: center; transition: all 0.25s ease;
        box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.06);
      }
      .gc-feature-row-card:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: rgba(34, 197, 94, 0.45);
        box-shadow: 0 12px 36px rgba(2, 6, 23, 0.45), 0 0 24px rgba(34, 197, 94, 0.12);
      }
      .gc-feature-row-icon {
        font-size: 1.75rem; margin-bottom: 0.55rem;
        filter: drop-shadow(0 0 12px rgba(34, 197, 94, 0.45));
      }
      .gc-feature-row-card h4 { margin: 0 0 0.45rem 0; color: #ffffff; font-size: 0.92rem; font-weight: 700; }
      .gc-feature-row-card p { margin: 0; color: #94a3b8; font-size: 0.78rem; line-height: 1.45; }
      .gc-stat-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(7,26,26,0.85));
        border: 1px solid rgba(34, 197, 94, 0.22);
        border-radius: 16px; padding: 1.25rem 1rem; text-align: center;
        transition: all 0.25s ease; min-height: 130px;
        position: relative; overflow: hidden;
      }
      .gc-stat-card::before {
        content: ""; position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(6,182,212,0.06));
        opacity: 0; transition: opacity 0.25s ease;
      }
      .gc-stat-card:hover::before { opacity: 1; }
      .gc-stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(6, 182, 212, 0.45);
        box-shadow: 0 14px 40px rgba(2, 6, 23, 0.5), 0 0 28px rgba(34, 197, 94, 0.15);
      }
      .gc-stat-value {
        font-size: clamp(1.8rem, 3vw, 2.4rem); font-weight: 900; color: #ffffff;
        background: linear-gradient(90deg, #22c55e, #14b8a6, #06b6d4);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: gcStatPulse 3s ease-in-out infinite;
      }
      @keyframes gcStatPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
      }
      .gc-stat-label { margin-top: 0.35rem; color: #d1d5db; font-size: 0.88rem; font-weight: 600; }
      .gc-landing-section { margin: 1.5rem 0 1rem 0; }
      .gc-landing-section-title {
        font-size: 1.45rem; font-weight: 800; color: #ffffff;
        margin: 0 0 0.35rem 0; letter-spacing: -0.02em;
      }
      .gc-landing-section-sub { color: #94a3b8; font-size: 0.92rem; margin: 0 0 1rem 0; }
      .gc-workflow { max-width: 760px; margin: 0 auto; }
      .gc-workflow-step {
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(7,26,26,0.8));
        border: 1px solid rgba(34, 197, 94, 0.22); border-left: 3px solid #22c55e;
        border-radius: 14px; padding: 1rem 1.15rem; margin: 0;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
      }
      .gc-workflow-step:hover {
        transform: translateX(4px);
        border-color: rgba(6, 182, 212, 0.45);
        box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45), 0 0 20px rgba(34, 197, 94, 0.1);
      }
      .gc-workflow-step-head { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.45rem; }
      .gc-workflow-badge {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; color: #22c55e;
        background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
        border-radius: 6px; padding: 0.15rem 0.45rem;
      }
      .gc-workflow-icon { font-size: 1.35rem; filter: drop-shadow(0 0 8px rgba(34,197,94,0.45)); }
      .gc-workflow-step h4 { margin: 0; color: #ffffff; font-size: 1rem; font-weight: 700; }
      .gc-workflow-step p { margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.86rem; line-height: 1.5; }
      .gc-workflow-list { margin: 0.4rem 0 0 0; padding-left: 1.1rem; color: #cbd5e1; font-size: 0.84rem; line-height: 1.55; }
      .gc-workflow-list li { margin: 0.15rem 0; }
      .gc-workflow-arrow {
        text-align: center; color: #22c55e; font-size: 1.25rem; line-height: 1;
        padding: 0.35rem 0; text-shadow: 0 0 14px rgba(34,197,94,0.55);
      }
      .gc-why-card {
        background: linear-gradient(160deg, rgba(15,23,42,0.92), rgba(7,26,26,0.82));
        border: 1px solid rgba(148,163,184,0.18); border-radius: 14px;
        padding: 1.1rem; min-height: 130px; transition: all 0.22s ease;
      }
      .gc-why-card:hover {
        transform: translateY(-3px);
        border-color: rgba(34, 197, 94, 0.4);
        box-shadow: 0 10px 30px rgba(2,6,23,0.4), 0 0 22px rgba(34,197,94,0.1);
      }
      .gc-why-icon { font-size: 1.5rem; margin-bottom: 0.4rem; filter: drop-shadow(0 0 10px rgba(34,197,94,0.4)); }
      .gc-why-card h4 { margin: 0 0 0.35rem 0; color: #f8fafc; font-size: 0.92rem; font-weight: 700; }
      .gc-why-card p { margin: 0; color: #94a3b8; font-size: 0.82rem; line-height: 1.45; }
      .gc-about-block {
        background: linear-gradient(145deg, rgba(15,23,42,0.88), rgba(7,26,26,0.78));
        border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 14px;
        padding: 1.15rem 1.25rem; margin-bottom: 0.75rem;
      }
      .gc-about-block h4 { margin: 0 0 0.4rem 0; color: #22c55e; font-size: 0.95rem; font-weight: 700; }
      .gc-about-block p { margin: 0; color: #d1d5db; font-size: 0.92rem; line-height: 1.6; }
      .gc-action-col .stButton > button { margin-top: 0 !important; }
      .gc-stat-card.gc-fade-in, .gc-workflow-compact-step { opacity: 1 !important; }
      .gc-about-mission {
        background: linear-gradient(120deg, rgba(7,26,26,0.9), rgba(15,23,42,0.85));
        border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 18px;
        padding: 2rem; text-align: center; margin-bottom: 1.5rem;
      }
      .gc-about-mission h3 { margin: 0; color: #22c55e; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.12em; }
      .gc-about-mission p { margin: 0.75rem 0 0 0; color: #d1d5db; font-size: 1.15rem; line-height: 1.65; }
      .stApp { background: linear-gradient(180deg, #030712 0%, #020617 55%, #071A1A 100%); }
      .gc-feature-chips {
        display: flex; flex-wrap: wrap; gap: 0.55rem; justify-content: center;
        margin-top: 1.6rem;
      }
      .gc-chip {
        display: inline-flex; align-items: center; padding: 0.42rem 0.85rem;
        border-radius: 999px; font-size: 0.86rem; font-weight: 600; color: #d1fae5;
        background: rgba(6, 78, 59, 0.45); border: 1px solid rgba(52, 211, 153, 0.35);
        box-shadow: 0 0 16px rgba(52, 211, 153, 0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .gc-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 22px rgba(52, 211, 153, 0.28);
      }
      .gc-home-section { margin-top: 2rem; margin-bottom: 1.5rem; }
      .gc-public-hide-sidebar section[data-testid="stSidebar"] { display: none; }
      .gc-public-hide-sidebar [data-testid="collapsedControl"] { display: none; }

      .gc-hero-content { position: relative; z-index: 2; }
      .gc-hero-premium h1 {
        margin: 0; font-size: 2.45rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.03em;
      }
      .gc-hero-premium p {
        margin: 0.65rem 0 0 0; color: #cbd5e1; font-size: 1.08rem; max-width: 720px; line-height: 1.55;
      }
      .gc-hero-visual {
        position: relative; min-height: 220px; border-radius: 16px;
        border: 1px solid rgba(52,211,153,0.2);
        background: linear-gradient(145deg, rgba(15,23,42,0.75), rgba(2,6,23,0.9));
        backdrop-filter: blur(12px);
        padding: 1rem; overflow: hidden;
      }
      .gc-hero-visual::after {
        content: "AI + Carbon Intelligence";
        position: absolute; bottom: 12px; left: 14px; color: #86efac; font-size: 0.78rem; letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .gc-orbit {
        position: absolute; border: 1px dashed rgba(52,211,153,0.35); border-radius: 50%;
        animation: gcSpin 18s linear infinite;
      }
      .gc-orbit-1 { width: 140px; height: 140px; left: 50%; top: 42%; transform: translate(-50%,-50%); }
      .gc-orbit-2 { width: 90px; height: 90px; left: 50%; top: 42%; transform: translate(-50%,-50%); animation-duration: 12s; }
      @keyframes gcSpin { from { transform: translate(-50%,-50%) rotate(0deg); } to { transform: translate(-50%,-50%) rotate(360deg); } }
      .gc-core-dot {
        position: absolute; left: 50%; top: 42%; width: 18px; height: 18px; border-radius: 50%;
        transform: translate(-50%,-50%); background: #34d399; box-shadow: 0 0 24px #34d399;
      }

      .gc-section-title {
        font-size: 1.45rem; font-weight: 700; color: #f1f5f9; margin: 1.4rem 0 0.8rem 0;
      }
      .gc-checklist { color: #cbd5e1; line-height: 1.9; font-size: 0.98rem; }
      .gc-checklist li { list-style: none; margin: 0.2rem 0; }

      .gc-card, .gc-glass-panel {
        background: linear-gradient(145deg, rgba(15,23,42,0.82), rgba(17,24,39,0.72));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 16px; padding: 1.1rem; margin-bottom: 0.85rem;
        backdrop-filter: blur(10px);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
      }
      .gc-card:hover, .gc-glass-panel:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 14px 30px rgba(2,6,23,0.45);
        border-color: rgba(52,211,153,0.35);
      }

      .gc-kpi-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(17,24,39,0.82));
        border: 1px solid transparent;
        border-radius: 14px; padding: 0.9rem 1rem; min-height: 118px; margin-bottom: 0.75rem;
        background-clip: padding-box;
        box-shadow: inset 0 0 0 1px rgba(148,163,184,0.18);
        transition: all 0.22s ease;
      }
      .gc-kpi-card:hover {
        box-shadow: 0 0 0 1px rgba(52,211,153,0.45), 0 12px 24px rgba(2,6,23,0.35);
        transform: translateY(-2px);
      }
      .gc-kpi-icon { font-size: 1.25rem; margin-bottom: 0.35rem; }
      .gc-kpi-title { color: #94a3b8; font-size: 0.84rem; font-weight: 500; }
      .gc-kpi-value { color: #f8fafc; font-size: 1.5rem; font-weight: 800; margin: 0.15rem 0; }
      .gc-kpi-subtitle { color: #64748b; font-size: 0.76rem; }

      .gc-feature-card {
        background: rgba(15,23,42,0.72); border: 1px solid rgba(148,163,184,0.2);
        border-radius: 14px; padding: 1rem; min-height: 150px; margin-bottom: 0.8rem;
        transition: all 0.22s ease;
      }
      .gc-feature-card:hover { border-color: #34d399; transform: translateY(-3px); }
      .gc-feature-icon { font-size: 1.35rem; margin-bottom: 0.35rem; }
      .gc-feature-card h4 { margin: 0.15rem 0 0.4rem 0; color: #e2e8f0; font-size: 1rem; }
      .gc-feature-card p { margin: 0; color: #94a3b8; font-size: 0.88rem; line-height: 1.45; }

      .gc-stack-card {
        background: rgba(30,41,59,0.72); border: 1px solid rgba(148,163,184,0.2);
        border-radius: 12px; padding: 0.95rem; text-align: center; margin-bottom: 0.75rem; color: #e2e8f0;
        transition: all 0.2s ease;
      }
      .gc-stack-card:hover { border-color: #34d399; transform: scale(1.02); }

      .gc-impact-card {
        background: linear-gradient(160deg, rgba(6,78,59,0.25), rgba(15,23,42,0.85));
        border: 1px solid rgba(52,211,153,0.22); border-radius: 14px; padding: 1rem; min-height: 120px;
      }
      .gc-impact-card h4 { margin: 0 0 0.4rem 0; color: #86efac; font-size: 0.95rem; }
      .gc-impact-card p { margin: 0; color: #cbd5e1; font-size: 0.88rem; }

      .gc-timeline { border-left: 2px solid rgba(52,211,153,0.35); margin-left: 0.4rem; padding-left: 1rem; }
      .gc-timeline-item {
        margin-bottom: 0.9rem; padding: 0.75rem 0.9rem; border-radius: 12px;
        background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.16);
      }
      .gc-timeline-item strong { color: #a7f3d0; }

      .gc-user-box {
        display: flex; gap: 0.65rem; align-items: center; padding: 0.65rem;
        margin-bottom: 0.8rem; border: 1px solid rgba(52,211,153,0.2); border-radius: 12px;
        background: rgba(15,23,42,0.85);
      }
      .gc-avatar {
        width: 38px; height: 38px; border-radius: 999px; display: flex; align-items: center;
        justify-content: center; background: linear-gradient(135deg, #065f46, #134e4a);
        border: 1px solid rgba(52,211,153,0.35); font-size: 1rem;
      }
      .gc-user-name { color: #e2e8f0; font-weight: 700; font-size: 0.92rem; }
      .gc-user-role { color: #94a3b8; font-size: 0.74rem; text-transform: capitalize; }

      .gc-auth-visual {
        min-height: 360px; border-radius: 16px; position: relative; overflow: hidden;
        border: 1px solid rgba(52,211,153,0.22);
        background: linear-gradient(160deg, #020617, #0f172a 55%, #064e3b);
      }
      .gc-auth-visual .gc-particles span { background: rgba(110,231,183,0.75); }
      .gc-auth-card {
        background: rgba(15,23,42,0.72); border: 1px solid rgba(148,163,184,0.2);
        border-radius: 14px; padding: 0.85rem 1rem; margin-bottom: 0.7rem;
      }
      .gc-auth-card h3 { margin: 0; color: #ecfdf5; font-size: 1.15rem; }

      .gc-profile-hero {
        background: linear-gradient(120deg, rgba(6,78,59,0.35), rgba(15,23,42,0.9));
        border: 1px solid rgba(52,211,153,0.25); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem;
      }
      .gc-profile-hero h2 { margin: 0; color: #f8fafc; font-size: 1.5rem; }
      .gc-profile-hero p { margin: 0.35rem 0 0 0; color: #94a3b8; }

      .gc-page-header {
        background: linear-gradient(120deg, rgba(15,23,42,0.9), rgba(2,6,23,0.95));
        border: 1px solid rgba(148,163,184,0.18); border-radius: 14px;
        padding: 0.9rem 1.1rem; margin-bottom: 1rem;
      }
      .gc-page-header h2 { margin: 0; color: #f8fafc; font-size: 1.55rem; }
      .gc-page-header p { margin: 0.25rem 0 0 0; color: #94a3b8; font-size: 0.9rem; }

      .gc-badge {
        display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600;
      }
      h1, h2, h3 { letter-spacing: 0.01em; }

      div.stButton > button {
        border-radius: 10px; transition: box-shadow 0.15s ease, border-color 0.15s ease;
      }
      .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button:hover,
      .gc-hero-cta-marker + div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        transform: none !important;
      }

      @media (max-width: 900px) {
        .gc-hero-premium { border-radius: 0; }
        .gc-hero-center { padding: 2.25rem 1rem 1.75rem 1rem; }
        .gc-hero-title-gradient { font-size: clamp(2.75rem, 11vw, 3.75rem); }
        .gc-hero-plant { font-size: 2.25rem; }
        .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] {
          padding: 0.75rem 0.55rem;
        }
        .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button,
        .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] .stButton > button p {
          font-size: 0.78rem !important;
        }
        .gc-public-navbar-marker + div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child .stButton > button {
          font-size: 0.88rem !important;
        }
        .gc-landing-section { margin: 1.1rem 0 0.75rem 0; }
      }
    </style>
    """
