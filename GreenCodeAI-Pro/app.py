"""
app.py — GreenCode AI Pro: Streamlit dashboard for ML sustainability insights.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import analyze_uploaded_source
from carbon_tracker import estimate_carbon_footprint, live_adjusted_estimate
from code_metrics import CodeMetrics, compute_code_metrics
from github_analyzer import analyze_github_repository
from language_detector import language_badge_html
from auth.auth_utils import (
    check_session_expiry,
    ensure_default_admin,
    is_logged_in,
    logout,
    touch_activity,
)
from auth.forgot_password import render_forgot_password
from auth.login import render_login_form
from auth.profile import render_profile_page
from auth.signup import render_signup_form
from ai_assistant import build_assistant_context, generate_assistant_reply
from memory_optimizer import check_and_optimize_memory
from sustainability_insights import (
    determine_green_winner,
    sustainability_chart_values,
    sustainability_comparison_table_rows,
)
from metrics_insights import (
    build_file_comparison_profile,
    compute_quality_insights,
)
from monitor import sample_system_metrics
from report_generator import generate_pdf_report
from suggestions import build_suggestions
from sustainability_score import compute_green_score, score_status_label
from utils import append_analysis_history, dark_theme_css, load_analysis_history, save_analysis_report

UPLOAD_TYPES = ["py", "java", "js", "ts", "cpp", "c", "h", "cs", "go", "php"]


def _init_session_state() -> None:
    """Ensure Streamlit session keys exist before the dashboard reads them."""
    defaults: dict = {
        "analysis": None,
        "carbon_base": None,
        "suggestions": [],
        "metric_history": [],
        "last_report_text": "",
        "code_metrics": None,
        "quality_insights": None,
        "green_score": None,
        "language": "Python",
        "pdf_bytes": None,
        "compare_a": None,
        "compare_b": None,
        "repo_result": None,
        "logged_in": False,
        "username": "",
        "user_id": None,
        "user_name": "",
        "user_email": "",
        "user_role": "user",
        "last_activity_time": None,
        "session_warned": False,
        "login_attempts": 0,
        "nav_page": "📊 Dashboard",
        "show_auth_tabs": True,
        "ui_auth_tab": 0,
        "ui_show_learn": False,
        "ui_load_demo": False,
        "public_page": "home",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _run_full_analysis(text: str, logical_name: str, duration_hours: float) -> None:
    """Analyze source, compute metrics/score, save TXT + PDF + history."""
    with st.spinner("Analyzing code, metrics, and carbon footprint..."):
        analysis = analyze_uploaded_source(text, filename=logical_name)
        metrics = compute_code_metrics(text, logical_name)
        quality = compute_quality_insights(metrics, text, len(analysis.issues))
        carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
        mem = check_and_optimize_memory(threshold_percent=80.0)
        green_score = compute_green_score(carbon, len(analysis.issues), metrics, mem.memory_percent)
        suggestions = build_suggestions(analysis.issues)
        report_path = save_analysis_report(analysis, carbon, suggestions, metrics=metrics, quality=quality)
        append_analysis_history(
            logical_name,
            analysis.language,
            carbon,
            len(analysis.issues),
            green_score,
            user_id=st.session_state.get("user_id"),
        )
        st.session_state.analysis = analysis
        st.session_state.carbon_base = carbon
        st.session_state.suggestions = suggestions
        st.session_state.code_metrics = metrics
        st.session_state.quality_insights = quality
        st.session_state.green_score = green_score
        st.session_state.language = analysis.language
        st.session_state.metric_history = []
        st.session_state.last_report_text = report_path.read_text(encoding="utf-8")
        st.session_state.last_mem_result = mem
        try:
            pdf_path = generate_pdf_report(
                analysis, carbon, suggestions, analysis.language, green_score, metrics=metrics, quality=quality
            )
            st.session_state.pdf_bytes = pdf_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            st.session_state.pdf_bytes = None
            st.warning(f"PDF export skipped: {exc}")
    st.success(f"Analysis complete. Report saved to `{report_path}`.")


def _load_builtin_sample(duration_hours: float) -> None:
    """Load sample_train.py from the project folder and run analysis."""
    sample_path = Path(__file__).resolve().parent / "sample_train.py"
    if not sample_path.is_file():
        st.error("sample_train.py was not found next to app.py.")
        return
    text = sample_path.read_text(encoding="utf-8", errors="replace")
    _run_full_analysis(text, sample_path.name, duration_hours)


def _single_gauge_figure(title: str, value: float, bar_color: str = "#34d399") -> go.Figure:
    """Build one modern gauge chart."""
    v = min(100.0, max(0.0, float(value)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            title={"text": title, "font": {"size": 14, "color": "#d1d5db"}},
            number={"font": {"size": 26, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar": {"color": bar_color},
                "bgcolor": "#111827",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#1f2937"},
                    {"range": [50, 80], "color": "#111827"},
                    {"range": [80, 100], "color": "#0b1220"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=16, r=16, t=36, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d1d5db"},
    )
    return fig


def _quality_badge_color(label: str) -> str:
    """Return accent color for a quality label."""
    return {
        "Excellent": "#34d399",
        "Good": "#60a5fa",
        "Average": "#fbbf24",
        "Needs Improvement": "#f87171",
    }.get(label, "#94a3b8")


def _render_quality_indicator(title: str, score: int, label: str) -> None:
    """Render a KPI card with quality label and progress bar."""
    color = _quality_badge_color(label)
    st.markdown(
        f"""
        <div class="gc-kpi-card gc-fade-in">
          <div class="gc-kpi-title">{title}</div>
          <div class="gc-kpi-value">{score}/100</div>
          <div class="gc-kpi-subtitle" style="color:{color};font-weight:600;">{label}</div>
          <div style="margin-top:0.65rem;background:#1f2937;border-radius:999px;height:8px;overflow:hidden;">
            <div style="width:{min(100, max(0, score))}%;height:8px;background:{color};border-radius:999px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plotly_dark_layout(**kwargs) -> dict:
    """Shared dark-theme layout for Plotly charts."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#cbd5e1"},
        margin=dict(l=20, r=20, t=50, b=20),
    )
    base.update(kwargs)
    return base


def _sustainability_comparison_chart(profile_a: dict, profile_b: dict) -> go.Figure:
    """Bar chart comparing sustainability score and resource efficiency."""
    vals = sustainability_chart_values(profile_a, profile_b)
    categories = ["Sustainability Score", "Resource Efficiency"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=profile_a["name"],
            x=categories,
            y=[vals["a_sustainability"], vals["a_resource"]],
            marker_color="#60a5fa",
        )
    )
    fig.add_trace(
        go.Bar(
            name=profile_b["name"],
            x=categories,
            y=[vals["b_sustainability"], vals["b_resource"]],
            marker_color="#34d399",
        )
    )
    fig.update_layout(
        **_plotly_dark_layout(title="Sustainability Comparison", barmode="group"),
        yaxis=dict(range=[0, 100], title="Score (0–100)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360,
    )
    return fig


def _carbon_impact_comparison_chart(profile_a: dict, profile_b: dict) -> go.Figure:
    """Bar chart comparing CO₂ emissions and energy usage."""
    vals = sustainability_chart_values(profile_a, profile_b)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=profile_a["name"],
            x=["CO₂ (kg)", "Energy (kWh)"],
            y=[vals["a_co2"], vals["a_energy"]],
            marker_color="#60a5fa",
        )
    )
    fig.add_trace(
        go.Bar(
            name=profile_b["name"],
            x=["CO₂ (kg)", "Energy (kWh)"],
            y=[vals["b_co2"], vals["b_energy"]],
            marker_color="#34d399",
        )
    )
    fig.update_layout(
        **_plotly_dark_layout(title="Carbon & Energy Impact"),
        barmode="group",
        yaxis_title="Estimated impact",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360,
    )
    return fig


def _render_sustainability_issue_checklist(checklist: list[dict], title: str = "Green Software Issues") -> None:
    """Render sustainability issue checklist with pass/fail indicators."""
    st.markdown(f"**{title}**")
    for item in checklist:
        icon = "🔴" if item["detected"] else "🟢"
        status = "Detected" if item["detected"] else "Not detected"
        detail = f" — {item['detail']}" if item.get("detected") and item.get("detail") else ""
        st.markdown(
            f"""
            <div style="padding:0.45rem 0.65rem;margin:0.25rem 0;background:rgba(15,23,42,0.55);
            border:1px solid rgba(52,211,153,0.12);border-radius:10px;font-size:0.9rem;">
              {icon} <strong>{item['label']}</strong> · {status}{detail}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_optimization_cards(suggestions: list[str]) -> None:
    """Render green optimization suggestions as KPI-style cards."""
    if not suggestions:
        st.info("No optimization suggestions — script follows green software practices.")
        return
    cols = st.columns(min(3, len(suggestions)))
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="gc-kpi-card gc-fade-in" style="min-height:110px;">
                  <div class="gc-kpi-icon">🌿</div>
                  <div class="gc-kpi-title">AI Optimization</div>
                  <div class="gc-kpi-value" style="font-size:0.95rem;line-height:1.35;">{suggestion}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _comparison_radar_chart(dims: dict, name_a: str, name_b: str) -> go.Figure:
    """Radar chart comparing efficiency, sustainability, complexity, and carbon."""
    categories = ["Efficiency", "Sustainability", "Complexity", "Carbon Impact"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[dims["a_efficiency"], dims["a_sustainability"], dims["a_complexity"], dims["a_carbon"], dims["a_efficiency"]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name_a,
            line_color="#60a5fa",
            fillcolor="rgba(96,165,250,0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[dims["b_efficiency"], dims["b_sustainability"], dims["b_complexity"], dims["b_carbon"], dims["b_efficiency"]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name_b,
            line_color="#34d399",
            fillcolor="rgba(52,211,153,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(bgcolor="rgba(15,23,42,0.45)", radialaxis=dict(range=[0, 100], showgrid=True, gridcolor="#334155")),
        showlegend=True,
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def _comparison_bar_chart(dims: dict, name_a: str, name_b: str) -> go.Figure:
    """Grouped bar chart for comparison dimensions."""
    categories = ["Efficiency", "Sustainability", "Complexity", "Carbon Impact"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=name_a, x=categories, y=[dims["a_efficiency"], dims["a_sustainability"], dims["a_complexity"], dims["a_carbon"]], marker_color="#60a5fa"))
    fig.add_trace(go.Bar(name=name_b, x=categories, y=[dims["b_efficiency"], dims["b_sustainability"], dims["b_complexity"], dims["b_carbon"]], marker_color="#34d399"))
    fig.update_layout(
        barmode="group",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 100], title="Score (0–100)"),
    )
    return fig


def _comparison_score_chart(profile_a: dict, profile_b: dict) -> go.Figure:
    """Bar chart for sustainability and quality scores."""
    labels = ["Sustainability", "Maintainability", "Readability", "Code Quality"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=profile_a["name"],
            x=labels,
            y=[
                profile_a["score"],
                profile_a["quality"].maintainability_score,
                profile_a["quality"].readability_score,
                profile_a["quality"].code_quality_score,
            ],
            marker_color="#60a5fa",
        )
    )
    fig.add_trace(
        go.Bar(
            name=profile_b["name"],
            x=labels,
            y=[
                profile_b["score"],
                profile_b["quality"].maintainability_score,
                profile_b["quality"].readability_score,
                profile_b["quality"].code_quality_score,
            ],
            marker_color="#34d399",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Score Comparison",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#cbd5e1"},
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def _language_pie_chart(languages: dict[str, int]) -> go.Figure:
    """Pie chart for repository language distribution."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(languages.keys()),
                values=list(languages.values()),
                hole=0.45,
                marker=dict(colors=["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#fb7185"]),
            )
        ]
    )
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
    )
    return fig


def _complexity_distribution_chart(distribution: list[dict]) -> go.Figure:
    """Bar chart for cyclomatic complexity buckets."""
    fig = go.Figure(
        go.Bar(
            x=[d["bucket"] for d in distribution],
            y=[d["count"] for d in distribution],
            marker_color=["#34d399", "#60a5fa", "#fbbf24", "#f87171"],
        )
    )
    fig.update_layout(
        title="Complexity Distribution",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#cbd5e1"},
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Files",
    )
    return fig


def _history_line_chart(df: pd.DataFrame) -> go.Figure:
    """Plot live telemetry history with modern dark style."""
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="Waiting for live samples...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "#94a3b8"},
        )
        fig.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)",
            font={"color": "#cbd5e1"},
        )
        return fig

    x = df.index
    fig.add_trace(go.Scatter(x=x, y=df["cpu"], name="CPU %", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=x, y=df["ram"], name="RAM %", line=dict(color="#c084fc", width=2)))
    fig.add_trace(go.Scatter(x=x, y=df["gpu"], name="GPU %", line=dict(color="#4ade80", width=2)))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["co2_live"],
            name="Live CO2 (kg)",
            line=dict(color="#f59e0b", width=2.5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=55, t=24, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Utilization %", range=[0, 105], gridcolor="#334155"),
        yaxis2=dict(title="CO2 kg", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
        hovermode="x unified",
        transition={"duration": 350, "easing": "cubic-in-out"},
    )
    return fig


def _render_metric_card(title: str, value: str, subtitle: str = "", icon: str = "") -> None:
    """Render a glass KPI card with optional icon."""
    icon_html = f'<div class="gc-kpi-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="gc-kpi-card gc-fade-in">
          {icon_html}
          <div class="gc-kpi-title">{title}</div>
          <div class="gc-kpi-value">{value}</div>
          <div class="gc-kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_auth_visual_panel(title: str, subtitle: str) -> None:
    """CSS-only auth side panel — no stock imagery."""
    st.markdown(
        f"""
        <div class="gc-auth-visual gc-fade-in">
          <div class="gc-hero-glow"></div>
          <div class="gc-particles">
            <span></span><span></span><span></span><span></span><span></span>
          </div>
          <div style="position:relative;z-index:2;padding:1.4rem;">
            <div class="gc-orbit gc-orbit-1"></div>
            <div class="gc-orbit gc-orbit-2"></div>
            <div class="gc-core-dot"></div>
            <h3 style="margin:9rem 0 0 0;color:#ecfdf5;font-size:1.2rem;">{title}</h3>
            <p style="margin:0.4rem 0 0 0;color:#94a3b8;font-size:0.9rem;">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _set_public_page(page: str) -> None:
    """Navigate between public landing pages."""
    st.session_state.public_page = page
    st.rerun()


def _render_public_navbar(current: str) -> None:
    """Sticky glassmorphism top navbar for unauthenticated visitors."""
    st.markdown('<div class="gc-public-navbar-marker"></div>', unsafe_allow_html=True)
    brand_col, center_col, right_col = st.columns([2.5, 2.5, 2.5])
    with brand_col:
        if st.button("🌱 GreenCode AI Pro", key="nav_brand_home", use_container_width=True):
            _set_public_page("home")
    with center_col:
        nc1, nc2 = st.columns(2)
        with nc1:
            if st.button(
                "Home",
                key="nav_home",
                use_container_width=True,
                type="primary" if current == "home" else "secondary",
            ):
                _set_public_page("home")
        with nc2:
            if st.button(
                "About",
                key="nav_about",
                use_container_width=True,
                type="primary" if current == "about" else "secondary",
            ):
                _set_public_page("about")
    with right_col:
        nr1, nr2 = st.columns(2)
        with nr1:
            if st.button(
                "Login",
                key="nav_login",
                use_container_width=True,
                type="primary" if current == "login" else "secondary",
            ):
                _set_public_page("login")
        with nr2:
            if st.button(
                "Sign Up",
                key="nav_signup",
                use_container_width=True,
                type="primary" if current == "signup" else "secondary",
            ):
                _set_public_page("signup")
    st.markdown('<hr class="gc-nav-divider">', unsafe_allow_html=True)


def _render_landing_stat_card(value: str, label: str) -> None:
    """Premium gradient KPI card for the landing page."""
    st.markdown(
        f"""
        <div class="gc-stat-card gc-fade-in">
          <div class="gc-stat-value">{value}</div>
          <div class="gc-stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_premium_hero() -> None:
    """Centered premium SaaS hero — title, subtitle, and description."""
    st.markdown(
        """
        <section class="gc-hero-premium gc-hero-saas gc-hero-center">
          <div class="gc-hero-fx" aria-hidden="true">
            <div class="gc-hero-network"></div>
            <div class="gc-hero-glow gc-hero-glow-left"></div>
            <div class="gc-hero-glow gc-hero-glow-right"></div>
            <div class="gc-particles">
              <span></span><span></span><span></span><span></span><span></span>
            </div>
          </div>
          <div class="gc-hero-content-center">
            <div class="gc-hero-title-wrap">
              <span class="gc-hero-plant" aria-hidden="true">🌱</span>
              <h1 class="gc-hero-title-gradient">GreenCode AI Pro</h1>
            </div>
            <p class="gc-hero-subtitle">
              Build Sustainable, Efficient, and Green Software with AI-Powered Insights
            </p>
            <p class="gc-hero-description">
              Analyze code, track carbon emissions, monitor resources, and receive
              AI-powered optimization recommendations.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_cta_buttons() -> None:
    """Compact SaaS Login / Sign Up buttons directly below the hero."""
    st.markdown('<div class="gc-hero-cta-marker"></div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔐 Login", key="landing_action_login", use_container_width=True):
                _set_public_page("login")
        with b2:
            if st.button("📝 Sign Up", key="landing_action_signup", use_container_width=True):
                _set_public_page("signup")


def _render_what_is_section() -> None:
    """What is GreenCode AI Pro — overview panel."""
    st.markdown(
        """
        <div class="gc-info-panel">
          <h2>What is GreenCode AI Pro?</h2>
          <p>
            GreenCode AI Pro is an AI-powered sustainability platform that helps developers analyze
            source code, monitor system resources, estimate carbon emissions, and improve software
            efficiency through intelligent optimization recommendations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_step_card(step: str, icon: str, title: str, intro: str, bullets: list[str]) -> str:
    """Build HTML for one how-it-works step card."""
    body_parts: list[str] = []
    if intro:
        body_parts.append(f"<p>{intro}</p>")
    if bullets:
        items = "".join(f"<li>{b}</li>" for b in bullets)
        body_parts.append(f"<ul>{items}</ul>")
    body = "".join(body_parts)
    return f"""
    <div class="gc-step-card">
      <div class="gc-step-card-head">
        <span class="gc-step-badge">STEP {step}</span>
        <span class="gc-step-icon">{icon}</span>
        <h4>{title}</h4>
      </div>
      {body}
    </div>
    """


def _render_how_it_works_section() -> None:
    """Detailed How It Works cards with icons."""
    st.markdown(
        '<div class="gc-landing-section-title">How It Works</div>',
        unsafe_allow_html=True,
    )
    steps = [
        (
            "1",
            "📂",
            "Upload Source Code",
            "Upload Python, Java, JavaScript, C, C++, C#, PHP, Go and other supported files.",
            [],
        ),
        (
            "2",
            "🔍",
            "Code Analysis",
            "Detect inefficient coding patterns such as:",
            [
                "Large batch sizes",
                "Missing mixed precision",
                "Inefficient DataLoader settings",
                "Full fine-tuning",
                "Resource-heavy operations",
            ],
        ),
        (
            "3",
            "📊",
            "Code Metrics",
            "Generate:",
            ["Lines of Code", "File Statistics", "Complexity Metrics", "Sustainability Score"],
        ),
        (
            "4",
            "🌍",
            "Carbon Analysis",
            "Estimate:",
            ["Electricity Usage", "CO₂ Emissions", "Energy Consumption", "Operational Cost"],
        ),
        (
            "5",
            "💻",
            "Resource Monitoring",
            "Track:",
            ["CPU Usage", "RAM Usage", "GPU Usage", "Memory Efficiency"],
        ),
        (
            "6",
            "🤖",
            "AI Optimization",
            "Suggest:",
            ["LoRA", "Quantization", "Mixed Precision", "Better Resource Usage", "Performance Improvements"],
        ),
        (
            "7",
            "📄",
            "Reporting",
            "Generate:",
            ["TXT Reports", "PDF Reports", "Sustainability Reports"],
        ),
    ]
    left_steps = steps[0::2]
    right_steps = steps[1::2]
    c1, c2 = st.columns(2)
    with c1:
        for step in left_steps:
            st.markdown(_render_step_card(*step), unsafe_allow_html=True)
    with c2:
        for step in right_steps:
            st.markdown(_render_step_card(*step), unsafe_allow_html=True)


def _render_supported_features_section() -> None:
    """Supported features grid for the landing page."""
    st.markdown(
        '<div class="gc-landing-section-title">Supported Features</div>',
        unsafe_allow_html=True,
    )
    features = [
        ("🌱", "Carbon Tracking"),
        ("📊", "Sustainability Scoring"),
        ("💻", "Multi-Language Analysis"),
        ("📂", "GitHub Repository Analysis"),
        ("⚡", "Real-Time Monitoring"),
        ("📄", "PDF Report Generation"),
        ("🤖", "AI Optimization Suggestions"),
    ]
    cols = st.columns(3)
    for idx, (icon, title) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="gc-feature-grid-card">
                  <div class="gc-feature-grid-icon">{icon}</div>
                  <h4>{title}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_home_tech_stack_section() -> None:
    """Technology stack cards on the home page."""
    st.markdown(
        '<div class="gc-landing-section-title">Technology Stack</div>',
        unsafe_allow_html=True,
    )
    stack = ["Python", "Streamlit", "CodeCarbon", "Plotly", "Pandas", "Psutil", "GitPython"]
    cols = st.columns(4)
    for idx, item in enumerate(stack):
        with cols[idx % 4]:
            st.markdown(f'<div class="gc-tech-home-card">{item}</div>', unsafe_allow_html=True)


def _render_workflow_section() -> None:
    """Compact workflow steps for How GreenCode AI Pro Works."""
    st.markdown(
        """
        <div class="gc-landing-section">
          <div class="gc-landing-section-title">How GreenCode AI Pro Works</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    steps = [
        "Upload Code",
        "Analyze Code",
        "Calculate Metrics",
        "Estimate Carbon Impact",
        "Monitor Resources",
        "Generate AI Suggestions",
        "Export Reports",
    ]
    parts = ['<div class="gc-workflow-compact">']
    for idx, title in enumerate(steps):
        parts.append(f'<div class="gc-workflow-compact-step">{title}</div>')
        if idx < len(steps) - 1:
            parts.append('<div class="gc-workflow-compact-arrow">↓</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_landing_statistics() -> None:
    """Platform KPI statistics section."""
    st.markdown(
        """
        <div class="gc-landing-section">
          <div class="gc-landing-section-title">Platform Statistics</div>
          <div class="gc-landing-section-sub">Enterprise-grade sustainability intelligence at scale.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _render_landing_stat_card("20+", "Supported Checks")
    with s2:
        _render_landing_stat_card("100+", "Analyses")
    with s3:
        _render_landing_stat_card("Real-Time", "Monitoring")
    with s4:
        _render_landing_stat_card("PDF", "Reporting")


def _render_home_landing() -> None:
    """Render landing: hero → CTA → info sections → workflow → statistics."""
    _render_premium_hero()
    if not is_logged_in():
        _render_hero_cta_buttons()
    _render_what_is_section()
    _render_how_it_works_section()
    _render_supported_features_section()
    _render_home_tech_stack_section()
    _render_workflow_section()
    _render_landing_statistics()


def _render_login_page() -> None:
    """Dedicated login page — separate from home landing."""
    _render_public_navbar("login")
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>Login</h2>
          <p>Sign in to access your sustainability workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1.15])
    with left:
        _render_auth_visual_panel(
            "Secure Workspace Access",
            "Monitor carbon, analyze code, and optimize with AI insights.",
        )
    with right:
        st.markdown('<div class="gc-auth-card"><h3>Welcome Back</h3></div>', unsafe_allow_html=True)
        render_login_form()
        render_forgot_password()


def _render_signup_page() -> None:
    """Dedicated signup page — separate from home landing."""
    _render_public_navbar("signup")
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>Sign Up</h2>
          <p>Create your account and start building greener software.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1.15])
    with left:
        _render_auth_visual_panel(
            "Start Your Green Journey",
            "Create a profile and track your sustainability impact over time.",
        )
    with right:
        st.markdown('<div class="gc-auth-card"><h3>Create Account</h3></div>', unsafe_allow_html=True)
        render_signup_form()


def _render_public_app() -> None:
    """Route unauthenticated visitors across Home, About, Login, and Sign Up."""
    page = st.session_state.get("public_page", "home")
    if page == "login":
        _render_login_page()
    elif page == "signup":
        _render_signup_page()
    elif page == "about":
        _render_public_navbar("about")
        _render_about_page()
    else:
        _render_public_navbar("home")
        _render_home_landing()


def _render_about_page() -> None:
    """Simple About page: overview, mission, goals, stack, and key features."""
    st.markdown(
        """
        <div class="gc-page-header">
          <h2>About GreenCode AI Pro</h2>
          <p>AI-powered sustainability for modern software teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sections = [
        (
            "Project Overview",
            "GreenCode AI Pro is an AI-powered platform that analyzes code, estimates carbon "
            "emissions, monitors system resources, and delivers optimization recommendations.",
        ),
        (
            "Mission",
            "Help developers create energy-efficient and sustainable software.",
        ),
        (
            "Sustainability Goals",
            "Reduce carbon footprint, improve energy efficiency, and promote green computing practices.",
        ),
    ]
    for title, body in sections:
        st.markdown(
            f"""
            <div class="gc-about-block">
              <h4>{title}</h4>
              <p>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gc-landing-section-title">Technology Stack</div>', unsafe_allow_html=True)
    stack = ["Python", "Streamlit", "Plotly", "CodeCarbon", "Pandas", "GitPython", "Psutil"]
    cols = st.columns(4)
    for i, item in enumerate(stack):
        with cols[i % 4]:
            st.markdown(f'<div class="gc-stack-card">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="gc-landing-section-title">Key Features</div>', unsafe_allow_html=True)
    features = [
        "Carbon Tracking & Emissions Estimation",
        "Sustainability Score & Code Metrics",
        "GitHub Repository Analyzer",
        "AI Optimization Suggestions",
        "Real-Time Resource Monitoring",
        "PDF & TXT Report Export",
    ]
    for feat in features:
        st.markdown(
            f'<div class="gc-workflow-compact-step" style="margin-bottom:0.45rem;">✔ {feat}</div>',
            unsafe_allow_html=True,
        )


def _render_history_page() -> None:
    """User-specific analysis history with search, sort, and filter."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>🕓 Analysis History</h2>
          <p>Search, filter, and export your personal sustainability analyses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_id = st.session_state.get("user_id")
    df = load_analysis_history(user_id=str(user_id) if user_id else None)
    if df.empty or len(df.dropna(how="all")) == 0:
        st.info("No history yet. Run an analysis from Dashboard.")
        return
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        search = st.text_input("Search filename or language", key="hist_search")
    with c2:
        lang_filter = st.multiselect(
            "Filter language",
            sorted(df["Language"].dropna().unique().tolist()),
            key="hist_lang",
        )
    with c3:
        sort_col = st.selectbox("Sort by", ["Date", "Green Score", "CO2", "Issues Count"], key="hist_sort")
    view = df.copy()
    if search:
        mask = view["Filename"].astype(str).str.contains(search, case=False, na=False) | view["Language"].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]
    if lang_filter:
        view = view[view["Language"].isin(lang_filter)]
    view = view.sort_values(sort_col, ascending=st.checkbox("Ascending", value=False, key="hist_asc"))
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download my history CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="my_analysis_history.csv",
        mime="text/csv",
    )


def _render_reports_section() -> None:
    """TXT and PDF report actions."""
    analysis = st.session_state.analysis
    carbon_base = st.session_state.carbon_base
    st.markdown("### Reports")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.session_state.last_report_text:
            st.text_area("Latest report preview", st.session_state.last_report_text, height=220)
    with col_b:
        if st.session_state.last_report_text:
            st.download_button(
                label="Download TXT Report",
                data=st.session_state.last_report_text.encode("utf-8"),
                file_name="greencode_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        if st.session_state.pdf_bytes:
            st.download_button(
                label="Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name="greencode_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        if st.button("Re-save report", use_container_width=True):
            if analysis is not None and carbon_base is not None:
                p = save_analysis_report(
                    analysis,
                    carbon_base,
                    st.session_state.suggestions,
                    metrics=st.session_state.get("code_metrics"),
                    quality=st.session_state.get("quality_insights"),
                )
                st.session_state.last_report_text = p.read_text(encoding="utf-8")
                st.toast(f"Saved to {p}")


def _render_dashboard_cards() -> None:
    """Render premium KPI cards for core metrics."""
    carbon = st.session_state.get("carbon_base")
    score = st.session_state.get("green_score")
    mem = st.session_state.get("last_mem_result")
    cpu = f"{getattr(mem, 'memory_percent', 0):.0f}%"
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _render_metric_card("Green Score", f"{score}/100" if score is not None else "--", icon="🌿")
    with c2:
        _render_metric_card("CO2 Emission", f"{carbon.co2_kg:.2f} kg" if carbon else "--", icon="🌍")
    with c3:
        _render_metric_card("Energy Usage", f"{carbon.energy_kwh:.2f} kWh" if carbon else "--", icon="⚡")
    with c4:
        _render_metric_card("Cost", f"₹{carbon.cost_inr:.2f}" if carbon else "--", icon="💰")
    with c5:
        _render_metric_card("RAM Usage", cpu, icon="🧠")
    with c6:
        _render_metric_card("Language", st.session_state.get("language", "Python"), icon="💻")


@st.fragment(run_every=timedelta(seconds=2))
def _render_live_dashboard() -> None:
    """Auto-refreshing panel for system metrics and charts."""
    snap = sample_system_metrics(interval=0.1)
    mem = check_and_optimize_memory(threshold_percent=80.0)
    st.session_state.last_mem_result = mem
    base = st.session_state.get("carbon_base")

    if base is not None and st.session_state.get("analysis") is not None:
        e_live, co2_live, cost_live = live_adjusted_estimate(base, snap.cpu_percent, snap.ram_percent, snap.gpu_percent)
    else:
        e_live = co2_live = cost_live = 0.0

    row = {
        "cpu": snap.cpu_percent,
        "ram": snap.ram_percent,
        "gpu": snap.gpu_percent if snap.gpu_percent is not None else 0.0,
        "co2_live": co2_live,
        "energy_live": e_live,
        "cost_live": cost_live,
        "ts": time.time(),
    }
    st.session_state.metric_history.append(row)
    if len(st.session_state.metric_history) > 200:
        st.session_state.metric_history = st.session_state.metric_history[-200:]

    st.markdown("#### Live Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CPU Usage", f"{snap.cpu_percent:.1f}%")
    m2.metric("RAM Usage", f"{snap.ram_percent:.1f}%")
    m3.metric("GPU Usage", f"{snap.gpu_percent:.1f}%" if snap.gpu_percent is not None else "N/A")
    m4.metric("Live CO2", f"{co2_live:.2f} kg" if base else "--")
    m5.metric("Live Cost", f"₹{cost_live:.2f}" if base else "--")

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(_single_gauge_figure("CPU %", snap.cpu_percent, "#38bdf8"), use_container_width=True)
    g2.plotly_chart(_single_gauge_figure("RAM %", snap.ram_percent, "#c084fc"), use_container_width=True)
    g3.plotly_chart(_single_gauge_figure("GPU %", float(snap.gpu_percent or 0.0), "#4ade80"), use_container_width=True)
    df = pd.DataFrame(st.session_state.metric_history).reset_index(drop=True)
    st.plotly_chart(_history_line_chart(df), use_container_width=True)


def _render_dashboard_page(duration_hours: float) -> None:
    """Render premium dashboard UI while keeping existing functionality."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>📊 Dashboard</h2>
          <p>Analyze code, monitor live resources, and track sustainability metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Language Detected: {language_badge_html(st.session_state.get('language', 'Python'))}",
        unsafe_allow_html=True,
    )

    if st.session_state.pop("ui_load_demo", False):
        _load_builtin_sample(duration_hours)

    st.markdown('<div class="gc-section-title">Key Metrics</div>', unsafe_allow_html=True)
    _render_dashboard_cards()

    st.markdown(
        '<div class="gc-glass-panel gc-fade-in"><h4 style="margin:0;color:#e2e8f0;">Analyze Script</h4></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a training script",
        type=UPLOAD_TYPES,
        help="Supported: Python, Java, JavaScript, TypeScript, C, C++, C#, Go, PHP",
    )
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        if st.button("Run Analysis", type="primary"):
            _run_full_analysis(text, uploaded.name, duration_hours)

    analysis = st.session_state.analysis
    carbon_base = st.session_state.carbon_base
    if analysis is not None and carbon_base is not None:
        st.session_state.carbon_base = estimate_carbon_footprint(analysis, duration_hours=duration_hours)

    if st.session_state.green_score is not None:
        st.progress(st.session_state.green_score / 100.0, text="Sustainability Progress")
        st.plotly_chart(
            _single_gauge_figure("Green Score", float(st.session_state.green_score), "#22c55e"),
            use_container_width=True,
        )

    st.markdown(
        '<div class="gc-glass-panel gc-fade-in"><h4 style="margin:0;color:#e2e8f0;">Live Monitoring</h4></div>',
        unsafe_allow_html=True,
    )
    _render_live_dashboard()

    st.markdown(
        '<div class="gc-glass-panel gc-fade-in"><h4 style="margin:0;color:#e2e8f0;">Code Issues & Suggestions</h4></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Code Issues")
    if analysis is not None:
        if analysis.issues:
            for issue in analysis.issues:
                with st.expander(f"{issue.title} ({issue.code})"):
                    st.write(issue.detail)
        else:
            st.success("No issues detected by current rules.")
        with st.expander("View analyzed source"):
            st.code(analysis.raw_text, language="python")
    else:
        st.info("Upload or load a sample script to start.")

    st.markdown("#### Optimization Suggestions")
    if st.session_state.suggestions:
        for s in st.session_state.suggestions:
            st.markdown(f"- ✅ **{s}**")
    else:
        st.caption("Suggestions appear after analysis.")

    _render_reports_section()


def _render_carbon_page() -> None:
    """Render dedicated carbon tracking page."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>🌍 Carbon Tracking</h2>
          <p>Monitor energy, emissions, and cost estimates from your latest analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    carbon = st.session_state.get("carbon_base")
    score = st.session_state.get("green_score")
    if carbon is None:
        st.info("No carbon data yet. Run an analysis first.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        _render_metric_card("Energy", f"{carbon.energy_kwh} kWh")
    with c2:
        _render_metric_card("CO2", f"{carbon.co2_kg} kg")
    with c3:
        _render_metric_card("Cost", f"₹{carbon.cost_inr}")
    st.caption(f"{carbon.method} · {carbon.notes}")
    if score is not None:
        st.info(f"Green Score: {score}/100 ({score_status_label(int(score))})")


def _render_code_metrics_page() -> None:
    """Render dedicated code metrics page with quality dashboard."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>🧮 Code Metrics</h2>
          <p>Structural insights and engineering quality analysis from your analyzed source file.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics: CodeMetrics | None = st.session_state.get("code_metrics")
    if metrics is None:
        st.info("Run an analysis on the Dashboard to view code metrics.")
        return

    quality = st.session_state.get("quality_insights")
    if quality is None and st.session_state.get("analysis") is not None:
        analysis = st.session_state.analysis
        src = ""
        try:
            src = Path(analysis.file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        if src:
            quality = compute_quality_insights(metrics, src, len(analysis.issues))
            st.session_state.quality_insights = quality

    st.markdown(language_badge_html(metrics.language), unsafe_allow_html=True)
    st.markdown("#### Core Metrics")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        _render_metric_card("Lines of Code", str(metrics.total_lines), "Total lines", "📄")
    with r1c2:
        _render_metric_card("Blank Lines", str(metrics.blank_lines), "Whitespace", "⬜")
    with r1c3:
        _render_metric_card("Comment Lines", str(metrics.comment_lines), "Documentation", "💬")
    with r1c4:
        _render_metric_card("Code Lines", str(metrics.code_lines), "Executable", "⚡")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        _render_metric_card("Functions", str(metrics.functions), "Defined functions", "ƒ")
    with r2c2:
        _render_metric_card("Classes", str(metrics.classes), "Defined types", "◆")
    with r2c3:
        _render_metric_card("Imports", str(metrics.imports), "Dependencies", "⤴")

    if quality is not None:
        st.markdown("#### Quality Analysis")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(
                f"""
                <div class="gc-kpi-card gc-fade-in">
                  <div class="gc-kpi-title">Cyclomatic Complexity</div>
                  <div class="gc-kpi-value">{quality.cyclomatic_complexity}</div>
                  <div class="gc-kpi-subtitle" style="color:{_quality_badge_color(quality.complexity_label)};font-weight:600;">
                    {quality.complexity_label}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q2:
            _render_quality_indicator("Maintainability", quality.maintainability_score, quality.maintainability_label)
        with q3:
            _render_quality_indicator("Readability", quality.readability_score, quality.readability_label)
        with q4:
            _render_quality_indicator("Code Quality", quality.code_quality_score, quality.code_quality_label)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(_single_gauge_figure("Maintainability", quality.maintainability_score), use_container_width=True)
        with g2:
            st.plotly_chart(_single_gauge_figure("Readability", quality.readability_score, "#60a5fa"), use_container_width=True)
        with g3:
            st.plotly_chart(_single_gauge_figure("Code Quality", quality.code_quality_score, "#a78bfa"), use_container_width=True)


def _render_comparison_page(duration_hours: float) -> None:
    """Render sustainability-focused script comparison dashboard."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>⚖️ Script Comparison</h2>
          <p>Compare which training script is greener — carbon impact, energy usage, and green AI practices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Upload two ML training scripts · Supported: Python, Java, JavaScript, TypeScript, C, C++, C#, Go, PHP")
    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("File A (Baseline)", type=UPLOAD_TYPES, key="cmp_a")
    with c2:
        up_b = st.file_uploader("File B (Candidate)", type=UPLOAD_TYPES, key="cmp_b")
    if st.button("Compare Sustainability", type="primary"):
        if up_a is None or up_b is None:
            st.error("Upload both files.")
        else:
            with st.spinner("Analyzing carbon impact, energy usage, and green software practices..."):
                ta = up_a.getvalue().decode("utf-8", errors="replace")
                tb = up_b.getvalue().decode("utf-8", errors="replace")
                st.session_state.compare_a = build_file_comparison_profile("File A", ta, up_a.name, duration_hours)
                st.session_state.compare_b = build_file_comparison_profile("File B", tb, up_b.name, duration_hours)

    profile_a = st.session_state.get("compare_a")
    profile_b = st.session_state.get("compare_b")
    if not (profile_a and profile_b):
        return

    st.markdown("#### How Green Is Each Script?")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _render_metric_card(
            "File A Sustainability",
            f"{profile_a['score']}/100",
            f"Grade {profile_a.get('sustainability_grade', '—')} · {profile_a.get('sustainability_status', '')}",
            "🌱",
        )
    with k2:
        _render_metric_card(
            "File B Sustainability",
            f"{profile_b['score']}/100",
            f"Grade {profile_b.get('sustainability_grade', '—')} · {profile_b.get('sustainability_status', '')}",
            "🌱",
        )
    with k3:
        _render_metric_card(
            "File A CO₂",
            f"{profile_a['carbon'].co2_kg:.4f} kg",
            f"Energy {profile_a['carbon'].energy_kwh:.4f} kWh · ₹{profile_a['carbon'].cost_inr:.2f}",
            "☁️",
        )
    with k4:
        _render_metric_card(
            "File B CO₂",
            f"{profile_b['carbon'].co2_kg:.4f} kg",
            f"Energy {profile_b['carbon'].energy_kwh:.4f} kWh · ₹{profile_b['carbon'].cost_inr:.2f}",
            "☁️",
        )

    st.markdown("#### Sustainability Comparison Table")
    st.dataframe(
        pd.DataFrame(sustainability_comparison_table_rows(profile_a, profile_b)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Sustainability & Carbon Charts")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(_sustainability_comparison_chart(profile_a, profile_b), use_container_width=True)
    with ch2:
        st.plotly_chart(_carbon_impact_comparison_chart(profile_a, profile_b), use_container_width=True)

    st.markdown("#### Issues Found")
    ia, ib = st.columns(2)
    with ia:
        st.markdown(f"**File A — {profile_a['name']}** · {profile_a['issues']} issue(s)")
        _render_sustainability_issue_checklist(profile_a.get("issue_checklist", []))
    with ib:
        st.markdown(f"**File B — {profile_b['name']}** · {profile_b['issues']} issue(s)")
        _render_sustainability_issue_checklist(profile_b.get("issue_checklist", []))

    st.markdown("#### Optimization Suggestions")
    sa, sb = st.columns(2)
    with sa:
        st.markdown(f"**File A — {profile_a['name']}**")
        _render_optimization_cards(profile_a.get("green_suggestions", []))
    with sb:
        st.markdown(f"**File B — {profile_b['name']}**")
        _render_optimization_cards(profile_b.get("green_suggestions", []))

    winner, reasons = determine_green_winner(profile_a, profile_b)
    reason_html = "".join(f"<li>{r}</li>" for r in reasons)
    st.markdown(
        f"""
        <div class="gc-kpi-card gc-fade-in" style="margin-top:0.75rem;border-color:rgba(52,211,153,0.35);">
          <div class="gc-kpi-title">🏆 Final Recommendation</div>
          <div class="gc-kpi-value" style="font-size:1.35rem;">Winner: {winner}</div>
          <div class="gc-kpi-subtitle" style="margin-top:0.5rem;">Why this script is greener:</div>
          <ul style="color:#cbd5e1;margin:0.5rem 0 0 1rem;">{reason_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if profile_a["carbon"].co2_kg > 0:
        delta = (profile_a["carbon"].co2_kg - profile_b["carbon"].co2_kg) / profile_a["carbon"].co2_kg * 100.0
        if delta > 0:
            st.success(f"Candidate reduces estimated carbon by {delta:.1f}% vs baseline.")
        elif delta < 0:
            st.warning(f"Candidate increases estimated carbon by {abs(delta):.1f}% vs baseline.")


def _render_github_page(duration_hours: float) -> None:
    """Render sustainability-focused GitHub repository analyzer."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>🐙 GitHub Analyzer</h2>
          <p>Scan a repository for carbon impact, energy efficiency, and green AI training practices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repository")
    if st.button("Analyze Repository Sustainability", type="primary"):
        if not repo_url.strip():
            st.error("Enter a GitHub repository URL.")
        else:
            with st.spinner("Scanning repository for sustainability and carbon signals..."):
                st.session_state.repo_result = analyze_github_repository(repo_url, duration_hours=duration_hours)

    result = st.session_state.get("repo_result")
    if result is None:
        return
    if result.error:
        st.error(result.error)
        if result.metadata.name:
            st.caption(f"Repository: {result.metadata.owner}/{result.metadata.name}")
        return

    meta = result.metadata
    repo_label = f"{meta.owner}/{meta.name}" if meta.name else result.repo_url
    st.markdown(f"#### Repository: **{repo_label}**")
    if meta.description:
        st.caption(meta.description)

    st.markdown("#### Sustainability Overview")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            _single_gauge_figure("Sustainability Score", result.sustainability_score, "#34d399"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            _single_gauge_figure("Carbon Impact Score", result.carbon_impact_score, "#60a5fa"),
            use_container_width=True,
        )
    with g3:
        _render_metric_card(
            "Repository Health",
            f"{result.health_score}/100",
            f"Grade {getattr(result, 'sustainability_grade', '—')} · {getattr(result, 'sustainability_status', '')}",
            "💚",
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _render_metric_card("Energy Efficiency", f"{result.energy_efficiency_score}/100", "Lower energy per file", "⚡")
    with k2:
        _render_metric_card("Resource Efficiency", f"{result.resource_efficiency_score}/100", "Compute & memory usage", "🔋")
    with k3:
        _render_metric_card("Issues Found", str(result.aggregate_issues), f"Across {result.total_files} scanned files", "⚠️")
    with k4:
        _render_metric_card(
            "Sustainability Grade",
            getattr(result, "sustainability_grade", "—"),
            getattr(result, "sustainability_status", ""),
            "📊",
        )

    st.markdown("#### Issues Found — Green Software Checklist")
    _render_sustainability_issue_checklist(
        getattr(result, "issue_checklist", []),
        "Repository-wide sustainability violations",
    )

    if result.resource_heavy_files:
        st.warning("Resource-heavy scripts: " + ", ".join(result.resource_heavy_files[:6]))

    st.markdown("#### Optimization Suggestions")
    suggestions = getattr(result, "green_suggestions", None) or result.recommendations
    _render_optimization_cards(suggestions)

    if result.file_results:
        with st.expander("View per-file sustainability scan (training scripts)"):
            scan_df = pd.DataFrame(result.file_results)
            show_cols = [c for c in ["file", "score", "resource_efficiency", "issues", "co2_kg", "energy_kwh"] if c in scan_df.columns]
            st.dataframe(scan_df[show_cols], use_container_width=True, hide_index=True)


def _render_ai_assistant_page() -> None:
    """Render AI assistant page."""
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>🤖 AI Assistant</h2>
          <p>Ask sustainability questions and get optimization guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Local assistant by default; uses API keys automatically if available.")
    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = []
    for msg in st.session_state.ai_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Why is my carbon usage high?")
    if prompt:
        st.session_state.ai_chat.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            answer = generate_assistant_reply(
                prompt,
                history=st.session_state.ai_chat[:-1],
                context=build_assistant_context(st.session_state),
            )
        st.session_state.ai_chat.append({"role": "assistant", "content": answer})
        st.rerun()


def _render_sidebar() -> tuple[str, float]:
    """Render modern sidebar with icon navigation."""
    username = st.session_state.get("username", "user") or "user"
    initial = username[0].upper()
    with st.sidebar:
        st.markdown(
            """
            <div class="gc-sidebar-logo">🌱 GreenCode AI Pro</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="gc-user-box">
              <div class="gc-avatar">{initial}</div>
              <div>
                <div class="gc-user-name">{username}</div>
                <div class="gc-user-role">{st.session_state.get('user_role', 'user')}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Navigation")
        nav_page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📊 Dashboard",
                "🌍 Carbon Tracking",
                "🧮 Code Metrics",
                "🕓 Analysis History",
                "⚖️ Comparison",
                "🐙 GitHub Analyzer",
                "🤖 AI Assistant",
                "ℹ️ About",
                "👤 Profile",
                "🚪 Logout",
            ],
            key="nav_page",
            label_visibility="collapsed",
        )
        duration_hours = st.slider("Training duration (hours)", 0.25, 24.0, 1.0, 0.25)
        if st.button("Load built-in sample", use_container_width=True):
            _load_builtin_sample(duration_hours)
    return nav_page, duration_hours


def main() -> None:
    """Configure Streamlit app and route pages."""
    st.set_page_config(page_title="GreenCode AI Pro", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
    st.markdown(dark_theme_css(), unsafe_allow_html=True)
    _init_session_state()
    ensure_default_admin()

    if not is_logged_in():
        st.markdown(
            """
            <style>
              section[data-testid="stSidebar"] { display: none !important; }
              [data-testid="collapsedControl"] { display: none !important; }
              .stApp { background: linear-gradient(180deg, #030712 0%, #020617 55%, #071A1A 100%); }
              .gc-public-landing .block-container { padding-top: 1rem !important; overflow: visible !important; }
              .gc-public-landing header[data-testid="stHeader"] { background: transparent; }
              .gc-public-landing [data-testid="stAppViewContainer"] { overflow: visible !important; }
              .gc-public-landing .main { overflow: visible !important; }
            </style>
            <div class="gc-public-landing"></div>
            """,
            unsafe_allow_html=True,
        )
        _render_public_app()
        return

    expired, warn = check_session_expiry()
    if warn and not st.session_state.get("session_warned"):
        st.warning("Your session will expire in about 2 minutes due to inactivity.")
        st.session_state.session_warned = True
    if expired:
        logout()
        st.warning("Logged out automatically after 30 minutes of inactivity.")
        st.rerun()
    touch_activity()

    nav_page, duration_hours = _render_sidebar()

    if nav_page == "🚪 Logout":
        logout()
        st.rerun()
    elif nav_page == "🏠 Home":
        _render_home_landing()
    elif nav_page == "📊 Dashboard":
        _render_dashboard_page(duration_hours)
    elif nav_page == "🌍 Carbon Tracking":
        _render_carbon_page()
    elif nav_page == "🧮 Code Metrics":
        _render_code_metrics_page()
    elif nav_page == "🕓 Analysis History":
        _render_history_page()
    elif nav_page == "⚖️ Comparison":
        _render_comparison_page(duration_hours)
    elif nav_page == "🐙 GitHub Analyzer":
        _render_github_page(duration_hours)
    elif nav_page == "🤖 AI Assistant":
        _render_ai_assistant_page()
    elif nav_page == "ℹ️ About":
        _render_about_page()
    elif nav_page == "👤 Profile":
        render_profile_page()


if __name__ == "__main__":
    main()
