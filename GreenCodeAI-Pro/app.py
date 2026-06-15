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
from memory_optimizer import check_and_optimize_memory
from monitor import sample_system_metrics
from report_generator import generate_pdf_report
from suggestions import build_suggestions
from sustainability_score import compute_green_score, score_status_label
from utils import (
    append_analysis_history,
    dark_theme_css,
    load_analysis_history,
    save_analysis_report,
)

UPLOAD_TYPES = ["py", "java", "js", "ts", "cpp"]


def _init_session_state() -> None:
    """
    Ensure Streamlit session keys exist before the dashboard reads them.
    """
    defaults: dict = {
        "analysis": None,
        "carbon_base": None,
        "suggestions": [],
        "metric_history": [],
        "last_report_text": "",
        "code_metrics": None,
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
        "nav_page": "Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _run_full_analysis(text: str, logical_name: str, duration_hours: float) -> None:
    """
    Analyze source, compute metrics/score, save TXT + PDF + history.

    Args:
        text: Source code text.
        logical_name: Display filename.
        duration_hours: Training hours for carbon scaling.
    """
    with st.spinner("Analyzing code, metrics, and carbon footprint…"):
        analysis = analyze_uploaded_source(text, filename=logical_name)
        metrics = compute_code_metrics(text, logical_name)
        carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
        mem = check_and_optimize_memory(threshold_percent=80.0)
        green_score = compute_green_score(
            carbon, len(analysis.issues), metrics, mem.memory_percent
        )
        suggestions = build_suggestions(analysis.issues)
        report_path = save_analysis_report(analysis, carbon, suggestions)
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
        st.session_state.green_score = green_score
        st.session_state.language = analysis.language
        st.session_state.metric_history = []
        st.session_state.last_report_text = report_path.read_text(encoding="utf-8")
        st.session_state.last_mem_result = mem
        try:
            pdf_path = generate_pdf_report(
                analysis,
                carbon,
                suggestions,
                analysis.language,
                green_score,
                metrics=metrics,
            )
            st.session_state.pdf_bytes = pdf_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            st.session_state.pdf_bytes = None
            st.warning(f"PDF export skipped: {exc}")
    st.success(f"Analysis complete. Report saved to `{report_path}`.")


def _load_builtin_sample(duration_hours: float) -> None:
    """
    Load ``sample_train.py`` from the project folder and run analysis.

    Args:
        duration_hours: Hours assumed for carbon scaling.
    """
    sample_path = Path(__file__).resolve().parent / "sample_train.py"
    if not sample_path.is_file():
        st.error("sample_train.py was not found next to app.py.")
        return
    text = sample_path.read_text(encoding="utf-8", errors="replace")
    _run_full_analysis(text, sample_path.name, duration_hours)


def _single_gauge_figure(title: str, value: float, bar_color: str = "#00d4aa") -> go.Figure:
    """
    Build one semi-circular gauge (0–100) for a single metric.

    Args:
        title: Label shown on the gauge.
        value: Percent value to display.
        bar_color: Gauge bar hex color.

    Returns:
        Plotly figure for one indicator.
    """
    v = min(100.0, max(0.0, float(value)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            title={"text": title, "font": {"size": 14, "color": "#c9d1d9"}},
            number={"font": {"size": 28, "color": "#f0f2f6"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": bar_color},
                "bgcolor": "#161b22",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=16, r=16, t=40, b=16),
        paper_bgcolor="rgba(14,17,23,0)",
        font={"color": "#c9d1d9"},
    )
    return fig


def _history_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plot live telemetry history (CPU, RAM, GPU, live CO₂).

    Args:
        df: DataFrame built from ``metric_history`` rows.

    Returns:
        Plotly line figure.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="Waiting for live samples…",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "#8b949e"},
        )
        fig.update_layout(
            height=320,
            paper_bgcolor="rgba(14,17,23,0)",
            plot_bgcolor="rgba(22,27,34,0.6)",
            font={"color": "#c9d1d9"},
        )
        return fig

    x = df.index
    fig.add_trace(go.Scatter(x=x, y=df["cpu"], name="CPU %", line=dict(color="#58a6ff")))
    fig.add_trace(go.Scatter(x=x, y=df["ram"], name="RAM %", line=dict(color="#d2a8ff")))
    fig.add_trace(go.Scatter(x=x, y=df["gpu"], name="GPU %", line=dict(color="#56d364")))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["co2_live"],
            name="Live CO₂ (kg, scaled)",
            line=dict(color="#ffa657"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=50, t=40, b=40),
        paper_bgcolor="rgba(14,17,23,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font={"color": "#c9d1d9"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Utilization %", range=[0, 105], gridcolor="#30363d"),
        yaxis2=dict(
            title="CO₂ kg",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
    )
    return fig


def _ai_assistant_reply(question: str) -> str:
    """
    Answer sustainability questions via OpenAI/Gemini if keys exist, else local rules.

    Args:
        question: User question text.

    Returns:
        Assistant response string.
    """
    q = question.strip().lower()
    if not q:
        return "Ask me about carbon usage, LoRA, mixed precision, or energy-saving tips."

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if openai_key:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are GreenCode AI, a concise ML sustainability tutor.",
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=400,
            )
            return resp.choices[0].message.content or "No response."
        except Exception as exc:  # noqa: BLE001
            pass  # fall through to local

    if gemini_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(
                f"You are GreenCode AI sustainability tutor. Answer briefly:\n{question}"
            )
            return resp.text or "No response."
        except Exception:
            pass

    # Local rule-based fallback
    if "carbon" in q and ("high" in q or "why" in q):
        return (
            "**Why carbon usage may be high:** large batch sizes, full fine-tuning, missing "
            "mixed precision, zero DataLoader workers, and long training duration all increase "
            "GPU/CPU energy. Check the Issues panel and enable AMP, LoRA, and sensible workers."
        )
    if "lora" in q:
        return (
            "**LoRA (Low-Rank Adaptation)** trains small adapter matrices instead of all weights, "
            "cutting memory and energy versus full fine-tuning while keeping good quality on many tasks."
        )
    if "reduce" in q and "energy" in q:
        return (
            "To **reduce energy**: use mixed precision (FP16/BF16), parameter-efficient fine-tuning "
            "(LoRA), smaller batches with gradient accumulation, more DataLoader workers, "
            "early stopping, and quantization where accuracy allows."
        )
    if "mixed precision" in q or "amp" in q:
        return (
            "**Mixed precision** runs matmuls in FP16/BF16 on supported GPUs, often ~1.5–2× faster "
            "with lower power per step. In PyTorch use `torch.cuda.amp.autocast` and `GradScaler`."
        )
    if "quant" in q:
        return (
            "**Quantization** stores weights in INT8/4-bit formats to shrink memory and speed "
            "inference; training-time QLoRA combines quantization with LoRA for efficient fine-tuning."
        )
    return (
        "I can help with **carbon usage**, **LoRA**, **mixed precision**, **quantization**, and "
        "**energy reduction**. Try: 'Why is my carbon usage high?' or 'What is LoRA?'"
    )


@st.fragment(run_every=timedelta(seconds=2))
def render_live_dashboard() -> None:
    """
    Auto-refreshing panel: system metrics, memory optimizer, gauges, and charts.
    """
    snap = sample_system_metrics(interval=0.1)
    mem = check_and_optimize_memory(threshold_percent=80.0)
    st.session_state.last_mem_result = mem

    base = st.session_state.get("carbon_base")

    if base is not None and st.session_state.get("analysis") is not None:
        e_live, co2_live, cost_live = live_adjusted_estimate(
            base,
            snap.cpu_percent,
            snap.ram_percent,
            snap.gpu_percent,
        )
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

    st.subheader("System metrics (live)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CPU usage", f"{snap.cpu_percent:.1f}%")
    c2.metric("RAM usage", f"{snap.ram_percent:.1f}%")
    if snap.gpu_percent is not None:
        c3.metric("GPU usage", f"{snap.gpu_percent:.1f}%")
    else:
        c3.metric("GPU usage", "N/A")
    if base is not None:
        c4.metric("Live CO₂ (scaled)", f"{co2_live:.2f} kg")
        c5.metric("Live cost (scaled)", f"₹{cost_live:.2f}")
    else:
        c4.metric("Live CO₂ (scaled)", "—")
        c5.metric("Live cost (scaled)", "—")

    if mem.gc_triggered:
        st.warning(
            f"Garbage collection triggered · Memory {mem.memory_percent:.0f}% · "
            f"Objects collected: {mem.objects_collected} · GC count: {mem.gc_count}"
        )
    else:
        st.caption(
            f"Memory {mem.memory_percent:.0f}% · GC count {mem.gc_count} · "
            f"Live refresh ~2s · GPU: {snap.gpu_name or 'not detected'}"
        )

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(_single_gauge_figure("CPU %", snap.cpu_percent), use_container_width=True)
    g2.plotly_chart(_single_gauge_figure("RAM %", snap.ram_percent, "#d2a8ff"), use_container_width=True)
    gpu_show = float(snap.gpu_percent) if snap.gpu_percent is not None else 0.0
    g3.plotly_chart(_single_gauge_figure("GPU %", gpu_show, "#56d364"), use_container_width=True)

    hist = st.session_state.metric_history
    df = pd.DataFrame(hist)
    if not df.empty:
        df = df.reset_index(drop=True)
    st.plotly_chart(_history_line_chart(df), use_container_width=True)


def _render_header_cards() -> None:
    """
    Show language badge and green score cards when analysis exists.
    """
    lang = st.session_state.get("language", "Python")
    score = st.session_state.get("green_score")
    h1, h2, h3 = st.columns([2, 2, 2])
    with h1:
        st.markdown(
            f"**Language detected:** {language_badge_html(lang)}",
            unsafe_allow_html=True,
        )
    with h2:
        if score is not None:
            status = score_status_label(int(score))
            st.metric("Green Score", f"{score}/100", delta=status)
        else:
            st.metric("Green Score", "—")
    with h3:
        user = st.session_state.get("username", "")
        st.metric("Welcome", user or "user")


def _render_auth_gate() -> None:
    """
    Pre-login screen with Login and Sign Up tabs.
    """
    st.markdown(
        """
        <div class="gc-card" style="max-width:520px;margin:2rem auto;text-align:center;">
          <h2 style="margin:0;">GreenCode AI Pro</h2>
          <p style="color:#8b949e;">Sign up or log in to access the sustainability dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    with tab_login:
        col_l, col_c, col_r = st.columns([1, 1.4, 1])
        with col_c:
            render_login_form()
            render_forgot_password()
    with tab_signup:
        col_l, col_c, col_r = st.columns([1, 1.4, 1])
        with col_c:
            render_signup_form()


def _render_history_page() -> None:
    """
    User-specific analysis history with search, filter, and sort.
    """
    st.subheader("Analysis history")
    user_id = st.session_state.get("user_id")
    df = load_analysis_history(user_id=str(user_id) if user_id else None)
    if df.empty or len(df.dropna(how="all")) == 0:
        st.info("No history yet. Run an analysis from the Dashboard to save your first report.")
        return
    search = st.text_input("Search filename or language", key="hist_search")
    lang_filter = st.multiselect(
        "Filter by language",
        sorted(df["Language"].dropna().unique().tolist()),
        key="hist_lang",
    )
    view = df.copy()
    if search:
        mask = view["Filename"].astype(str).str.contains(search, case=False, na=False) | view[
            "Language"
        ].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]
    if lang_filter:
        view = view[view["Language"].isin(lang_filter)]
    sort_col = st.selectbox("Sort by", ["Date", "Green Score", "CO2", "Issues Count"], key="hist_sort")
    ascending = st.checkbox("Ascending", value=False, key="hist_asc")
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download my history CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="my_analysis_history.csv",
        mime="text/csv",
    )


def _render_issues_and_suggestions() -> None:
    """
    Display detected issues, source preview, and optimization suggestions.
    """
    analysis = st.session_state.analysis
    st.subheader("Code issues")
    if analysis is not None:
        if analysis.issues:
            for issue in analysis.issues:
                sev = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
                with st.expander(f"{sev} **{issue.title}** (`{issue.code}`)", expanded=False):
                    st.write(issue.detail)
        else:
            st.success("No issues detected by the current rules.")
        lang = (analysis.language or "python").lower()
        code_lang = "python" if lang == "python" else lang.replace("++", "pp")
        with st.expander("View analyzed source"):
            st.code(analysis.raw_text, language=code_lang if code_lang != "c++" else "cpp")
    else:
        st.warning("No script loaded yet.")

    st.subheader("Optimization suggestions")
    if st.session_state.suggestions:
        for s in st.session_state.suggestions:
            st.markdown(f"- ✔ **{s}**")
    else:
        st.caption("Suggestions appear after you run an analysis.")


def _render_reports_section() -> None:
    """
    TXT preview, download buttons, PDF export, and re-save.
    """
    analysis = st.session_state.analysis
    carbon_base = st.session_state.carbon_base
    st.subheader("Reports")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.session_state.last_report_text:
            st.text_area("Latest report preview", st.session_state.last_report_text, height=200)
    with col_b:
        if st.session_state.last_report_text:
            st.download_button(
                label="Download report.txt",
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
        if st.button("Re-save report to disk", use_container_width=True):
            if analysis is not None and carbon_base is not None:
                p = save_analysis_report(analysis, carbon_base, st.session_state.suggestions)
                st.session_state.last_report_text = p.read_text(encoding="utf-8")
                st.toast(f"Saved to {p}")


def main() -> None:
    """
    Configure Streamlit, login gate, sidebar, and tabbed dashboard.
    """
    st.set_page_config(
        page_title="GreenCode AI Pro",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(dark_theme_css(), unsafe_allow_html=True)
    _init_session_state()
    ensure_default_admin()

    if not is_logged_in():
        _render_auth_gate()
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

    st.markdown(
        """
        <div class="gc-card">
          <h1 style="margin:0;">GreenCode AI Pro</h1>
          <p style="margin:0.4rem 0 0 0; color:#8b949e;">
            Real-time sustainability dashboard · Multi-language analysis · Carbon insights
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Navigation")
        st.caption(f"Welcome, **{st.session_state.get('username', '')}**")
        nav_page = st.radio(
            "Menu",
            [
                "Dashboard",
                "Profile",
                "Analysis History",
                "GitHub Analyzer",
                "AI Assistant",
            ],
            key="nav_page",
        )
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()
        st.markdown("---")
        st.caption("Live charts refresh every ~2s on Dashboard.")
        duration_hours = st.slider(
            "Training duration (hours)", min_value=0.25, max_value=24.0, value=1.0, step=0.25
        )
        st.markdown("---")
        if st.button("Load built-in sample", use_container_width=True):
            _load_builtin_sample(duration_hours)
        st.markdown("---")
        with st.expander("About"):
            st.markdown(
                """
                **GreenCode AI Pro** analyzes training scripts (Python, Java, JS, TS, C++),
                estimates CO₂ with **CodeCarbon**, monitors **CPU/RAM/GPU**, and tracks history.

                Optional: set `OPENAI_API_KEY` or `GEMINI_API_KEY` for smarter AI answers.
                """
            )

    if nav_page == "Profile":
        render_profile_page()
        return

    if nav_page == "Analysis History":
        _render_history_page()
        return

    if nav_page == "GitHub Analyzer":
        st.subheader("GitHub repository analyzer")
        repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/org/repo")
        if st.button("Analyze repository", type="primary"):
            with st.spinner("Cloning and scanning repository (may take a minute)…"):
                st.session_state.repo_result = analyze_github_repository(
                    repo_url, duration_hours=duration_hours
                )
        result = st.session_state.get("repo_result")
        if result is not None:
            if result.error:
                st.error(result.error)
            else:
                g1, g2, g3 = st.columns(3)
                g1.metric("Repository score", f"{result.repository_score}/100")
                g2.metric("Total files scanned", result.total_files)
                g3.metric("Total lines", result.total_lines)
                st.write("**Detected languages:**", result.languages)
                st.metric("Aggregate issues", result.aggregate_issues)
                if result.file_results:
                    st.dataframe(pd.DataFrame(result.file_results), use_container_width=True)
        return

    if nav_page == "AI Assistant":
        st.subheader("Ask GreenCode AI")
        st.caption("Local tutor by default · Set OPENAI_API_KEY or GEMINI_API_KEY for cloud AI.")
        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat = []
        for msg in st.session_state.ai_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        prompt = st.chat_input("Why is my carbon usage high?")
        if prompt:
            st.session_state.ai_chat.append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                answer = _ai_assistant_reply(prompt)
            st.session_state.ai_chat.append({"role": "assistant", "content": answer})
            st.rerun()
        return

    tabs = st.tabs(
        [
            "Overview",
            "Carbon Tracking",
            "Code Metrics",
            "Comparison",
        ]
    )

    with tabs[0]:
        _render_header_cards()

        uploaded = st.file_uploader(
            "Upload a training script",
            type=UPLOAD_TYPES,
            help="Supported: .py, .java, .js, .ts, .cpp",
        )
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8", errors="replace")
            if st.button("Run analysis on uploaded file", type="primary"):
                _run_full_analysis(text, uploaded.name, duration_hours)

        analysis = st.session_state.analysis
        carbon_base = st.session_state.carbon_base
        if analysis is not None and carbon_base is not None:
            st.session_state.carbon_base = estimate_carbon_footprint(
                analysis, duration_hours=duration_hours
            )
        carbon_base = st.session_state.carbon_base

        if st.session_state.green_score is not None:
            st.progress(st.session_state.green_score / 100.0, text="Sustainability progress")
            st.plotly_chart(
                _single_gauge_figure("Green Score", float(st.session_state.green_score), "#3fb950"),
                use_container_width=True,
            )

        render_live_dashboard()
        _render_issues_and_suggestions()
        _render_reports_section()

    # --- Carbon tab ---
    with tabs[1]:
        st.subheader("Carbon tracking (from code analysis)")
        carbon_base = st.session_state.carbon_base
        if carbon_base is not None:
            b1, b2, b3 = st.columns(3)
            b1.metric("Electricity (est.)", f"{carbon_base.energy_kwh} kWh")
            b2.metric("CO₂ (est.)", f"{carbon_base.co2_kg} kg")
            b3.metric("Electricity cost (est.)", f"₹{carbon_base.cost_inr}")
            st.caption(f"{carbon_base.method} · {carbon_base.notes}")
            if st.session_state.green_score is not None:
                st.info(
                    f"Green Score: **{st.session_state.green_score}/100** "
                    f"({score_status_label(int(st.session_state.green_score))})"
                )
        else:
            st.info("Upload a file or load the sample to see carbon metrics.")

    # --- Code metrics tab ---
    with tabs[2]:
        st.subheader("Code metrics")
        metrics: CodeMetrics | None = st.session_state.get("code_metrics")
        if metrics is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total lines", metrics.total_lines)
            m2.metric("Code lines", metrics.code_lines)
            m3.metric("Comment lines", metrics.comment_lines)
            m4.metric("Blank lines", metrics.blank_lines)
            m5, m6, m7 = st.columns(3)
            m5.metric("Functions", metrics.functions)
            m6.metric("Classes", metrics.classes)
            m7.metric("Imports", metrics.imports)
            st.caption(f"Language: {metrics.language}")
        else:
            st.info("Run an analysis to see code metrics.")

    # --- History moved to sidebar "Analysis History" ---

    # --- Comparison tab ---
    with tabs[3]:
        st.subheader("Script comparison mode")
        c1, c2 = st.columns(2)
        with c1:
            up_a = st.file_uploader("File A (baseline)", type=UPLOAD_TYPES, key="cmp_a")
        with c2:
            up_b = st.file_uploader("File B (candidate)", type=UPLOAD_TYPES, key="cmp_b")

        if st.button("Compare scripts", type="primary"):
            if up_a is None or up_b is None:
                st.error("Upload both files to compare.")
            else:
                with st.spinner("Comparing…"):
                    ta = up_a.getvalue().decode("utf-8", errors="replace")
                    tb = up_b.getvalue().decode("utf-8", errors="replace")
                    aa = analyze_uploaded_source(ta, up_a.name)
                    ab = analyze_uploaded_source(tb, up_b.name)
                    ma = compute_code_metrics(ta, up_a.name)
                    mb = compute_code_metrics(tb, up_b.name)
                    ca = estimate_carbon_footprint(aa, duration_hours=duration_hours)
                    cb = estimate_carbon_footprint(ab, duration_hours=duration_hours)
                    sa = compute_green_score(ca, len(aa.issues), ma, 70.0)
                    sb = compute_green_score(cb, len(ab.issues), mb, 70.0)
                    st.session_state.compare_a = {
                        "name": up_a.name,
                        "lines": ma.total_lines,
                        "co2": ca.co2_kg,
                        "energy": ca.energy_kwh,
                        "issues": len(aa.issues),
                        "score": sa,
                    }
                    st.session_state.compare_b = {
                        "name": up_b.name,
                        "lines": mb.total_lines,
                        "co2": cb.co2_kg,
                        "energy": cb.energy_kwh,
                        "issues": len(ab.issues),
                        "score": sb,
                    }

        a = st.session_state.get("compare_a")
        b = st.session_state.get("compare_b")
        if a and b:
            cmp_df = pd.DataFrame(
                [
                    {"Metric": "Lines", "A": a["lines"], "B": b["lines"]},
                    {"Metric": "CO₂ (kg)", "A": a["co2"], "B": b["co2"]},
                    {"Metric": "Energy (kWh)", "A": a["energy"], "B": b["energy"]},
                    {"Metric": "Issues", "A": a["issues"], "B": b["issues"]},
                    {"Metric": "Green Score", "A": a["score"], "B": b["score"]},
                ]
            )
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            if a["co2"] > 0:
                carbon_delta = (a["co2"] - b["co2"]) / a["co2"] * 100.0
                if carbon_delta > 0:
                    st.success(f"Carbon reduced by **{carbon_delta:.1f}%** (B vs A)")
                elif carbon_delta < 0:
                    st.warning(f"Carbon increased by **{abs(carbon_delta):.1f}%** (B vs A)")
                else:
                    st.info("Carbon estimate unchanged between versions.")
            score_delta = b["score"] - a["score"]
            st.metric("Score improvement (B − A)", f"{score_delta:+d} points")


if __name__ == "__main__":
    main()
