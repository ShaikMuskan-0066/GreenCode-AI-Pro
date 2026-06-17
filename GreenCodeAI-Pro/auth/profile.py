"""
profile.py — User profile, account statistics, and admin dashboard cards.
"""

from __future__ import annotations

import streamlit as st

from auth.auth_utils import find_user_by_id, get_admin_stats, is_admin, update_user_profile
from utils import get_user_account_stats


def _profile_kpi(icon: str, title: str, value: str, subtitle: str = "") -> None:
    """Render a profile statistics KPI card."""
    st.markdown(
        f"""
        <div class="gc-kpi-card gc-fade-in">
          <div class="gc-kpi-icon">{icon}</div>
          <div class="gc-kpi-title">{title}</div>
          <div class="gc-kpi-value">{value}</div>
          <div class="gc-kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_page() -> None:
    """
    Show profile details, edit forms, account stats, and optional admin panel.
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("Please log in to view your profile.")
        return

    user = find_user_by_id(str(user_id))
    if user is None:
        st.error("User record not found.")
        return

    username = user.get("username", "user")
    initial = (username[0] if username else "U").upper()
    display_name = user.get("name") or username

    st.markdown(
        f"""
        <div class="gc-profile-hero gc-fade-in">
          <div style="display:flex;gap:0.9rem;align-items:center;">
            <div class="gc-avatar" style="width:52px;height:52px;font-size:1.3rem;">{initial}</div>
            <div>
              <h2>Welcome, {display_name}</h2>
              <p>@{username} · {user.get("role", "user")}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gc-section-title">Account Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="gc-glass-panel gc-fade-in">
              <div class="gc-kpi-title">Full Name</div>
              <div class="gc-kpi-value" style="font-size:1.2rem;">{user.get("name", "—")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="gc-glass-panel gc-fade-in">
              <div class="gc-kpi-title">Username</div>
              <div class="gc-kpi-value" style="font-size:1.2rem;">{user.get("username", "—")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="gc-glass-panel gc-fade-in">
              <div class="gc-kpi-title">Email</div>
              <div class="gc-kpi-value" style="font-size:1.2rem;">{user.get("email", "—")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="gc-glass-panel gc-fade-in">
              <div class="gc-kpi-title">Registration</div>
              <div class="gc-kpi-value" style="font-size:1.2rem;">{user.get("registration_date", "—")}</div>
              <div class="gc-kpi-subtitle">Last login: {user.get("last_login") or "Never"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gc-section-title">Your Sustainability Metrics</div>', unsafe_allow_html=True)
    stats = get_user_account_stats(str(user_id))
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _profile_kpi("📊", "Total Analyses", str(stats["analyses_completed"]), "Completed scans")
    with s2:
        _profile_kpi("🌿", "Average Green Score", f"{stats['average_score']:.1f}", "Across all analyses")
    with s3:
        _profile_kpi("♻️", "Carbon Saved", f"{stats['total_carbon_saved']:.2f} kg", "Estimated vs baseline")
    with s4:
        _profile_kpi("📄", "Reports Generated", str(stats["analyses_completed"]), "TXT and PDF exports")

    st.markdown('<div class="gc-section-title">Recent Activity</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="gc-glass-panel gc-fade-in">
          <div class="gc-kpi-title">Highest Green Score</div>
          <div class="gc-kpi-value">{stats["highest_score"]}</div>
          <div class="gc-kpi-subtitle">Total carbon tracked: {stats["total_carbon_tracked"]:.2f} kg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gc-section-title">Edit Profile</div>', unsafe_allow_html=True)

    with st.form("edit_name_form"):
        new_name = st.text_input("Change full name", value=user.get("name", ""))
        save_name = st.form_submit_button("Update name")
    if save_name:
        ok, msg = update_user_profile(str(user_id), name=new_name)
        if ok:
            st.session_state.user_name = new_name.strip()
            st.success(msg)
            st.rerun()
        st.error(msg)

    with st.form("edit_password_form"):
        new_pw = st.text_input("New password", type="password")
        confirm_pw = st.text_input("Confirm new password", type="password")
        save_pw = st.form_submit_button("Change password")
    if save_pw:
        if new_pw != confirm_pw:
            st.error("Passwords do not match.")
        else:
            ok, msg = update_user_profile(str(user_id), password=new_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if is_admin():
        st.divider()
        _render_admin_dashboard()


def _render_admin_dashboard() -> None:
    """
    Admin-only cards: all users, analyses, reports, active users.
    """
    st.markdown(
        """
        <div class="gc-page-header gc-fade-in">
          <h2>Admin Dashboard</h2>
          <p>Platform-wide usage and user activity overview.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stats = get_admin_stats()
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        _profile_kpi("👥", "Total Users", str(stats["total_users"]))
    with a2:
        _profile_kpi("📊", "Total Analyses", str(stats["total_analyses"]))
    with a3:
        _profile_kpi("📄", "Reports Generated", str(stats["total_reports"]))
    with a4:
        _profile_kpi("⚡", "Active Users (7d)", str(stats["active_users_7d"]))

    users_df_data = [
        {
            "Username": u.get("username"),
            "Name": u.get("name"),
            "Email": u.get("email"),
            "Role": u.get("role"),
            "Analyses": u.get("analyses_count", 0),
            "Last login": u.get("last_login") or "—",
        }
        for u in stats.get("users", [])
    ]
    if users_df_data:
        import pandas as pd

        st.dataframe(pd.DataFrame(users_df_data), use_container_width=True, hide_index=True)
