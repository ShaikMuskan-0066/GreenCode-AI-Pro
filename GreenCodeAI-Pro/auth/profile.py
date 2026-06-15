"""
profile.py — User profile, account statistics, and admin dashboard cards.
"""

from __future__ import annotations

import streamlit as st

from auth.auth_utils import find_user_by_id, get_admin_stats, is_admin, update_user_profile
from utils import get_user_account_stats


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

    st.subheader("User profile")
    st.markdown(f"### Welcome, **{user.get('username')}**")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Full name", user.get("name", "—"))
        st.metric("Username", user.get("username", "—"))
    with c2:
        st.metric("Email", user.get("email", "—"))
        st.metric("Registration", user.get("registration_date", "—"))
    st.metric("Last login", user.get("last_login") or "Never")

    st.divider()
    st.subheader("Account statistics")
    stats = get_user_account_stats(str(user_id))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Analyses completed", stats["analyses_completed"])
    s2.metric("Average green score", f"{stats['average_score']:.1f}")
    s3.metric("Highest score", stats["highest_score"])
    s4.metric("Total carbon tracked (kg)", f"{stats['total_carbon_tracked']:.2f}")
    st.caption(f"Estimated carbon savings vs baseline: **{stats['total_carbon_saved']:.2f} kg**")

    st.divider()
    st.subheader("Edit profile")

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
    st.subheader("Admin dashboard")
    stats = get_admin_stats()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total users", stats["total_users"])
    a2.metric("Total analyses", stats["total_analyses"])
    a3.metric("Reports generated", stats["total_reports"])
    a4.metric("Active users (7d)", stats["active_users_7d"])

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
