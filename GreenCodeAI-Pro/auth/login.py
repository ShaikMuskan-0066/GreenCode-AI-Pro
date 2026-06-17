"""
login.py — Streamlit login form with attempt limiting.
"""

from __future__ import annotations

import streamlit as st

from auth.auth_utils import authenticate_user, login_session


MAX_LOGIN_ATTEMPTS = 5


def render_login_form() -> None:
    """
    Render the login form inside the Login tab.

    Sets session state on successful authentication.
    """
    attempts = int(st.session_state.get("login_attempts", 0))
    if attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("Too many failed attempts. Please wait and try again later or use Sign Up.")
        return

    with st.form("login_form", clear_on_submit=False):
        st.markdown("#### Sign in to continue")
        username = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.checkbox("Remember me", key="remember_me")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if submitted:
        if not username.strip() or not password:
            st.error("Please enter username and password.")
            return
        ok, message, user = authenticate_user(username.strip(), password)
        if ok and user:
            login_session(user)
            st.success(f"Welcome, {user.get('username')}!")
            st.rerun()
        st.session_state.login_attempts = attempts + 1
        st.error(message)
