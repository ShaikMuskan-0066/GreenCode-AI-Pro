"""
forgot_password.py — Educational password reset flow (token-based, no email).
"""

from __future__ import annotations

import streamlit as st

from auth.auth_utils import create_reset_token, reset_password_with_token


def render_forgot_password() -> None:
    """
    Render forgot-password and reset-password steps in an expander.
    """
    with st.expander("Forgot Password?", expanded=False):
        st.caption("Educational reset flow — token is shown on screen (no email integration).")

        step = st.radio("Step", ["Request token", "Reset password"], horizontal=True, label_visibility="collapsed")

        if step == "Request token":
            with st.form("forgot_request"):
                identifier = st.text_input("Username or Email")
                req = st.form_submit_button("Generate reset token", use_container_width=True)
            if req:
                ok, message, token = create_reset_token(identifier)
                if ok:
                    st.success(message)
                    st.info("Copy the token above, then switch to **Reset password**.")
                    st.session_state.reset_token_hint = token
                else:
                    st.error(message)

        else:
            with st.form("forgot_reset"):
                token = st.text_input(
                    "Reset token",
                    value=st.session_state.get("reset_token_hint", ""),
                )
                new_password = st.text_input("New password", type="password")
                confirm = st.text_input("Confirm new password", type="password")
                reset = st.form_submit_button("Reset password", use_container_width=True)
            if reset:
                if new_password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = reset_password_with_token(token.strip(), new_password)
                    if ok:
                        st.success(message)
                        st.session_state.pop("reset_token_hint", None)
                    else:
                        st.error(message)
