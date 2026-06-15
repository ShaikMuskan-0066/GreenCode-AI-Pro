"""
signup.py — Streamlit user registration form.
"""

from __future__ import annotations

import streamlit as st

from auth.auth_utils import create_user, email_exists, is_strong_password, is_valid_email, login_session, username_exists


def render_signup_form() -> None:
    """
    Render the Sign Up form with validation and auto-login on success.
    """
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="Muskan Sharma")
        username = st.text_input("Username", placeholder="muskan")
        email = st.text_input("Email", placeholder="muskan@gmail.com")
        password = st.text_input("Password", type="password", placeholder="Min. 8 characters")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    if submitted:
        if not full_name.strip():
            st.error("Full name is required.")
            return
        if not username.strip():
            st.error("Username is required.")
            return
        if username_exists(username):
            st.error("Username already taken.")
            return
        if not is_valid_email(email):
            st.error("Please enter a valid email address.")
            return
        if email_exists(email):
            st.error("Email already registered.")
            return
        ok_pw, pw_msg = is_strong_password(password)
        if not ok_pw:
            st.error(pw_msg)
            return
        if password != confirm:
            st.error("Password and Confirm Password do not match.")
            return

        success, message, user = create_user(full_name, username, email, password)
        if success and user:
            login_session(user)
            st.success(message)
            st.balloons()
            st.rerun()
        st.error(message)
