"""
auth_utils.py — User database, password hashing, sessions, and reset tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

SESSION_TIMEOUT_MINUTES = 30
SESSION_WARNING_MINUTES = 28
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def project_root() -> Path:
    """
    Return GreenCode AI Pro project root (parent of ``auth/``).

    Returns:
        Absolute path to project root.
    """
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """
    Ensure ``data/`` exists and return its path.

    Returns:
        Path to the data directory.
    """
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def users_db_path() -> Path:
    """
    Path to ``data/users.json``.

    Returns:
        Absolute path to the user database file.
    """
    return data_dir() / "users.json"


def reset_tokens_path() -> Path:
    """
    Path to ``data/reset_tokens.json``.

    Returns:
        Absolute path to password-reset token storage.
    """
    return data_dir() / "reset_tokens.json"


def hash_password(password: str) -> str:
    """
    Hash a password with PBKDF2-SHA256 and a random salt (stdlib only).

    Args:
        password: Plain-text password.

    Returns:
        Stored hash string ``salt$hexdigest``.
    """
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored PBKDF2 hash.

    Args:
        password: Plain-text password attempt.
        stored_hash: Value from ``password_hash`` field.

    Returns:
        True if the password matches.
    """
    try:
        salt, expected = stored_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
        ).hex()
        return secrets.compare_digest(digest, expected)
    except (ValueError, AttributeError):
        return False


def is_valid_email(email: str) -> bool:
    """
    Basic email format validation.

    Args:
        email: Email address string.

    Returns:
        True if format looks valid.
    """
    return bool(EMAIL_PATTERN.match(email.strip()))


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Check minimum password rules (8+ characters).

    Args:
        password: Candidate password.

    Returns:
        Tuple (ok, error_message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    return True, ""


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """
    Load JSON from disk with a safe default.

    Args:
        path: File path.
        default: Returned when file is missing or invalid.

    Returns:
        Parsed dict.
    """
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return default


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Write JSON to disk with UTF-8 encoding.

    Args:
        path: Destination file.
        data: Serializable dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_all_users() -> list[dict[str, Any]]:
    """
    Load all user records from ``data/users.json``.

    Returns:
        List of user dicts.
    """
    data = _load_json(users_db_path(), {"users": []})
    users = data.get("users", [])
    return users if isinstance(users, list) else []


def save_all_users(users: list[dict[str, Any]]) -> None:
    """
    Persist the full user list to ``data/users.json``.

    Args:
        users: List of user records.
    """
    _save_json(users_db_path(), {"users": users})


def find_user_by_username(username: str) -> dict[str, Any] | None:
    """
    Find a user by username (case-insensitive).

    Args:
        username: Username to search.

    Returns:
        User dict or None.
    """
    key = username.strip().lower()
    for user in load_all_users():
        if str(user.get("username", "")).lower() == key:
            return user
    return None


def find_user_by_email(email: str) -> dict[str, Any] | None:
    """
    Find a user by email (case-insensitive).

    Args:
        email: Email to search.

    Returns:
        User dict or None.
    """
    key = email.strip().lower()
    for user in load_all_users():
        if str(user.get("email", "")).lower() == key:
            return user
    return None


def find_user_by_id(user_id: str) -> dict[str, Any] | None:
    """
    Find a user by ``user_id``.

    Args:
        user_id: UUID string.

    Returns:
        User dict or None.
    """
    for user in load_all_users():
        if user.get("user_id") == user_id:
            return user
    return None


def username_exists(username: str) -> bool:
    """
    Return True if username is already registered.

    Args:
        username: Candidate username.

    Returns:
        True if taken.
    """
    return find_user_by_username(username) is not None


def email_exists(email: str) -> bool:
    """
    Return True if email is already registered.

    Args:
        email: Candidate email.

    Returns:
        True if taken.
    """
    return find_user_by_email(email) is not None


def create_user(
    full_name: str,
    username: str,
    email: str,
    password: str,
    role: str = "user",
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Register a new user with hashed password.

    Args:
        full_name: Display name.
        username: Unique username.
        email: Unique email.
        password: Plain password (will be hashed).
        role: ``user`` or ``admin``.

    Returns:
        Tuple (success, message, user_dict_or_none).
    """
    username = username.strip()
    email = email.strip()
    full_name = full_name.strip()

    if not full_name:
        return False, "Full name is required.", None
    if not username:
        return False, "Username is required.", None
    if not is_valid_email(email):
        return False, "Invalid email format.", None
    if username_exists(username):
        return False, "Username already exists.", None
    if email_exists(email):
        return False, "Email already registered.", None

    ok, msg = is_strong_password(password)
    if not ok:
        return False, msg, None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = {
        "user_id": str(uuid.uuid4()),
        "name": full_name,
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "role": role,
        "registration_date": now,
        "last_login": None,
        "analyses_count": 0,
    }
    users = load_all_users()
    users.append(user)
    save_all_users(users)
    return True, "Account created successfully.", user


def authenticate_user(username: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Validate login credentials and update last login.

    Args:
        username: Submitted username.
        password: Submitted password.

    Returns:
        Tuple (success, message, user_dict_or_none).
    """
    user = find_user_by_username(username)
    if user is None:
        return False, "Invalid username or password.", None
    if not verify_password(password, str(user.get("password_hash", ""))):
        return False, "Invalid username or password.", None

    users = load_all_users()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, u in enumerate(users):
        if u.get("user_id") == user.get("user_id"):
            users[idx]["last_login"] = now
            user = users[idx]
            break
    save_all_users(users)
    return True, "Login successful.", user


def update_user_profile(user_id: str, name: str | None = None, password: str | None = None) -> tuple[bool, str]:
    """
    Update a user's display name and/or password.

    Args:
        user_id: Target user UUID.
        name: New full name (optional).
        password: New plain password (optional).

    Returns:
        Tuple (success, message).
    """
    users = load_all_users()
    updated = False
    for idx, u in enumerate(users):
        if u.get("user_id") != user_id:
            continue
        if name is not None and name.strip():
            users[idx]["name"] = name.strip()
            updated = True
        if password is not None:
            ok, msg = is_strong_password(password)
            if not ok:
                return False, msg
            users[idx]["password_hash"] = hash_password(password)
            updated = True
        break
    if not updated:
        return False, "User not found or no changes provided."
    save_all_users(users)
    return True, "Profile updated successfully."


def increment_user_analyses_count(user_id: str) -> None:
    """
    Bump ``analyses_count`` after a successful analysis.

    Args:
        user_id: User UUID.
    """
    users = load_all_users()
    for idx, u in enumerate(users):
        if u.get("user_id") == user_id:
            users[idx]["analyses_count"] = int(u.get("analyses_count", 0)) + 1
            break
    save_all_users(users)


def ensure_default_admin() -> None:
    """
    Create default admin (admin / admin123) if no admin account exists.
    """
    users = load_all_users()
    if any(u.get("username") == "admin" for u in users):
        return
    create_user(
        full_name="Administrator",
        username="admin",
        email="admin@greencode.ai",
        password="admin123",
        role="admin",
    )


def load_reset_tokens() -> list[dict[str, Any]]:
    """
    Load password-reset tokens from disk.

    Returns:
        List of token dicts.
    """
    data = _load_json(reset_tokens_path(), {"tokens": []})
    tokens = data.get("tokens", [])
    return tokens if isinstance(tokens, list) else []


def save_reset_tokens(tokens: list[dict[str, Any]]) -> None:
    """
    Persist reset tokens to disk.

    Args:
        tokens: Token record list.
    """
    _save_json(reset_tokens_path(), {"tokens": tokens})


def create_reset_token(username_or_email: str) -> tuple[bool, str, str | None]:
    """
    Generate a temporary reset token for a username or email.

    Args:
        username_or_email: Identifier entered by the user.

    Returns:
        Tuple (success, message, token_or_none).
    """
    user = find_user_by_username(username_or_email) or find_user_by_email(username_or_email)
    if user is None:
        return False, "No account found with that username or email.", None

    token = secrets.token_urlsafe(24)
    expires = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    tokens = [t for t in load_reset_tokens() if t.get("username") != user.get("username")]
    tokens.append(
        {
            "token": token,
            "username": user.get("username"),
            "expires": expires,
        }
    )
    save_reset_tokens(tokens)
    return True, f"Reset token generated (valid 1 hour). Token: {token}", token


def reset_password_with_token(token: str, new_password: str) -> tuple[bool, str]:
    """
    Reset password using a valid token.

    Args:
        token: Reset token string.
        new_password: New plain password.

    Returns:
        Tuple (success, message).
    """
    ok, msg = is_strong_password(new_password)
    if not ok:
        return False, msg

    tokens = load_reset_tokens()
    now = datetime.now()
    match = None
    for t in tokens:
        if t.get("token") != token:
            continue
        try:
            expires = datetime.strptime(str(t.get("expires")), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if now <= expires:
            match = t
            break

    if match is None:
        return False, "Invalid or expired reset token."

    user = find_user_by_username(str(match.get("username", "")))
    if user is None:
        return False, "User not found."

    success, message = update_user_profile(str(user["user_id"]), password=new_password)
    if success:
        remaining = [t for t in tokens if t.get("token") != token]
        save_reset_tokens(remaining)
    return success, message


def is_logged_in() -> bool:
    """
    Check whether the session is authenticated.

    Returns:
        True if ``logged_in`` is set in session state.
    """
    return bool(st.session_state.get("logged_in"))


def is_admin() -> bool:
    """
    Return True if the logged-in user has admin role.

    Returns:
        True for admin accounts.
    """
    return st.session_state.get("user_role") == "admin"


def login_session(user: dict[str, Any]) -> None:
    """
    Populate session state after successful authentication.

    Args:
        user: User record dict.
    """
    st.session_state.logged_in = True
    st.session_state.user_id = user.get("user_id")
    st.session_state.username = user.get("username")
    st.session_state.user_name = user.get("name")
    st.session_state.user_email = user.get("email")
    st.session_state.user_role = user.get("role", "user")
    st.session_state.login_attempts = 0
    touch_activity()


def logout() -> None:
    """
    Clear authentication and session activity from session state.
    """
    for key in (
        "logged_in",
        "user_id",
        "username",
        "user_name",
        "user_email",
        "user_role",
        "last_activity_time",
        "session_warned",
    ):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.logged_in = False


def touch_activity() -> None:
    """
    Record the current time as last user activity (for session timeout).
    """
    st.session_state.last_activity_time = datetime.now().timestamp()
    st.session_state.session_warned = False


def check_session_expiry() -> tuple[bool, bool]:
    """
    Evaluate inactivity timeout (30 minutes) and warning (28 minutes).

    Returns:
        Tuple (expired, show_warning).
    """
    if not is_logged_in():
        return False, False
    last = st.session_state.get("last_activity_time")
    if last is None:
        touch_activity()
        return False, False

    elapsed_min = (datetime.now().timestamp() - float(last)) / 60.0
    if elapsed_min >= SESSION_TIMEOUT_MINUTES:
        return True, False
    if elapsed_min >= SESSION_WARNING_MINUTES:
        return False, True
    return False, False


def get_admin_stats() -> dict[str, Any]:
    """
    Aggregate admin dashboard statistics.

    Returns:
        Dict with users count, analyses, reports, active users.
    """
    from utils import load_all_user_reports  # local import avoids circular at module load

    users = load_all_users()
    reports = load_all_user_reports()
    active = sum(
        1
        for u in users
        if u.get("last_login")
        and _parse_dt(str(u["last_login"])) > datetime.now() - timedelta(days=7)
    )
    return {
        "total_users": len(users),
        "total_analyses": len(reports),
        "total_reports": len(reports),
        "active_users_7d": active,
        "users": users,
    }


def _parse_dt(value: str) -> datetime:
    """
    Parse a stored datetime string.

    Args:
        value: ``YYYY-MM-DD HH:MM:SS`` string.

    Returns:
        Parsed datetime or epoch on failure.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.min
