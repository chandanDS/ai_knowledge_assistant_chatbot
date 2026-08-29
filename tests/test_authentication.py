import pytest

from auth.authentication import (
    authenticate_credentials,
    streamlit_login_enabled,
)


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("admin", "admin123"),
        (" chandan ", "1234"),
        ("testuser", "test123"),
    ],
)
def test_authenticate_credentials_accepts_valid_demo_users(username, password):
    assert authenticate_credentials(username, password) is True


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("admin", "wrong"),
        ("ADMIN", "admin123"),
        ("unknown", "password"),
        ("", ""),
        (None, None),
    ],
)
def test_authenticate_credentials_rejects_invalid_values(username, password):
    assert authenticate_credentials(username, password) is False


def test_streamlit_login_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("STREAMLIT_LOGIN_ENABLED", raising=False)

    assert streamlit_login_enabled() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_streamlit_login_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("STREAMLIT_LOGIN_ENABLED", value)

    assert streamlit_login_enabled() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_streamlit_login_can_be_enabled(monkeypatch, value):
    monkeypatch.setenv("STREAMLIT_LOGIN_ENABLED", value)

    assert streamlit_login_enabled() is True
