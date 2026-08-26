import pytest

from auth.authentication import authenticate_credentials


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
