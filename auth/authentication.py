"""Small authentication helpers for the learning application.

The credentials are intentionally hardcoded for this local demo. A real
application should use password hashes and an external identity provider.
"""

import hmac


USERS = {
    "chandan": "1234",
    "admin": "admin123",
    "testuser": "test123",
}


def authenticate_credentials(username: str, password: str) -> bool:
    """Return True only when the supplied demo credentials match."""
    normalized_username = (username or "").strip()
    expected_password = USERS.get(normalized_username)
    if expected_password is None:
        return False
    return hmac.compare_digest(expected_password, password or "")
