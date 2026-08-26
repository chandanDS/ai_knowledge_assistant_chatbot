from io import BytesIO
from urllib.error import URLError

import healthcheck


class FakeResponse:
    def __init__(self, status=200, body=b"ok"):
        self.status = status
        self._body = BytesIO(body)

    def getcode(self):
        return self.status

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_build_health_url_uses_streamlit_port(monkeypatch):
    monkeypatch.delenv("CHATBOT_HEALTH_URL", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "8765")

    assert healthcheck.build_health_url() == "http://127.0.0.1:8765/_stcore/health"


def test_build_health_url_prefers_explicit_url(monkeypatch):
    monkeypatch.setenv("CHATBOT_HEALTH_URL", "https://example.test/ready/")

    assert healthcheck.build_health_url() == "https://example.test/ready"


def test_check_health_accepts_streamlit_ok(monkeypatch):
    monkeypatch.setattr(healthcheck, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    assert healthcheck.check_health("http://test", 1) == (True, "Chatbot is healthy.")


def test_check_health_rejects_non_200_response(monkeypatch):
    monkeypatch.setattr(
        healthcheck,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(status=503, body=b"unavailable"),
    )

    healthy, message = healthcheck.check_health("http://test", 1)
    assert healthy is False
    assert "503" in message


def test_check_health_rejects_unexpected_body(monkeypatch):
    monkeypatch.setattr(
        healthcheck, "urlopen", lambda *_args, **_kwargs: FakeResponse(body=b"starting")
    )

    assert healthcheck.check_health("http://test", 1)[0] is False


def test_check_health_handles_unreachable_server(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(healthcheck, "urlopen", unavailable)

    healthy, message = healthcheck.check_health("http://test", 1)
    assert healthy is False
    assert "unreachable" in message


def test_invalid_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CHATBOT_HEALTH_TIMEOUT_SECONDS", "not-a-number")

    assert healthcheck.health_timeout_seconds() == healthcheck.DEFAULT_TIMEOUT_SECONDS
