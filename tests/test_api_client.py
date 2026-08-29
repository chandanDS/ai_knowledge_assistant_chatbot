import httpx
import pytest

from ui.api_client import ChatbotApiClient, ChatbotApiError


def test_local_mode_sends_no_cloud_run_auth_header(
    monkeypatch,
):
    monkeypatch.setenv("CHATBOT_API_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("CHATBOT_API_AUTH_MODE", "none")
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={"status": "ready"},
            request=httpx.Request("GET", kwargs["url"]),
        )

    monkeypatch.setattr(httpx, "request", fake_request)

    result = ChatbotApiClient().check_readiness()

    assert result == {"status": "ready"}
    assert captured["headers"] == {}


def test_google_mode_sends_identity_token_in_serverless_header(
    monkeypatch,
):
    monkeypatch.setenv(
        "CHATBOT_API_BASE_URL",
        "https://chatbot-api-example.run.app/",
    )
    monkeypatch.setenv("CHATBOT_API_AUTH_MODE", "google")
    captured = {}

    monkeypatch.setattr(
        ChatbotApiClient,
        "_google_identity_token",
        lambda self: "signed-id-token",
    )

    def fake_request(**kwargs):
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={"status": "ready"},
            request=httpx.Request("GET", kwargs["url"]),
        )

    monkeypatch.setattr(httpx, "request", fake_request)

    client = ChatbotApiClient()
    client.check_readiness()

    assert client.audience == "https://chatbot-api-example.run.app"
    assert captured["headers"] == {
        "X-Serverless-Authorization": "Bearer signed-id-token"
    }


def test_unknown_authentication_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("CHATBOT_API_AUTH_MODE", "password")

    with pytest.raises(ChatbotApiError, match="Unsupported"):
        ChatbotApiClient().check_readiness()


def test_http_error_preserves_status_code(monkeypatch):
    monkeypatch.setenv("CHATBOT_API_AUTH_MODE", "none")

    def fake_request(**kwargs):
        return httpx.Response(
            403,
            json={"detail": "Forbidden"},
            request=httpx.Request("GET", kwargs["url"]),
        )

    monkeypatch.setattr(httpx, "request", fake_request)

    with pytest.raises(ChatbotApiError) as captured:
        ChatbotApiClient().check_readiness()

    assert captured.value.status_code == 403
