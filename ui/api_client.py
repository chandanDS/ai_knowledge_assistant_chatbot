import os
from typing import Any

import httpx


class ChatbotApiError(RuntimeError):
    """Raised when the FastAPI chatbot cannot process a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class ChatbotApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "CHATBOT_API_BASE_URL",
            "http://localhost:8000",
        ).rstrip("/")

        self.timeout = float(
            os.getenv("CHATBOT_API_TIMEOUT_SECONDS", "120")
        )

        self.auth_mode = os.getenv(
            "CHATBOT_API_AUTH_MODE",
            "none",
        ).strip().lower()

        self.audience = os.getenv(
            "CHATBOT_API_AUDIENCE",
            self.base_url,
        ).rstrip("/")

    def _google_identity_token(self) -> str:
        try:
            from google.auth.exceptions import GoogleAuthError
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token
        except ImportError as exc:
            raise ChatbotApiError(
                "Cloud Run authentication requires the "
                "google-auth package."
            ) from exc

        try:
            return id_token.fetch_id_token(
                Request(),
                self.audience,
            )
        except GoogleAuthError as exc:
            raise ChatbotApiError(
                "Unable to obtain a Google identity token for "
                "the FastAPI Cloud Run service."
            ) from exc

    def _authentication_headers(self) -> dict[str, str]:
        if self.auth_mode == "none":
            return {}

        if self.auth_mode == "google":
            token = self._google_identity_token()
            return {
                "X-Serverless-Authorization": f"Bearer {token}",
            }

        raise ChatbotApiError(
            "Unsupported CHATBOT_API_AUTH_MODE. "
            "Expected 'none' or 'google'."
        )

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"

        supplied_headers = kwargs.pop("headers", {})
        headers = {
            **self._authentication_headers(),
            **supplied_headers,
        }

        try:
            response = httpx.request(
                method=method,
                url=url,
                timeout=self.timeout,
                headers=headers,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise ChatbotApiError(
                f"Unable to reach chatbot API: {exc}"
            ) from exc

        if response.status_code == 204:
            return None

        if response.is_error:
            try:
                body = response.json()
                message = (
                    body.get("error", {}).get("message")
                    or body.get("detail")
                    or response.text
                )
            except ValueError:
                message = response.text

            raise ChatbotApiError(
                f"Chatbot API returned HTTP "
                f"{response.status_code}: {message}",
                status_code=response.status_code,
            )

        return response.json()

    def check_readiness(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/readiness",
        )

    def create_conversation(
        self,
        user_id: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/conversations",
            json={"user_id": user_id},
        )

    def send_message(
        self,
        conversation_id: str,
        content: str,
        model: str = "Automatic",
        temperature: float = 0,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            (
                f"/api/v1/conversations/"
                f"{conversation_id}/messages"
            ),
            json={
                "content": content,
                "model": model,
                "temperature": temperature,
            },
        )

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        self._request(
            "DELETE",
            f"/api/v1/conversations/{conversation_id}",
        )
