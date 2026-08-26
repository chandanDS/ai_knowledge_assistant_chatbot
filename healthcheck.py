import os
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_HOST = "127.0.0.1"
DEFAULT_STREAMLIT_PORT = "8501"
DEFAULT_TIMEOUT_SECONDS = 5


def build_health_url() -> str:
    explicit_url = os.getenv("CHATBOT_HEALTH_URL")

    if explicit_url:
        return explicit_url.rstrip("/")

    host = os.getenv(
        "CHATBOT_HEALTH_HOST",
        DEFAULT_HOST,
    )

    port = (
        os.getenv("PORT")
        or os.getenv("STREAMLIT_SERVER_PORT")
        or DEFAULT_STREAMLIT_PORT
    )

    return f"http://{host}:{port}/_stcore/health"


def health_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "CHATBOT_HEALTH_TIMEOUT_SECONDS",
        str(DEFAULT_TIMEOUT_SECONDS),
    )

    try:
        timeout = float(raw_timeout)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS

    if timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS

    return timeout


def check_health(
    url: str,
    timeout_seconds: float,
) -> tuple[bool, str]:
    try:
        with urlopen(
            url,
            timeout=timeout_seconds,
        ) as response:
            status_code = response.getcode()
            body = response.read().decode(
                "utf-8",
                errors="replace",
            ).strip().lower()

    except HTTPError as exc:
        return (
            False,
            f"Chatbot health check returned HTTP {exc.code}.",
        )

    except URLError as exc:
        return (
            False,
            f"Chatbot is unreachable: {exc.reason}",
        )

    except OSError as exc:
        return (
            False,
            f"Chatbot is unreachable: {exc}",
        )

    if status_code != 200:
        return (
            False,
            f"Chatbot health check returned HTTP {status_code}.",
        )

    if body != "ok":
        return (
            False,
            f"Unexpected health-check response: {body!r}.",
        )

    return True, "Chatbot is healthy."


def main() -> int:
    healthy, message = check_health(
        build_health_url(),
        health_timeout_seconds(),
    )

    print(message)

    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())