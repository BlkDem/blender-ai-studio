"""Local OpenAI-compatible endpoint handling and safe transport diagnostics."""

from urllib.parse import urlsplit, urlunsplit


def is_loopback(url: str) -> bool:
    return urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}


def normalize_local_url(url: str) -> str:
    url = url.strip().rstrip("/")
    parsed = urlsplit(url)
    # Only Ollama's conventional ports: arbitrary gateways may use a root API.
    if is_loopback(url) and parsed.port in {11434, 11400} and parsed.path in {"", "/api", "/v1"}:
        return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))
    return url


def connection_hint(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or "configured host"
    endpoint = f"{parsed.scheme}://{host}"
    if parsed.port:
        endpoint += f":{parsed.port}"
    endpoint += parsed.path
    hint = f"Cannot reach {endpoint}. Check that the server is running and the host/port are correct."
    if is_loopback(url):
        hint += (
            " For Ollama, check /api/tags and use an OpenAI base URL ending in /v1"
            " (normally port 11434). Windows and WSL have different localhost networks;"
            " a Windows-only listener may not be reachable from WSL."
        )
    return hint
