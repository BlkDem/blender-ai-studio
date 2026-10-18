import httpx
import pytest

from app.core.errors import ProviderError
from app.llm.connection import connection_hint, is_loopback, normalize_local_url
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.parametrize("suffix", ["", "/", "/api", "/v1/"])
def test_ollama_url(suffix):
    assert normalize_local_url("http://127.0.0.1:11434" + suffix) == "http://127.0.0.1:11434/v1"


def test_other_gateway_unchanged():
    assert normalize_local_url("https://example.org/api") == "https://example.org/api"
    assert not is_loopback("https://localhost.example.org")
    assert not is_loopback("https://example.org/localhost")


def test_proxy_defaults():
    assert not OpenAICompatibleProvider(base_url="http://localhost:11434").trust_env
    assert OpenAICompatibleProvider().trust_env
    assert OpenAICompatibleProvider(base_url="http://localhost:11434", trust_env=True).trust_env


def test_hint_redacts_credentials_and_query():
    hint = connection_hint("http://user:secret@localhost:11434/v1?key=private")
    assert "secret" not in hint and "private" not in hint
    assert "WSL" in hint and "11434" in hint


@pytest.mark.asyncio
async def test_connection_failure_is_actionable():
    provider = OpenAICompatibleProvider(base_url="http://localhost:11434")

    async def fail(request):
        raise httpx.ConnectError("sensitive upstream text", request=request)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(ProviderError) as error:
            await provider._post("http://localhost:11434/v1/chat/completions", {})
        assert "ConnectError" in str(error.value)
        assert "sensitive" not in str(error.value)
        assert "WSL" in error.value.hint
    finally:
        await provider.close()
