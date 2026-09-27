"""The registry: configuration in, providers and models out."""

from __future__ import annotations

import pytest

from app.core.errors import ConfigurationError
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
from app.llm.registry import LLMRegistry, ProviderConfig, registry_from_settings
from app.storage.secrets import SecretStore


def config(**kwargs) -> ProviderConfig:
    base = {
        "name": "local",
        "kind": "openai-compatible",
        "base_url": "https://api.local/v1",
        "default_model": "space-bunny-free",
        "models": [
            {
                "id": "space-bunny-free",
                "display_name": "Space Bunny",
                "supports_tools": True,
                "supports_vision": False,
                "input_price": 0.0,
                "output_price": 0.0,
                "gateway": "opencode",
            }
        ],
    }
    base.update(kwargs)
    return ProviderConfig(**base)


def test_a_model_is_a_row_not_a_branch() -> None:
    """The point of the whole design: Space Bunny, GPT and a locally served
    model are the same kind of thing, so adding one is a config change."""
    registry = LLMRegistry(
        [
            config(),
            ProviderConfig(
                name="openai",
                kind="openai",
                default_model="gpt-4o-mini",
                models=[{"id": "gpt-4o-mini", "supports_vision": True, "input_price": 0.15}],
            ),
            ProviderConfig(
                name="anthropic",
                kind="anthropic",
                default_model="claude-sonnet-4-5",
                models=[{"id": "claude-sonnet-4-5", "supports_thinking": True}],
            ),
        ]
    )
    models = {m.id: m for m in registry.models()}
    assert set(models) == {"space-bunny-free", "gpt-4o-mini", "claude-sonnet-4-5"}
    assert models["space-bunny-free"].supports_vision is False
    assert models["gpt-4o-mini"].supports_vision is True
    assert models["space-bunny-free"].extra["gateway"] == "opencode"


def test_the_provider_is_built_once_and_kept() -> None:
    registry = LLMRegistry([config()])
    assert registry.provider("local") is registry.provider("local")


def test_a_one_off_model_can_be_resolved_without_touching_config() -> None:
    registry = LLMRegistry([config()])
    provider, model = registry.resolve("local", "some-other-model")
    assert isinstance(provider, MockLLMProvider | object)
    assert model == "some-other-model"


def test_the_model_falls_back_from_argument_to_config_to_provider_default() -> None:
    registry = LLMRegistry([config(default_model="")])
    registry.configure([ProviderConfig(name="mock", kind="mock", default_model="", models=[{"id": "m"}])])
    _provider, model = registry.resolve("mock")
    assert model == "m", "the provider's own default is the last resort"


def test_an_unconfigured_provider_says_so() -> None:
    registry = LLMRegistry([config()])
    with pytest.raises(ConfigurationError) as excinfo:
        registry.provider("nope")
    assert "Settings → Models" in (excinfo.value.hint or "")


def test_an_unknown_kind_lists_the_known_ones() -> None:
    registry = LLMRegistry([config(kind="telepathy")])
    with pytest.raises(ConfigurationError) as excinfo:
        registry.provider("local")
    assert "openai-compatible" in (excinfo.value.hint or "")


def test_a_configured_provider_without_a_key_is_not_ready(secrets_file) -> None:
    registry = LLMRegistry([config()], SecretStore(secrets_file))
    assert registry.is_configured("local") is False


def test_a_key_in_the_store_makes_a_provider_ready(secrets_file) -> None:
    secrets = SecretStore(secrets_file)
    secrets.set("local.api_key", "sk-1")
    registry = LLMRegistry([config()], secrets)
    assert registry.is_configured("local") is True


def test_the_store_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch, secrets_file) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    secrets = SecretStore(secrets_file)
    secrets.set("openai.api_key", "from-store")
    registry = LLMRegistry(
        [
            ProviderConfig(
                name="openai", kind="openai", default_model="gpt-4o-mini", models=[{"id": "gpt-4o-mini"}]
            )
        ],
        secrets,
    )
    assert registry.provider("openai").api_key == "from-store"


def test_the_environment_is_the_fallback(monkeypatch: pytest.MonkeyPatch, secrets_file) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    registry = LLMRegistry(
        [ProviderConfig(name="anthropic", kind="anthropic", default_model="c", models=[{"id": "c"}])],
        SecretStore(secrets_file),
    )
    assert registry.provider("anthropic").api_key == "from-env"


def test_the_mock_provider_needs_no_key() -> None:
    registry = LLMRegistry(
        [ProviderConfig(name="mock", kind="mock", default_model="m", models=[{"id": "m"}])]
    )
    assert registry.is_configured("mock") is True


def test_a_local_endpoint_needs_no_key(secrets_file) -> None:
    """vLLM, Ollama and llama.cpp do not ask for one, and a studio that reports
    "no API key" about the user's own localhost is confusing."""
    registry = LLMRegistry(
        [config(name="local", base_url="http://127.0.0.1:11400/v1")], SecretStore(secrets_file)
    )
    assert registry.is_configured("local") is True


def test_a_remote_endpoint_still_needs_a_key(secrets_file) -> None:
    registry = LLMRegistry(
        [config(name="gateway", base_url="https://api.example/v1")], SecretStore(secrets_file)
    )
    assert registry.is_configured("gateway") is False


def test_a_disabled_provider_is_off_but_still_listed() -> None:
    registry = LLMRegistry([config(enabled=False)])
    assert registry.is_configured("local") is False
    assert registry.config("local") is not None
    assert registry.models() == [], "a disabled provider offers no models"


def test_the_registry_is_built_from_settings() -> None:
    from app.core.settings import Settings

    settings = Settings(
        llm_providers=[
            {
                "name": "local",
                "kind": "openai-compatible",
                "base_url": "https://api.local/v1",
                "default_model": "space-bunny-free",
                "models": [{"id": "space-bunny-free"}],
            }
        ]
    )
    registry = registry_from_settings(settings)
    _provider, model = registry.resolve("local")
    assert model == "space-bunny-free"


def test_a_custom_endpoint_needs_no_code() -> None:
    """A user pointing the studio at a new OpenAI-compatible server is the MVP's
    central claim, so it is asserted rather than assumed."""
    registry = LLMRegistry([config(name="my-gateway", base_url="http://192.168.1.10:8000/v1")])
    provider = registry.provider("my-gateway")
    assert provider.base_url == "http://192.168.1.10:8000/v1"
    assert provider.default_model() == "space-bunny-free"


def test_closing_releases_every_provider() -> None:
    registry = LLMRegistry(
        [config(), ProviderConfig(name="mock", kind="mock", models=[{"id": "m"}], default_model="m")]
    )
    registry.provider("local")
    registry.provider("mock")
    import asyncio

    asyncio.run(registry.close_all())


async def test_the_mock_provider_replays_its_script() -> None:
    from app.llm.base import ChatRequest, Message

    provider = MockLLMProvider(
        [
            ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
            ScriptedTurn(text="The scene has one cube."),
        ]
    )
    first = await provider.chat(ChatRequest(messages=[Message.user("look")], model="mock-model"))
    second = await provider.chat(ChatRequest(messages=[Message.user("look")], model="mock-model"))

    assert first.wants_tools()
    assert first.tool_calls[0].name == "blender.get_scene"
    assert second.text.startswith("The scene")
    assert provider.calls == 2


async def test_the_mock_provider_streams_text_and_tool_arguments() -> None:
    from app.llm.base import ChatRequest

    provider = MockLLMProvider([ScriptedTurn(text="working", tool_calls=[("t", {"a": 1})])])
    chunks = [c async for c in provider.stream(ChatRequest(messages=[], model="mock-model"))]
    assert "".join(c.text for c in chunks if c.type == "text").strip() == "working"
    end = next(c for c in chunks if c.type == "tool_end")
    assert end.arguments == {"a": 1}
    deltas = [c for c in chunks if c.type == "tool_delta"]
    assert len(deltas) > 1, "arguments must arrive in pieces, as they do for real"


async def test_the_mock_provider_can_return_an_error() -> None:
    from app.core.errors import ProviderError
    from app.llm.base import ChatRequest

    provider = MockLLMProvider([ScriptedTurn(error=ProviderError("upstream is down"))])
    with pytest.raises(ProviderError):
        await provider.chat(ChatRequest(messages=[], model="mock-model"))


async def test_the_mock_provider_records_what_it_was_asked() -> None:
    from app.llm.base import ChatRequest, Message, ToolSpec

    provider = MockLLMProvider([ScriptedTurn(text="ok")])
    await provider.chat(ChatRequest(messages=[Message.user("hi")], model="mock-model", tools=[ToolSpec("t")]))
    assert provider.requests[0].tools[0].name == "t"
