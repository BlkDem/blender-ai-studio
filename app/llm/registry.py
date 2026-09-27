"""Providers and models, from configuration.

The registry is the only place that knows which providers exist and how to build
one. Everything else — the agent, the GUI, the benchmark — asks for a
``provider + model`` pair and gets an :class:`~app.llm.base.LLMProvider`.

A model is a row, not a branch. "Space Bunny", "GPT", "Claude", "JEV" and a
locally served model are all the same thing: an entry in a provider's model list
with its own capabilities and prices. There is no ``if model == ...`` anywhere in
this project, and that is the property that makes swapping a model a settings
change rather than a code change.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import ConfigurationError
from app.llm.base import LLMProvider
from app.llm.models import ModelInfo
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.mock import MockLLMProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider, OpenAIProvider
from app.storage.secrets import SecretStore, key_name

logger = logging.getLogger(__name__)

#: provider name -> (class, default base url, env var holding the key)
PROVIDER_TYPES: dict[str, tuple[type[LLMProvider], str, str]] = {
    "openai": (OpenAIProvider, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": (AnthropicProvider, "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "gemini": (GeminiProvider, "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
    "openai-compatible": (OpenAICompatibleProvider, "", ""),
    "mock": (MockLLMProvider, "mock://", ""),
}


@dataclass(slots=True)
class ProviderConfig:
    """One configured provider.

    ``models`` is a list of :class:`ModelInfo` dicts so a user can add a model
    this project has never heard of from a config file or the GUI.
    """

    name: str
    kind: str = "openai-compatible"
    enabled: bool = True
    base_url: str = ""
    default_model: str = ""
    models: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "enabled": self.enabled,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "models": self.models,
            "options": self.options,
        }


class LLMRegistry:
    """Builds providers, remembers them, and knows every model on offer."""

    def __init__(self, configs: Iterable[ProviderConfig] = (), secrets: SecretStore | None = None) -> None:
        self._configs: dict[str, ProviderConfig] = {config.name: config for config in configs}
        self._secrets = secrets
        self._providers: dict[str, LLMProvider] = {}

    # --- configuration -----------------------------------------------------

    def configure(self, configs: Iterable[ProviderConfig]) -> None:
        for config in configs:
            self._configs[config.name] = config
        self._providers.clear()

    def add(self, config: ProviderConfig) -> None:
        self.configure([config])

    def remove(self, name: str) -> None:
        self._configs.pop(name, None)
        self._providers.pop(name, None)

    def configs(self) -> list[ProviderConfig]:
        return list(self._configs.values())

    def config(self, name: str) -> ProviderConfig | None:
        return self._configs.get(name)

    def is_configured(self, name: str) -> bool:
        """Whether this provider could work: configured, enabled, and keyed.

        Used by ``--check`` and by the GUI's status bar. Deliberately does not
        make a network call: a provider that is configured correctly but is down
        is configured.
        """
        config = self._configs.get(name)
        if config is None or not config.enabled:
            return False
        if config.kind == "mock":
            return True
        if self._api_key(name):
            return True
        # A server on this machine -- vLLM, Ollama, llama.cpp -- usually needs no
        # key, and telling a user "no API key" about their own localhost is a
        # confusing way to say "ready".
        return _is_local_url(config.base_url)

    # --- providers ---------------------------------------------------------

    def provider(self, name: str) -> LLMProvider:
        """The provider, built once and kept."""
        if name in self._providers:
            return self._providers[name]
        config = self._configs.get(name)
        if config is None:
            raise ConfigurationError(
                f"No provider named '{name}' is configured",
                hint="Add it in Settings → Models.",
                known=sorted(self._configs),
            )
        provider = self._build(config)
        self._providers[name] = provider
        return provider

    def _build(self, config: ProviderConfig) -> LLMProvider:
        kind = config.kind
        entry = PROVIDER_TYPES.get(kind)
        if entry is None:
            raise ConfigurationError(
                f"Unknown provider kind '{kind}'",
                hint=f"Use one of: {', '.join(sorted(PROVIDER_TYPES))}",
            )
        provider_class, default_base, key_env = entry
        catalog = [
            model if isinstance(model, ModelInfo) else ModelInfo.from_dict(model) for model in config.models
        ]
        if catalog:
            for model in catalog:
                model.provider = config.name

        base_url = config.base_url or default_base
        options = dict(config.options)
        # One rule for every kind: an explicit default, then an option, then the
        # first model in the catalog. Without this, a provider whose catalog was
        # edited would keep answering with a model that is no longer listed.
        selected = config.default_model or options.get("model") or (catalog[0].id if catalog else "")
        if selected:
            options["model"] = selected
        if kind == "openai":
            return OpenAIProvider(
                api_key=self._api_key(config.name) or os.environ.get(key_env, ""),
                base_url=base_url,
                model=options.get("model", ""),
                models=catalog or None,
                **options.get("kwargs", {}),
            )
        if kind == "anthropic":
            return AnthropicProvider(
                api_key=self._api_key(config.name) or os.environ.get(key_env, ""),
                model=options.get("model", ""),
                models=catalog or None,
            )
        if kind == "gemini":
            return GeminiProvider(
                api_key=self._api_key(config.name) or os.environ.get(key_env, ""),
                model=options.get("model", ""),
                models=catalog or None,
            )
        if kind == "mock":
            return MockLLMProvider(model=options.get("model", "mock-model"), models=catalog or None)

        return OpenAICompatibleProvider(
            api_key=self._api_key(config.name),
            base_url=base_url,
            model=options.get("model", ""),
            models=catalog or None,
            **options.get("kwargs", {}),
        )

    def set_provider(self, name: str, provider: LLMProvider) -> None:
        """Use a specific provider instance for a name.

        For a test, and for a user who wants to force one model without editing
        configuration. The instance is kept until the next
        :meth:`configure`, so a benchmark can pin every model it compares.
        """
        self._providers[name] = provider

    def _api_key(self, name: str) -> str:
        """Secrets first, environment second.

        A key set in the GUI wins, because that is the one the user just typed; the
        environment is the fallback for a headless run.
        """
        config = self._configs.get(name)
        kind = config.kind if config else ""
        env_var = PROVIDER_TYPES.get(kind, ("", "", ""))[2]
        if self._secrets is not None:
            stored = self._secrets.get(key_name(name))
            if stored:
                return stored
        if env_var:
            return os.environ.get(env_var, "")
        return os.environ.get(f"{name.upper().replace('-', '_')}_API_KEY", "")

    # --- models ------------------------------------------------------------

    def models(self, provider: str | None = None) -> list[ModelInfo]:
        """Every model on offer, from configuration only. No network."""
        configs = [self._configs[provider]] if provider else list(self._configs.values())
        catalog: list[ModelInfo] = []
        for config in configs:
            if not config.enabled:
                continue
            for entry in config.models:
                model = entry if isinstance(entry, ModelInfo) else ModelInfo.from_dict(entry)
                model.provider = config.name
                catalog.append(model)
        return catalog

    def find(self, provider: str, model_id: str) -> ModelInfo | None:
        return next(
            (m for m in self.models(provider) if m.id == model_id),
            None,
        )

    def resolve(self, provider: str, model_id: str = "") -> tuple[LLMProvider, str]:
        """A provider and the model to use with it.

        The model comes from the caller, then the provider's configuration, then
        the provider's own default — in that order, so a one-off model can be
        passed to the benchmark without editing anything.
        """
        instance = self.provider(provider)
        model = model_id or self._configs[provider].default_model or instance.default_model()
        if not model:
            raise ConfigurationError(
                f"No model selected for '{provider}'",
                hint="Pick a model in the toolbar, or set a default in Settings → Models.",
            )
        return instance, model

    async def list_models_live(self, provider: str) -> list[ModelInfo]:
        """Ask the provider what it serves. Used by the Models panel's refresh."""
        return await self.provider(provider).list_models()

    async def close_all(self) -> None:
        for provider in list(self._providers.values()):
            try:
                await provider.close()
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("closing %s failed", provider.name, exc_info=True)
        self._providers.clear()


def _is_local_url(url: str) -> bool:
    """Whether this base URL points at this machine."""
    if not url:
        return False
    lowered = url.lower()
    return any(
        host in lowered for host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "host.docker.internal")
    )


def registry_from_settings(settings: Any, secrets: SecretStore | None = None) -> LLMRegistry:
    """Build a registry from :class:`app.core.settings.Settings`."""
    configs: list[ProviderConfig] = []
    for entry in settings.llm_providers or []:
        if isinstance(entry, ProviderConfig):
            configs.append(entry)
        else:
            configs.append(
                ProviderConfig(
                    name=str(entry.get("name", "")),
                    kind=str(entry.get("kind", "openai-compatible")),
                    enabled=bool(entry.get("enabled", True)),
                    base_url=str(entry.get("base_url", "")),
                    default_model=str(entry.get("default_model", "")),
                    models=list(entry.get("models") or []),
                    options=dict(entry.get("options") or {}),
                )
            )
    return LLMRegistry(configs, secrets)
