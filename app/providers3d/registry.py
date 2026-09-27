"""Which 3D providers exist, and which one the model should be offered.

The same shape as the LLM registry and for the same reason: the agent asks for a
3D capability, and which service satisfies it is a configuration decision. Swapping
Tripo for Meshy later is a new entry here, not a change to the agent.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.errors import ConfigurationError
from app.core.events import EventBus, EventType
from app.core.settings import ThreeDConfig
from app.providers3d.base import ThreeDProvider
from app.providers3d.mock import MockThreeDProvider
from app.providers3d.tripo import TripoProvider
from app.storage.secrets import SecretStore, key_name

logger = logging.getLogger(__name__)

#: provider name -> (class, environment variable holding the key)
PROVIDER_TYPES: dict[str, type[ThreeDProvider]] = {
    "tripo": TripoProvider,
    "mock": MockThreeDProvider,
}


class ThreeDRegistry:
    """Holds the configured 3D providers."""

    def __init__(self, bus: EventBus | None = None, config: ThreeDConfig | None = None) -> None:
        self._bus = bus
        self._config = config or ThreeDConfig()
        self._providers: dict[str, ThreeDProvider] = {}
        self._secrets: SecretStore | None = None

    def use_secrets(self, secrets: SecretStore) -> None:
        self._secrets = secrets
        self._providers.clear()

    def configure(self, config: ThreeDConfig) -> None:
        self._config = config
        self._providers.clear()

    @property
    def config(self) -> ThreeDConfig:
        return self._config

    def add(self, provider: ThreeDProvider) -> None:
        self._providers[provider.name] = provider

    def all(self) -> dict[str, ThreeDProvider]:
        return dict(self._providers)

    def any_enabled(self) -> bool:
        """Whether the model should be offered 3D generation at all.

        Needs a provider that is enabled *and* keyed: a tool that is always going
        to fail wastes a step and teaches the model that the tool is noise.
        """
        return any(p.enabled and p.is_configured() for p in self._providers.values())

    def provider(self, name: str = "") -> ThreeDProvider:
        name = name or self._config.provider
        if not name:
            raise ConfigurationError(
                "No 3D provider is selected",
                hint="Pick one in Settings → Tripo.",
            )
        built = self._providers.get(name)
        if built is not None:
            return built
        provider = self._build(name)
        self._providers[name] = provider
        return provider

    def _build(self, name: str) -> ThreeDProvider:
        provider_class = PROVIDER_TYPES.get(name)
        if provider_class is None:
            raise ConfigurationError(
                f"Unknown 3D provider '{name}'",
                hint=f"Known providers: {', '.join(sorted(PROVIDER_TYPES))}",
            )
        config = self._config
        if provider_class is MockThreeDProvider:
            return MockThreeDProvider()
        return provider_class(
            api_key=self._api_key(name),
            model=config.model,
            quality=config.quality,
            poll_interval=config.poll_interval,
            poll_timeout=config.poll_timeout,
        )

    def _api_key(self, name: str) -> str:
        if self._secrets is not None:
            stored = self._secrets.get(key_name(name))
            if stored:
                return stored
        return ""

    def describe(self) -> list[dict[str, Any]]:
        return [provider.capabilities() for provider in self._providers.values()]

    async def close_all(self) -> None:
        for provider in list(self._providers.values()):
            try:
                await provider.close()
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("closing %s failed", provider.name, exc_info=True)
        self._providers.clear()

    def _publish(self, **payload: Any) -> None:
        if self._bus is not None:
            self._bus.emit(EventType.INFO, **payload)
