"""Errors with stable codes.

The GUI shows these to a person, so every one carries a ``code`` that survives a
round trip through a log line or the database, and a ``hint`` when there is a
concrete next step to suggest. Provider and transport failures are translated
into these at the edge: nothing above the provider layer should have to know what
an httpx exception looks like.
"""

from __future__ import annotations

from typing import Any


class StudioError(Exception):
    """Base class. ``code`` is what the UI and the logs key off."""

    code = "STUDIO_ERROR"

    def __init__(self, message: str, *, hint: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload

    def user_text(self) -> str:
        """What to put in front of a person: the problem, and what to do about it."""
        return f"{self.message}\n{self.hint}" if self.hint else self.message


class ConfigurationError(StudioError):
    code = "CONFIGURATION"


class ProviderError(StudioError):
    """An LLM or 3D provider failed. ``provider`` says which."""

    code = "PROVIDER_ERROR"

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["provider"] = self.provider
        payload["retryable"] = self.retryable
        return payload


class AuthenticationError(ProviderError):
    code = "PROVIDER_AUTH"

    def __init__(self, provider: str, message: str = "The provider rejected the API key") -> None:
        super().__init__(message, provider=provider, hint="Check the API key in Settings → Models.")


class RateLimitError(ProviderError):
    code = "PROVIDER_RATE_LIMIT"

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(
            f"{provider} is rate limiting this client",
            provider=provider,
            retryable=True,
            retry_after=retry_after,
        )
        self.retry_after = retry_after


class MCPError(StudioError):
    """The MCP server could not be reached, or refused something."""

    code = "MCP_ERROR"


class MCPToolError(MCPError):
    """A tool ran and failed. The server's own code is kept in ``tool_code``."""

    code = "MCP_TOOL_ERROR"

    def __init__(self, tool: str, message: str, *, tool_code: str = "", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.tool = tool
        self.tool_code = tool_code

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["tool"] = self.tool
        if self.tool_code:
            payload["tool_code"] = self.tool_code
        return payload


class AgentError(StudioError):
    code = "AGENT_ERROR"


class BudgetExceeded(AgentError):
    code = "BUDGET_EXCEEDED"

    def __init__(self, limit: str, value: float, cap: float) -> None:
        super().__init__(
            f"Agent stopped: {limit} limit reached ({value:g} of {cap:g}).",
            hint="Raise the limit in Settings → Agent, or narrow the request.",
            limit=limit,
            value=value,
            cap=cap,
        )


class Cancelled(StudioError):
    code = "CANCELLED"

    def __init__(self, reason: str = "Cancelled by the user") -> None:
        super().__init__(reason)


class ThreeDError(StudioError):
    code = "THREED_ERROR"


class StorageError(StudioError):
    code = "STORAGE_ERROR"


class NotImplementedCapability(ProviderError):
    """A provider is registered but does not do this.

    Explicit on purpose: the alternative is a stub that returns something
    plausible, and a plausible-looking fake result is worse than a refusal when
    the result was going to be imported into a scene.
    """

    code = "NOT_IMPLEMENTED"

    def __init__(self, provider: str, capability: str) -> None:
        super().__init__(
            f"{provider} does not implement {capability}",
            provider=provider,
            hint="Pick a different provider, or enable one that supports it.",
        )
