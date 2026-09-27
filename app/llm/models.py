"""Model metadata, and the pricing that goes with it.

Capabilities are data, not a naming convention. Whether a model can see images or
call tools is a property of that deployment of that model, and guessing from a
name is how a studio ends up promising a vision workflow to a text-only model —
or hiding tools from one that can use them. So a model is a row: what it can do,
what it costs, and where to read that from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelInfo:
    """Everything the app needs to know about one model.

    Prices are per million tokens, which is how every provider quotes them and how
    a human compares them.
    """

    id: str
    #: Filled in by the registry from the provider's configuration, so a model
    #: row does not have to repeat the provider it is listed under.
    provider: str = ""
    display_name: str = ""
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_images: bool = False
    supports_thinking: bool = False
    context_window: int = 128_000
    max_output_tokens: int = 4096
    input_price: float = 0.0
    output_price: float = 0.0
    #: What this deployment is called by the user. A gateway's own name for a
    #: hosted model, which is the only thing the user can select.
    description: str = ""
    #: Anything a provider needs that is not here: a gateway id, a region.
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """What the model selector shows."""
        name = self.display_name or self.id
        return f"{name}  ·  {self.provider}" if self.provider else name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "display_name": self.display_name,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "supports_streaming": self.supports_streaming,
            "supports_images": self.supports_images,
            "supports_thinking": self.supports_thinking,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "description": self.description,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelInfo:
        """Build from a stored or configured row.

        Unknown keys are kept in ``extra`` rather than dropped: a configuration
        file may carry a field a future version understands, and losing it would
        mean a round trip through the database changes the model.
        """
        known = {name for name in cls.__dataclass_fields__}
        extra: dict[str, Any] = dict(raw.get("extra") or {})
        for key, value in raw.items():
            if key not in known and key != "extra":
                extra[key] = value
        return cls(
            **{key: value for key, value in raw.items() if key in known and key != "extra"},
            extra=extra,
        )


def cost_of(
    model: ModelInfo,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_tokens: int = 0,
) -> float:
    """What a request cost, in dollars.

    Cached input is billed separately by several providers and is usually much
    cheaper; it is subtracted from the input count before the input price is
    applied, so a long system prompt that keeps being cached is not billed at
    full price on every turn.
    """
    billable_input = max(0, input_tokens - cached_tokens)
    return (
        billable_input / 1_000_000 * model.input_price
        + output_tokens / 1_000_000 * model.output_price
    )
