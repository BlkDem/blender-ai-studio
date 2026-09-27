"""The conversation format every provider speaks.

Three APIs disagree about almost everything: roles, where a system prompt lives,
how a tool result is addressed, whether a tool call carries an id. Rather than
let that leak upward, this module defines one format and each provider translates
into and out of it. The agent above this line never sees a vendor shape.

The format is close to OpenAI's, because that is the shape with the fewest
special cases, and it is not a thin wrapper around it: the parts that matter are
the ones where providers genuinely differ, and those are explicit fields rather
than conventions.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.core.errors import ProviderError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.llm.models import ModelInfo


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """One tool the model wants to run.

    ``id`` matters: Anthropic, Gemini and OpenAI all use it to match a result to
    the call that asked for it, and getting it wrong produces a conversation
    where the model is arguing with a result it never requested.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Providers stream arguments as partial JSON. Kept for the GUI; providers
    #: assemble it themselves before yielding the completed call.
    partial: str = ""

    @classmethod
    def new(cls, name: str, arguments: dict[str, Any]) -> ToolCall:
        return cls(id=f"call_{uuid.uuid4().hex[:16]}", name=name, arguments=arguments)

    def arguments_text(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass(slots=True)
class ContentPart:
    """One piece of a message: text, an image, or a provider-specific extra.

    ``extra`` carries what a provider needs sent back verbatim — a Gemini thought
    signature, for instance. It is kept as received and echoed unchanged, because
    the alternative is reimplementing a provider's thinking bookkeeping.
    """

    type: str  # "text" | "image"
    text: str = ""
    # base64 payload and mime type, for vision.
    data: str | None = None
    mime_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text_part(cls, text: str) -> ContentPart:
        return cls(type="text", text=text)

    @classmethod
    def image_part(cls, data: str, mime_type: str) -> ContentPart:
        return cls(type="image", data=data, mime_type=mime_type)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    parts: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: For a tool result: the call being answered.
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str = ""

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=text)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        return cls(role=Role.ASSISTANT, content=text, tool_calls=list(tool_calls))

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=call.id, name=call.name)

    def text(self) -> str:
        return self.content or "".join(part.text for part in self.parts if part.type == "text")

    def to_dict(self) -> dict[str, Any]:
        """For storage and for a provider that wants plain dictionaries."""
        payload: dict[str, Any] = {"role": str(self.role), "content": self.text()}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments} for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Message:
        return cls(
            role=Role(raw.get("role", "user")),
            content=raw.get("content", "") or "",
            tool_calls=[
                ToolCall(id=call["id"], name=call["name"], arguments=call.get("arguments") or {})
                for call in raw.get("tool_calls") or []
            ],
            tool_call_id=raw.get("tool_call_id"),
            reasoning=raw.get("reasoning", "") or "",
        )


@dataclass(slots=True)
class ToolSpec:
    """A tool as the model should see it.

    The JSON Schema is passed through untouched. Provider-specific reshaping is
    the provider's job, and the studio never invents constraints the server did
    not declare.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    @classmethod
    def from_json_schema(cls, name: str, schema: dict[str, Any], description: str = "") -> ToolSpec:
        return cls(name=name, description=description, parameters=schema or {"type": "object"})

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def to_gemini(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _clean_schema_for_gemini(self.parameters),
        }


def _clean_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop the JSON Schema keywords Gemini rejects.

    It accepts a subset of OpenAPI: an empty ``properties`` with no ``required``,
    or a ``$schema``/``additionalProperties`` left in by another server, comes
    back as a 400 with a message about the offending field. Filtering here keeps
    a third-party MCP server's schema from breaking the whole request.
    """
    unsupported = {
        "$schema",
        "additionalProperties",
        "$id",
        "definitions",
        "$defs",
        "const",
        "examples",
        "default",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    cleaned = {key: value for key, value in schema.items() if key not in unsupported}
    if "type" not in cleaned:
        cleaned["type"] = "object"
    if cleaned.get("type") == "object" and not cleaned.get("properties"):
        cleaned.pop("required", None)
        cleaned["properties"] = {}
    for key in ("properties", "$defs", "definitions"):
        if isinstance(cleaned.get(key), dict):
            cleaned[key] = {
                name: _clean_schema_for_gemini(value) if isinstance(value, dict) else value
                for name, value in cleaned[key].items()
            }
    if isinstance(cleaned.get("items"), dict):
        cleaned["items"] = _clean_schema_for_gemini(cleaned["items"])
    return cleaned


@dataclass(slots=True)
class ChatRequest:
    """One turn. The agent builds these; providers consume them.

    The message list is copied on construction. The agent appends to its own list
    as the loop goes, and a provider that keeps the request — for a retry, for a
    log line, for a stream it reads later — would otherwise watch that list grow
    underneath it and see a conversation that never happened.
    """

    messages: list[Message]
    model: str
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    #: "auto" lets the model decide; "any" forces a tool call; "none" forbids one.
    tool_choice: str = "auto"
    stop: list[str] = field(default_factory=list)
    timeout: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = list(self.messages)
        self.tools = list(self.tools)


@dataclass(slots=True)
class Usage:
    """What one request cost, in the provider's own units."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    #: Kept apart from cost because 3D credits are not dollars and must never be
    #: added to a token bill.
    credits: float = 0.0

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class ToolResult:
    """What came back from running a tool, on its way back to the model."""

    call: ToolCall
    content: str
    is_error: bool = False
    error_code: str = ""

    def message(self) -> Message:
        return Message.tool_result(self.call, self.content)


@dataclass(slots=True)
class ChatResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    model: str = ""
    reasoning: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class StreamChunk:
    """One increment of a streaming response.

    Text and tool arguments arrive separately, and in Anthropic's case a tool
    argument arrives as many partial JSON fragments, so both are explicit here
    and the provider assembles them.
    """

    type: str  # "start" | "text" | "tool_start" | "tool_delta" | "tool_end" | "usage" | "end" | "reasoning"
    text: str = ""
    call_id: str = ""
    name: str = ""
    partial: str = ""
    arguments: dict[str, Any] | None = None
    usage: Usage | None = None
    finish_reason: str = ""
    reasoning: str = ""


class LLMProvider(ABC):
    """What the agent is allowed to assume about a language model.

    Deliberately small. Everything vendor-shaped stays inside the subclass, and
    anything a second provider cannot honour is either optional here or reported
    through :class:`~app.core.errors.NotImplementedCapability` rather than
    emulated.
    """

    #: Short, stable identifier used in configuration and in the database.
    name: str = ""
    #: True when the provider can be used at all right now.
    available: bool = True

    def __init__(self, api_key: str = "", base_url: str = "", **options: Any) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.options = options
        self._client: Any = None

    @abstractmethod
    def default_model(self) -> str:
        """The model to use when the user has not chosen one."""

    @abstractmethod
    def models(self) -> list[str]:
        """Model ids this provider can serve, from configuration or a live call."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """One non-streaming turn."""

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """One streaming turn, chunk by chunk."""

    async def list_models(self) -> list[ModelInfo]:  # noqa: F821 - resolved at runtime
        """Metadata for :meth:`models`, filled in by the registry.

        Kept here as a default so a provider that only knows its own catalog does
        not have to import the model layer.
        """
        from app.llm.models import ModelInfo

        return [ModelInfo(id=model, provider=self.name) for model in self.models()]

    async def close(self) -> None:
        """Release sockets. Called when the app exits or the model changes."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    def require_key(self) -> str:
        if not self.api_key:
            from app.core.errors import ConfigurationError

            raise ConfigurationError(
                f"No API key for {self.name or type(self).__name__}",
                hint=f"Add it in Settings → Models, or set {self.name.upper()}_API_KEY.",
            )
        return self.api_key


def ensure_not_empty(response: ChatResponse, provider: str) -> ChatResponse:
    """A response with neither text nor tool calls is a provider bug, not an answer.

    Reported rather than returned, because an empty assistant turn is what an
    agent loop turns into an infinite "the model said nothing, ask again".
    """
    if not response.text and not response.tool_calls:
        raise ProviderError(
            f"{provider} returned an empty response",
            provider=provider,
            hint="Check the model id, and whether it supports the tools you sent.",
        )
    return response
