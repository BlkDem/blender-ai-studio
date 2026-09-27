"""Anthropic Messages API.

The two things that make this provider different from an OpenAI-compatible one,
and the reasons it is its own class rather than a base URL:

* the system prompt is a top-level field, not a message;
* a tool result is a ``user`` message whose content is a ``tool_result`` block,
  matched to the call by ``tool_use_id``.

Streaming assembles ``input_json_delta`` fragments, because a long argument list
arrives in pieces and each piece is not valid JSON on its own.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.errors import AuthenticationError, ProviderError, RateLimitError
from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
    ensure_not_empty,
)
from app.llm.models import ModelInfo, cost_of

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
#: The version header is required and is not the SDK's to negotiate; this is the
#: value the API documents.
API_VERSION = "2023-06-01"
#: Anthropic requires max_tokens on every request and rejects the request without
#: one, so this is a floor rather than a preference.
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str = "",
        *,
        model: str = "claude-sonnet-4-5",
        models: list[ModelInfo] | None = None,
        timeout: float = 120.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(api_key=api_key, base_url=API_URL)
        self.default_model_id = model
        self.model_catalog = models or [
            ModelInfo(
                id=model,
                provider=self.name,
                display_name=model,
                supports_vision=True,
                supports_images=True,
                supports_thinking=True,
                context_window=200_000,
                max_output_tokens=8192,
                input_price=3.0,
                output_price=15.0,
            )
        ]
        self.timeout = timeout
        self.max_tokens = max_tokens

    def default_model(self) -> str:
        return self.default_model_id

    def models(self) -> list[str]:
        return [model.id for model in self.model_catalog]

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": self.require_key(),
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout, read=300.0),
            )
        return self._client

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = _to_wire(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.max_tokens,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop
        if request.tools:
            payload["tools"] = [tool.to_anthropic() for tool in request.tools]
            if request.tool_choice == "any":
                payload["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
        payload.update(request.extra)
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        try:
            response = await self._http().post("/v1/messages", json=self._payload(request, stream=False))
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}", provider=self.name, retryable=True) from exc
        if response.status_code >= 400:
            raise _error_for(self.name, response.status_code, response.content)
        payload = _json(response)
        parsed = _parse_message(payload, request.model)
        parsed.latency_ms = (time.perf_counter() - started) * 1000
        return ensure_not_empty(parsed, self.name)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        yield StreamChunk(type="start")
        usage = Usage()
        finish = ""
        text: list[str] = []
        reasoning: list[str] = []
        # Anthropic streams content blocks by index, so arguments have to be
        # accumulated per index and only become a ToolCall at block stop.
        blocks: dict[int, dict[str, Any]] = {}
        partial: dict[int, str] = {}
        usage_sent = False

        try:
            async with self._http().stream(
                "POST", "/v1/messages", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise _error_for(self.name, response.status_code, body)

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = _json_line(line[5:].strip())
                    if event is None:
                        continue
                    kind = event.get("type")

                    if kind == "message_start":
                        reported = (event.get("message") or {}).get("usage") or {}
                        usage.input_tokens = int(reported.get("input_tokens") or 0)
                        usage.output_tokens = int(reported.get("output_tokens") or 0)
                        usage.cached_tokens = int(reported.get("cache_read_input_tokens") or 0)

                    elif kind == "content_block_start":
                        index = int(event.get("index", 0))
                        block = event.get("content_block") or {}
                        blocks[index] = {"type": block.get("type"), "id": block.get("id"), "name": block.get("name")}
                        partial[index] = ""
                        if block.get("type") == "tool_use":
                            yield StreamChunk(
                                type="tool_start",
                                call_id=block.get("id") or "",
                                name=block.get("name") or "",
                            )

                    elif kind == "content_block_delta":
                        index = int(event.get("index", 0))
                        delta = event.get("delta") or {}
                        delta_kind = delta.get("type")
                        if delta_kind == "text_delta":
                            piece = delta.get("text") or ""
                            text.append(piece)
                            yield StreamChunk(type="text", text=piece)
                        elif delta_kind == "input_json_delta":
                            piece = delta.get("partial_json") or ""
                            partial[index] = partial.get(index, "") + piece
                            yield StreamChunk(
                                type="tool_delta",
                                call_id=(blocks.get(index) or {}).get("id") or "",
                                name=(blocks.get(index) or {}).get("name") or "",
                                partial=piece,
                            )
                        elif delta_kind == "thinking_delta":
                            piece = delta.get("thinking") or ""
                            reasoning.append(piece)
                            yield StreamChunk(type="reasoning", text=piece)

                    elif kind == "content_block_stop":
                        index = int(event.get("index", 0))
                        block = blocks.get(index) or {}
                        if block.get("type") == "tool_use":
                            yield StreamChunk(
                                type="tool_end",
                                call_id=block.get("id") or "",
                                name=block.get("name") or "",
                                arguments=_parse_arguments(partial.get(index, "")),
                            )

                    elif kind == "message_delta":
                        finish = (event.get("delta") or {}).get("stop_reason") or finish
                        # Cumulative, not a delta: the last one wins.
                        reported = event.get("usage") or {}
                        if reported.get("output_tokens") is not None:
                            usage.output_tokens = int(reported["output_tokens"])

                    elif kind == "message_stop":
                        if not usage_sent:
                            usage_sent = True
                            yield StreamChunk(type="usage", usage=usage)

                    elif kind == "error":
                        raise ProviderError(
                            str((event.get("error") or {}).get("message") or "stream error"),
                            provider=self.name,
                        )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic stream failed: {exc}", provider=self.name, retryable=True) from exc

        if not usage_sent:
            yield StreamChunk(type="usage", usage=usage)
        yield StreamChunk(
            type="end", text="".join(text), finish_reason=finish, reasoning="".join(reasoning)
        )
        logger.debug("%s stream finished in %.0f ms", self.name, (time.perf_counter() - started) * 1000)

    def price(self, model_id: str, usage: Usage) -> Usage:
        model = next((m for m in self.model_catalog if m.id == model_id), None)
        if model is not None:
            usage.cost_usd = cost_of(
                model, usage.input_tokens, usage.output_tokens, cached_tokens=usage.cached_tokens
            )
        return usage


def _to_wire(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Into (system, messages).

    Consecutive tool results are merged into one user message: the API rejects
    two ``tool_result`` blocks split across messages, and a model that asked for
    three tools in one turn gets exactly that case.
    """
    system_chunks = [m.text() for m in messages if m.role is Role.SYSTEM]
    wire: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            wire.append({"role": "user", "content": pending_results.copy()})
            pending_results.clear()

    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content,
                    **({"is_error": True} if message.content.startswith("Error:") else {}),
                }
            )
            continue
        flush()
        if message.role is Role.ASSISTANT and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            wire.append({"role": "assistant", "content": blocks})
        else:
            wire.append({"role": "assistant" if message.role is Role.ASSISTANT else "user",
                         "content": message.text()})
    flush()
    return "\n\n".join(chunk for chunk in system_chunks if chunk), wire


def _parse_message(payload: dict[str, Any], model_id: str) -> ChatResponse:
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            text.append(block.get("text") or "")
        elif kind == "thinking":
            reasoning.append(block.get("thinking") or "")
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=block.get("input") or {},
                )
            )
    reported = payload.get("usage") or {}
    return ChatResponse(
        text="".join(text),
        reasoning="".join(reasoning),
        tool_calls=calls,
        usage=Usage(
            input_tokens=int(reported.get("input_tokens") or 0),
            output_tokens=int(reported.get("output_tokens") or 0),
            cached_tokens=int(reported.get("cache_read_input_tokens") or 0),
        ),
        finish_reason=payload.get("stop_reason") or "",
        model=payload.get("model") or model_id,
        raw=payload,
    )


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": raw}


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("Anthropic did not return JSON", provider="anthropic") from exc
    return payload if isinstance(payload, dict) else {}


def _json_line(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_for(provider: str, status: int, body: bytes) -> ProviderError:
    text = ""
    try:
        payload = json.loads(body)
        error = payload.get("error", payload)
        text = error.get("message") or json.dumps(payload)[:200]
    except (json.JSONDecodeError, AttributeError, TypeError):
        text = body.decode("utf-8", errors="replace")[:200]
    if status in (401, 403):
        return AuthenticationError(provider, text or "The provider rejected the API key")
    if status == 429:
        return RateLimitError(provider)
    return ProviderError(text or f"HTTP {status}", provider=provider, status=status)
