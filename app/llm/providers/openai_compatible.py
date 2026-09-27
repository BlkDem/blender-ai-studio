"""OpenAI-compatible chat completions.

One provider covers OpenAI itself and everything that copies its wire format:
a gateway hosting Space Bunny or JEV, vLLM, Ollama's OpenAI endpoint, LM Studio.
That is the point of the class — a user can add an endpoint in the GUI and drive
Blender with whatever model they have, without a code change and without this
project knowing what any of those models are.

Deliberately the chat-completions API rather than the newer Responses API: it is
the one every compatible server implements, and tool calling behaves the same
across all of them.
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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: Streaming needs a read timeout, not a total one: a long generation is silent
#: between tokens, and a 30s read timeout kills a healthy slow model.
STREAM_READ_TIMEOUT = 300.0


class OpenAICompatibleProvider(LLMProvider):
    """Chat, streaming and tool calling over the OpenAI wire format."""

    name = "openai-compatible"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str = "gpt-4o-mini",
        models: list[ModelInfo] | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
        self.default_model_id = model
        self.model_catalog = models or [ModelInfo(id=model, provider=self.name)]
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    # --- catalog -----------------------------------------------------------

    def default_model(self) -> str:
        return self.default_model_id

    def models(self) -> list[str]:
        return [model.id for model in self.model_catalog]

    async def list_models(self) -> list[ModelInfo]:
        """Prefer the server's own list, merged over the configured catalog.

        A gateway can serve a model the studio has never heard of, and hiding it
        until someone edits a config file would make the model selector a lie.
        """
        known = {model.id: model for model in self.model_catalog}
        try:
            response = await self._http().get("/models")
        except (httpx.HTTPError, ProviderError) as exc:
            logger.info("could not list models from %s: %s", self.base_url, exc)
            return list(known.values())
        payload = _json(response)
        for entry in payload.get("data", []) or []:
            model_id = entry.get("id")
            if not model_id or model_id in known:
                continue
            known[model_id] = ModelInfo(id=model_id, provider=self.name, display_name=model_id)
        return list(known.values())

    # --- transport ---------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json", **self.extra_headers}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, read=STREAM_READ_TIMEOUT),
            )
        return self._client

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_to_wire(message) for message in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = [tool.to_openai() for tool in request.tools]
            payload["tool_choice"] = request.tool_choice if request.tool_choice != "auto" else "auto"
        if stream:
            # Without this the final usage is absent and a streamed run's cost is
            # a guess, which is exactly what the cost column must not be.
            payload["stream_options"] = {"include_usage": True}
        payload.update(request.extra)
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        response = await self._post("/chat/completions", self._payload(request, stream=False))
        payload = _json(response)
        parsed = _parse_completion(payload, request.model)
        parsed.latency_ms = (time.perf_counter() - started) * 1000
        return ensure_not_empty(parsed, self.name)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        yield StreamChunk(type="start")
        usage: Usage | None = None
        finish = ""
        text: list[str] = []
        calls: dict[int, ToolCall] = {}
        collected: dict[int, str] = {}

        try:
            async with self._http().stream(
                "POST", "/chat/completions", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise _error_for(self.name, response.status_code, body)

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = _json_line(data)
                    if chunk is None:
                        continue
                    if chunk.get("usage"):
                        usage = _usage_from_openai(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        for piece in delta.get("content") or "":
                            text.append(piece)
                            yield StreamChunk(type="text", text=piece)
                        for call in delta.get("tool_calls") or []:
                            index = int(call.get("index", 0))
                            call_id = call.get("id")
                            function = call.get("function") or {}
                            if call_id:
                                calls[index] = ToolCall(id=call_id, name=function.get("name", ""))
                                collected[index] = ""
                                yield StreamChunk(
                                    type="tool_start", call_id=call_id, name=function.get("name", "")
                                )
                            arguments = function.get("arguments")
                            if arguments:
                                collected[index] = collected.get(index, "") + arguments
                                yield StreamChunk(
                                    type="tool_delta",
                                    call_id=calls[index].id if index in calls else "",
                                    name=calls[index].name if index in calls else "",
                                    partial=arguments,
                                )
                    reasoning = _reasoning_from(chunk)
                    if reasoning:
                        yield StreamChunk(type="reasoning", text=reasoning)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name} request failed: {exc}", provider=self.name, retryable=True
            ) from exc

        for index, call in calls.items():
            call.arguments = _parse_arguments(collected.get(index, ""))
            yield StreamChunk(type="tool_end", call_id=call.id, name=call.name, arguments=call.arguments)

        if usage is not None:
            yield StreamChunk(type="usage", usage=usage)
        yield StreamChunk(
            type="end",
            text="".join(text),
            finish_reason=finish,
            arguments=None,
        )
        logger.debug("%s stream finished in %.0f ms", self.name, (time.perf_counter() - started) * 1000)

    # --- pricing -----------------------------------------------------------

    def price(self, model_id: str, usage: Usage) -> Usage:
        """Attach a cost, when the model is in the catalog.

        An unknown model is left at zero rather than guessed: a made-up price on a
        benchmark table is worse than a blank one.
        """
        model = next((m for m in self.model_catalog if m.id == model_id), None)
        if model is not None:
            usage.cost_usd = cost_of(
                model, usage.input_tokens, usage.output_tokens, cached_tokens=usage.cached_tokens
            )
        return usage

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._http().post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name} request failed: {exc}", provider=self.name, retryable=True
            ) from exc
        if response.status_code >= 400:
            raise _error_for(self.name, response.status_code, response.content)
        return response


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI proper, so configuration can name it without a base URL."""

    name = "openai"

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key, base_url=kwargs.pop("base_url", DEFAULT_BASE_URL), model=model, **kwargs
        )
        self.model_catalog = kwargs.get("models") or _OPENAI_CATALOG


#: Prices and capabilities as published by OpenAI, in dollars per million tokens.
#: A deployment the studio has never heard of still works; it just shows no price.
_OPENAI_CATALOG = [
    ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        supports_vision=True,
        supports_images=True,
        context_window=128_000,
        max_output_tokens=16_384,
        input_price=0.15,
        output_price=0.60,
    ),
    ModelInfo(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        supports_vision=True,
        supports_images=True,
        supports_thinking=True,
        context_window=128_000,
        max_output_tokens=16_384,
        input_price=2.50,
        output_price=10.00,
    ),
    ModelInfo(
        id="o3-mini",
        provider="openai",
        display_name="o3-mini",
        supports_thinking=True,
        context_window=200_000,
        max_output_tokens=100_000,
        input_price=1.10,
        output_price=4.40,
    ),
]


# --- wire format ------------------------------------------------------------


def _to_wire(message: Message) -> dict[str, Any]:
    """One message into OpenAI's shape.

    A tool result is its own message with ``tool_call_id``; an assistant message
    that asked for tools keeps its ``tool_calls`` array, and the empty string
    that goes with it is required by the API even when there is no prose.
    """
    if message.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    if message.role is Role.ASSISTANT and message.tool_calls:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_text()},
                }
                for call in message.tool_calls
            ],
        }
        if message.reasoning:
            payload["reasoning_content"] = message.reasoning
        return payload
    text = message.content or message.text()
    if message.parts:
        # A message with a picture is a list of parts, not a string. The data
        # URI is the form every OpenAI-compatible server understands, including
        # llama.cpp and vLLM.
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{part.mime_type or 'image/png'};base64,{part.data or ''}"},
            }
            for part in message.parts
            if part.type == "image"
        )
        return {"role": str(message.role), "content": content}
    return {"role": str(message.role), "content": text}


def _parse_completion(payload: dict[str, Any], model_id: str) -> ChatResponse:
    choices = payload.get("choices") or []
    if not choices:
        return ChatResponse(model=model_id, raw=payload, finish_reason="no_choices")
    message = choices[0].get("message") or {}
    calls = [
        ToolCall(
            id=(call.get("id") or f"call_{index}"),
            name=(call.get("function") or {}).get("name", ""),
            arguments=_parse_arguments((call.get("function") or {}).get("arguments", "")),
        )
        for index, call in enumerate(message.get("tool_calls") or [])
    ]
    return ChatResponse(
        text=message.get("content") or "",
        reasoning=message.get("reasoning_content") or "",
        tool_calls=calls,
        usage=_usage_from_openai(payload.get("usage") or {}),
        finish_reason=choices[0].get("finish_reason") or "",
        model=payload.get("model") or model_id,
        raw=payload,
    )


def _usage_from_openai(raw: dict[str, Any]) -> Usage:
    details = raw.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
    )


def _reasoning_from(chunk: dict[str, Any]) -> str:
    """Reasoning text, under whichever key this deployment uses.

    Reasoning models disagree on the field name, and an OpenAI-compatible server
    is as likely to use a vendor's as OpenAI's. Both are read; neither is required.
    """
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        for key in ("reasoning_content", "reasoning"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Tool arguments are JSON on the wire and a dict here.

    Models occasionally emit a bare string or a truncated fragment; an empty
    dict plus whatever text arrived is more useful than an exception that ends
    the run, and the agent will pass the raw text back as the error result.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": raw}


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"{response.request.url} did not return JSON", provider="openai-compatible"
        ) from exc
    return payload if isinstance(payload, dict) else {}


def _json_line(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_for(provider: str, status: int, body: bytes) -> ProviderError:
    """Turn an HTTP status into something a person can act on.

    The provider's own message is kept: "model not found" and "insufficient
    quota" need different fixes, and paraphrasing them into one string loses the
    only useful part.
    """
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
    if status == 404:
        return ProviderError(
            text or "No such endpoint or model",
            provider=provider,
            hint="Check the base URL and the model id.",
        )
    return ProviderError(text or f"HTTP {status}", provider=provider, status=status)
