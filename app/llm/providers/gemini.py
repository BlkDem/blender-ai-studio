"""Google Gemini, generateContent.

Three differences from the OpenAI shape, all of which this class handles:

* the system prompt is ``systemInstruction``, not a message;
* roles are ``user`` and ``model`` only — there is no assistant or tool role, so
  a tool result becomes a ``user`` message holding a ``functionResponse`` part;
* Gemini 3 models reason internally and require the returned ``thoughtSignature``
  to be sent back on the matching part. Unknown parts are therefore kept in
  ``ContentPart.extra`` and echoed verbatim, which is the only way to satisfy that
  without reimplementing the model's thinking bookkeeping.
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

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str = "",
        *,
        model: str = DEFAULT_MODEL,
        models: list[ModelInfo] | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=API_ROOT)
        self.default_model_id = model
        self.model_catalog = models or [
            ModelInfo(
                id=model,
                provider=self.name,
                display_name=model,
                supports_vision=True,
                supports_images=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=8192,
                input_price=0.30,
                output_price=2.50,
            )
        ]
        self.timeout = timeout

    def default_model(self) -> str:
        return self.default_model_id

    def models(self) -> list[str]:
        return [model.id for model in self.model_catalog]

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=API_ROOT,
                headers={"x-goog-api-key": self.require_key(), "content-type": "application/json"},
                timeout=httpx.Timeout(self.timeout, read=300.0),
            )
        return self._client

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system, contents = _to_wire(request.messages)
        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            config["stopSequences"] = request.stop
        if request.tools:
            config["tools"] = [{"functionDeclarations": [tool.to_gemini() for tool in request.tools]}]
            if request.tool_choice == "any":
                config["tool_calling_config"] = {"mode": "ANY"}
            elif request.tool_choice == "none":
                config["tool_calling_config"] = {"mode": "NONE"}
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if config:
            payload["generationConfig"] = config
        payload.update(request.extra)
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        url = f"/models/{request.model}:generateContent"
        try:
            response = await self._http().post(url, json=self._payload(request))
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {exc}", provider=self.name, retryable=True) from exc
        if response.status_code >= 400:
            raise _error_for(self.name, response.status_code, response.content)
        payload = _json(response)
        parsed = _parse_candidates(payload, request.model)
        parsed.latency_ms = (time.perf_counter() - started) * 1000
        return ensure_not_empty(parsed, self.name)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        yield StreamChunk(type="start")
        text: list[str] = []
        reasoning: list[str] = []
        finish = ""
        usage = Usage()
        calls: list[ToolCall] = []
        usage_sent = False

        try:
            async with self._http().stream(
                "POST",
                f"/models/{request.model}:streamGenerateContent",
                params={"alt": "sse"},
                json=self._payload(request),
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
                    for candidate in event.get("candidates") or []:
                        finish = candidate.get("finishReason") or finish
                        for part in (candidate.get("content") or {}).get("parts") or []:
                            if part.get("text") and not part.get("thought"):
                                piece = _accumulate(text, part["text"])
                                if piece:
                                    yield StreamChunk(type="text", text=piece)
                            elif part.get("text") and part.get("thought"):
                                reasoning.append(part["text"])
                                yield StreamChunk(type="reasoning", text=part["text"])
                            call = part.get("functionCall")
                            if call:
                                tool = ToolCall(
                                    id=call.get("id") or f"call_{len(calls)}",
                                    name=call.get("name") or "",
                                    arguments=call.get("args") or {},
                                )
                                calls.append(tool)
                                yield StreamChunk(type="tool_start", call_id=tool.id, name=tool.name)
                                yield StreamChunk(
                                    type="tool_end",
                                    call_id=tool.id,
                                    name=tool.name,
                                    arguments=tool.arguments,
                                )
                    reported = event.get("usageMetadata") or {}
                    if reported:
                        usage = _usage_from_gemini(reported)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini stream failed: {exc}", provider=self.name, retryable=True) from exc

        if not usage_sent:
            usage_sent = True
            yield StreamChunk(type="usage", usage=usage)
        yield StreamChunk(type="end", text="".join(text), finish_reason=finish, reasoning="".join(reasoning))
        logger.debug("%s stream finished in %.0f ms", self.name, (time.perf_counter() - started) * 1000)

    def price(self, model_id: str, usage: Usage) -> Usage:
        model = next((m for m in self.model_catalog if m.id == model_id), None)
        if model is not None:
            usage.cost_usd = cost_of(model, usage.input_tokens, usage.output_tokens)
        return usage


def _accumulate(collected: list[str], piece: str) -> str:
    """Add a text chunk, tolerating both streaming conventions.

    ``streamGenerateContent`` sends deltas, but a gateway in front of it may send
    the whole text so far on every chunk. Treating a chunk that starts with what we
    already have as a replacement handles the second case; anything else is a
    delta and is appended. Returning the piece to emit — empty when this chunk
    said nothing new — keeps the caller from re-sending a repeat.
    """
    joined = "".join(collected)
    if not joined:
        collected.append(piece)
        return piece
    if piece.startswith(joined):
        collected.clear()
        collected.append(piece)
        return piece[len(joined) :]
    if joined.endswith(piece):
        return ""
    collected.append(piece)
    return piece


def _to_wire(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Into (systemInstruction, contents).

    ``user`` and ``model`` are the only roles. A tool result is sent as a
    ``user`` turn holding a ``functionResponse``, which is what the API expects
    even though it reads oddly next to the other providers.
    """
    system_chunks = [m.text() for m in messages if m.role is Role.SYSTEM]
    contents: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            contents.append({"role": "user", "parts": pending.copy()})
            pending.clear()

    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            response: dict[str, Any] = {
                "name": message.name or "",
                "response": {"result": message.content},
            }
            if message.tool_call_id:
                # Gemini 3 matches on the exact id it handed out.
                response["id"] = message.tool_call_id
            pending.append({"functionResponse": response})
            continue
        flush()
        parts: list[dict[str, Any]] = []
        if message.role is Role.ASSISTANT and message.tool_calls:
            for call in message.tool_calls:
                # Not named `part`: an annotated assignment fixes the name's type
                # for the whole function, which would then be wrong for the
                # ContentPart loop below.
                call_part: dict[str, Any] = {"functionCall": {"name": call.name, "args": call.arguments}}
                if call.id:
                    call_part["functionCall"]["id"] = call.id
                parts.append(call_part)
            if message.content:
                parts.insert(0, {"text": message.content})
        else:
            for part in message.parts or []:
                if part.type == "image" and part.data:
                    parts.append(
                        {"inlineData": {"mimeType": part.mime_type or "image/png", "data": part.data}}
                    )
                elif part.text:
                    piece: dict[str, Any] = {"text": part.text}
                    # A thought signature, or anything else the provider wants
                    # echoed verbatim on the next turn.
                    piece.update(part.extra)
                    parts.append(piece)
            if not parts and message.content:
                parts = [{"text": message.content}]
        if parts:
            contents.append({"role": "model" if message.role is Role.ASSISTANT else "user", "parts": parts})
    flush()
    return "\n\n".join(chunk for chunk in system_chunks if chunk), contents


def _parse_candidates(payload: dict[str, Any], model_id: str) -> ChatResponse:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ChatResponse(model=model_id, raw=payload, finish_reason="no_candidates")
    candidate = candidates[0]
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for part in (candidate.get("content") or {}).get("parts") or []:
        if part.get("text"):
            (reasoning if part.get("thought") else text).append(part["text"])
        call = part.get("functionCall")
        if call:
            calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{len(calls)}",
                    name=call.get("name") or "",
                    arguments=call.get("args") or {},
                )
            )
    return ChatResponse(
        text="".join(text),
        reasoning="".join(reasoning),
        tool_calls=calls,
        usage=_usage_from_gemini(payload.get("usageMetadata") or {}),
        finish_reason=candidate.get("finishReason") or "",
        model=model_id,
        raw=payload,
    )


def _usage_from_gemini(raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(raw.get("promptTokenCount") or 0),
        output_tokens=int(raw.get("candidatesTokenCount") or 0),
        cached_tokens=int(raw.get("cachedContentTokenCount") or 0),
    )


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("Gemini did not return JSON", provider="gemini") from exc
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
    if status in (400, 401, 403) and "api key" in text.lower():
        return AuthenticationError(provider, text)
    if status in (401, 403):
        return AuthenticationError(provider, text or "The provider rejected the API key")
    if status == 429:
        return RateLimitError(provider)
    return ProviderError(text or f"HTTP {status}", provider=provider, status=status)
