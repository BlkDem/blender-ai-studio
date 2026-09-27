"""A scripted provider, for tests and for the first run before any key exists.

It replays a list of responses and records what it was asked, which is what the
agent's tests need: a tool call followed by a final answer, without a network or
an API key. It is not a stub pretending to be a real provider — it says what it
is, and it is never registered by default.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    StreamChunk,
    ToolCall,
    Usage,
)
from app.llm.models import ModelInfo, cost_of


class ScriptedTurn:
    """One reply the scripted provider will give.

    Either text, or tool calls, or both — a model that narrates before it acts is
    ordinary and the agent has to handle it.
    """

    def __init__(
        self,
        text: str = "",
        tool_calls: Sequence[tuple[str, dict[str, Any]]] = (),
        *,
        usage: Usage | None = None,
        finish_reason: str = "stop",
        reasoning: str = "",
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.tool_calls = list(tool_calls)
        self.usage = usage or Usage(input_tokens=20, output_tokens=10)
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.error = error

    def to_response(self, model_id: str) -> ChatResponse:
        return ChatResponse(
            text=self.text,
            tool_calls=[ToolCall.new(name, args) for name, args in self.tool_calls],
            usage=self.usage,
            finish_reason=self.finish_reason,
            model=model_id,
            reasoning=self.reasoning,
        )


class MockLLMProvider(LLMProvider):
    """Replays :class:`ScriptedTurn` objects, one per request.

    Scripted, not random: an agent test that depended on timing would be a flaky
    test, and a benchmark comparing two models must be able to compare *the same*
    model twice.
    """

    name = "mock"

    def __init__(
        self,
        script: Sequence[ScriptedTurn] = (),
        *,
        model: str = "mock-model",
        price_per_million: float = 0.0,
        models: list[ModelInfo] | None = None,
    ) -> None:
        super().__init__(api_key="mock", base_url="mock://")
        self.script = list(script)
        self.model_catalog = models or [
            ModelInfo(
                id=model,
                provider=self.name,
                display_name=f"{model} (scripted)",
                supports_vision=True,
                context_window=1_000_000,
                input_price=price_per_million,
                output_price=price_per_million * 3,
            )
        ]
        self.default_model_id = model
        #: Everything the agent asked, so a test can assert on the request too.
        self.requests: list[ChatRequest] = []
        #: How many scripted turns have been consumed.
        self.calls = 0

    def script_turns(self, *turns: ScriptedTurn) -> MockLLMProvider:
        self.script = list(turns)
        return self

    def default_model(self) -> str:
        return self.default_model_id

    def models(self) -> list[str]:
        return [model.id for model in self.model_catalog]

    def _next_turn(self) -> ScriptedTurn:
        if not self.script:
            return ScriptedTurn(text="The scripted provider has no more turns.")
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[index]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        turn = self._next_turn()
        if turn.error is not None:
            raise turn.error
        started = time.perf_counter()
        response = turn.to_response(request.model)
        self._price(request.model, response.usage)
        response.latency_ms = (time.perf_counter() - started) * 1000
        return response

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        turn = self._next_turn()
        if turn.error is not None:
            raise turn.error
        yield StreamChunk(type="start")
        # Word by word, with the space *before* each word after the first, so the
        # reassembled text is byte-identical to what was scripted. A trailing
        # space here would quietly break every exact-match assertion downstream.
        words = [word for word in turn.text.split(" ") if word]
        for index, word in enumerate(words):
            yield StreamChunk(type="text", text=(" " if index else "") + word)
        for name, arguments in turn.tool_calls:
            call = ToolCall.new(name, arguments)
            yield StreamChunk(type="tool_start", call_id=call.id, name=call.name)
            # Split the JSON so the agent's partial-argument handling is exercised
            # by default rather than only by the provider tests.
            serialised = json.dumps(arguments)
            for index in range(0, len(serialised), 7):
                yield StreamChunk(type="tool_delta", call_id=call.id, name=call.name, partial=serialised[index : index + 7])
            yield StreamChunk(type="tool_end", call_id=call.id, name=call.name, arguments=arguments)
        usage = turn.usage
        self._price(request.model, usage)
        yield StreamChunk(type="usage", usage=usage)
        yield StreamChunk(type="end", text=turn.text, finish_reason=turn.finish_reason)

    def _price(self, model_id: str, usage: Usage) -> None:
        model = next((m for m in self.model_catalog if m.id == model_id), None)
        if model is not None:
            usage.cost_usd = cost_of(model, usage.input_tokens, usage.output_tokens)
