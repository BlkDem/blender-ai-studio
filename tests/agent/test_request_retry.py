"""Request retries never replay visible content or tool work."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.agent import Agent
from app.core.cost import Budget
from app.core.errors import ProviderError, RateLimitError
from app.core.events import EventType
from app.llm.base import ChatResponse, StreamChunk, Usage
from tests.agent.test_agent import FakeMCP


class Provider:
    name = "test"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(self, request):
        chunks = await self.chat(request)
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def make(outcomes, **kwargs):
    return Agent(Provider(outcomes), "test", FakeMCP(), **kwargs)


@pytest.mark.asyncio
async def test_retry_success(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.core.agent.asyncio.sleep", sleep)
    agent = make([ProviderError("busy", retryable=True), RateLimitError("test", 1000),
                  ChatResponse(text="done", usage=Usage(input_tokens=12))], stream=False)
    result = await agent.run("hello", run_id="retry")
    assert result.text == "done"
    assert result.steps == 1
    assert result.totals.input_tokens == 12
    assert [call.args[0] for call in sleep.call_args_list] == [3, 30]
    retries = [e for e in agent.bus.history() if e.type == EventType.REQUEST_RETRY]
    assert [e.payload["attempt"] for e in retries] == [2, 3]
    assert all(e.run_id == "retry" and e.payload["max_attempts"] == 11 for e in retries)


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable,expected", [(True, 11), (False, 1)])
async def test_failure_visible(monkeypatch, retryable, expected):
    sleep = AsyncMock()
    monkeypatch.setattr("app.core.agent.asyncio.sleep", sleep)
    error = ProviderError("upstream failed", retryable=retryable, hint="Try another model")
    agent = make([error] * 11, stream=False)
    with pytest.raises(ProviderError):
        await agent.run("hello")
    assert agent.provider.calls == expected
    assert [call.args[0] for call in sleep.call_args_list] == (
        list(range(3, 31, 3)) if retryable else []
    )
    failed = [e for e in agent.bus.history() if e.type == EventType.RUN_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["detail"] == "upstream failed"
    assert failed[0].payload["hint"] == "Try another model"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk", [StreamChunk(type="text", text="partial"),
    StreamChunk(type="reasoning", text="thinking"),
    StreamChunk(type="tool_start", call_id="call", name="tool")])
async def test_partial_never_retried(chunk):
    agent = make([[chunk, ProviderError("lost", retryable=True)]])
    with pytest.raises(ProviderError):
        await agent.run("hello")
    assert agent.provider.calls == 1
    assert not any(e.type == EventType.REQUEST_RETRY for e in agent.bus.history())


@pytest.mark.asyncio
async def test_cancel_during_backoff():
    agent = make([ProviderError("busy", retryable=True)], stream=False)
    waiting = asyncio.Event()
    agent.bus.subscribe(lambda e: waiting.set() if e.type == EventType.REQUEST_RETRY else None)
    task = asyncio.create_task(agent.run("hello", run_id="cancel"))
    await asyncio.wait_for(waiting.wait(), 1)
    assert agent.cancel("cancel")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.provider.calls == 1
    assert not agent.is_running()
    assert agent.bus.history()[-1].payload["reason"] == "cancelled"


@pytest.mark.asyncio
async def test_backoff_inside_timeout():
    agent = make([ProviderError("busy", retryable=True)], stream=False,
                 budget=Budget(max_seconds=0.01))
    result = await agent.run("hello")
    assert not result.finished
    assert agent.provider.calls == 1
    assert agent.bus.history()[-1].type == EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_empty_stream_is_failure_and_usage_accounted(monkeypatch):
    monkeypatch.setattr("app.core.agent.asyncio.sleep", AsyncMock())
    agent = make([[StreamChunk(type="usage", usage=Usage(input_tokens=9))] for _ in range(11)])
    with pytest.raises(ProviderError, match="empty response"):
        await agent.run("hello")
    assert agent.provider.calls == 11
    assert agent.cost.totals.input_tokens == 99
    assert agent.bus.history()[-1].type == EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_empty_stream_retries_then_succeeds(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.core.agent.asyncio.sleep", sleep)
    agent = make([
        [StreamChunk(type="usage", usage=Usage(input_tokens=9))],
        [StreamChunk(type="text", text="done")],
    ])
    result = await agent.run("hello")
    assert result.text == "done"
    assert agent.provider.calls == 2
    assert agent.cost.totals.input_tokens == 9
    sleep.assert_awaited_once_with(3)
    assert any(e.type == EventType.REQUEST_RETRY for e in agent.bus.history())


@pytest.mark.asyncio
async def test_nonstream_provider_empty_response_retries(monkeypatch):
    from app.llm.base import ensure_not_empty

    monkeypatch.setattr("app.core.agent.asyncio.sleep", AsyncMock())

    class ValidatingProvider(Provider):
        async def chat(self, request):
            return ensure_not_empty(await super().chat(request), self.name)

    agent = Agent(ValidatingProvider([ChatResponse(), ChatResponse(text="done")]),
                  "test", FakeMCP(), stream=False)
    result = await agent.run("hello")
    assert result.text == "done"
    assert agent.provider.calls == 2
