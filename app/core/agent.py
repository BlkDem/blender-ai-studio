"""The agent: a model, some tools, and a loop.

One rule shapes this class: the agent knows about *tools*, not about Blender. It
asks an LLM what to do, runs what the LLM asked for, feeds the results back, and
stops when the model answers. Whether a tool comes from an MCP server, from a 3D
provider or is defined here makes no difference to the loop, which is what lets a
model change without touching it.

Everything observable goes through the event bus, and everything expensive is
checked against a budget *before* it happens. A model that has misunderstood a
tool result and decides to try again forty times should cost a stop event, not a
bill.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.cost import Budget, CostTotals, CostTracker
from app.core.errors import AgentError, BudgetExceeded, Cancelled, MCPError, StudioError
from app.core.events import Event, EventBus, EventType, new_run_id
from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from app.llm.models import ModelInfo
from app.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

#: The instructions the studio ships. A user's own system prompt is appended, not
#: substituted: the base text is about how to use *these* tools, which stays true
#: whatever model is running.
BASE_SYSTEM_PROMPT = """\
You are an AI assistant controlling Blender through tools.

The Blender tools are reached over MCP. Their names and JSON schemas are the
contract; use them as they are described rather than guessing arguments.

Work like this:
1. Understand what was asked.
2. Inspect the scene before a complex change, so you act on what is there.
3. Plan, then call one tool at a time, reading each result before the next.
4. Use generate_3d_asset when a specialist 3D generation service is the right
   tool for the object, rather than approximating it with primitives.
5. Verify a result you are unsure about, when there is a way to.
6. Report what actually happened.

Never claim an operation succeeded unless a tool said so. If a tool failed, say
what failed and what you would try next. Do not repeat a call that has already
failed for the same reason; change something first.

Keep answers short. The user is watching a 3D viewport, not reading an essay.\
"""


@dataclass(slots=True)
class LocalTool:
    """A tool the studio provides itself.

    3D generation is one of these: it is a capability of the application, not
    something an MCP server exposes, and the model should not have to know the
    difference.
    """

    name: str
    description: str
    schema: dict[str, Any]
    handler: Any

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.schema)


@dataclass
class RunResult:
    """What one run produced, for the caller and for the database."""

    run_id: str
    text: str = ""
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolResult] = field(default_factory=list)
    totals: CostTotals = field(default_factory=CostTotals)
    steps: int = 0
    finished: bool = False
    stopped_because: str = ""
    images: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def tool_names(self) -> list[str]:
        return [call.call.name for call in self.tool_calls]


class Agent:
    """Runs one conversation turn, with tools, under a budget."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        mcp: MCPManager,
        *,
        bus: EventBus | None = None,
        model_info: ModelInfo | None = None,
        system_prompt: str = "",
        budget: Budget | None = None,
        local_tools: Sequence[LocalTool] = (),
        stream: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_result_chars: int = 8000,
        studio: Any = None,
        conversation_id: str = "",
    ) -> None:
        self.provider = provider
        self.model = model
        self.mcp = mcp
        self.bus = bus or EventBus()
        self.cost = CostTracker(model_info)
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self.local_tools = list(local_tools)
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        #: How much of one tool result goes back to the model. A whole scene's
        #: object list is fine in the transcript and ruinous in the prompt.
        self.max_result_chars = max_result_chars
        #: Where a run is written down. Optional, and deliberately so: a test that
        #: only cares about the loop should not need a database, but every real
        #: run belongs in one -- a benchmark is a recorded run, and a transcript a
        #: user cannot scroll back through is not a conversation.
        self.studio = studio
        self.conversation_id = conversation_id
        self._runs: dict[str, asyncio.Task[Any]] = {}

    # --- tools -------------------------------------------------------------

    def tools(self) -> list[ToolSpec]:
        """MCP tools and the studio's own, in one list for the model."""
        specs = list(self.mcp.tool_specs())
        known = {spec.name for spec in specs}
        specs.extend(tool.spec() for tool in self.local_tools if tool.name not in known)
        return specs

    def _local_tool(self, name: str) -> LocalTool | None:
        return next((tool for tool in self.local_tools if tool.name == name), None)

    # --- system prompt -----------------------------------------------------

    def build_system_prompt(self) -> str:
        parts = [BASE_SYSTEM_PROMPT]
        if self.system_prompt:
            parts.append(self.system_prompt.strip())
        server_notes = self.mcp.tool_instructions()
        if server_notes:
            parts.append(server_notes)
        catalog = self.tools()
        if catalog:
            lines = [f"- {spec.name}: {spec.description.splitlines()[0] if spec.description else ''}".rstrip(": ")
                     for spec in catalog]
            parts.append("Tools available now:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    # --- running -----------------------------------------------------------

    async def run(
        self,
        user_text: str,
        *,
        history: Sequence[Message] = (),
        run_id: str = "",
        conversation_id: str = "",
    ) -> RunResult:
        """One user turn, to a final answer.

        Raises :class:`~app.core.errors.Cancelled` when the user stops it, and
        :class:`~app.core.errors.BudgetExceeded` when a limit is reached; both
        are published as events first, because a stop with no visible reason is
        the thing a user cannot act on.
        """
        run_id = run_id or new_run_id()
        result = RunResult(run_id=run_id)
        # Registered before anything can fail, so Stop works from the first token
        # rather than only after the first step.
        current = asyncio.current_task()
        if current is not None:
            self._runs[run_id] = current
        self.budget.reset()
        started = time.perf_counter()
        self._publish(run_id, EventType.RUN_STARTED, model=self.model, provider=self.provider.name,
                      tools=[spec.name for spec in self.tools()])

        messages: list[Message] = [Message.system(self.build_system_prompt()), *history, Message.user(user_text)]
        result.messages = list(messages)
        if self.studio is not None:
            self.conversation_id = await self._conversation()
            await self._store(conversation_id=self.conversation_id, role="user", content=user_text, run_id=run_id)

        try:
            while True:
                self.budget.check(result.totals)
                response = await self._one_turn(run_id, messages, result)
                messages.append(
                    Message.assistant(response.text, response.tool_calls)
                )
                if self.studio is not None and (response.text or response.tool_calls):
                    await self._store(
                        conversation_id=self.conversation_id,
                        role="assistant",
                        content=response.text,
                        run_id=run_id,
                        reasoning=response.reasoning,
                    )
                result.steps += 1
                self.budget.steps += 1

                if not response.tool_calls:
                    result.text = response.text
                    result.finished = True
                    break

                for call in response.tool_calls:
                    # Checked per call, not per step: a model that asks for six
                    # tools in one turn should stop at the fifth, not after the
                    # whole batch has already run.
                    self.budget.check_call(result.totals)
                    tool_result = await self._run_tool(run_id, call, result)
                    result.tool_calls.append(tool_result)
                    messages.append(tool_result.message())
                    result.messages = messages
        except asyncio.CancelledError:
            self._publish(run_id, EventType.RUN_FAILED, reason="cancelled",
                          steps=result.steps, **result.totals.to_dict())
            raise
        except BudgetExceeded as exc:
            result.stopped_because = exc.message
            self._publish(run_id, EventType.RUN_FAILED, reason=exc.code, detail=exc.message,
                          **result.totals.to_dict())
            return result
        except (AgentError, MCPError) as exc:
            result.stopped_because = exc.user_text()
            self._publish(run_id, EventType.RUN_FAILED, reason=exc.code, detail=exc.message,
                          **result.totals.to_dict())
            raise
        finally:
            self._runs.pop(run_id, None)
            result.elapsed_s = time.perf_counter() - started
            result.totals = self.cost.totals

        self._publish(run_id, EventType.RUN_FINISHED, text=result.text, steps=result.steps,
                      elapsed_s=round(result.elapsed_s, 2), **result.totals.to_dict())
        return result

    async def _one_turn(self, run_id: str, messages: list[Message], result: RunResult) -> ChatResponse:
        """One LLM call, streamed or not."""
        request = ChatRequest(
            messages=messages,
            model=self.model,
            tools=self.tools(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.stream:
            response = await self.provider.chat(request)
            if response.text:
                self._publish(run_id, EventType.TEXT, text=response.text)
            await self._account(run_id, response, result)
            return response

        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[str, ToolCall] = {}
        arguments: dict[str, str] = {}
        usage = None
        finish = ""
        from app.llm.base import Usage

        async for chunk in self.provider.stream(request):
            if chunk.type == "start":
                continue
            if chunk.type == "text":
                text.append(chunk.text)
                self._publish(run_id, EventType.TOKEN, text=chunk.text)
            elif chunk.type == "reasoning":
                reasoning.append(chunk.text)
                self._publish(run_id, EventType.THINKING, text=chunk.text)
            elif chunk.type == "tool_start":
                calls[chunk.call_id] = ToolCall(id=chunk.call_id, name=chunk.name)
                arguments[chunk.call_id] = ""
            elif chunk.type == "tool_delta":
                arguments[chunk.call_id] = arguments.get(chunk.call_id, "") + chunk.partial
                self._publish(run_id, EventType.TOOL_STARTED, tool=chunk.name, call_id=chunk.call_id,
                              streaming=True)
            elif chunk.type == "tool_end":
                call = calls.get(chunk.call_id)
                if call is not None:
                    call.arguments = chunk.arguments or _parse(arguments.get(chunk.call_id, ""))
            elif chunk.type == "usage":
                usage = chunk.usage
            elif chunk.type == "end":
                finish = chunk.finish_reason

        response = ChatResponse(
            text="".join(text),
            reasoning="".join(reasoning),
            tool_calls=list(calls.values()),
            usage=usage or Usage(),
            finish_reason=finish,
            model=self.model,
        )
        await self._account(run_id, response, result)
        if response.text:
            self._publish(run_id, EventType.MESSAGE_COMPLETE, text=response.text)
        return response

    async def _account(self, run_id: str, response: ChatResponse, result: RunResult) -> None:
        """Price one request, check the per-request limit, publish the cost."""
        if self.model is not None:
            self.cost.set_model(self.cost.model)
        cost = self.cost.add_usage(response.usage)
        result.totals = self.cost.totals
        if self.studio is not None:
            # Recorded before the budget is checked: the request has already been
            # billed, and a cost report that omits the expensive one is exactly
            # the report that cannot be trusted.
            await self._record_usage(run_id, response, cost)
        self.budget.check_request(cost)
        self._publish(
            run_id,
            EventType.USAGE,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(response.latency_ms, 1),
        )

    # --- tools -------------------------------------------------------------

    async def _run_tool(self, run_id: str, call: ToolCall, result: RunResult) -> ToolResult:
        """Run one tool call and describe the outcome to the model."""
        self._publish(run_id, EventType.TOOL_STARTED, tool=call.name, call_id=call.id,
                      arguments=call.arguments)
        local = self._local_tool(call.name)
        error_code = ""
        try:
            if local is not None:
                outcome_text, images, credits, usd = await self._run_local(local, call, result)
                is_error = outcome_text.startswith("Error:")
            else:
                outcome = await self.mcp.call_tool(call.name, call.arguments)
                outcome_text = outcome.text
                images = outcome.images
                is_error = outcome.is_error
                error_code = outcome.error_code
                credits = usd = 0.0
        except asyncio.CancelledError:
            raise
        except (StudioError, OSError, ValueError) as exc:
            outcome_text = f"Error: {exc}"
            images, credits, usd, is_error = [], 0.0, 0.0, True
            error_code = exc.code

        body = self._for_model(outcome_text)

        if self.studio is not None:
            record = await self.studio.tool_calls.start(run_id, call.name, call.arguments)
            await self.studio.tool_calls.finish(
                record.id, result=body, error_code=error_code, is_error=is_error
            )
        self.cost.add_tool_call(is_error)
        self.cost.add_3d(credits, usd)
        result.totals = self.cost.totals
        result.images.extend(images)

        body = self._for_model(outcome_text)
        self._publish(
            run_id,
            EventType.TOOL_FAILED if is_error else EventType.TOOL_FINISHED,
            tool=call.name,
            call_id=call.id,
            is_error=is_error,
            text=body,
            images=len(images),
        )
        self._publish(run_id, EventType.COST, **self.cost.totals.to_dict())
        return ToolResult(call=call, content=body, is_error=is_error)

    async def _run_local(
        self, tool: LocalTool, call: ToolCall, result: RunResult
    ) -> tuple[str, list[str], float, float]:
        outcome = await tool.handler(call.arguments, run_id=call.id)
        if isinstance(outcome, dict):
            return (
                json.dumps(outcome, ensure_ascii=False, indent=2, default=str),
                list(outcome.get("images") or []),
                float(outcome.get("credits") or 0.0),
                float(outcome.get("cost_usd") or 0.0),
            )
        return (str(outcome), [], 0.0, 0.0)

    def _for_model(self, text: str) -> str:
        """What goes back into the prompt.

        Truncated here rather than in the transport, because the transport's job
        is to report what the tool said and this is the point where a transcript
        turns into context.
        """
        if len(text) <= self.max_result_chars:
            return text
        half = self.max_result_chars // 2
        return (
            text[:half]
            + f"\n… {len(text) - self.max_result_chars} characters omitted; "
            "ask for a narrower result if you need the detail …\n"
            + text[-half:]
        )

    # --- persistence -------------------------------------------------------

    async def _conversation(self) -> str:
        """A conversation to write into, created on first use.

        The repository mints the id; this only has to remember it for the rest of
        the run.
        """
        if self.conversation_id:
            return self.conversation_id
        conversation = await self.studio.conversations.create(None, "")
        self.conversation_id = conversation.id
        return self.conversation_id

    async def _store(
        self, *, conversation_id: str, role: str, content: str, run_id: str, reasoning: str = ""
    ) -> None:
        if not conversation_id:
            return
        await self.studio.messages.add(
            conversation_id, role, content, run_id=run_id, reasoning=reasoning or None
        )

    async def _record_usage(self, run_id: str, response: ChatResponse, cost: float) -> None:
        from app.storage.repositories import LLMRequestRecord

        await self.studio.usage.record(
            LLMRequestRecord(
                id=f"req_{uuid.uuid4().hex[:12]}",
                run_id=run_id,
                provider=self.provider.name,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=response.usage.cached_tokens,
                cost_usd=cost,
                latency_ms=response.latency_ms,
                is_error=False,
                error=None,
                created_at=time.time(),
                messages=len(response.raw.get("messages", [])) if response.raw else 0,
            )
        )

    # --- cancellation ------------------------------------------------------

    def cancel(self, run_id: str = "") -> bool:
        """Stop a run. Without an id, the most recent one."""
        if not run_id and self._runs:
            run_id = next(reversed(self._runs))
        task = self._runs.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    # --- events ------------------------------------------------------------

    def _publish(self, run_id: str, event_type: EventType, **payload: Any) -> None:
        event = Event(type=event_type, run_id=run_id, payload=payload)
        with contextlib.suppress(RuntimeError):
            self.bus.emit(event.type, run_id=run_id, **payload)


def _parse(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": raw}


def role_of(message: Message) -> str:
    return str(message.role) if message.role is not Role.ASSISTANT else "assistant"
