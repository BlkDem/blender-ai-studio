"""Benchmark: the same task, several models, the same starting point.

The whole value of a benchmark here is that the runs are *comparable*, and
everything about isolation serves that. If model B sees the scene model A left
behind, the comparison measures memory rather than capability, and no amount of
data in the table fixes it.

So a run is a copy of the initial ``.blend``, and the studio says out loud which
kind of reset it actually managed. ``verified`` means the file was opened. ``copy_only``
means a fresh copy was staged and the operator opened it. ``unverified`` means
neither, and the row is marked so nobody reads it as evidence.

What the benchmark will not do is declare a winner. It records what happened —
time, tokens, money, tool calls, errors — and leaves the judgement to a person,
who scores geometry, materials and instruction following by looking at the
result.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.agent import Agent
from app.core.cost import Budget, CostTotals
from app.core.errors import StudioError
from app.core.events import EventBus, EventType
from app.llm.models import ModelInfo
from app.llm.registry import LLMRegistry
from app.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

#: How a run's starting scene was established. Stored per run, because a
#: comparison between a verified and an unverified run is not a comparison.
RESET_VERIFIED = "verified"
RESET_COPY_ONLY = "copy_only"
RESET_UNVERIFIED = "unverified"


@dataclass
class BenchmarkTask:
    """One thing to ask every model."""

    prompt: str
    name: str = ""

    def label(self) -> str:
        return self.name or self.prompt[:40]


@dataclass
class ModelSpec:
    """A model to compare, by name."""

    provider: str
    model: str = ""

    def label(self) -> str:
        return f"{self.model or 'default'} @ {self.provider}"


@dataclass
class RunOutcome:
    """What one run did."""

    model: ModelSpec
    task: BenchmarkTask
    status: str = "pending"
    run_id: str = ""
    duration_s: float = 0.0
    totals: CostTotals = field(default_factory=CostTotals)
    mcp_calls: int = 0
    tool_errors: int = 0
    final_scene: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    scene_reset: str = RESET_UNVERIFIED
    blend_file: str = ""

    def row(self) -> dict[str, Any]:
        return {
            "model": self.model.label(),
            "provider": self.model.provider,
            "model_id": self.model.model,
            "task": self.task.label(),
            "status": self.status,
            "duration_s": round(self.duration_s, 1),
            "input_tokens": self.totals.input_tokens,
            "output_tokens": self.totals.output_tokens,
            "llm_cost_usd": round(self.totals.llm_usd, 4),
            "three_d_credits": self.totals.three_d_credits,
            "three_d_cost_usd": round(self.totals.three_d_usd, 4),
            "total_cost_usd": round(self.totals.llm_usd + self.totals.three_d_usd, 4),
            "mcp_calls": self.mcp_calls,
            "tool_errors": self.tool_errors,
            "scene_reset": self.scene_reset,
            "error": self.error,
        }


@dataclass
class Comparison:
    """A table. No verdict, on purpose."""

    tasks: list[BenchmarkTask]
    outcomes: list[RunOutcome] = field(default_factory=list)

    def columns(self) -> list[str]:
        return ["Model", "Status", "Time", "Tokens", "Cost", "Tool calls", "Errors", "Scene reset"]

    def rows(self) -> list[list[str]]:
        table: list[list[str]] = []
        for outcome in self.outcomes:
            table.append(
                [
                    outcome.model.label(),
                    outcome.status,
                    f"{outcome.duration_s:.1f}s",
                    str(outcome.totals.tokens),
                    f"${outcome.totals.llm_usd + outcome.totals.three_d_usd:.4f}",
                    str(outcome.mcp_calls),
                    str(outcome.tool_errors),
                    outcome.scene_reset,
                ]
            )
        return table

    def to_text(self) -> str:
        header = "  ".join(f"{column:<14}" for column in self.columns())
        lines = [header, "-" * len(header)]
        for row in self.rows():
            lines.append("  ".join(f"{cell:<14}" for cell in row))
        warnings = [o for o in self.outcomes if o.scene_reset != RESET_VERIFIED]
        if warnings:
            lines.append("")
            lines.append(f"{len(warnings)} run(s) were not scene-isolated; their results are not comparable.")
        return "\n".join(lines)

    def fastest(self) -> RunOutcome | None:
        finished = [o for o in self.outcomes if o.status == "ok"]
        return min(finished, key=lambda o: o.duration_s) if finished else None


class BenchmarkRunner:
    """Runs a suite: every model, every task, one fresh scene each."""

    def __init__(
        self,
        llm: LLMRegistry,
        mcp: MCPManager,
        *,
        bus: EventBus | None = None,
        workdir: Path | None = None,
        system_prompt: str = "",
        budget: Budget | None = None,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.bus = bus or EventBus()
        self.workdir = workdir or (Path.home() / ".local" / "share" / "blender-ai-studio" / "benchmarks")
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # --- isolation ---------------------------------------------------------

    def stage_scene(self, blend: Path, model: ModelSpec, task_index: int) -> Path:
        """A private copy of the starting file, so no run can see another's work."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in model.label())[:60]
        destination = self.workdir / f"run-{safe}-{task_index}-{uuid.uuid4().hex[:6]}.blend"
        shutil.copy2(blend, destination)
        return destination

    async def reset_scene(self, staged: Path) -> str:
        """Get Blender to the staged file, and report honestly whether it worked.

        blender-mcp has no "open a file" tool, and embedding Blender's own API in
        a GUI client is exactly the boundary this project is built to avoid. So
        the studio does the only honest thing available: stage the file, ask
        through the studio's own capability (``execute_python``, off by default),
        and say which of the three states it reached.
        """
        if not staged.exists():
            return RESET_UNVERIFIED
        tools = self.mcp.tools()
        if "blender.execute_python" not in tools:
            # The operator has not enabled execute_python, so the reset is the
            # copy alone. Recorded as such rather than claimed.
            return RESET_COPY_ONLY
        code = f"import bpy\nbpy.ops.wm.open_mainfile(filepath={str(staged)!r})\nresult = True"
        outcome = await self.mcp.call_tool("blender.execute_python", {"code": code}, timeout=120.0)
        if outcome.is_error:
            logger.warning("could not open the staged scene: %s", outcome.text[:120])
            return RESET_COPY_ONLY
        return RESET_VERIFIED

    # --- running -----------------------------------------------------------

    async def run_task_for_model(
        self, task: BenchmarkTask, model: ModelSpec, *, blend: Path | None = None, task_index: int = 0
    ) -> RunOutcome:
        outcome = RunOutcome(model=model, task=task)
        if self._cancelled:
            outcome.status = "cancelled"
            return outcome
        try:
            provider, resolved = self.llm.resolve(model.provider, model.model)
        except StudioError as exc:
            outcome.status = "failed"
            outcome.error = exc.user_text()
            return outcome
        # Record what actually ran, not what was asked for: a row that says
        # "default" cannot be compared with a row that names its model.
        outcome.model = ModelSpec(model.provider, resolved)

        if blend is not None:
            if not blend.exists():
                # A path that is not there is a configuration mistake, and the
                # row says so rather than the run dying on a copy error.
                outcome.error = f"the starting file {blend} does not exist; this run is not isolated"
            else:
                staged = self.stage_scene(blend, model, task_index)
                outcome.blend_file = str(staged)
                outcome.scene_reset = await self.reset_scene(staged)

        agent = Agent(
            provider,
            resolved,
            self.mcp,
            bus=self.bus,
            model_info=self.llm.find(model.provider, resolved) or ModelInfo(id=resolved),
            system_prompt=self.system_prompt,
            budget=self._fresh_budget(),
        )
        started = time.perf_counter()
        try:
            result = await agent.run(task.prompt)
        except asyncio.CancelledError:
            outcome.status = "cancelled"
            outcome.duration_s = time.perf_counter() - started
            raise
        except StudioError as exc:
            outcome.status = "failed"
            outcome.error = exc.user_text()
            outcome.duration_s = time.perf_counter() - started
            return outcome
        outcome.duration_s = time.perf_counter() - started
        outcome.run_id = result.run_id
        outcome.totals = result.totals
        outcome.mcp_calls = result.totals.tool_calls
        outcome.tool_errors = result.totals.tool_errors
        outcome.status = "ok" if result.finished else "stopped"
        if result.stopped_because:
            outcome.error = result.stopped_because
        outcome.final_scene = await self.scene_snapshot()
        outcome.transcript = [
            {"role": message.role.value, "content": message.content[:2000]}
            for message in result.messages
            if message.role.value in ("user", "assistant")
        ]
        return outcome

    async def run_suite(
        self,
        tasks: Sequence[BenchmarkTask],
        models: Sequence[ModelSpec],
        *,
        blend: Path | None = None,
        persist: Any = None,
        suite_id: str = "",
    ) -> Comparison:
        """Every model against every task, in order, and one table at the end.

        Sequential by design: two models driving one Blender at the same time
        would interleave their tool calls, and the resulting scenes would belong
        to neither.
        """
        comparison = Comparison(tasks=list(tasks))
        for task_index, task in enumerate(tasks):
            for model in models:
                if self._cancelled:
                    break
                self.bus.emit(EventType.INFO, message=f"benchmark: {model.label()} on {task.label()}")
                outcome = await self.run_task_for_model(task, model, blend=blend, task_index=task_index)
                comparison.outcomes.append(outcome)
                if persist is not None:
                    await persist(comparison, outcome)
            if self._cancelled:
                break
        return comparison

    async def scene_snapshot(self) -> dict[str, Any]:
        """What the scene looks like now, for the record.

        Objects, transforms and the render engine. Not a screenshot: the studio
        is not a renderer, and ``blender.render_preview`` is available to whoever
        wants a picture.
        """
        outcome = await self.mcp.call_tool("blender.get_objects", {"limit": 200})
        if outcome.is_error:
            return {"error": outcome.error_code or outcome.text[:200]}
        try:
            payload = json.loads(outcome.text)
        except json.JSONDecodeError:
            return {"error": "unreadable object list"}
        return {
            "objects": [
                {
                    "name": obj.get("name"),
                    "type": obj.get("type"),
                    "location": obj.get("location"),
                    "dimensions": obj.get("dimensions"),
                }
                for obj in payload.get("objects", [])
            ],
            "total": payload.get("total", 0),
        }

    def _fresh_budget(self) -> Budget:
        """A copy per run.

        The budget carries counters, so sharing one would make the second model
        inherit the first model's step count and stop early.
        """
        budget = self.budget
        return Budget(
            max_steps=budget.max_steps,
            max_tool_calls=budget.max_tool_calls,
            max_seconds=budget.max_seconds,
            max_request_cost=budget.max_request_cost,
            max_session_cost=budget.max_session_cost,
            max_3d_credits=budget.max_3d_credits,
        )

    async def run_tasks_sequentially(self, tasks: Sequence[BenchmarkTask]) -> AsyncIterator[RunOutcome]:
        for task in tasks:
            yield await self.run_task_for_model(task, ModelSpec(provider="", model=""))


async def store_run(persist: Any, comparison: Comparison, outcome: RunOutcome) -> None:  # pragma: no cover
    """Kept as a seam for the benchmark panel; the real work is in its storage."""
    with contextlib.suppress(Exception):
        await persist(comparison, outcome)
