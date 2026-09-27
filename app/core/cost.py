"""Money, in two currencies that must not be added together.

A token bill and a 3D credit are both "cost" and neither is the other. A
benchmark that sums them produces a number that means nothing, so they are
tracked separately from the first line of code and reported separately.

The tracker is also the budget. Every limit in :class:`~app.core.settings.AgentConfig`
is checked here, and the check raises rather than warns: a runaway loop has to stop
at a number the user chose, not at an amount they find out later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.errors import BudgetExceeded
from app.llm.base import Usage
from app.llm.models import ModelInfo, cost_of


@dataclass(slots=True)
class CostTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    llm_usd: float = 0.0
    three_d_credits: float = 0.0
    three_d_usd: float = 0.0
    requests: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_usd(self) -> float:
        """Only meaningful as a sum when both are real money.

        Kept as a property with the caveat in its name, because a single "cost"
        column is what a user will look for first.
        """
        return self.llm_usd + self.three_d_usd

    def to_dict(self) -> dict[str, float | int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "tokens": self.tokens,
            "llm_usd": round(self.llm_usd, 6),
            "three_d_credits": self.three_d_credits,
            "three_d_usd": round(self.three_d_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
        }


@dataclass
class Budget:
    """The limits, and the state they are checked against."""

    max_steps: int = 30
    max_tool_calls: int = 50
    max_seconds: float = 900.0
    max_request_cost: float = 2.0
    max_session_cost: float = 5.0
    max_3d_credits: int = 200

    steps: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.steps = 0
        self.started_at = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check(self, totals: CostTotals) -> None:
        """Raise when a limit is reached. Called before every step."""
        if self.steps >= self.max_steps:
            raise BudgetExceeded("agent steps", self.steps, self.max_steps)
        self.check_call(totals)

    def check_call(self, totals: CostTotals) -> None:
        """The limits that apply between two tool calls in one step.

        Separate from :meth:`check` because the step counter is incremented as
        soon as the model has answered: re-checking it here would refuse the
        tools of a step that had already been counted, which is a limit of zero
        tools per turn wearing a hat.
        """
        if totals.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("tool calls", totals.tool_calls, self.max_tool_calls)
        if self.elapsed() > self.max_seconds:
            raise BudgetExceeded("wall-clock seconds", self.elapsed(), self.max_seconds)
        if totals.llm_usd > self.max_session_cost:
            raise BudgetExceeded("session LLM cost", totals.llm_usd, self.max_session_cost)
        if totals.three_d_credits > self.max_3d_credits:
            raise BudgetExceeded("3D credits", totals.three_d_credits, self.max_3d_credits)

    def check_request(self, cost: float) -> None:
        if cost > self.max_request_cost:
            raise BudgetExceeded("single request cost", cost, self.max_request_cost)


class CostTracker:
    """Accumulates usage for one run and prices it from the model catalog."""

    def __init__(self, model: ModelInfo | None = None) -> None:
        self.model = model
        self.totals = CostTotals()

    def price(self, model: ModelInfo | None = None) -> ModelInfo | None:
        return model or self.model

    def add_usage(self, usage: Usage) -> float:
        """Record one request. Returns what it cost."""
        model = self.price()
        cost = cost_of(model, usage.input_tokens, usage.output_tokens, cached_tokens=usage.cached_tokens) if model else 0.0
        if cost == 0.0 and usage.cost_usd:
            # A provider that knows its own pricing better than the catalog did.
            cost = usage.cost_usd
        self.totals.input_tokens += usage.input_tokens
        self.totals.output_tokens += usage.output_tokens
        self.totals.cached_tokens += usage.cached_tokens
        self.totals.llm_usd += cost
        self.totals.requests += 1
        return cost

    def add_tool_call(self, is_error: bool = False) -> None:
        self.totals.tool_calls += 1
        if is_error:
            self.totals.tool_errors += 1

    def add_3d(self, credits: float = 0.0, usd: float = 0.0) -> None:
        self.totals.three_d_credits += credits
        self.totals.three_d_usd += usd

    def set_model(self, model: ModelInfo | None) -> None:
        self.model = model
