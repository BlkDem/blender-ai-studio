"""A scripted 3D provider.

Mirrors :class:`~app.llm.providers.mock.MockLLMProvider`: the agent's 3D path can
be tested without a Tripo account, a wait and a credit. Scripted rather than
random, so a benchmark can compare the same provider twice.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from app.providers3d.base import ThreeDProvider, failed_task, succeeded_task
from app.providers3d.models import AssetRequest, AssetResult, ProviderTask, TaskStatus


class MockThreeDProvider(ThreeDProvider):
    """Fakes a generation, with a controllable number of polls."""

    name = "mock-3d"
    display_name = "Mock 3D provider"

    def __init__(
        self,
        *,
        credits: int = 100,
        polls_before_done: int = 1,
        fail_with: str = "",
        usd_per_credit: float = 0.0,
        **options: Any,
    ) -> None:
        super().__init__(api_key="mock", base_url="mock://", model="mock-model", **options)
        self.enabled = True
        self.credits = credits
        self.polls_before_done = polls_before_done
        self.fail_with = fail_with
        self.usd_per_credit = usd_per_credit
        #: What was asked, so a test can assert on the request too.
        self.requests: list[AssetRequest] = []
        self.polls = 0
        self._pending = 0

    def supported_kinds(self) -> set[str]:
        return {"text_to_3d", "image_to_3d"}

    async def create(self, request: AssetRequest, *, run_id: str = "") -> ProviderTask:
        self.requests.append(request)
        if self.fail_with:
            return failed_task(ProviderTask.new(self.name, request, run_id=run_id), self.fail_with)
        self._pending = self.polls_before_done
        task = ProviderTask.new(self.name, request, run_id=run_id)
        task.provider_task_id = f"mock-{len(self.requests)}"
        task.status = TaskStatus.RUNNING
        task.credits = self.estimate_credits(request)
        task.cost_usd = self.estimate_usd(task.credits)
        return task

    async def status(self, task: ProviderTask) -> ProviderTask:
        self.polls += 1
        task.updated_at = time.time()
        if self._pending > 0:
            self._pending -= 1
            task.status = TaskStatus.RUNNING
            task.progress = min(0.9, 0.3 + 0.2 * (self.polls_before_done - self._pending))
            return task
        return succeeded_task(
            task,
            AssetResult(
                url=f"mock://assets/{task.provider_task_id}.glb",
                format="glb",
                credits=task.credits,
                cost_usd=task.cost_usd,
            ),
        )

    async def download(self, task: ProviderTask, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"glTF-fixture")
        return destination

    def estimate_credits(self, request: AssetRequest) -> int:
        return self.credits

    def estimate_usd(self, credits: int) -> float:
        return credits * self.usd_per_credit

    async def close(self) -> None:
        return None

    async def wait_slowly(self, task: ProviderTask, seconds: float = 0.01) -> ProviderTask:
        """A wait that yields, so cancellation has something to interrupt."""
        await asyncio.sleep(seconds)
        return await self.status(task)
