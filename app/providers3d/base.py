"""What every 3D provider must be able to do.

The studio knows one thing about 3D generation: it takes a while, costs credits,
and comes back as a file. Everything else — REST paths, task ids, polling
cadence, a credit system that freezes and then deducts — is the provider's
problem, and the agent must not have to care.

The interface is therefore four methods, and a capability question. Providers
answer :meth:`is_configured` honestly: a provider without a key is registered and
visible but not offered to the model, because a tool that always fails is worse
than an absent one.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.errors import NotImplementedCapability, ThreeDError
from app.providers3d.models import AssetRequest, AssetResult, ProviderTask, TaskStatus


class ThreeDProvider(ABC):
    """One 3D generation service."""

    #: Short, stable identifier used in configuration and in the database.
    name: str = ""
    #: Human-readable, for the Models panel.
    display_name: str = ""
    enabled: bool = False

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        *,
        model: str = "",
        quality: str = "medium",
        poll_interval: float = 5.0,
        poll_timeout: float = 900.0,
        **options: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.quality = quality
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.options = options
        self._client: Any = None

    # --- capability --------------------------------------------------------

    def is_configured(self) -> bool:
        """Whether this provider could run a task right now."""
        return bool(self.api_key)

    def supported_kinds(self) -> set[str]:
        return {"text_to_3d"}

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name or self.name,
            "enabled": self.enabled,
            "ready": self.is_configured(),
            "model": self.model,
            "kinds": sorted(self.supported_kinds()),
        }

    # --- the four operations ----------------------------------------------

    @abstractmethod
    async def create(self, request: AssetRequest, *, run_id: str = "") -> ProviderTask:
        """Submit a generation. Returns immediately with a task in flight."""

    @abstractmethod
    async def status(self, task: ProviderTask) -> ProviderTask:
        """Ask the provider about one task. Never blocks."""

    @abstractmethod
    async def download(self, task: ProviderTask, destination: Path) -> Path:
        """Fetch the finished model to a local file."""

    @abstractmethod
    async def close(self) -> None:
        """Release sockets."""

    # --- shared behaviour --------------------------------------------------

    async def wait_for(
        self,
        task: ProviderTask,
        *,
        on_progress: Callable[[ProviderTask], None] | None = None,
        interval: float | None = None,
        timeout: float | None = None,
    ) -> ProviderTask:
        """Poll until the task settles.

        Cancellable: the wait is a plain await on a sleep, so cancelling the task
        running it stops the polling immediately, which is what a user pressing
        Stop on a two-minute generation expects.
        """
        import asyncio

        # A floor, because a configured interval of zero would otherwise spin the
        # event loop as fast as it can go while a long task runs elsewhere.
        delay = max(0.05, interval if interval is not None else self.poll_interval)
        deadline = time.monotonic() + (timeout if timeout is not None else self.poll_timeout)
        while True:
            if task.status.terminal:
                return task
            if time.monotonic() > deadline:
                task.error = f"Timed out after {timeout or self.poll_timeout:g}s"
                return task
            task = await self.status(task)
            if on_progress is not None:
                on_progress(task)
            if task.status.terminal:
                return task
            await asyncio.sleep(delay)

    async def text_to_3d(self, request: AssetRequest, *, run_id: str = "") -> ProviderTask:
        await self._require("text_to_3d")
        return await self.create(request, run_id=run_id)

    async def image_to_3d(self, request: AssetRequest, *, run_id: str = "") -> ProviderTask:
        if "image_to_3d" not in self.supported_kinds():
            raise NotImplementedCapability(self.name or type(self).__name__, "image to 3D")
        return await self.create(request, run_id=run_id)

    async def _require(self, capability: str) -> bool:
        if capability not in self.supported_kinds():
            raise NotImplementedCapability(self.name or type(self).__name__, capability)
        return True

    # --- cost --------------------------------------------------------------

    def estimate_credits(self, request: AssetRequest) -> int:
        """What a task should cost, in this provider's credits.

        An estimate, and the GUI says so. It exists so a budget can stop a
        runaway plan before it is submitted, not to be an invoice.
        """
        return 0

    def estimate_usd(self, credits: int) -> float:
        """Credits in dollars, when the provider prices them.

        Zero by default rather than a guess: a wrong dollar figure in a cost
        report is worse than an honest blank.
        """
        return 0.0

    def require_key(self) -> str:
        if not self.api_key:
            raise ThreeDError(
                f"No API key for {self.name or type(self).__name__}",
                hint="Add it in Settings → Models.",
            )
        return self.api_key


def failed_task(task: ProviderTask, reason: str) -> ProviderTask:
    task.status = TaskStatus.FAILED
    task.error = reason
    task.updated_at = time.time()
    return task


def succeeded_task(task: ProviderTask, result: AssetResult) -> ProviderTask:
    task.status = TaskStatus.SUCCEEDED
    task.result = result
    task.progress = 1.0
    task.updated_at = time.time()
    return task
