"""What a 3D provider is asked for, and what it gives back.

Modelled on a job, not a function. Text-to-3D is minutes of a provider's compute
and comes back as a URL, so the vocabulary here is a task with a status, a cost in
credits, and a download — and an import step that belongs to Blender rather than
to the provider.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)


class AssetKind(StrEnum):
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"


@dataclass
class AssetRequest:
    """What to generate."""

    prompt: str
    image_url: str | None = None
    model: str = ""
    quality: str = "medium"
    texture: bool = True
    #: Provider-specific extras, passed through untouched. This is the escape
    #: hatch that lets a new provider option be used before this class grows a
    #: field for it.
    options: dict[str, Any] = field(default_factory=dict)

    def kind(self) -> AssetKind:
        return AssetKind.IMAGE_TO_3D if self.image_url else AssetKind.TEXT_TO_3D

    def validate(self) -> None:
        from app.core.errors import ThreeDError

        if not self.prompt.strip():
            raise ThreeDError("A 3D generation request needs a prompt")
        if len(self.prompt) > 2000:
            raise ThreeDError(
                "That prompt is too long for a 3D provider",
                hint="Describe the object in a sentence or two.",
            )


@dataclass
class AssetResult:
    """What came back."""

    url: str = ""
    format: str = "glb"
    #: What the provider says it cost, in its own credits.
    credits: int = 0
    cost_usd: float = 0.0
    #: Assets Tripo returns, for a human who wants to know what else exists.
    extras: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.url


@dataclass
class ProviderTask:
    """One generation, from the studio's point of view.

    ``id`` is the studio's; ``provider_task_id`` is the provider's. Keeping both
    means a task can be found again after a restart without depending on one
    provider's id format.
    """

    id: str
    provider: str
    kind: str
    status: TaskStatus = TaskStatus.QUEUED
    provider_task_id: str = ""
    prompt: str = ""
    model: str = ""
    progress: float = 0.0
    message: str = ""
    credits: int = 0
    cost_usd: float = 0.0
    result: AssetResult | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: str = ""

    @classmethod
    def new(
        cls, provider: str, request: AssetRequest, *, run_id: str = "", provider_task_id: str = ""
    ) -> ProviderTask:
        return cls(
            id=f"asset_{uuid.uuid4().hex[:12]}",
            provider=provider,
            kind=str(request.kind()),
            provider_task_id=provider_task_id,
            prompt=request.prompt,
            model=request.model,
            run_id=run_id,
            status=TaskStatus.RUNNING if provider_task_id else TaskStatus.QUEUED,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "kind": self.kind,
            "status": str(self.status),
            "provider_task_id": self.provider_task_id,
            "prompt": self.prompt,
            "model": self.model,
            "progress": self.progress,
            "credits": self.credits,
            "cost_usd": self.cost_usd,
            "url": self.result.url if self.result else "",
            "error": self.error,
        }
