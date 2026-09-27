"""The types the benchmark panel and the CLI share."""

from __future__ import annotations

import json
from typing import Any

from app.benchmark.runner import (
    RESET_COPY_ONLY,
    RESET_UNVERIFIED,
    RESET_VERIFIED,
    BenchmarkTask,
    Comparison,
    ModelSpec,
    RunOutcome,
)

__all__ = [
    "BenchmarkTask",
    "Comparison",
    "ModelSpec",
    "RESET_COPY_ONLY",
    "RESET_UNVERIFIED",
    "RESET_VERIFIED",
    "RunOutcome",
    "load_suite",
    "save_suite",
]


def save_suite(path: Any, tasks: list[BenchmarkTask], models: list[ModelSpec], name: str) -> None:
    """A suite on disk, so a comparison can be repeated by someone else.

    Plain JSON, versioned, and holding a model by *name* rather than by id:
    ids are deployment-specific, and a suite that cannot be run again is a
    document rather than a benchmark.
    """
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": name,
                "tasks": [{"name": t.name, "prompt": t.prompt} for t in tasks],
                "models": [{"provider": m.provider, "model": m.model} for m in models],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_suite(path: Any) -> tuple[str, list[BenchmarkTask], list[ModelSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = [BenchmarkTask(prompt=item["prompt"], name=item.get("name", "")) for item in payload["tasks"]]
    models = [ModelSpec(provider=item["provider"], model=item.get("model", "")) for item in payload["models"]]
    return payload.get("name", path.stem), tasks, models
