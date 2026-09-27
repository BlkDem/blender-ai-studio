"""Tripo.

The real v3 REST API, as documented: submit a generation, get a task id, poll for
the result, download the GLB. Nothing about the agent or the GUI knows that this
provider uses a ``code``/``data`` envelope or calls its states ``success`` and
``banned`` — the translation to the studio's vocabulary happens here.

Two things are worth knowing about the API and are handled explicitly:

* a status of ``banned`` means the *prompt* violated the content policy, which is
  the user's prompt to fix, not a transient failure;
* credits are frozen at submission and only deducted on success, so a failed task
  costs nothing — which is why the estimate is an estimate and the truth comes
  from ``credits_consumed``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.errors import ProviderError, ThreeDError
from app.providers3d.base import ThreeDProvider, failed_task
from app.providers3d.models import AssetRequest, AssetResult, ProviderTask, TaskStatus

logger = logging.getLogger(__name__)

API_ROOT = "https://openapi.tripo3d.ai/v3"


def _format_of(url: str) -> str:
    """The file extension in a URL, ignoring the query string.

    Tripo hands back a presigned link, and ``Path(url).suffix`` on one of those
    is ``.glb?Policy=...&Signature=...`` -- a format nobody can name a file
    after.
    """
    suffix = Path(url.split("?", 1)[0]).suffix
    return suffix.lstrip(".").lower() or "glb"


DEFAULT_MODEL = "v3.1-20260211"

#: Provider status -> the studio's. Anything unknown is left as running, because
#: guessing "failed" on an unrecognised state would hide a task that is fine.
_STATUS_MAP = {
    "queued": TaskStatus.QUEUED,
    "running": TaskStatus.RUNNING,
    "success": TaskStatus.SUCCEEDED,
    "failed": TaskStatus.FAILED,
    "banned": TaskStatus.FAILED,
    "expired": TaskStatus.FAILED,
    "cancelled": TaskStatus.CANCELLED,
}

#: Credit deltas the API documents. Used only to refuse a plan that is obviously
#: over budget before it is submitted; the real figure comes back with the task.
_CREDIT_BASE = 100
_CREDIT_NO_TEXTURE = -10
_CREDIT_DETAILED_TEXTURE = 10
_CREDIT_DETAILED_GEOMETRY = 20
_CREDIT_LOW_POLY = 10
_CREDIT_PARTS = 20

_QUALITY_GEOMETRY = {"draft": "standard", "medium": "standard", "high": "detailed"}
_QUALITY_TEXTURE = {"draft": "standard", "medium": "standard", "high": "detailed"}


class TripoProvider(ThreeDProvider):
    name = "tripo"
    display_name = "Tripo"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = API_ROOT,
        *,
        model: str = DEFAULT_MODEL,
        quality: str = "medium",
        poll_interval: float = 3.0,
        poll_timeout: float = 600.0,
        transport: Any = None,
        **options: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            quality=quality,
            # The API recommends 1-2 s. Three is a compromise: fast enough to feel
            # live, gentle enough not to be the reason a task is rate limited.
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
            **options,
        )
        self.enabled = True
        #: Injected by the tests, so they exercise the real client -- its headers
        #: and its base URL -- rather than a hand-built one that agrees with them.
        self._transport = transport

    def supported_kinds(self) -> set[str]:
        return {"text_to_3d", "image_to_3d"}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.require_key()}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, read=120.0),
                transport=self._transport,
            )
        return self._client

    # --- requests ----------------------------------------------------------

    def _generation_body(self, request: AssetRequest) -> dict[str, Any]:
        quality = request.quality or self.quality
        body: dict[str, Any] = {
            "prompt": request.prompt,
            "model": request.model or self.model or DEFAULT_MODEL,
            "texture": request.texture,
            "pbr": request.texture,
            "texture_quality": _QUALITY_TEXTURE.get(quality, "standard"),
            "geometry_quality": _QUALITY_GEOMETRY.get(quality, "standard"),
        }
        if request.image_url:
            body["image_url"] = request.image_url
        for key in (
            "smart_low_poly",
            "quad",
            "face_limit",
            "generate_parts",
            "export_uv",
            "texture",
            "pbr",
            "orientation",
            "style",
        ):
            if key in request.options:
                body[key] = request.options[key]
        return body

    async def create(self, request: AssetRequest, *, run_id: str = "") -> ProviderTask:
        task = ProviderTask.new(self.name, request, run_id=run_id)
        try:
            request.validate()
        except ThreeDError as exc:
            # The prompt comes from a model, so a bad one is data to report back,
            # not an exception to propagate into the agent loop.
            return failed_task(task, exc.message)
        endpoint = "/generation/image-to-model" if request.image_url else "/generation/text-to-model"
        try:
            response = await self._http().post(endpoint, json=self._generation_body(request))
        except httpx.HTTPError as exc:
            return failed_task(task, f"Could not reach Tripo: {exc}")
        try:
            payload = _envelope(response, self.name)
        except ProviderError as exc:
            # A refusal from the API is the task's outcome, not an exception in the
            # agent's loop: the model asked for an asset and needs to be told no.
            return failed_task(task, exc.message)
        task.provider_task_id = str(payload.get("task_id") or "")
        if not task.provider_task_id:
            return failed_task(task, "Tripo accepted the request but returned no task id")
        task.status = TaskStatus.RUNNING
        task.credits = self.estimate_credits(request)
        task.cost_usd = self.estimate_usd(task.credits)
        task.updated_at = time.time()
        logger.info("Tripo task %s created for %r", task.provider_task_id, request.prompt[:40])
        return task

    async def status(self, task: ProviderTask) -> ProviderTask:
        if not task.provider_task_id:
            return failed_task(task, "This task has no provider id, so it cannot be polled")
        try:
            response = await self._http().get(f"/tasks/{task.provider_task_id}")
        except httpx.HTTPError as exc:
            # One failed poll is not a failed task.
            task.error = f"Could not reach Tripo: {exc}"
            task.updated_at = time.time()
            return task
        try:
            payload = _envelope(response, self.name)
        except ProviderError as exc:
            # Left running: a refusal on one poll says nothing about the task, and
            # the next poll may well succeed. The error travels with it.
            task.error = exc.message
            task.updated_at = time.time()
            return task
        # _envelope has already unwrapped "data".
        return self._apply(task, payload)

    def _apply(self, task: ProviderTask, data: dict[str, Any]) -> ProviderTask:
        raw_status = str(data.get("status") or "").lower()
        task.status = _STATUS_MAP.get(raw_status, task.status)
        progress = data.get("progress")
        if isinstance(progress, (int, float)):
            # The API sends 0-100; the studio works in 0-1.
            task.progress = max(0.0, min(1.0, float(progress) / 100.0))
        consumed = data.get("credits_consumed")
        if isinstance(consumed, (int, float)):
            task.credits = int(consumed)
            task.cost_usd = self.estimate_usd(task.credits)
        task.updated_at = time.time()

        if task.status is TaskStatus.SUCCEEDED:
            output = data.get("output") or {}
            url = str(output.get("model_url") or output.get("pbr_model_url") or output.get("model") or "")
            if not url:
                return failed_task(task, "Tripo reported success but returned no model URL")
            task.result = AssetResult(
                url=url,
                # The download URL is a presigned S3 link: everything after "?"
                # is signature, not filename, so the suffix has to be taken from
                # the path alone. Reading the whole string yields a "format" of
                # "glb?policy=eyj..." and a file named after a signature.
                format=_format_of(url),
                credits=task.credits,
                cost_usd=task.cost_usd,
                extras={k: v for k, v in output.items() if k != "model_url"},
            )
        elif task.status is TaskStatus.FAILED:
            reason = str(data.get("error") or data.get("message") or raw_status or "unknown error")
            if raw_status == "banned":
                # The content policy refused the prompt: that is the user's to fix.
                task.error = f"Tripo refused this prompt under its content policy ({reason})"
            else:
                task.error = reason
        return task

    async def download(self, task: ProviderTask, destination: Path) -> Path:
        if task.result is None or task.result.is_empty():
            raise ThreeDError(
                "This task has no asset to download yet",
                hint="Wait for the generation to finish in the Tasks panel.",
                task_id=task.id,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with self._http().stream("GET", task.result.url) as response:
                if response.status_code >= 400:
                    raise ThreeDError(
                        f"Tripo's file server refused the download (HTTP {response.status_code})",
                        hint="The asset may have expired; regenerate it.",
                    )
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise ThreeDError(f"Could not download the asset: {exc}") from exc
        logger.info("Tripo asset saved to %s", destination)
        return destination

    # --- cost --------------------------------------------------------------

    def estimate_credits(self, request: AssetRequest) -> int:
        quality = request.quality or self.quality
        credits = _CREDIT_BASE
        if not request.texture:
            credits += _CREDIT_NO_TEXTURE
        if _QUALITY_TEXTURE.get(quality) == "detailed":
            credits += _CREDIT_DETAILED_TEXTURE
        if _QUALITY_GEOMETRY.get(quality) == "detailed":
            credits += _CREDIT_DETAILED_GEOMETRY
        if request.options.get("smart_low_poly"):
            credits += _CREDIT_LOW_POLY
        if request.options.get("generate_parts"):
            credits += _CREDIT_PARTS
        return max(0, credits)

    def estimate_usd(self, credits: int) -> float:
        """Tripo sells credits, not dollars, and publishes no per-credit price in
        the API docs. Left at zero: a fabricated dollar figure in a cost report is
        worse than an honest blank, and the credits column is the real number."""
        return 0.0

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def _envelope(response: httpx.Response, provider: str) -> dict[str, Any]:
    """Unwrap Tripo's ``{"code": 0, "data": {...}}`` envelope.

    A non-zero ``code`` is an error even with HTTP 200, which is the API's own
    convention and a classic source of silent failures if ignored.
    """
    if response.status_code >= 400:
        raise ProviderError(
            f"Tripo returned HTTP {response.status_code}: {_message(response)}",
            provider=provider,
            hint=_billing_hint(_message(response)),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("Tripo did not return JSON", provider=provider) from exc
    if not isinstance(payload, dict):
        return {}
    code = payload.get("code")
    if isinstance(code, int) and code != 0:
        # Tripo's errors carry a "suggestion" with what to do about them, which is
        # the most useful part of the response; pass it through as the hint.
        suggestion = payload.get("suggestion")
        raise ProviderError(
            f"Tripo refused the request: {_message(response)}",
            provider=provider,
            hint=str(suggestion) if suggestion else None,
            code=code,
        )
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _billing_hint(message: str) -> str | None:
    """What to do about the refusals that are not the code's fault.

    "Not enough credit" arrives as a 403 with a perfectly valid request behind
    it, and without a hint it reads as a bug in the client rather than an empty
    account.
    """
    lowered = message.lower()
    if "credit" in lowered or "balance" in lowered or "quota" in lowered:
        return (
            "The request was built and accepted; the account has no credit for it. "
            "Top up the Tripo balance, or use a cheaper quality."
        )
    if "api key" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        return "Check the Tripo API key in Settings → Models."
    return None


def _message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("msg") or payload.get("message") or payload.get("error") or payload)[:200]
    return str(payload)[:200]
