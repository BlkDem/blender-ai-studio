"""3D providers: request construction, parsing, polling, cost, errors.

Tripo is tested against a mock transport with the API's own documented response
shapes, because the interesting failures here are translation failures — a status
the studio maps wrong, an envelope it forgets to unwrap, a credit figure it
invents.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import ConfigurationError, NotImplementedCapability, ThreeDError
from app.providers3d.mock import MockThreeDProvider
from app.providers3d.models import AssetRequest, ProviderTask, TaskStatus
from app.providers3d.registry import ThreeDRegistry
from app.providers3d.tripo import TripoProvider
from app.storage.secrets import SecretStore


def tripo_with(handler, **kwargs) -> TripoProvider:
    """A provider whose HTTP goes to ``handler``, through its own client."""
    return TripoProvider(api_key="tsk-test", transport=httpx.MockTransport(handler), **kwargs)


# --- creating a task --------------------------------------------------------


async def test_the_request_is_built_the_way_the_api_documents() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "task_abc"}})

    task = await tripo_with(handler).create(AssetRequest(prompt="a medieval wooden chest"))

    assert seen["path"] == "/v3/generation/text-to-model"
    assert seen["auth"] == "Bearer tsk-test"
    assert seen["body"]["prompt"] == "a medieval wooden chest"
    assert seen["body"]["texture"] is True
    assert seen["body"]["pbr"] is True
    assert task.provider_task_id == "task_abc"
    assert task.status is TaskStatus.RUNNING


async def test_an_image_request_goes_to_the_image_endpoint() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "t"}})

    await tripo_with(handler).create(
        AssetRequest(prompt="a chest", image_url="https://example.com/chest.png")
    )
    assert seen["path"] == "/v3/generation/image-to-model"


async def test_a_non_zero_code_is_an_error_even_with_http_200() -> None:
    """Tripo's own convention. Ignoring it is how a failed submission looks like
    a task that is quietly running forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 1001, "msg": "insufficient credits"})

    task = await tripo_with(handler).create(AssetRequest(prompt="chest"))
    assert task.status is TaskStatus.FAILED
    assert "insufficient credits" in task.error


async def test_an_http_error_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": 401, "msg": "bad key"})

    task = await tripo_with(handler).create(AssetRequest(prompt="chest"))
    assert task.status is TaskStatus.FAILED
    assert "401" in task.error


async def test_an_empty_account_says_so_rather_than_looking_like_a_bug() -> None:
    """Tripo answers 403 'not enough credit' to a perfectly valid request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"code": 4003, "message": "You don't have enough credit to create this task"}
        )

    task = await tripo_with(handler).create(AssetRequest(prompt="chest"))
    assert task.status is TaskStatus.FAILED
    assert "enough credit" in task.error


async def test_a_network_failure_does_not_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    task = await tripo_with(handler).create(AssetRequest(prompt="chest"))
    assert task.status is TaskStatus.FAILED
    assert "Could not reach Tripo" in task.error


async def test_an_empty_prompt_is_refused_before_spending_credits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("a request was submitted for an empty prompt")

    task = await tripo_with(handler).create(AssetRequest(prompt="   "))
    assert task.status is TaskStatus.FAILED
    assert "prompt" in task.error.lower()


# --- reading status ---------------------------------------------------------


def _task_response(status: str, **extra) -> httpx.Response:
    data = {"task_id": "task_abc", "status": status, "progress": extra.pop("progress", 0)}
    data.update(extra)
    return httpx.Response(200, json={"code": 0, "data": data})


async def test_a_running_task_reports_progress_as_a_fraction() -> None:
    provider = tripo_with(lambda request: _task_response("running", progress=42))
    task = await provider.status(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    )
    assert task.status is TaskStatus.RUNNING
    assert task.progress == pytest.approx(0.42)


async def test_a_successful_task_becomes_an_asset() -> None:
    body = {
        "code": 0,
        "data": {
            "task_id": "task_abc",
            "status": "success",
            "progress": 100,
            "credits_consumed": 130.0,
            "output": {
                "model_url": "https://cdn.tripo3d.ai/output/model_pbr.glb",
                "rendered_image_url": "https://cdn.tripo3d.ai/output/preview.png",
            },
        },
    }
    provider = tripo_with(lambda request: httpx.Response(200, json=body))
    task = await provider.status(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    )
    assert task.status is TaskStatus.SUCCEEDED
    assert task.progress == 1.0
    assert task.credits == 130
    assert task.result is not None
    assert task.result.url.endswith("model_pbr.glb")
    assert task.result.format == "glb"
    assert "rendered_image_url" in task.result.extras, "the preview is worth keeping"


async def test_success_without_a_url_is_a_failure() -> None:
    provider = tripo_with(lambda request: _task_response("success", progress=100))
    task = await provider.status(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    )
    assert task.status is TaskStatus.FAILED
    assert "no model URL" in task.error


async def test_a_banned_task_explains_that_the_prompt_was_refused() -> None:
    """``banned`` is the content policy, which is the user's prompt to fix — not
    a transient failure to retry."""
    provider = tripo_with(lambda request: _task_response("banned", error="policy violation"))
    task = await provider.status(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    )
    assert task.status is TaskStatus.FAILED
    assert "content policy" in task.error


async def test_a_failed_poll_is_not_a_failed_task() -> None:
    """One unreachable poll must not lose a task that is still running."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("blip")
        return _task_response("running", progress=10)

    provider = tripo_with(handler)
    task = ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    first = await provider.status(task)
    assert first.status is not TaskStatus.FAILED
    second = await provider.status(first)
    assert second.progress == pytest.approx(0.1)


async def test_an_unknown_status_leaves_the_task_alone() -> None:
    """Guessing 'failed' on a state the studio has not seen would hide a task
    that is perfectly fine."""
    provider = tripo_with(lambda request: _task_response("hibernating"))
    task = await provider.status(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="task_abc")
    )
    assert task.status is not TaskStatus.FAILED


async def test_polling_stops_when_the_task_settles() -> None:
    states = ["running", "running", "success"]
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        polls["n"] += 1
        status = states.pop(0) if states else "success"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "task_id": "t",
                    "status": status,
                    "progress": 100,
                    "output": {"model_url": "https://cdn/model.glb"},
                },
            },
        )

    provider = tripo_with(handler, poll_interval=0.0)
    task = await provider.wait_for(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="t"), timeout=5
    )
    assert task.status is TaskStatus.SUCCEEDED
    assert polls["n"] == 3, "polling stopped as soon as the task settled"


async def test_polling_gives_up_at_the_timeout() -> None:
    provider = tripo_with(
        lambda request: _task_response("running", progress=5), poll_interval=0.01, poll_timeout=0.05
    )
    task = await provider.wait_for(ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="t"))
    assert "Timed out" in task.error


async def test_polling_reports_progress_to_the_caller() -> None:
    seen: list[float] = []
    states = ["running", "success"]
    body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body["data"] = {
            "task_id": "t",
            "status": states.pop(0),
            "progress": 80,
            "output": {"model_url": "https://cdn/model.glb"},
        }
        return httpx.Response(200, json={"code": 0, **body})

    provider = tripo_with(handler, poll_interval=0.0)
    task = await provider.wait_for(
        ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="t"),
        on_progress=lambda t: seen.append(t.progress),
        timeout=5,
    )
    assert seen and seen[0] == pytest.approx(0.8)
    assert task.status is TaskStatus.SUCCEEDED


# --- downloading ------------------------------------------------------------


async def test_an_asset_is_written_where_it_was_asked_for(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"glTF-binary")

    provider = tripo_with(handler)
    task = ProviderTask.new("tripo", AssetRequest(prompt="x"), provider_task_id="t")
    task.status = TaskStatus.SUCCEEDED
    task.result = __import__("app.providers3d.models", fromlist=["AssetResult"]).AssetResult(
        url="https://cdn/model.glb"
    )
    destination = tmp_path / "assets" / "chest.glb"
    written = await provider.download(task, destination)
    assert written.read_bytes() == b"glTF-binary"


async def test_downloading_before_the_asset_exists_is_refused(tmp_path: Path) -> None:
    provider = tripo_with(lambda request: httpx.Response(200))
    task = ProviderTask.new("tripo", AssetRequest(prompt="x"))
    with pytest.raises(ThreeDError) as excinfo:
        await provider.download(task, tmp_path / "out.glb")
    assert "Tasks panel" in (excinfo.value.hint or "")


# --- cost -------------------------------------------------------------------


def test_the_estimate_follows_the_documented_credit_rules() -> None:
    provider = TripoProvider(api_key="k")
    assert provider.estimate_credits(AssetRequest(prompt="x")) == 100
    assert provider.estimate_credits(AssetRequest(prompt="x", texture=False)) == 90
    assert provider.estimate_credits(AssetRequest(prompt="x", quality="high")) == 130
    assert provider.estimate_credits(AssetRequest(prompt="x", quality="high", texture=False)) == 120
    assert provider.estimate_credits(AssetRequest(prompt="x", options={"smart_low_poly": True})) == 110


def test_credits_are_not_turned_into_a_made_up_dollar_figure() -> None:
    """Tripo sells credits and publishes no per-credit price in its API docs. A
    fabricated number in a cost report is worse than a blank."""
    assert TripoProvider(api_key="k").estimate_usd(100) == 0.0


# --- the registry -----------------------------------------------------------


def test_a_provider_without_a_key_is_registered_but_not_offered() -> None:
    registry = ThreeDRegistry()
    registry.add(TripoProvider(api_key=""))
    assert registry.all(), "the Models panel should still show it"
    assert registry.any_enabled() is False, "the model must not be offered a tool that always fails"


def test_a_keyed_provider_is_offered() -> None:
    registry = ThreeDRegistry()
    registry.add(TripoProvider(api_key="tsk-1"))
    assert registry.any_enabled() is True


def test_a_key_comes_from_the_secret_store(secrets_file: Path) -> None:
    secrets = SecretStore(secrets_file)
    secrets.set("tripo.api_key", "tsk-secret")
    registry = ThreeDRegistry()
    registry.use_secrets(secrets)
    assert registry.provider("tripo").api_key == "tsk-secret"


def test_wiring_the_secrets_rebuilds_a_provider_built_without_a_key(secrets_file: Path) -> None:
    """The key was typed, so the capability must appear.

    A registry that cleared its providers and stopped there left a provider
    holding an empty key, and generate_3d_asset stayed hidden from the model for
    ever -- a stored key and a working key looked identical from the outside.
    """
    registry = ThreeDRegistry()
    assert registry.any_enabled() is False
    before = registry.provider("tripo")
    assert before.is_configured() is False

    secrets = SecretStore(secrets_file)
    secrets.set("tripo.api_key", "tsk-secret")
    registry.use_secrets(secrets)

    assert registry.any_enabled() is True, "the capability must be offered once a key exists"
    assert registry.provider("tripo").api_key == "tsk-secret"


def test_an_unknown_provider_says_which_are_known() -> None:
    registry = ThreeDRegistry()
    with pytest.raises(ConfigurationError) as excinfo:
        registry.provider("meshy")
    assert "tripo" in (excinfo.value.hint or "")


def test_a_provider_without_the_capability_says_so() -> None:
    class TextOnly(MockThreeDProvider):
        def supported_kinds(self) -> set[str]:
            return {"text_to_3d"}

    import asyncio

    with pytest.raises(NotImplementedCapability) as excinfo:
        asyncio.run(TextOnly().image_to_3d(AssetRequest(prompt="x", image_url="https://i/x.png")))
    assert "image to 3D" in excinfo.value.message


# --- the mock ---------------------------------------------------------------


async def test_the_mock_provider_produces_a_task_and_credits() -> None:
    provider = MockThreeDProvider(credits=50, usd_per_credit=0.01)
    task = await provider.create(AssetRequest(prompt="a chest"))
    assert task.provider_task_id == "mock-1"
    assert task.credits == 50
    assert task.cost_usd == pytest.approx(0.5)

    done = await provider.wait_for(task, interval=0.0)
    assert done.status is TaskStatus.SUCCEEDED
    assert done.result is not None and done.result.url.endswith(".glb")


async def test_the_mock_provider_can_fail_on_purpose() -> None:
    provider = MockThreeDProvider(fail_with="the provider is out of credits")
    task = await provider.create(AssetRequest(prompt="x"))
    assert task.status is TaskStatus.FAILED
    assert "out of credits" in task.error
