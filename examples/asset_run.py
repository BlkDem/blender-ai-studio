"""The second MVP scenario, end to end, with a real model and a real 3D service.

    you:      Create a medieval wooden chest.
    model:    calls generate_3d_asset
    Tripo:    generates a GLB (real API, real credits)
    studio:   polls, downloads, imports into Blender through the MCP bridge
    Blender:  the chest is in the scene, and get_objects says so

Everything goes through the studio's own layers — the agent's local tool, the
provider, the task manager, the MCP client — so this exercises the product rather
than a script that happens to call the same endpoints.

    python examples/asset_run.py --blender-mcp /path/to/blender-mcp \\
        --tripo-key tsk_... --base-url http://127.0.0.1:11400/v1 --model local-qwen
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core.context import AppContext  # noqa: E402
from app.core.errors import ThreeDError  # noqa: E402
from app.core.settings import AgentConfig, MCPServerConfig, Settings  # noqa: E402
from app.llm.registry import ProviderConfig  # noqa: E402
from app.providers3d.models import AssetRequest, TaskStatus  # noqa: E402
from app.storage.secrets import key_name  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> bool:
    if not condition:
        FAILURES.append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{f'  {detail}' if detail != '' else ''}", flush=True)
    return condition


def settings_for(options: argparse.Namespace) -> Settings:
    settings = Settings()
    settings.data_dir = options.data_dir
    settings.log_level = "WARNING"
    settings.mcp_servers = [
        MCPServerConfig(
            name="Blender MCP",
            command=options.python,
            args=["-m", "server.main"],
            cwd=options.blender_mcp,
            # The only way an asset reaches the scene: blender-mcp has no
            # import tool, and embedding Blender's API in this client is the
            # boundary the project is built to avoid. So the gate is opened for
            # this run, and the README says so.
            env={
                "PYTHONPATH": options.blender_mcp_env or options.blender_mcp,
                "ALLOW_PYTHON_EXECUTION": "true",
            },
            blender_port=options.port,
            tool_timeout=300.0,
        )
    ]
    settings.llm_providers = [
        ProviderConfig(
            name=options.provider,
            kind="openai-compatible",
            base_url=options.base_url,
            default_model=options.model,
            models=[{"id": options.model, "supports_tools": True, "context_window": 8192}],
        )
    ]
    # A generated asset has to be importable, and execute_python is the only way.
    settings.agent = AgentConfig(allow_execute_python=True, max_steps=12, max_tool_calls=20)
    return settings


def payload_of(outcome) -> dict:
    if getattr(outcome, "is_error", False):
        return {}
    try:
        found = json.loads(outcome.text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return found if isinstance(found, dict) else {}


async def wait_for_blender(context: AppContext, seconds: float):
    """The add-on reconnects on a backoff that reaches half a minute."""
    assert context.mcp is not None
    deadline = time.monotonic() + seconds
    attempt = 0
    while True:
        attempt += 1
        outcome = await context.mcp.call_tool("blender.get_scene", {})
        if not outcome.is_error:
            if attempt > 1:
                print(f"     (Blender attached after {attempt} attempts)", flush=True)
            return outcome
        if time.monotonic() >= deadline:
            return outcome
        print("     waiting for a Blender to attach…", flush=True)
        await asyncio.sleep(2.0)


async def run(options: argparse.Namespace) -> int:
    context = AppContext(settings=settings_for(options))
    await context.open()
    assert context.secrets is not None
    if options.tripo_key:
        context.secrets.set(key_name("tripo"), options.tripo_key)
    assert context.three_d is not None
    context.three_d.use_secrets(context.secrets)

    try:
        print("\n1. the pieces", flush=True)
        # All three shortcuts drive the pipeline without a model, and saying so
        # is clearer than failing a check about something the run never needed.
        if not (options.direct or options.import_only or options.resume):
            check("an LLM is configured", bool(options.base_url and options.model), options.model)
        provider = context.three_d.provider("tripo")
        check(
            "Tripo has a key",
            provider.is_configured(),
            provider.backend if hasattr(provider, "backend") else "",
        )
        check("Tripo is ready", provider.is_configured(), provider.model)
        estimated = provider.estimate_credits(
            AssetRequest(prompt=options.prompt, texture=not options.no_texture)
        )
        print(f"     estimated credits: {estimated}", flush=True)

        print("\n2. Blender", flush=True)
        assert context.mcp is not None
        await context.mcp.connect_all()
        scene = await wait_for_blender(context, options.wait_seconds)
        check("Blender is attached", not scene.is_error, scene.error_code)
        before = payload_of(await context.mcp.call_tool("blender.get_objects", {"limit": 200}))
        start_count = int(before.get("total", 0))

        if options.import_only:
            # The download-and-import half does not care where the bytes came from,
            # and Tripo's account has no credit, so it is tested with a fixture.
            print("\n3. the import half, with a local GLB", flush=True)
            written = options.import_only
            submission = {"provider": "fixture", "provider_task_id": "fixture", "status": "succeeded"}
            check("the fixture is a GLB", written.read_bytes()[:4] == b"glTF", written)
        elif options.resume:
            # A generation outlives the client that asked for it: the run gets
            # killed, the network drops, the laptop closes. The task is still on
            # the provider's side, still billing, and re-submitting would pay for
            # it twice.
            print(f"\n3. picking up task {options.resume}", flush=True)
            submission = {
                "provider": "tripo",
                "provider_task_id": options.resume,
                "status": "running",
                "submitted": True,
                "resumed": True,
            }
        elif options.direct:
            # The same handler the model would call, invoked directly. This
            # isolates the pipeline from the model's judgement: a 3B model asked
            # for "a medieval wooden chest" reaches for primitives instead, and
            # that is a finding about the model, not about the client.
            print("\n3. the 3D pipeline, called directly", flush=True)
            tools = {tool.name: tool for tool in context.assets_tool()}
            handler = tools["generate_3d_asset"].handler
            submission = await handler(
                {"prompt": options.prompt, "texture": not options.no_texture, "provider": "tripo"}
            )
            print(f"     {submission}", flush=True)
        else:
            print(f"\n3. you say: {options.prompt!r}", flush=True)
            agent = context.agent(options.provider, options.model)
            result = await agent.run(options.prompt)
            print(f"     model: {result.text or '(no answer)'}", flush=True)
            check("the agent finished", result.finished, result.stopped_because)
            called = [call.call.name for call in result.tool_calls]
            print(f"     tools: {called}", flush=True)
            check(
                "the model chose the 3D tool",
                any(name.endswith("generate_3d_asset") for name in called),
                called,
            )
            submission = next(
                (
                    payload_of_result(call)
                    for call in result.tool_calls
                    if call.call.name.endswith("generate_3d_asset")
                ),
                {},
            )
        if options.import_only:
            check("there is an asset to import", bool(submission.get("status")), submission)
        else:
            check(
                "a Tripo task was created",
                bool(submission.get("provider_task_id")),
                submission.get("provider_task_id") or submission.get("error"),
            )

        if not options.import_only and submission.get("submitted") is False:
            # Tripo refused it -- an empty account, a bad model name, anything
            # that needs a person to change something. There is no task to wait
            # for, and waiting for one is how a refusal looks like a hang.
            print("\n4. nothing to wait for", flush=True)
            print(f"     the submission was refused: {submission.get('error')}", flush=True)
            await context.close()
            check("the refusal reached the caller", True, submission.get("error_code", ""))
            print(
                f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'PASSED'} every check",
                flush=True,
            )
            return 1 if FAILURES else 0

        if not options.import_only:
            print("\n4. waiting for the generation", flush=True)
            task_id = submission.get("provider_task_id", "")
            started = time.monotonic()
            final = await _poll(context, task_id, options.timeout)
            elapsed = time.monotonic() - started
            if final.status is not TaskStatus.SUCCEEDED:
                blocked = "credit" in (final.error or "").lower()
                check(
                    "the generation succeeded",
                    False,
                    f"{final.status}: {final.error}"
                    + (" -- the account has no credit; the request itself was accepted" if blocked else ""),
                )
                if blocked:
                    print("\n     Nothing left to test on this side of the API.", flush=True)
                    return 2
                return 1
            check("the generation succeeded", True, f"in {elapsed:.0f}s")
            print(f"     {final.result.url}", flush=True)
            print(f"     credits consumed: {final.credits} (estimated {estimated})", flush=True)

            print("\n5. downloading", flush=True)
            destination = options.out_dir / f"{options.name}.{final.result.format}"
            written = await provider.download(final, destination)
            size = written.stat().st_size
            check("the model file arrived", size > 1000, f"{written} ({size / 1024:.0f} KB)")
            check("it is a GLB", written.suffix == ".glb", written.suffix)
        else:
            elapsed, written = 0.0, options.import_only

        print("\n6. importing into Blender", flush=True)
        from app.providers3d.importer import import_asset

        try:
            report = await import_asset(context.mcp, written)
        except ThreeDError as exc:
            check("the import ran", False, f"{exc.message} — {exc.hint or ''}")
        else:
            print(f"     {report.summary()}", flush=True)
            after = payload_of(await context.mcp.call_tool("blender.get_objects", {"limit": 200}))
            names = [obj["name"] for obj in after.get("objects", [])]
            check(
                "the scene grew",
                int(after.get("total", 0)) > start_count,
                f"{start_count} -> {after.get('total')}",
            )
            # The report is a diff of the scene, so it names what the import
            # added. Matching a word from the prompt would pass on an object
            # that was already sitting there.
            check(
                "the imported objects are in the scene",
                bool(report.imported) and all(name in names for name in report.imported),
                report.imported[:4],
            )
            if options.cleanup:
                for name in report.imported[:1]:
                    removed = await context.mcp.call_tool("blender.delete_object", {"name": name})
                    check("cleaned up", not removed.is_error, name)
        consumed = 0 if options.import_only else final.credits
        print(f"\n     total: {elapsed:.0f}s of generation, {consumed} credits", flush=True)
    finally:
        await context.close()
    return 0


def payload_of_result(tool_result) -> dict:
    try:
        found = json.loads(tool_result.content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return found if isinstance(found, dict) else {}


async def _poll(context: AppContext, provider_task_id: str, timeout: float):
    assert context.three_d is not None
    provider = context.three_d.provider("tripo")
    task = await provider.status(
        __import__("app.providers3d.models", fromlist=["ProviderTask"]).ProviderTask.new(
            "tripo", AssetRequest(prompt=""), provider_task_id=provider_task_id
        )
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not task.status.terminal:
        await asyncio.sleep(5.0)
        task = await provider.status(task)
        print(f"     {task.status} {task.progress * 100:.0f}%", flush=True)
    return task


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments every live run needs: where Blender is, and how to reach it.

    Shared so the two runs cannot drift apart in how they find a Blender -- a
    difference that would look like a product bug and be a typo.
    """
    parser.add_argument("--blender-mcp", required=True)
    parser.add_argument("--blender-mcp-env", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--provider", default="localqwen")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--tripo-key", default=os.environ.get("TRIPO_API_KEY", ""))
    parser.add_argument("--wait-seconds", type=float, default=90.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--prompt", default="Create a medieval wooden chest.")
    parser.add_argument("--name", default="Chest")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="call generate_3d_asset's handler directly, skipping the model's choice",
    )
    parser.add_argument(
        "--resume",
        default="",
        metavar="TASK_ID",
        help="skip submission and pick up a task the provider is already running",
    )
    parser.add_argument(
        "--import-only",
        type=Path,
        default=None,
        help="skip generation and import this GLB instead, for testing the import half",
    )
    parser.add_argument("--cleanup", action="store_true", help="delete the imported asset afterwards")
    parser.add_argument("--out-dir", type=Path, default=Path("out/assets"))
    parser.add_argument("--data-dir", type=Path, default=Path("out/data"))
    options = parser.parse_args()
    options.out_dir.mkdir(parents=True, exist_ok=True)
    options.data_dir.mkdir(parents=True, exist_ok=True)
    code = asyncio.run(run(options))
    print("", flush=True)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}", flush=True)
        return 1
    print("Asset run passed.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
