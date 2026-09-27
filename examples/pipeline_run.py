"""The 3D path against a real Blender, with a provider that is not Tripo.

Tripo's account has no credit, so the live half of the pipeline is exercised with
a provider that succeeds -- against the real MCP server, the real Windows Blender
and the real import. What is left untested by this is Tripo's HTTP, which is
tested against Tripo directly, and a billing refusal, which is what Tripo gives
instead of a model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.context import AppContext, three_d_tool  # noqa: E402
from app.core.settings import Settings, ThreeDConfig  # noqa: E402
from app.providers3d.mock import MockThreeDProvider  # noqa: E402
from app.providers3d.registry import ThreeDRegistry  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail != "" else ""), flush=True)
    if not condition:
        FAILURES.append(label)
    return condition


def settings_for(options: argparse.Namespace) -> Settings:
    from examples.asset_run import settings_for as build

    return build(options)


async def run(options: argparse.Namespace) -> int:
    from app.providers3d.importer import can_import

    settings = settings_for(options)
    settings.three_d = ThreeDConfig(provider="mock-3d", poll_interval=0.2, poll_timeout=120.0)
    settings.three_d.download_dir = options.assets

    context = AppContext(settings=settings)
    # The real registry goes in: the real one, minus Tripo, which is the one that
    # cannot afford to run.
    # A real GLB: the placeholder bytes this mock normally serves are not a file
    # Blender can open, and a live run that "succeeds" without importing anything
    # would be worth nothing.
    provider = MockThreeDProvider(polls_before_done=3, credits=100, download_from=options.fixture)
    registry = ThreeDRegistry(bus=context.bus, config=settings.three_d)
    registry.add(provider)
    context.three_d = registry

    await context.open()
    assert context.mcp is not None
    await context.mcp.connect_all()

    print("\n1. Blender", flush=True)
    attached = False
    for _ in range(int(options.wait_seconds // 2) or 1):
        scene = await context.mcp.call_tool("blender.get_scene", {})
        if not scene.is_error:
            attached = True
            break
        await asyncio.sleep(2)
    check("Blender is attached", attached)
    check("there is a way to import", can_import(context.mcp), "blender.execute_python")
    before = await context.mcp.call_tool("blender.get_objects", {"limit": 300})
    start_count = int(json.loads(before.text).get("total", 0))

    print("\n2. the tool the model sees", flush=True)
    tool = three_d_tool(
        registry,
        settings.three_d,
        tasks=context.tasks,
        on_ready=context._import_generated,  # noqa: SLF001 - the same path the product uses
        download_dir=context.asset_dir(),
    )
    check("it is a tool", tool.name == "generate_3d_asset", tool.name)

    print("\n3. asking for an asset", flush=True)
    started = time.monotonic()
    answer = await tool.handler({"prompt": options.prompt})
    print(f"     {answer}", flush=True)
    check(
        "the model got an answer at once",
        time.monotonic() - started < 2.0,
        f"{time.monotonic() - started:.2f}s",
    )
    check("it was submitted", answer.get("submitted") is True, answer.get("status"))
    check("there is a task to watch", bool(answer.get("studio_task_id")), answer.get("studio_task_id"))

    print("\n4. the task runs", flush=True)
    assert context.tasks is not None
    finished = await context.tasks.wait(answer["studio_task_id"], timeout=120)
    print(f"     {finished.state}: {finished.detail or finished.error}", flush=True)
    check("it succeeded", str(finished.state) == "succeeded", finished.error)
    check("the cost is on the task", finished.credits == 100, finished.credits)
    check(
        "the download is where the studio can see it",
        Path(str((finished.result or {}).get("path", ""))).exists(),
        (finished.result or {}).get("path"),
    )

    print("\n5. the scene", flush=True)
    after = await context.mcp.call_tool("blender.get_objects", {"limit": 300})
    objects = [obj["name"] for obj in json.loads(after.text).get("objects", [])]
    check(
        "the scene grew",
        int(json.loads(after.text).get("total", 0)) > start_count,
        f"{start_count} -> {json.loads(after.text).get('total')}",
    )
    check(
        "the generated object is in it",
        bool((finished.result or {}).get("objects")),
        (finished.result or {}).get("objects"),
    )
    check(
        "and it is the name Blender knows it by",
        any(name in objects for name in (finished.result or {}).get("objects", [])),
        [n for n in objects if n in (finished.result or {}).get("objects", [])],
    )

    if options.cleanup:
        print("\n6. cleaning up", flush=True)
        for name in (finished.result or {}).get("objects", []):
            removed = await context.mcp.call_tool("blender.delete_object", {"name": name})
            check(f"removed {name}", not removed.is_error, removed.error_code or "")

    await context.close()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'PASSED'} every check", flush=True)
    return 1 if FAILURES else 0


def main() -> int:
    from examples.asset_run import add_common_arguments

    parser = argparse.ArgumentParser(description="The 3D pipeline against a real Blender.")
    add_common_arguments(parser)
    parser.add_argument("--prompt", default="a medieval wooden chest")
    parser.add_argument("--assets", type=Path, default=Path("/mnt/c/Users/maxim/blender-mcp-demo/out"))
    parser.add_argument("--cleanup", action="store_true", help="delete the generated asset afterwards")
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/opencode/studio-data-pipeline"))
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("/mnt/c/Users/maxim/blender-mcp-demo/out/Fixture.glb"),
        help="a real GLB for the mock provider to serve",
    )
    options = parser.parse_args()
    options.data_dir.mkdir(parents=True, exist_ok=True)
    options.assets.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
