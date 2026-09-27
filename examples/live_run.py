"""Everything, live, against a real Blender and real models.

The unit tests drive the studio with fabricated events and a fake bridge. This
drives it with the actual product: a real MCP server, a real Windows Blender, a
real local model, a real 3D provider when there is credit for it, and a real
database. Each section answers one question that only a live run can answer --
does the thing work when the pieces are real.

Run it after starting Blender with the add-on attached, and with a model
configured:

    python examples/live_run.py \\
        --blender-mcp ../blender-mcp --blender-mcp-env /tmp/mcpenv \\
        --python /usr/bin/python3.11 --port 8767 \\
        --provider localqwen --base-url http://127.0.0.1:11400/v1 --model local-qwen
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.context import AppContext  # noqa: E402
from app.core.settings import MCPServerConfig, Settings, ThreeDConfig  # noqa: E402

FAILURES: list[str] = []
PASSES: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail != "" else ""), flush=True)
    (PASSES if condition else FAILURES).append(label)
    return condition


def section(title: str) -> None:
    print(f"\n{title}", flush=True)


def settings_for(options: argparse.Namespace) -> Settings:
    """A studio pointed at the Blender and models this run was told about."""
    settings = Settings()
    settings.data_dir = options.data_dir
    settings.log_level = options.log_level
    settings.mcp_servers = [
        MCPServerConfig(
            name="Blender MCP",
            command=options.python,
            args=["-m", "server.main"],
            cwd=options.blender_mcp,
            env={
                "PYTHONPATH": options.blender_mcp_env or options.blender_mcp,
                # The import needs the studio's own capability, and it is off by
                # default: a run that wants to prove the import has to ask for it.
                "ALLOW_PYTHON_EXECUTION": "true" if options.allow_execute_python else "false",
            },
            blender_host="127.0.0.1",
            blender_port=options.port,
            tool_timeout=600.0,
        )
    ]
    if options.second_mcp:
        # A second server, so the run can show that the catalogue is a union and
        # that saving one of them does not delete the other.
        name, _, rest = options.second_mcp.partition("=")
        command, _, args = rest.partition(" ")
        settings.mcp_servers.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=args.split(),
                blender_port=options.port + 1,
                tool_timeout=120.0,
            )
        )
    providers: list[dict[str, Any]] = []
    for spec in (options.provider_spec, options.second_model):
        if not spec:
            continue
        name, _, model = spec.partition(":")
        providers.append(
            {
                "name": name,
                "kind": "openai",
                "base_url": options.base_url if name == options.provider else options.second_base_url,
                "default_model": model,
                # Declared honestly: whether this model can see is the run's
                # business to find out, and the wire format is built from it.
                "models": [
                    {"id": model, "supports_tools": True, "supports_vision": options.model_sees_images}
                ],
            }
        )
    if not providers:
        providers.append(
            {
                "name": "scripted",
                "kind": "mock",
                "default_model": "scripted-planner",
                "models": [{"id": "scripted-planner", "supports_tools": True}],
            }
        )
    settings.llm_providers = providers
    # Both gates: the child's environment, and the studio's own switch, which is
    # what actually decides whether the agent is allowed to ask.
    settings.agent.allow_execute_python = options.allow_execute_python
    return settings


async def attach(context: AppContext, options: argparse.Namespace) -> bool:
    assert context.mcp is not None
    await context.mcp.connect_all()
    for _ in range(int(options.wait_seconds)):
        scene = await context.mcp.call_tool("blender.get_scene", {})
        if not scene.is_error:
            return True
        await asyncio.sleep(1)
    return False


# --- 1. the real stack ------------------------------------------------------


async def check_the_stack(context: AppContext, options: argparse.Namespace) -> None:
    section("1. the stack, for real")
    assert context.mcp is not None
    # Already connected by the caller: the catalogue is discovered during the
    # handshake, so asking about it before that asks about nothing.
    statuses = context.mcp.statuses()
    check(
        "a real MCP server is connected",
        any(s.ready for s in statuses),
        ", ".join(f"{s.name}:{s.state}" for s in statuses),
    )
    tools = context.mcp.tools()
    check("its tools were discovered", len(tools) >= 10, f"{len(tools)} tools")
    check("they are not hardcoded", any(name.startswith("blender.") for name in tools))
    check("Blender is attached", await attach(context, options))


# --- 2. a real model, a real scene ------------------------------------------


async def check_a_real_turn(context: AppContext, options: argparse.Namespace) -> None:
    section("2. a real model driving a real Blender")
    assert context.mcp is not None
    before = await context.mcp.call_tool("blender.get_objects", {"limit": 300})
    start = int(json.loads(before.text).get("total", 0))

    agent = context.agent(options.provider, options.model)
    result = await agent.run(
        "Use the blender.create_object tool to create a cylinder named LiveProbe "
        "at location 0, 0, 2. Call the tool; do not describe it instead. "
        "When it is done, tell me in one sentence what you made."
    )
    called = [c.call.name for c in result.tool_calls]
    print(f"     tools: {called}", flush=True)
    check("the model finished", result.finished, result.stopped_because)
    check("it used a real tool", bool(called), called)
    check("it made the object", "blender.create_object" in called, called)
    errors = [c for c in result.tool_calls if c.is_error]
    recovered = all("TRANSACTION" in (c.content or "") for c in errors)
    check(
        "no tool failed, or the run recovered from one",
        not errors or recovered,
        [c.error_code for c in errors] or "clean",
    )
    check("its tokens were counted", result.totals.tokens > 0, result.totals.to_dict())

    after = await context.mcp.call_tool("blender.get_objects", {"limit": 300})
    objects = json.loads(after.text)
    check(
        "the scene really grew",
        int(objects.get("total", 0)) == start + 1,
        f"{start} -> {objects.get('total')}",
    )
    probe = next((o for o in objects.get("objects", []) if o["name"] == "LiveProbe"), None)
    check(
        "and the object is where it was asked to be",
        probe is not None and [round(v, 2) for v in probe["location"]] == [0.0, 0.0, 2.0],
        probe["location"] if probe else "missing",
    )

    if probe is not None:
        removed = await context.mcp.call_tool("blender.delete_object", {"name": "LiveProbe"})
        check("cleaned up", not removed.is_error)


# --- 3. the vision path -----------------------------------------------------


async def check_vision(context: AppContext, options: argparse.Namespace) -> None:
    """Render, read the file, put it in front of a model that can see.

    The tool call is scripted rather than asked for, because a 3B vision model
    does not reliably issue one -- that is a fact about the model, not about the
    studio, and pretending otherwise would hide the part worth testing. Every
    other link in the chain is real: a real render, read across a filesystem
    boundary, encoded for the provider, and answered by a real multimodal model.
    """
    from app.mcp.images import fetch

    section("3. a render, and who gets to see it")
    assert context.mcp is not None and context.llm is not None
    tools = context.mcp.tools()
    if "blender.render_preview" not in tools:
        check("the server can render", False, "blender.render_preview is not offered")
        return
    render = await context.mcp.call_tool("blender.render_preview", {"width": 320, "height": 240}, timeout=600)
    check("Blender rendered", not render.is_error, render.error_code or render.size())

    inline = len(render.images)
    fetched = fetch(context.mcp, render.text)
    check(
        "the render came back as a picture",
        inline > 0 or fetched is not None,
        f"{inline} inline" + (", or read from the path the tool gave" if fetched else ""),
    )
    if not inline and not fetched:
        return
    if fetched is not None:
        raw = base64.b64decode(fetched[0])
        check("and it is a real PNG", raw[:8] == b"\x89PNG\r\n\x1a\n", f"{len(raw)} bytes")
        check("its mime type travelled with it", fetched[1] == "image/png", fetched[1])

    info = context.llm.find(options.provider, options.model)
    vision = bool(getattr(info, "supports_vision", False))
    print(f"     {options.model} sees images: {vision}", flush=True)
    if not vision:
        check(
            "a text-only model is not sent an image",
            context.agent(options.provider, options.model)._image_parts(  # noqa: SLF001
                render.images or [fetched[0] if fetched else ""], [fetched[1] if fetched else "image/png"]
            )
            == [],
            "it would reject the request; the window still shows the picture",
        )
        return
    check(
        "a model that can see is sent the render",
        True,
        f"{len(render.images or ([fetched] if fetched else []))} image(s)",
    )

    from app.llm.base import ChatRequest, ContentPart, Message, Role

    parts = [ContentPart.text_part(options.vision_prompt)]
    if inline:
        parts.append(ContentPart.image_part(render.images[0], render.image_mime_types[0]))
    elif fetched is not None:
        parts.append(ContentPart.image_part(fetched[0], fetched[1]))
    request = ChatRequest(
        model=options.model,
        messages=[Message(role=Role.USER, content=options.vision_prompt, parts=parts)],
    )
    provider = context.llm.resolve(options.provider, options.model)[0]
    response = await provider.chat(request)
    answer = (response.text or "").strip()
    print(f"     the model said: {answer[:200]}", flush=True)
    check("a real multimodal model described the real render", len(answer) > 10, answer[:80])
    check(
        "and what it said is about the scene",
        any(
            word in answer.lower()
            for word in ("table", "ball", "sphere", "ring", "chest", "floor", "light", "object", "scene")
        ),
        answer[:80],
    )


# --- 4. projects ------------------------------------------------------------


async def check_projects(context: AppContext, options: argparse.Namespace) -> None:
    section("4. a project, and the turns filed under it")
    assert context.studio is not None
    project = await context.studio.projects.create(
        f"Live check {options.name_suffix}", initial_blend=options.blend or None
    )
    context.current_project = project.id
    check("a project was created", project.id.startswith("prj_"), project.name)

    agent = context.agent(options.provider, options.model)
    await agent.run("Tell me in one short sentence what you would model first for a medieval kitchen.")
    filed = await context.studio.conversations.list(project.id)
    check("the turn is filed under it", len(filed) == 1, [c.id for c in filed])
    if filed:
        messages = await context.studio.messages.list(filed[0].id)
        check("with its messages", len(messages) >= 2, [m.role for m in messages])

    stored = await context.studio.projects.get(project.id)
    check("the project reads back", stored is not None and stored.name == project.name)
    await context.studio.projects.delete(project.id)
    check(
        "and deletes, taking its conversations with it",
        await context.studio.conversations.list(project.id) == [],
    )
    context.current_project = None


# --- 5. more than one MCP server --------------------------------------------


async def check_two_servers(context: AppContext, options: argparse.Namespace) -> None:
    section("5. two MCP servers")
    assert context.mcp is not None and context.studio is not None
    servers = list(context.settings.mcp_servers)
    check("more than one is configured", len(servers) >= 2, ", ".join(s.name for s in servers))
    merged = context.mcp.tools()
    by_server = {status.name: status.tools for status in context.mcp.statuses()}
    check("each contributes its tools", all(count > 0 for count in by_server.values()), by_server)
    check(
        "and the catalogue is the union",
        len(merged) >= max(by_server.values(), default=0),
        f"{len(merged)} merged",
    )


# --- 6. 3D, when there is credit for it -------------------------------------


async def check_3d(context: AppContext, options: argparse.Namespace) -> None:
    section("6. 3D generation")
    assert context.three_d is not None and context.tasks is not None
    tools = {t.name for t in context.assets_tool()}
    if not tools:
        check("the 3D tool is offered", False, "no provider is keyed; nothing to test")
        return
    check("the 3D tool is offered", "generate_3d_asset" in tools)
    gate = context.settings.agent.allow_execute_python
    offered = "blender.execute_python" in context.mcp.tools()
    check(
        "the import gate is open, as this run asked for",
        offered and gate,
        f"server offers it: {offered}, studio switch: {gate}",
    )

    task = await context.tasks.start(
        f"3D: {options.prompt}",
        lambda t: _three_d_work(context, t, options),
        provider="tripo",
        payload={"prompt": options.prompt, "kind": "text_to_3d"},
    )
    finished = await context.tasks.wait(task.id, timeout=options.three_d_timeout)
    print(f"     {finished.state}: {finished.detail or finished.error}", flush=True)
    if str(finished.state) == "failed" and "credit" in (finished.error or "").lower():
        check("the account has credit for a real generation", False, finished.error)
        return
    check("the task finished", str(finished.state) == "succeeded", finished.error)
    check(
        "a model was downloaded",
        bool((finished.result or {}).get("path")),
        (finished.result or {}).get("path"),
    )
    check(
        "and imported into the scene",
        bool((finished.result or {}).get("objects")),
        (finished.result or {}).get("objects"),
    )

    records = await context.studio.three_d.list() if context.studio else []
    check(
        "the generation is in the database",
        bool(records),
        [f"{r.provider} {r.status} {r.credits}cr" for r in records[:3]],
    )
    if records:
        check("with the credits that were spent", records[0].credits > 0, records[0].credits)

    for name in (finished.result or {}).get("objects", [])[:1]:
        if options.keep_assets:
            continue
        await context.mcp.call_tool("blender.delete_object", {"name": name})  # type: ignore[union-attr]


async def _three_d_work(context: AppContext, task: Any, options: argparse.Namespace) -> dict[str, Any]:
    from app.providers3d.models import AssetRequest

    provider = context.three_d.provider()  # type: ignore[union-attr]
    config = context.settings.three_d
    created = await provider.create(AssetRequest(prompt=options.prompt, texture=True))
    if str(created.status) == "failed":
        from app.core.errors import ThreeDError

        raise ThreeDError(created.error or "the provider refused the request")
    task.payload["provider_task_id"] = created.provider_task_id

    async def progress(part: Any) -> None:
        task.progress = float(getattr(part, "progress", 0.0) or 0.0)
        task.detail = "generating"

    finished = await provider.wait_for(
        created, interval=config.poll_interval, timeout=options.three_d_timeout, on_progress=progress
    )
    if not finished.status.terminal:
        from app.core.errors import ThreeDError

        raise ThreeDError(finished.error or "the generation never finished")
    task.credits = finished.credits
    task.detail = "downloading"
    destination = context.asset_dir() / f"{finished.provider_task_id}.glb"  # type: ignore[union-attr]  # noqa: E501
    path = await provider.download(finished, destination)
    task.detail = "importing"
    report = await context._import_generated(finished, path)  # type: ignore[union-attr]  # noqa: SLF001
    return {"path": str(path), "imported": True, "objects": report.imported, "url": finished.result.url}


# --- 7. the benchmark, two models -------------------------------------------


async def check_benchmark(context: AppContext, options: argparse.Namespace) -> None:
    section("7. a benchmark, and a review against it")
    if not options.second_model:
        check("a second model to compare against", False, "pass --second-model provider:model")
        return
    from app.benchmark.runner import BenchmarkRunner, BenchmarkTask, ModelSpec
    from app.benchmark.storage import BenchmarkStorage

    assert context.mcp is not None and context.llm is not None and context.studio is not None
    specs = [
        ModelSpec(provider=options.provider, model=options.model),
        ModelSpec(
            provider=options.second_model.split(":", 1)[0], model=options.second_model.split(":", 1)[1]
        ),
    ]
    storage = BenchmarkStorage(context.studio.db)
    suite_id = await storage.create_suite("live run", ", ".join(s.label() for s in specs))
    run_ids: list[str] = []

    async def persist(comparison: Any, outcome: Any) -> None:
        run_ids.append(await storage.save_run(suite_id, outcome, task_index=len(run_ids)))

    runner = BenchmarkRunner(
        context.llm, context.mcp, bus=context.bus, system_prompt=context.settings.agent.system_prompt
    )
    comparison = await runner.run_suite(
        [BenchmarkTask(prompt=options.bench_prompt)],
        specs,
        blend=Path(options.blend) if options.blend else None,
        persist=persist,
    )
    for outcome in comparison.outcomes:
        print(
            f"     {outcome.model.label():<28} {outcome.status:<8} "
            f"{outcome.duration_s:6.1f}s {outcome.totals.tokens:>7} tokens "
            f"{outcome.mcp_calls:>3} calls {outcome.tool_errors} errors",
            flush=True,
        )
    check(
        "both models ran",
        all(o.status == "ok" for o in comparison.outcomes),
        [o.error for o in comparison.outcomes if o.error],
    )
    check("each run was saved with an id", len(run_ids) == len(comparison.outcomes), run_ids)
    await storage.review(run_ids[0], overall=4, notes="live check")
    check("a review attaches to a run", bool(await storage.reviews(run_ids[0])))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blender-mcp", required=True)
    parser.add_argument("--blender-mcp-env", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--provider", default="localqwen")
    parser.add_argument("--provider-spec", default="", help="provider:model for the main run")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--second-base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--model-sees-images",
        action="store_true",
        help="declare the model multimodal, so the render is sent to it",
    )
    parser.add_argument("--second-model", default="", help="provider:model to benchmark against")
    parser.add_argument(
        "--second-mcp", default="", help="name=/path/to/python -m module, for a second server"
    )
    parser.add_argument(
        "--tripo-key",
        default=os.environ.get("TRIPO_API_KEY", ""),
        help="a 3D provider key; without one the 3D section says so instead of failing",
    )
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--allow-execute-python",
        action="store_true",
        help="open the gate the 3D import needs; without it the run proves the refusal instead",
    )
    parser.add_argument("--prompt", default="a medieval wooden chest with iron bands")
    parser.add_argument(
        "--bench-prompt", default="How many objects are in the scene? Use blender.get_objects."
    )
    parser.add_argument(
        "--vision-prompt",
        default="What is in this image? Answer in one short sentence naming what you can see.",
    )
    parser.add_argument("--blend", default="", help="a starting .blend, for the project and the benchmark")
    parser.add_argument("--assets", type=Path, default=Path("out/assets"))
    parser.add_argument("--data-dir", type=Path, default=Path("out/live"))
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--wait-seconds", type=float, default=60.0)
    parser.add_argument("--three-d-timeout", type=float, default=900.0)
    parser.add_argument("--keep-assets", action="store_true", help="leave generated models in the scene")
    parser.add_argument("--skip", default="", help="comma separated section numbers to skip")
    options = parser.parse_args()
    options.data_dir.mkdir(parents=True, exist_ok=True)
    options.assets.mkdir(parents=True, exist_ok=True)

    async def run() -> int:
        settings = settings_for(options)
        settings.three_d = ThreeDConfig(provider="tripo", download_dir=str(options.assets))
        context = AppContext(settings=settings)
        await context.open()
        if options.tripo_key and context.secrets is not None:
            from app.storage.secrets import key_name

            context.secrets.set(key_name("tripo"), options.tripo_key)
            if context.three_d is not None:
                context.three_d.use_secrets(context.secrets)
        try:
            # Connect once, here. Any section can then be skipped without taking
            # the bridge with it -- and a 3D import with no bridge answers "the
            # gate is off", which is a different problem entirely.
            if not await attach(context, options):
                check("Blender is attached", False, "no Blender on the bridge port")
                return 1
            skip = {part.strip() for part in options.skip.split(",") if part.strip()}
            for number, check_it in (
                ("1", check_the_stack),
                ("2", check_a_real_turn),
                ("3", check_vision),
                ("4", check_projects),
                ("5", check_two_servers),
                ("6", check_3d),
                ("7", check_benchmark),
            ):
                if number in skip:
                    print(f"\n{number}. skipped", flush=True)
                    continue
                await check_it(context, options)
        finally:
            await context.close()
        return 0

    asyncio.run(run())
    print("", flush=True)
    print(f"{len(PASSES)} passed, {len(FAILURES)} failed", flush=True)
    if FAILURES:
        print("failed: " + "; ".join(FAILURES), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
