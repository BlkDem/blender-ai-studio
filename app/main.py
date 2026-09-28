"""``blender-ai-studio`` — the entry point.

Three ways in:

* ``blender-ai-studio`` opens the window;
* ``blender-ai-studio --check`` reports what is configured and what is not, and
  exits non-zero if the core path is broken;
* ``blender-ai-studio --prompt "..."`` runs one turn headlessly, which is how the
  acceptance script drives the whole stack without a display.

The diagnostic deliberately builds the same application the window does. A
``--check`` that assembles something simpler is a check that passes while the app
is broken.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from app.core.context import AppContext, setup_logging
from app.core.errors import ConfigurationError, StudioError
from app.core.settings import Settings

logger = logging.getLogger("app.main")

EXIT_OK = 0
EXIT_NOT_CONFIGURED = 1
EXIT_FAILED = 2


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="blender-ai-studio",
        description="A desktop AI client that drives Blender through the blender-mcp MCP server.",
    )
    parser.add_argument("--check", action="store_true", help="check the configuration and exit")
    parser.add_argument("--prompt", help="run one turn headlessly and print the result")
    parser.add_argument("--provider", default="", help="provider to use for --prompt")
    parser.add_argument("--model", default="", help="model id for --prompt")
    parser.add_argument("--list-tools", action="store_true", help="print the MCP tools and exit")
    parser.add_argument(
        "--benchmark",
        metavar="PROVIDER:MODEL,PROVIDER:MODEL",
        default="",
        help="run each model against each prompt and print the comparison",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="a file of prompts, one per line, for --benchmark (repeat --prompt too)",
    )
    parser.add_argument(
        "--blend",
        type=Path,
        default=None,
        help="a .blend to start each benchmark run from, so the runs are comparable",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--data-dir", type=Path, help="override the data directory")
    parser.add_argument("--log-level", default="", help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args(argv)


def build_settings(options: argparse.Namespace) -> Settings:
    settings = Settings()
    if options.data_dir:
        settings.data_dir = options.data_dir
    if options.log_level:
        settings.log_level = options.log_level.upper()
    return settings


async def check(context: AppContext, *, as_json: bool) -> int:
    """Report the state of every moving part. Never raises."""
    settings = context.settings
    lines: list[str] = []
    problems: list[str] = []

    lines.append(f"data directory   {settings.resolved_database_path} (database)")
    lines.append(
        f"secrets          {settings.resolved_secrets_path} (backend: {context.secrets.backend if context.secrets else '?'})"
    )

    # --- MCP ---------------------------------------------------------------
    lines.append("")
    lines.append("MCP servers")
    assert context.mcp is not None
    if not settings.mcp_servers:
        problems.append("no MCP server is configured")
        lines.append("  (none configured)")
    for config in settings.mcp_servers:
        if not config.enabled:
            lines.append(f"  {config.name}: disabled")
            continue
        if not config.command:
            problems.append(f"{config.name}: no command")
            lines.append(f"  {config.name}: no command")
            continue
        if not Path(config.command).exists() and not _on_path(config.command):
            problems.append(f"{config.name}: command not found ({config.command})")
            lines.append(f"  {config.name}: command not found: {config.command}")
            continue
        status = await context.mcp.connect(config.name)
        mark = "ok" if status.ready else "FAILED"
        lines.append(
            f"  {config.name}: {mark}"
            + (f" — {status.last_error}" if status.last_error else "")
            + (
                f" ({status.tools} tools, {status.resources} resources, {status.latency_ms:.0f} ms)"
                if status.ready
                else ""
            )
        )
        if status.ready:
            if not status.tools:
                lines.append(f"    note: {config.name} published no tools")
            for name in context.mcp.tool_names()[:6]:
                lines.append(f"    · {name}")
            more = len(context.mcp.tool_names()) - 6
            if more > 0:
                lines.append(f"    … and {more} more")
        else:
            problems.append(f"{config.name}: {status.last_error}")

    # --- Blender -----------------------------------------------------------
    if context.mcp is not None and context.mcp.any_connected():
        lines.append("")
        lines.append("Blender")
        scene = await context.mcp.call_tool("blender.get_scene", {})
        if scene.is_error:
            lines.append(f"  attached: no — {scene.error_code or 'the bridge has no add-on'}")
            problems.append("no Blender is attached to the bridge")
        else:
            lines.append("  attached: yes")
            lines.append(f"  scene: {scene.text.splitlines()[0][:100]}")

    # --- LLM ---------------------------------------------------------------
    lines.append("")
    lines.append("LLM providers")
    assert context.llm is not None
    if not context.llm.configs():
        problems.append("no LLM provider is configured")
        lines.append("  (none configured)")
    for provider in context.llm.configs():
        ready = context.llm.is_configured(provider.name)
        models = [model.id for model in context.llm.models(provider.name)]
        mark = "ok" if ready else "no API key"
        lines.append(f"  {provider.name} [{provider.kind}]: {mark}")
        lines.append(f"    base url     {provider.base_url or '(default)'}")
        lines.append(f"    default      {provider.default_model or '(first in catalog)'}")
        lines.append(f"    models       {', '.join(models) if models else '(none configured)'}")
        if not ready and provider.kind != "mock":
            problems.append(f"{provider.name}: no API key")

    # --- 3D ----------------------------------------------------------------
    lines.append("")
    lines.append("3D providers")
    assert context.three_d is not None
    described = context.three_d.describe()
    if not described:
        problems.append("no 3D provider is configured")
        lines.append("  (none configured)")
    for entry in described:
        mark = "ok" if entry["ready"] else "no API key"
        lines.append(f"  {entry['name']}: {mark} (kinds: {', '.join(entry['kinds'])})")
        if not entry["ready"]:
            problems.append(f"{entry['name']}: no API key")

    # --- agent -------------------------------------------------------------
    lines.append("")
    lines.append("Agent limits")
    agent = settings.agent
    lines.append(
        f"  steps {agent.max_steps} · tool calls {agent.max_tool_calls} · seconds {agent.max_seconds:g}"
    )
    lines.append(
        f"  LLM budget ${agent.max_request_cost:g} per request, ${agent.max_session_cost:g} per session"
    )
    lines.append(
        f"  3D credits {agent.max_3d_credits} · execute_python {'allowed' if agent.allow_execute_python else 'blocked'}"
    )

    payload = {
        "ok": not problems,
        "problems": problems,
        "lines": lines,
        "mcp": [
            {
                "name": status.name,
                "state": str(status.state),
                "server": status.server_name,
                "version": status.version,
                "tools": status.tools,
                "resources": status.resources,
                "latency_ms": round(status.latency_ms),
                "error": status.last_error,
            }
            for status in (context.mcp.statuses() if context.mcp else [])
        ],
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(lines))
        if problems:
            print("\nProblems:")
            for problem in problems:
                print(f"  ! {problem}")
        else:
            print("\nEverything checks out.")
    return EXIT_OK if not problems else EXIT_NOT_CONFIGURED


async def list_tools(context: AppContext, *, as_json: bool) -> int:
    assert context.mcp is not None
    assert context.llm is not None
    await context.mcp.connect_all()
    catalog = context.mcp.tools()
    if as_json:
        print(
            json.dumps(
                {
                    name: tool.to_dict()
                    if hasattr(tool, "to_dict")
                    else {"name": tool.name, "description": tool.description, "schema": tool.schema}
                    for name, tool in catalog.items()
                },
                indent=2,
            )
        )
        return EXIT_OK
    for name, tool in catalog.items():
        required = ", ".join(tool.required_arguments()) or "-"
        print(f"{name}\n    {tool.summary()}\n    required: {required}")
    print(f"\n{len(catalog)} tools")
    return EXIT_OK


async def run_prompt(context: AppContext, text: str, provider: str, model: str, *, as_json: bool) -> int:
    """One turn, headless. The whole stack, no window."""
    if not provider:
        registry = context.llm
        configured = registry.configs() if registry else []
        if not configured or registry is None:
            print("No LLM provider is configured.", file=sys.stderr)
            return EXIT_NOT_CONFIGURED
        provider = next(
            (c.name for c in configured if c.kind != "mock" and registry.is_configured(c.name)),
            configured[0].name,
        )

    assert context.mcp is not None
    assert context.llm is not None
    await context.mcp.connect_all()
    if not context.mcp.any_connected():
        print("No MCP server is connected; start Blender with the add-on attached.", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    agent = context.agent(provider, model)
    started = time.perf_counter()
    result = await agent.run(text)
    elapsed = time.perf_counter() - started

    if as_json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "text": result.text,
                    "tools": [call.call.name for call in result.tool_calls],
                    "errors": [call.call.name for call in result.tool_calls if call.is_error],
                    "finished": result.finished,
                    "stopped_because": result.stopped_because,
                    "totals": result.totals.to_dict(),
                    "elapsed_s": round(elapsed, 2),
                },
                indent=2,
            )
        )
    else:
        for call in result.tool_calls:
            mark = "!" if call.is_error else "+"
            print(f"  {mark} {call.call.name}: {call.content.splitlines()[0][:80] if call.content else ''}")
        print()
        print(result.text or result.stopped_because or "(no answer)")
        print()
        print(
            f"{len(result.tool_calls)} tool call(s) · {result.totals.tokens} tokens · "
            f"${result.totals.llm_usd:.4f} · {elapsed:.1f}s"
        )
    return EXIT_OK if result.finished else EXIT_FAILED


def _on_path(command: str) -> bool:
    if not command:
        return False
    if Path(command).is_absolute():
        return False
    import shutil

    return shutil.which(command) is not None


async def run_benchmark(context: AppContext, options: argparse.Namespace) -> int:
    """Every model against every prompt, in order, and one table at the end.

    Headless, because that is where a comparison belongs: it has to be
    repeatable, and a window is a bad place to keep the record of a thing whose
    whole purpose is being comparable next time.
    """
    from app.benchmark.runner import BenchmarkRunner, BenchmarkTask, ModelSpec
    from app.benchmark.storage import BenchmarkStorage

    specs = [
        ModelSpec(
            provider=item.split(":", 1)[0].strip(), model=item.split(":", 1)[1].strip() if ":" in item else ""
        )
        for item in options.benchmark.split(",")
        if item.strip()
    ]
    if not specs:
        print("--benchmark needs at least one provider:model.", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    prompts = [options.prompt] if options.prompt else []
    if options.prompts:
        prompts += [
            line.strip()
            for line in options.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not prompts:
        print("--benchmark needs prompts: --prompt, --prompts, or both.", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    assert context.mcp is not None
    assert context.llm is not None
    await context.mcp.connect_all()
    if not context.mcp.any_connected():
        print("No MCP server is connected; start Blender with the add-on attached.", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    tasks = [BenchmarkTask(prompt=prompt) for prompt in prompts]
    storage = BenchmarkStorage(context.studio.db) if context.studio else None
    suite_id = ""
    run_ids: list[str] = []

    async def persist(comparison: Any, outcome: Any) -> None:
        if storage is not None and suite_id:
            run_ids.append(await storage.save_run(suite_id, outcome, task_index=len(run_ids)))

    if storage is not None:
        suite_id = await storage.create_suite(
            name=f"{len(specs)} model(s) x {len(tasks)} task(s)",
            description=", ".join(spec.label() for spec in specs),
        )

    runner = BenchmarkRunner(
        context.llm,
        context.mcp,
        bus=context.bus,
        system_prompt=context.settings.agent.system_prompt,
    )
    comparison = await runner.run_suite(tasks, specs, blend=options.blend, persist=persist)

    if options.json:
        print(
            json.dumps(
                {
                    "suite_id": suite_id,
                    "run_ids": run_ids,
                    "columns": comparison.columns(),
                    "rows": [
                        outcome.row() | {"id": run}
                        for outcome, run in zip(comparison.outcomes, run_ids, strict=False)
                    ],
                },
                indent=2,
            )
        )
    else:
        print()
        print(
            f"{'Model':<28} {'Status':<10} {'Time':>7} {'Tokens':>8} {'Cost':>9} {'Calls':>6} {'Errors':>7}"
        )
        for outcome, run in zip(comparison.outcomes, run_ids, strict=False):
            totals = outcome.totals
            print(
                f"{outcome.model.label():<28} {outcome.status:<10} {outcome.duration_s:>6.1f}s "
                f"{totals.tokens:>8} ${totals.total_usd:>8.4f} {outcome.mcp_calls:>6} {outcome.tool_errors:>7}"
                + (f"  {run}" if run else "")
            )
        print()
        print("No model is ranked above another here. Score the runs by hand:")
        print(f"  {suite_id or '(not saved)'}")
    failed = [outcome for outcome in comparison.outcomes if outcome.status not in ("ok", "verified")]
    return EXIT_FAILED if failed else EXIT_OK


async def amain(options: argparse.Namespace) -> int:
    settings = build_settings(options)
    setup_logging(options.log_level or settings.log_level)
    context = AppContext(settings=settings)
    try:
        # open() wires the secret store into every registry; calling use_secrets
        # again here would drop the providers it had already built.
        await context.open()

        if options.check:
            return await check(context, as_json=options.json)
        if options.list_tools:
            return await list_tools(context, as_json=options.json)
        if options.benchmark:
            return await run_benchmark(context, options)
        if options.prompt:
            return await run_prompt(
                context, options.prompt, options.provider, options.model, as_json=options.json
            )
    except ConfigurationError as exc:
        print(f"Configuration problem: {exc.user_text()}", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    except StudioError as exc:
        print(f"{exc.code}: {exc.user_text()}", file=sys.stderr)
        return EXIT_FAILED
    else:
        # The window runs *inside* the try, so the context is still open while it
        # is up. Running it after the finally block handed it a closed database
        # and a torn-down MCP manager: the window looked fine and every write --
        # conversations, tool calls, usage, generated tasks -- failed quietly.
        from app.gui.main import run_gui

        return run_gui(context)
    finally:
        await context.close()


def main(argv: list[str] | None = None) -> int:
    options = arguments(argv)
    headless = options.check or options.prompt or options.benchmark or options.list_tools
    if not os.environ.get("QT_QPA_PLATFORM") and sys.platform.startswith("linux") and not headless:
        # A desktop app on a machine with no display should say so once, clearly,
        # rather than dumping a Qt plugin error.
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print(
                "No display found. Use --check, --prompt, --benchmark or --list-tools for headless runs.",
                file=sys.stderr,
            )
            return EXIT_NOT_CONFIGURED
    try:
        return asyncio.run(amain(options))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
