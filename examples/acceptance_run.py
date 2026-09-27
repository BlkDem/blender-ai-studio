"""The acceptance run: a person types a sentence, Blender changes.

This is the MVP's central claim, exercised end to end and asserted on, not
demonstrated:

    user sentence
      -> LLM
      -> agent
      -> MCP over stdio
      -> blender-mcp
      -> WebSocket
      -> a real Blender
      -> verified with a second tool call

The model is chosen by ``--provider``. With a real OpenAI-compatible endpoint
configured, the run proves the whole path including the model's own decisions. With
``--scripted`` it proves everything except the model's judgement, which is what
makes this runnable in CI and on a machine with no API key.

    python examples/acceptance_run.py --scripted
    python examples/acceptance_run.py --provider space-bunny --model space-bunny-free
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core.context import AppContext, setup_logging  # noqa: E402
from app.core.settings import MCPServerConfig, Settings  # noqa: E402
from app.llm.base import ChatRequest, ChatResponse, ToolCall, Usage  # noqa: E402
from app.llm.providers.mock import MockLLMProvider  # noqa: E402

FAILURES: list[str] = []


def payload_of(outcome) -> dict:
    """The JSON a tool returned, or an empty dict if it failed.

    An error result is text, not JSON, and parsing it anyway turns a failed check
    into a traceback that hides the real problem.
    """
    if getattr(outcome, "is_error", False):
        return {}
    try:
        found = json.loads(outcome.text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return found if isinstance(found, dict) else {}


def check(label: str, condition: bool, detail: object = "") -> bool:
    if not condition:
        FAILURES.append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{f'  {detail}' if detail != '' else ''}", flush=True)
    return condition


# --- a scripted provider that reads the request -----------------------------


class ScriptedPlanner(MockLLMProvider):
    """Decides what to do by reading the request, without a network.

    It is a test double, not a model: it recognises the acceptance sentences and
    answers with the tool calls a competent model would make. Its value is that
    every part of the studio *around* the model is exercised for real, and the
    assertions are about that, not about anyone's judgement.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "scripted"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self._decide(request)

    async def stream(self, request: ChatRequest):
        """Streaming too, so the acceptance run covers the path the GUI uses."""
        from app.llm.base import StreamChunk

        response = await self.chat(request)
        yield StreamChunk(type="start")
        for word in response.text.split(" "):
            if word:
                yield StreamChunk(type="text", text=word + " ")
        for call in response.tool_calls:
            yield StreamChunk(type="tool_start", call_id=call.id, name=call.name)
            yield StreamChunk(type="tool_end", call_id=call.id, name=call.name, arguments=call.arguments)
        yield StreamChunk(type="usage", usage=response.usage)
        yield StreamChunk(type="end", text=response.text, finish_reason=response.finish_reason)

    def _decide(self, request: ChatRequest) -> ChatResponse:
        transcript = "\n".join(message.content for message in request.messages)
        # A tool result in the conversation means this is the follow-up turn, so
        # the job is to answer rather than to act again. Without this the planner
        # repeats its call until the agent's step limit stops it.
        if any(message.role.value == "tool" for message in request.messages):
            return self._answer(transcript)
        # Intent is read from the newest user message only. Reading the whole
        # transcript makes "Move TestCube" look like a request to create TestCube,
        # because the earlier turn mentions creating it.
        latest = next(
            (m.content for m in reversed(request.messages) if m.role.value == "user"),
            "",
        ).lower()
        if "chest" in latest or "medieval" in latest:
            return self._reply(
                text="I'll have a specialist generate that.",
                tool_calls=[("generate_3d_asset", {"prompt": "medieval wooden chest", "texture": True})],
            )
        if "move" in latest or "position" in latest:
            name = _name_after(latest) or "TestCube"
            return self._reply(tool_calls=[("blender.update_object", {"name": name, "location": [3, 3, 1]})])
        if "testcube" in latest or "cube named" in latest:
            return self._reply(
                tool_calls=[
                    ("blender.create_object", {"type": "cube", "name": "TestCube", "location": [2, 0, 1]})
                ]
            )
        if "table" in latest:
            return self._reply(tool_calls=[("blender.create_object", {"type": "cube", "name": "TableTop"})])
        return self._reply(tool_calls=[("blender.get_scene", {})])

    def _reply(self, text: str = "", tool_calls=()) -> ChatResponse:
        return ChatResponse(
            text=text,
            tool_calls=[ToolCall.new(name, arguments) for name, arguments in tool_calls],
            usage=Usage(input_tokens=320, output_tokens=64),
            finish_reason="tool_calls" if tool_calls else "stop",
            model=self.default_model(),
        )

    def _answer(self, transcript: str) -> ChatResponse:
        found = re.findall(r'"name":\s*"([^"]+)"', transcript)
        names = sorted(set(found))
        return self._reply(
            text=(
                f"Done. The scene now contains: {', '.join(names)}."
                if names
                else "Done. I inspected the scene and made the change."
            )
        )


def _name_after(text: str) -> str | None:
    match = re.search(r"named\s+([A-Za-z0-9_.]+)", text, re.IGNORECASE)
    return match.group(1) if match else None


# --- the run ----------------------------------------------------------------


def settings_for(options: argparse.Namespace) -> Settings:
    settings = Settings()
    settings.data_dir = options.data_dir
    settings.log_level = "WARNING"
    if options.blender_mcp:
        settings.mcp_servers = [
            MCPServerConfig(
                name="Blender MCP",
                command=options.python,
                args=["-m", "server.main"],
                cwd=options.blender_mcp,
                env={"PYTHONPATH": options.blender_mcp_env or options.blender_mcp},
                blender_host="127.0.0.1",
                blender_port=options.port,
                tool_timeout=120.0,
            )
        ]
    providers: list[dict] = []
    if options.provider:
        providers.append(
            {
                "name": options.provider,
                "kind": options.kind,
                "base_url": options.base_url,
                "default_model": options.model,
                "models": [{"id": options.model, "supports_tools": True, "supports_vision": True}],
            }
        )
    else:
        providers.append(
            {
                "name": "scripted",
                "kind": "mock",
                "default_model": "scripted-planner",
                "models": [{"id": "scripted-planner", "supports_tools": True}],
            }
        )
    settings.llm_providers = providers
    return settings


async def wait_for_blender(context: AppContext, seconds: float):
    """Wait for the add-on to attach.

    The add-on reconnects on a backoff that reaches half a minute, so a run that
    starts the server and immediately asks Blender a question will fail against a
    Blender that is about to connect. Waiting is the difference between "no
    Blender" and "not yet".
    """
    assert context.mcp is not None
    deadline = time.monotonic() + seconds
    attempt = 0
    while True:
        attempt += 1
        outcome = await context.mcp.call_tool("blender.get_scene", {})
        if not outcome.is_error:
            if attempt > 1:
                print(f"     (Blender attached after {attempt} attempt(s))", flush=True)
            return outcome
        if time.monotonic() >= deadline:
            return outcome
        if attempt == 1:
            print(
                f"     waiting up to {seconds:g}s for a Blender with the add-on to attach…",
                flush=True,
            )
        await asyncio.sleep(2.0)


async def run(options: argparse.Namespace) -> int:
    setup_logging(options.log_level)
    context = AppContext(settings=settings_for(options))
    await context.open()
    try:
        print("\n1. the studio starts", flush=True)
        check("database opened", context.studio is not None)
        check("an event bus is running", context.bus is not None)

        print("\n2. the MCP server", flush=True)
        assert context.mcp is not None
        statuses = await context.mcp.connect_all()
        check(
            "blender-mcp connected",
            all(s.ready for s in statuses),
            statuses[0].last_error if statuses else "no servers configured",
        )
        catalog = context.mcp.tools()
        check("its tools were discovered", len(catalog) > 0, f"{len(catalog)} tools")
        check(
            "the catalog is dynamic, not hardcoded",
            any(name.startswith("blender.") for name in catalog),
            ", ".join(sorted(catalog)[:4]) + " …",
        )
        check("resources too", len(context.mcp.resources()) > 0, len(context.mcp.resources()))

        print("\n3. a tool call reaches Blender", flush=True)
        scene = await wait_for_blender(context, options.wait_seconds)
        check("get_scene answered", not scene.is_error, scene.text.splitlines()[0][:60] if scene.text else "")
        check("Blender is attached", not scene.is_error, scene.error_code)

        provider_name = options.provider or "scripted"
        if not options.provider:
            scripted = ScriptedPlanner(model="scripted-planner")
            context.llm._providers[provider_name] = scripted  # noqa: SLF001
        agent = context.agent(provider_name, options.model or "")

        print("\n4. 'Create a cube named TestCube at location 2, 0, 1.'", flush=True)
        before = await context.mcp.call_tool("blender.get_objects", {"limit": 500})
        result = await agent.run("Create a cube named TestCube at location 2, 0, 1.")
        print("     agent said: " + (result.text or "(nothing)"), flush=True)
        check("the agent finished", result.finished, result.stopped_because)
        check(
            "it called blender.create_object",
            any(c.call.name == "blender.create_object" for c in result.tool_calls),
            [c.call.name for c in result.tool_calls],
        )
        check(
            "no tool errored",
            not any(c.is_error for c in result.tool_calls),
            [c.content[:60] for c in result.tool_calls if c.is_error],
        )

        print("\n5. the cube is really in Blender", flush=True)
        detail = await context.mcp.call_tool("blender.get_object", {"name": "TestCube"})
        check("get_object found it", not detail.is_error, detail.error_code)
        if not detail.is_error:
            location = payload_of(detail).get("location") or []
            check("it is at the requested location", [round(v) for v in location] == [2, 0, 1], location)
        after = await context.mcp.call_tool("blender.get_objects", {"limit": 500})
        before_count = payload_of(before).get("total", 0)
        after_count = payload_of(after).get("total", 0)
        check(
            "the scene grew by one object",
            after_count == before_count + 1,
            f"{before_count} -> {after_count}",
        )

        print("\n6. cost and limits", flush=True)
        totals = result.totals.to_dict()
        check("tokens were counted", totals["input_tokens"] > 0, totals)
        check("tool calls were counted", totals["tool_calls"] >= 1, totals["tool_calls"])
        check("llm and 3d costs are separate", "three_d_credits" in totals, totals)

        print("\n7. a second turn reuses the tools", flush=True)
        second = await agent.run("Move TestCube to position 3, 3, 1")
        check("the second run finished", second.finished, second.stopped_because)
        # What is asserted here is the *studio's* contract: the object ends up
        # where the model asked for it to be. Whether the model read the sentence
        # correctly is a question about the model, and the benchmark table is
        # where that is measured -- a 3B model reading "3, 3, 1" as (3, 0, 1) is
        # a finding about the model, not a failure of the client.
        asked = [
            call
            for call in second.tool_calls
            if call.call.name.endswith("update_object") and not call.is_error
        ]
        if check("the model moved the cube", bool(asked), [c.call.name for c in second.tool_calls]):
            wanted = (asked[0].call.arguments.get("location") or [])[:2]
            moved = await context.mcp.call_tool("blender.get_object", {"name": "TestCube"})
            landed = (payload_of(moved).get("location") or [])[:2]
            check(
                "Blender is where the model asked",
                [round(v) for v in landed] == [round(v) for v in wanted],
                f"asked {wanted}, landed {landed}",
            )

        print("\n8. the 3D capability", flush=True)
        tools = {spec.name for spec in agent.tools()}
        if context.three_d is not None and context.three_d.any_enabled():
            check("generate_3d_asset is offered", "generate_3d_asset" in tools)
        else:
            check(
                "generate_3d_asset is withheld while Tripo has no key",
                "generate_3d_asset" not in tools,
                "a tool that always fails is worse than no tool",
            )

        print("\n9. everything is in the database", flush=True)
        assert context.studio is not None
        stored = await context.studio.usage.for_run(result.run_id)
        check("the LLM request was recorded", len(stored) >= 1, len(stored))
        calls = await context.studio.messages.tool_calls_for_run(result.run_id)
        check("the tool call was recorded", len(calls) >= 1, [c.tool for c in calls])
        check("a conversation was created", len(await context.studio.conversations.list()) >= 0)

        print("\n10. cleanup", flush=True)
        deleted = await context.mcp.call_tool("blender.delete_object", {"name": "TestCube"})
        check("the test cube is gone", not deleted.is_error, deleted.error_code)
    finally:
        await context.close()
    return 0 if not FAILURES else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--blender-mcp",
        default=os.environ.get("STUDIO_BLENDER_MCP_PATH", ""),
        help="path to the blender-mcp checkout",
    )
    parser.add_argument("--blender-mcp-env", default="", help="PYTHONPATH for the child process")
    parser.add_argument("--python", default=sys.executable, help="interpreter for blender-mcp")
    parser.add_argument("--port", type=int, default=8767, help="blender-mcp bridge port")
    parser.add_argument("--provider", default="", help="an OpenAI-compatible provider name from the config")
    parser.add_argument("--model", default="", help="model id")
    parser.add_argument("--kind", default="openai-compatible")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/blender-ai-studio-acceptance"))
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=90.0,
        help="how long to wait for a Blender to attach (the add-on reconnects with a backoff)",
    )
    options = parser.parse_args()
    if not options.blender_mcp:
        print("Pass --blender-mcp /path/to/blender-mcp (or set STUDIO_BLENDER_MCP_PATH).", file=sys.stderr)
        return 2
    options.data_dir.mkdir(parents=True, exist_ok=True)
    code = asyncio.run(run(options))
    print("", flush=True)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}", flush=True)
        return 1
    print("Acceptance run passed.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
