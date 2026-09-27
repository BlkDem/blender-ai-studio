"""The headless benchmark, driven the way the CLI drives it.

The point of the command is that a comparison can be repeated, so what is tested
here is the whole route: the arguments, the suite, the saved rows, and the ids the
table hands to whoever scores it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.benchmark.storage import BenchmarkStorage
from app.core.settings import Settings
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
from app.llm.registry import LLMRegistry, ProviderConfig
from app.mcp.models import ServerInfo, ToolDescriptor, ToolOutcome
from app.storage.repositories import Studio


class BenchMCP:
    """A bridge that answers, and records that it was connected."""

    def __init__(self) -> None:
        self.connected = False

    async def connect_all(self) -> None:
        self.connected = True

    def any_connected(self) -> bool:
        return self.connected

    def tools(self):  # type: ignore[override]
        return {"blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List")}

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [ToolSpec(name="blender.get_objects", description="List")]

    def tool_instructions(self) -> str:
        return ""

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        return ToolOutcome(
            call_id="c", tool=name, text=json.dumps({"objects": [{"name": "Cube"}], "total": 1})
        )

    def servers(self) -> list[ServerInfo]:
        return [ServerInfo(name="blender", connected=True)]


class FakeContext:
    """Just the surface ``run_benchmark`` uses."""

    def __init__(self, tmp_path: Path, db: object) -> None:
        self.studio = type("Studio", (), {"db": db})()
        self.mcp = BenchMCP()
        self.bus = None
        self.settings = Settings(data_dir=tmp_path)
        self.llm = LLMRegistry(
            [
                ProviderConfig(
                    name=name, kind="mock", default_model=f"{name}-model", models=[{"id": f"{name}-model"}]
                )
                for name in ("one", "two")
            ]
        )
        for name in ("one", "two"):
            self.llm.set_provider(
                name,
                MockLLMProvider(
                    [
                        ScriptedTurn(tool_calls=[("blender.get_objects", {})]),
                        ScriptedTurn(text="One object."),
                    ],
                    model=f"{name}-model",
                ),
            )


@pytest.fixture
async def cli(tmp_path: Path):
    from app.main import arguments, run_benchmark

    studio_instance = await Studio.open(tmp_path / "studio.db")
    context = FakeContext(tmp_path, studio_instance.db)
    storage = BenchmarkStorage(studio_instance.db)

    async def call(*argv: str) -> int:
        options = arguments(list(argv))
        options.data_dir = tmp_path
        return await run_benchmark(context, options)  # type: ignore[arg-type]

    yield call, storage, tmp_path
    studio_instance.close()


async def test_a_headless_comparison_prints_a_table_and_saves_every_run(cli, capsys) -> None:
    call, storage, _tmp_path = cli
    code = await call("--benchmark", "one:one-model,two:two-model", "--prompt", "what is here?")
    out = capsys.readouterr().out

    assert code == 0
    assert "one-model @ one" in out and "two-model @ two" in out
    assert "No model is ranked above another" in out, "the table stays a table, not a verdict"

    suites = await storage.suites()
    assert suites and suites[0]["runs"] == 2
    rows = await storage.runs(suites[0]["id"])
    shown = {line.split()[-1] for line in out.splitlines() if line.startswith(("one-model", "two-model"))}
    assert {row["id"] for row in rows} == shown, "the table shows the id of the run that happened"
    assert all(row["transcript"] for row in rows), "each run keeps what it did"
    assert all(row["mcp_calls"] == 1 for row in rows)


async def test_a_run_that_cannot_happen_makes_the_command_exit_non_zero(cli) -> None:
    call, _storage, _tmp_path = cli
    assert await call("--benchmark", "one:one-model", "--prompt", "what is here?") == 0
    # A provider that is not configured: the run fails, and a command that
    # printed a table anyway would read as a comparison that happened.
    assert await call("--benchmark", "ghost:ghost-model", "--prompt", "what is here?") != 0


async def test_a_benchmark_with_no_prompts_says_so(cli, capsys) -> None:
    call, _storage, _tmp_path = cli
    assert await call("--benchmark", "one:one-model") != 0
    assert "needs prompts" in capsys.readouterr().err


async def test_a_benchmark_with_no_models_says_so(cli, capsys) -> None:
    call, _storage, _tmp_path = cli
    assert await call("--benchmark", "  ", "--prompt", "what is here?") != 0
    assert "at least one provider:model" in capsys.readouterr().err


async def test_prompts_can_come_from_a_file(cli, capsys) -> None:
    call, storage, tmp_path = cli
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("# a comment\n\nwhat is here?\nand what is that?\n", encoding="utf-8")
    assert await call("--benchmark", "one:one-model", "--prompts", str(prompts)) == 0
    suites = await storage.suites()
    assert suites[0]["runs"] == 2, "one run per prompt; comments and blank lines are not prompts"


async def test_a_saved_run_can_be_reviewed(cli, capsys) -> None:
    """The ids in the table exist so a person can score the run they are looking at."""
    call, storage, _tmp_path = cli
    await call("--benchmark", "one:one-model", "--prompt", "what is here?")
    out = capsys.readouterr().out
    run_id = [line.split()[-1] for line in out.splitlines() if line.startswith("one-model")][0]

    await storage.review(run_id, geometry=3, overall=4, notes="clear enough")
    reviews = await storage.reviews(run_id)
    assert reviews[0]["overall"] == 4
    assert reviews[0]["notes"] == "clear enough"
