"""The documentation, checked against the code.

A document that is wrong is worse than no document: it is trusted, and it is
wrong. These tests read the docs and assert the claims in them, so a rename
cannot leave the README describing a method that no longer exists.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "CONTRIBUTING.md"]

#: What an example that never reaches Blender has to put in its own help text.
NO_BACKEND_SENTENCE = b"This check never talks to Blender."


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_document_exists_and_is_not_empty() -> None:
    """The set a newcomer is sent to: the README, four documents, contributing."""
    names = {path.relative_to(ROOT).as_posix() for path in DOCS}
    assert "README.md" in names
    assert "CONTRIBUTING.md" in names
    for expected in (
        "docs/architecture.md",
        "docs/providers.md",
        "docs/development.md",
        "docs/troubleshooting.md",
    ):
        assert expected in names, f"{expected} is linked from the README and must exist"
    for path in DOCS:
        assert len(read(path)) > 400, f"{path.name} is a stub"


def test_the_readme_links_to_every_document() -> None:
    readme = read(ROOT / "README.md")
    for path in DOCS:
        if path.name == "README.md":
            continue
        target = path.relative_to(ROOT).as_posix()
        assert f"]({target})" in readme, f"the README does not link {target}"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_every_relative_link_resolves(path: Path) -> None:
    text = read(path)
    for target in re.findall(r"\]\((?!https?:)([^)#]+)", text):
        if target.startswith("mailto:"):
            continue
        assert (path.parent / target).exists(), f"{path.name} links to a missing {target}"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_python_examples_are_valid_python(path: Path) -> None:
    """A snippet that will not paste is a snippet that will not be used."""
    for block in read(path).split("```python")[1:]:
        ast.parse(block.split("```")[0])


def test_the_architecture_document_names_real_pieces() -> None:
    """The point of the document is the map. Check the map against the map."""
    from app.core.events import EventType
    from app.core.settings import MCPServerConfig, Settings
    from app.core.task_manager import TaskState
    from app.llm.registry import PROVIDER_TYPES
    from app.providers3d.importer import EXECUTE_PYTHON, FUTURE_IMPORT_TOOL
    from app.providers3d.registry import PROVIDER_TYPES as THREE_D_TYPES

    text = read(ROOT / "docs/architecture.md")
    for module in (
        "context.py",
        "agent.py",
        "task_manager.py",
        "manager.py",
        "importer.py",
        "bridge.py",
    ):
        assert module in text, f"{module} exists and the document should say so"
    for tool in (EXECUTE_PYTHON, FUTURE_IMPORT_TOOL):
        assert tool in text
    for kind in PROVIDER_TYPES:
        assert kind in text or kind == "openai-compatible", f"the {kind} provider exists"
    assert set(THREE_D_TYPES) <= {"tripo", "mock"}
    assert set(TaskState) >= {"queued", "running", "succeeded", "failed", "cancelled"}
    assert len(list(EventType)) >= 19
    assert MCPServerConfig.model_fields["kind"].annotation is not None
    assert Settings.model_fields["mcp_servers"] is not None


def test_the_panel_count_matches_the_code() -> None:
    """It said six panels for a while, and the seventh is the projects page."""
    from app.gui.main_window import PAGES

    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    readme = read(ROOT / "README.md")
    said = f"{len(PAGES)} pages" in readme or f"{words.get(len(PAGES), '')} pages" in readme
    assert said, f"the README should say there are {len(PAGES)} pages"
    for page in PAGES:
        assert f"**{page}**" in readme, f"the README's panel table is missing {page}"


def test_every_example_script_runs_its_help() -> None:
    """A script whose arguments have drifted from its documentation is worse
    than no script, because it is what a broken report will be run from.

    A script that reaches Blender must say which backend it drives, so a report
    can be reproduced. One that never does may skip the flag, but has to say so
    in its help -- otherwise the exemption is silent, and the next script to
    need a backend quietly inherits it.
    """
    import subprocess
    import sys

    for script in sorted((ROOT / "examples").glob("*.py")):
        done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, timeout=120)
        assert done.returncode == 0, f"{script.name} --help failed: {done.stderr.decode()[:200]}"
        needs_backend = b"--blender-mcp" in done.stdout
        says_it_does_not = NO_BACKEND_SENTENCE in done.stdout
        assert needs_backend or says_it_does_not, (
            f"{script.name} neither takes --blender-mcp nor says '{NO_BACKEND_SENTENCE.decode()}'"
        )


def test_the_development_document_lists_the_commands_that_exist() -> None:
    import subprocess
    import sys

    documented = read(ROOT / "docs/development.md")
    help_text = subprocess.run(
        [sys.executable, "-m", "app.main", "--help"], capture_output=True, timeout=120
    ).stdout.decode()
    known = set(re.findall(r"(--[a-z][a-z-]+)", help_text))
    for flag in ("--check", "--prompt", "--list-tools", "--benchmark", "--blend", "--data-dir"):
        assert flag in known, f"{flag} is in the documentation, so it must exist"
    for flag in ("--check", "--prompt", "--list-tools", "--benchmark"):
        assert flag in documented


def test_the_troubleshooting_document_quotes_messages_that_still_exist() -> None:
    """The value of this document is that you can search for the text you saw."""
    from app.core.settings import DEFAULT_DATA_DIR
    from app.providers3d.importer import EXECUTE_PYTHON, FUTURE_IMPORT_TOOL
    from app.storage.migrations import LATEST_VERSION

    text = read(ROOT / "docs/troubleshooting.md")
    assert "No Blender instance is connected" in text, "the add-on's own refusal"
    assert "You don't have enough credit to create this task" in text, "Tripo's own refusal"
    assert "but no object appeared" in text, "the importer's own message"
    assert "STUDIO_AGENT__MAX_STEPS" in text
    assert "ALLOW_PYTHON_EXECUTION" in text
    assert "studio.db" in text
    assert "is no longer available to new users" in text, "a retired model, quoted as Google words it"
    # Built the way the window builds it, so the quote cannot drift from the message.
    from app.core.errors import BudgetExceeded

    reached = BudgetExceeded("agent steps", 30, 30).message
    assert "agent steps limit reached (30 of 30)" in reached
    assert reached.splitlines()[0] in text, "the step limit, quoted as the window words it"
    assert DEFAULT_DATA_DIR.name == "blender-ai-studio"
    assert EXECUTE_PYTHON.endswith("execute_python")
    assert FUTURE_IMPORT_TOOL.endswith("import_asset")
    assert LATEST_VERSION >= 3


def test_the_providers_document_matches_the_interfaces() -> None:
    from app.core.agent import LocalTool
    from app.llm.base import LLMProvider
    from app.llm.models import ModelInfo
    from app.providers3d.base import ThreeDProvider

    text = read(ROOT / "docs/providers.md")
    for method in LLMProvider.__abstractmethods__:
        assert method in text, f"LLMProvider requires {method}() and the guide should say so"
    for method in ThreeDProvider.__abstractmethods__:
        assert method in text, f"ThreeDProvider requires {method}() and the guide should say so"
    for field in LocalTool.__dataclass_fields__:
        assert field in text, f"LocalTool has a {field} field"
    for flag in ("supports_tools", "supports_vision", "context_window"):
        assert flag in ModelInfo.__dataclass_fields__, f"{flag} is a real capability flag"
        assert flag in text, f"and the guide mentions {flag}"


def test_no_document_promises_a_flag_that_does_not_exist() -> None:
    """The one failure mode specific to documentation: a flag that never was."""
    import subprocess
    import sys

    flags_claimed = {match for path in DOCS for match in re.findall(r"`(--[a-z][a-z-]+)", read(path))}
    # Flags that belong to other tools quoted in a command line: llama.cpp's in
    # the local-model example, ruff's and mypy's in the daily loop.
    foreign = {
        "--jinja",
        "--alias",
        "--ctx-size",
        "--help",
        "--fix",
        "--output-format",
        "--select",
        "--ignore",
    }
    flags_claimed -= foreign
    known: set[str] = set()
    # -m, not the file path: the entry point uses relative imports, so running
    # it as a script would fail and report no flags at all.
    targets = [["-m", "app.main"]] + [[str(script)] for script in sorted((ROOT / "examples").glob("*.py"))]
    for target in targets:
        done = subprocess.run([sys.executable, *target, "--help"], capture_output=True, timeout=120)
        known |= set(re.findall(r"(--[a-z][a-z-]+)", done.stdout.decode()))
    for flag in sorted(flags_claimed - known):
        # A flag that belongs to another tool (ruff, mypy, git) is fine.
        if flag in ("--version",):
            continue
        assert flag in known, f"{flag} is documented but no tool accepts it"
