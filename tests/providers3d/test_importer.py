"""Getting an asset into the scene.

The import is the half of the 3D pipeline that runs inside Blender, so the things
worth testing are the refusals: a disabled gate, a file Blender cannot see, and a
Blender that says no. Each has to arrive as an explanation rather than a
``RuntimeError`` from inside a tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ThreeDError
from app.mcp.manager import MCPManager
from app.mcp.models import ToolDescriptor, ToolOutcome
from app.providers3d.importer import (
    EXECUTE_PYTHON,
    FUTURE_IMPORT_TOOL,
    can_import,
    import_asset,
)

GLB = Path("/tmp/opencode/assets/Fixture.glb")


class ImportMCP(MCPManager):
    """A bridge that answers the two tools the import needs."""

    def __init__(self, *, tools: dict[str, ToolDescriptor] | None = None) -> None:
        super().__init__()
        self._catalog = tools or {
            EXECUTE_PYTHON: ToolDescriptor(name=EXECUTE_PYTHON, description="Run Python"),
            "blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List"),
        }
        self.calls: list[tuple[str, dict]] = []
        self.lookups = 0

    def tools(self):  # type: ignore[override]
        return dict(self._catalog)

    def tool_specs(self):  # type: ignore[override]
        from app.llm.base import ToolSpec

        return [ToolSpec(name=t.name, description=t.description) for t in self._catalog.values()]

    def tool_instructions(self) -> str:
        return ""

    async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
        self.calls.append((name, arguments or {}))
        if name == EXECUTE_PYTHON:
            # All the operator says. The report has to come from the scene diff.
            return ToolOutcome(call_id="1", tool=name, text=json.dumps({"result": {"FINISHED": ""}}))
        if name == FUTURE_IMPORT_TOOL:
            return ToolOutcome(
                call_id="1", tool=name, text=json.dumps({"imported": ["Imported_1"], "scene_total": 4})
            )
        if name == "blender.get_objects":
            self.lookups += 1
            names = ["Cube"] if self.lookups == 1 else ["Cube", "Imported_1", "Imported_2"]
            return ToolOutcome(
                call_id="1",
                tool=name,
                text=json.dumps({"objects": [{"name": n} for n in names], "total": len(names)}),
            )
        return ToolOutcome(call_id="1", tool=name, text="{}", is_error=True)


@pytest.fixture
def glb(tmp_path: Path) -> Path:
    path = tmp_path / "asset.glb"
    path.write_bytes(b"glTF" + b"\x00" * 32)
    return path


async def test_a_gated_blender_is_told_which_tool_is_missing(glb: Path) -> None:
    bridge = ImportMCP(
        tools={"blender.get_objects": ToolDescriptor(name="blender.get_objects", description="List")}
    )
    assert can_import(bridge) is False
    with pytest.raises(ThreeDError) as excinfo:
        await import_asset(bridge, glb)
    assert EXECUTE_PYTHON in (excinfo.value.hint or "")
    assert FUTURE_IMPORT_TOOL in (excinfo.value.hint or ""), "and names the tool that would replace it"


def test_a_path_blender_cannot_open_is_explained_as_a_filesystem_boundary() -> None:
    """A WSL-hosted studio driving a Windows Blender is the common case: the file
    is right there and Blender still cannot open it.

    There is no way to ask Blender in advance -- ``import os`` is blocked in the
    execution sandbox on purpose -- so the import is attempted and the error is
    read. The message has to name the boundary, or the user goes hunting for a
    corrupt file that is perfectly fine.
    """
    from app.providers3d.importer import _explain

    problem = _explain("RuntimeError: Error: Cannot open file", Path("/mnt/c/Users/x/out/a.glb"))
    assert "cannot see this filesystem" in problem.message
    assert "C:/Users/x/out/a.glb" in (problem.hint or ""), "and the path actually given to Blender"


def test_an_unreadable_file_elsewhere_is_a_different_problem() -> None:
    from app.providers3d.importer import _explain

    problem = _explain("Error: No such file", Path("/tmp/a.glb"))
    assert "could not read the asset" in problem.message
    assert "truncated" in (problem.hint or "")


def test_a_refusal_that_is_not_about_the_file_is_reported_as_a_refusal() -> None:
    from app.providers3d.importer import _explain

    problem = _explain("RuntimeError: poll() failed", Path("/tmp/a.glb"))
    assert "refused the import" in problem.message


async def test_a_missing_local_file_is_refused(glb: Path) -> None:
    with pytest.raises(ThreeDError) as excinfo:
        await import_asset(ImportMCP(), glb.parent / "nope.glb")
    assert "not on this machine" in excinfo.value.message


async def test_an_import_reports_what_arrived(glb: Path) -> None:
    """The operator only says FINISHED, so the report is a diff of the scene.
    A name the operator invents instead would be a guess presented as a fact."""
    bridge = ImportMCP()
    report = await import_asset(bridge, glb)
    assert report.imported == ["Imported_1", "Imported_2"]
    assert report.scene_total == 3
    assert report.object_count == 2
    assert report.via == EXECUTE_PYTHON
    assert report.summary()["new_objects"] == 2


async def test_an_import_that_brings_no_object_is_not_a_success(glb: Path) -> None:
    """A GLB with no geometry is a silent failure if reported as done."""

    class Empty(ImportMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
            if name == "blender.get_objects":
                return ToolOutcome(call_id="1", tool=name, text=json.dumps({"objects": [{"name": "Cube"}]}))
            return await super().call_tool(name, arguments, timeout=timeout)

    with pytest.raises(ThreeDError) as excinfo:
        await import_asset(Empty(), glb)
    assert "no object appeared" in excinfo.value.message


async def test_a_blender_that_refuses_raises_with_its_reason(glb: Path) -> None:
    class Refuses(ImportMCP):
        async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]
            if name == EXECUTE_PYTHON and "os.path.exists" not in (arguments or {}).get("code", ""):
                return ToolOutcome(call_id="1", tool=name, text="GLTF loader is missing", is_error=True)
            return await super().call_tool(name, arguments, timeout=timeout)

    with pytest.raises(ThreeDError) as excinfo:
        await import_asset(Refuses(), glb)
    assert "refused the import" in excinfo.value.message
    assert "GLTF loader is missing" in excinfo.value.message


async def test_the_future_import_tool_would_be_used_when_it_exists(glb: Path) -> None:
    """The path this project is written towards, exercised before it exists."""
    bridge = ImportMCP(
        tools={FUTURE_IMPORT_TOOL: ToolDescriptor(name=FUTURE_IMPORT_TOOL, description="Import an asset")}
    )
    report = await import_asset(bridge, glb)
    assert report.via == FUTURE_IMPORT_TOOL
    assert [name for name, _ in bridge.calls] == [FUTURE_IMPORT_TOOL], "no execute_python needed"
    assert EXECUTE_PYTHON not in bridge.tools(), "and the gate was never opened"


def test_a_wsl_path_is_given_to_blender_in_a_form_it_can_open() -> None:
    from app.providers3d.importer import _windows_path

    assert _windows_path(Path("/mnt/c/Users/x/out/a.glb")) == "C:/Users/x/out/a.glb"
    assert _windows_path(Path("/tmp/a.glb")) == "/tmp/a.glb", "a Linux path is left alone"
    assert _windows_path(Path("/mnt/d/out/a.glb")) == "D:/out/a.glb"
