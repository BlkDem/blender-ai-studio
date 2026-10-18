import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.context import AppContext
from app.core.settings import Settings


def context(tmp_path, *, allowed=True, available=True, report=None, error=False):
    ctx = AppContext(Settings(data_dir=tmp_path))
    ctx.settings.agent.allow_execute_python = allowed
    ctx.mcp = SimpleNamespace(
        tools=lambda: {"blender.execute_python": object()} if available else {},
        call_tool=AsyncMock(return_value=SimpleNamespace(
            is_error=error,
            text=json.dumps(report if report is not None else {
                "result": {"studio_save": True, "finished": True, "filepath": "C:/work/scene.blend"}
            }),
        )),
    )
    return ctx


@pytest.mark.asyncio
async def test_success_and_default_packing_without_workspace_update(tmp_path):
    ctx = context(tmp_path)
    ctx.current_project = "active"
    ctx.studio = SimpleNamespace(projects=SimpleNamespace(update=AsyncMock()))
    tool = ctx.blend_save_tools()[0]
    assert tool.name == "save_blend_file"
    assert tool.schema["required"] == ["path"]
    result = await tool.handler({"path": "/mnt/c/work/scene.blend"})
    assert result["saved"] is True
    assert result["workspace_updated"] is False
    assert "pack_all" in ctx.mcp.call_tool.call_args.args[1]["code"]
    ctx.studio.projects.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{}, {"path": "x.txt"}, {"path": 1}, {"path": "x.blend", "pack_resources": "false"}, {"path": "x.blend", "code": "bad"}])
async def test_arguments(tmp_path, arguments):
    ctx = context(tmp_path)
    assert (await ctx.blend_save_tools()[0].handler(arguments)).startswith("Error:")
    ctx.mcp.call_tool.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed,available", [(False, True), (True, False)])
async def test_permission_and_availability(tmp_path, allowed, available):
    ctx = context(tmp_path, allowed=allowed, available=available)
    assert (await ctx.blend_save_tools()[0].handler({"path": "x.blend"})).startswith("Error:")
    ctx.mcp.call_tool.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("report,error", [({"success": True}, False), ({"studio_save": True, "finished": False}, False), ("VALIDATION_ERROR file access", True)])
async def test_failure_is_model_readable(tmp_path, report, error):
    ctx = context(tmp_path, report=report, error=error)
    answer = await ctx.blend_save_tools()[0].handler({"path": "x.blend"})
    assert answer.startswith("Error:")
    assert "Do not claim success" in answer


@pytest.mark.asyncio
@pytest.mark.parametrize("pack", [True, False])
@pytest.mark.parametrize("root", ["/home/maksim/projects/blender-mcp", "/mnt/c/Users/maxim/projects/blender-mcp"])
async def test_generated_code_against_real_mcp_validator(tmp_path, root, pack):
    if not Path(root, "server/validation/python.py").is_file():
        pytest.skip("Optional external MCP checkout is not present")
    ctx = context(tmp_path)
    await ctx.blend_save_tools()[0].handler({"path": "/mnt/c/work/scene.blend", "pack_resources": pack})
    code = ctx.mcp.call_tool.call_args.args[1]["code"]
    script = (
        "import json,sys; from server.validation.python import validate_python; "
        "r=validate_python(json.loads(sys.stdin.read())); "
        "assert r.allowed, r.violations; "
        "assert not validate_python('bpy.data.libraries.write(\"x.blend\", set())').allowed; "
        "assert not validate_python('open(\"x\", \"w\")').allowed; "
        "assert not validate_python('bpy.ops.wm.open_mainfile(filepath=\"x.blend\")').allowed"
    )
    env = dict(os.environ, PYTHONPATH=root)
    outcome = subprocess.run([os.sys.executable, "-c", script], input=json.dumps(code), text=True, capture_output=True, cwd=root, env=env)
    assert outcome.returncode == 0, outcome.stderr
