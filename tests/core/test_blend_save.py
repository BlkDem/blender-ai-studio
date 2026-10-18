import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.blend_save import blender_path, save_blend


def bridge(report, error=False):
    return SimpleNamespace(
        tools=lambda: {"blender.execute_python": object()},
        call_tool=AsyncMock(return_value=SimpleNamespace(is_error=error, text=json.dumps(report))),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pack", [True, False])
async def test_verified_save(pack):
    path = "/mnt/c/work/it's a scene.blend"
    b = bridge({"result": {"result": {"studio_save": True, "finished": True, "filepath": "C:/work/it's a scene.blend"}}})
    assert await save_blend(b, path, pack=pack) == path
    code = b.call_tool.call_args.args[1]["code"]
    compile(code, "save", "exec")
    assert ("pack_all" in code) is pack
    assert "open_mainfile" not in code
    assert repr("C:/work/it's a scene.blend") in code


@pytest.mark.asyncio
@pytest.mark.parametrize("report", [{"success": True}, {"studio_save": True, "finished": False}, {"studio_save": True, "finished": True, "filepath": "wrong.blend"}])
async def test_unverified_rejected(report):
    with pytest.raises(RuntimeError):
        await save_blend(bridge(report), "/scene.blend")


@pytest.mark.asyncio
async def test_errors_and_gate():
    with pytest.raises(RuntimeError, match="failed"):
        await save_blend(bridge("pack error", True), "/scene.blend")
    b = bridge({})
    b.tools = lambda: {}
    with pytest.raises(RuntimeError, match="disabled"):
        await save_blend(b, "/scene.blend")
    b.call_tool.assert_not_called()


def test_paths():
    assert blender_path(r"C:\work\scene.blend") == "C:/work/scene.blend"
    assert blender_path("/mnt/c/work/scene.blend") == "C:/work/scene.blend"
    assert blender_path("/home/me/scene.blend") == "/home/me/scene.blend"
