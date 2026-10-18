from pathlib import Path

import pytest

from app.core.context import AppContext
from app.core.settings import Settings


def context(tmp_path: Path):
    return AppContext(settings=Settings(data_dir=tmp_path))


def test_read_tool_is_registered_with_safe_schema(tmp_path: Path) -> None:
    tool = context(tmp_path).file_read_tools()[0]
    assert tool.name == "read_text_file"
    assert tool.schema["required"] == ["path"]
    assert tool.schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_read_tool_returns_bounded_document(tmp_path: Path) -> None:
    path = tmp_path / "brief.md"
    path.write_text("# Island\nmake it beautiful", encoding="utf-8")
    result = await context(tmp_path).file_read_tools()[0].handler({"path": str(path), "max_chars": 12})
    assert result["filename"] == "brief.md"
    assert result["text"] == "# Island\nmak"
    assert result["truncated"] is True
    assert result["encoding"] == "utf-8"


@pytest.mark.asyncio
async def test_read_tool_rejects_extra_arguments_and_binary_files(tmp_path: Path) -> None:
    tool = context(tmp_path).file_read_tools()[0]
    assert (await tool.handler({"path": "x.md", "write": True})).startswith("Error:")
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"BLENDER-v300")
    result = await tool.handler({"path": str(blend)})
    assert "Blender/3D" in result


@pytest.mark.asyncio
async def test_read_tool_rejects_invalid_arguments(tmp_path: Path) -> None:
    tool = context(tmp_path).file_read_tools()[0]
    assert (await tool.handler({})).startswith("Error:")
    assert (await tool.handler({"path": "x.md", "max_chars": True})).startswith("Error:")
