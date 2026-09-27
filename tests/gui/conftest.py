"""Shared GUI fixtures.

The window fixture lives here because two test modules want a real window on a
real context -- the transcript and the projects -- and a fixture copied into both
would drift.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_widget_left_behind(qapp: QApplication) -> None:
    """Close every top-level widget once the test is done.

    A widget still alive when the interpreter tears Qt down takes the process with
    it: the suite finishes green and then dumps core, which reads like a flaky test
    and is not one.
    """
    yield
    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
async def window(qapp, database_path: Path):
    """A real window on a real context, with a fake bridge and a scripted model."""
    import json as _json

    from app.core.context import AppContext
    from app.core.settings import MCPServerConfig, Settings
    from app.gui.bridge import CoreThread
    from app.gui.main_window import MainWindow
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn
    from app.llm.registry import ProviderConfig
    from app.mcp.manager import MCPManager
    from app.mcp.models import ToolDescriptor

    class FakeMCP(MCPManager):
        def tool_specs(self):  # type: ignore[override]
            from app.llm.base import ToolSpec

            return [ToolSpec(name="blender.get_scene", description="Summarise the scene")]

        def tools(self):  # type: ignore[override]
            return {"blender.get_scene": ToolDescriptor(name="blender.get_scene", description="Summarise")}

        def tool_instructions(self) -> str:
            return ""

        async def call_tool(self, name, arguments=None, *, timeout=None):  # type: ignore[override]

            from app.mcp.models import ToolOutcome

            if name == "blender.get_scene":
                return ToolOutcome(
                    call_id="c",
                    tool=name,
                    text=_json.dumps(
                        {
                            "scene": "Scene",
                            "objects": [],
                            "objects_count": 0,
                            "objects_shown": 0,
                            "objects_truncated": False,
                            "render_engine": "CYCLES",
                            "frame": 1,
                        }
                    ),
                )
            return ToolOutcome(call_id="c", tool=name, text=_json.dumps({"ok": True}))

    settings = Settings(data_dir=database_path.parent)
    settings.mcp_servers = [MCPServerConfig(name="Blender MCP")]
    settings.llm_providers = [
        ProviderConfig(
            name="scripted",
            kind="mock",
            default_model="scripted-model",
            models=[{"id": "scripted-model", "supports_tools": True}],
        )
    ]
    context = await AppContext(settings=settings).open()
    context.mcp = FakeMCP(context.bus)
    provider = MockLLMProvider(
        [
            ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
            ScriptedTurn(text="The scene is empty."),
        ],
        model="scripted-model",
    )
    context.llm.set_provider("scripted", provider)

    core = CoreThread(context)
    core.start()
    win = MainWindow(context, core)
    win.show()
    win.start()  # the real entry point does this; the model selector fills from it
    for _ in range(20):
        qapp.processEvents()
        await asyncio.sleep(0.01)
    yield win
    # Qt tears widgets down at interpreter exit, and a live MainWindow meeting a
    # dead QApplication segfaults on the way out -- the suite would end in a core
    # dump with every test green. Close things in the order a person would.
    await context.close()
    win.close()
    win.deleteLater()
    core.stop()
    qapp.processEvents()
