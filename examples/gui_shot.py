"""Run one real turn through the window, with a real model, and photograph it.

The unit tests drive the GUI with fabricated events. This drives it with a real
agent, a real MCP server and a real model, then grabs the widget — so what the
screenshot shows is what a person would have seen, not a mock-up.

    python examples/gui_shot.py --blender-mcp /path/to/blender-mcp \
        --base-url http://127.0.0.1:11400/v1 --model local-qwen
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Before Qt, or the platform plugin is already chosen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core.context import AppContext  # noqa: E402
from app.core.settings import MCPServerConfig, Settings  # noqa: E402
from app.llm.registry import ProviderConfig  # noqa: E402


def settings_for(options: argparse.Namespace) -> Settings:
    settings = Settings()
    settings.data_dir = options.data_dir
    settings.log_level = "WARNING"
    settings.mcp_servers = [
        MCPServerConfig(
            name="Blender MCP",
            command=options.python,
            args=["-m", "server.main"],
            cwd=options.blender_mcp,
            env={"PYTHONPATH": options.blender_mcp_env or options.blender_mcp},
            blender_port=options.port,
            tool_timeout=180.0,
        )
    ]
    settings.llm_providers = [
        ProviderConfig(
            name=options.provider,
            kind="openai-compatible",
            base_url=options.base_url,
            default_model=options.model,
            models=[
                {
                    "id": options.model,
                    "display_name": options.model,
                    "supports_tools": True,
                    "supports_vision": False,
                    "context_window": 8192,
                    "input_price": 0.0,
                    "output_price": 0.0,
                }
            ],
        )
    ]
    return settings


async def run(options: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from app.gui.bridge import CoreThread
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Blender AI Studio")

    context = await AppContext(settings=settings_for(options)).open()
    core = CoreThread(context)
    core.start()
    window = MainWindow(context, core)
    window.resize(1280, 860)
    window.show()
    window.start()

    def pump(seconds: float) -> None:
        import time

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)

    async def settle(limit: float = 240.0) -> None:
        end = asyncio.get_running_loop().time() + limit
        while asyncio.get_running_loop().time() < end:
            app.processEvents()
            await asyncio.sleep(0.02)
            if window.chat.send.isEnabled() and "user:" in window.chat.transcript_text():
                return

    await settle(60.0)  # the MCP handshake and the model list
    print(f"model selector: {window.model_selector.count()} model(s)", flush=True)

    window.model_selector.setCurrentIndex(0)
    await settle(30.0)  # the add-on's reconnect

    print("sending:", options.prompt, flush=True)
    window.send(options.prompt)
    await settle()
    pump(0.5)

    shot = options.out
    shot.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(shot))
    print(f"screenshot: {shot} ({shot.stat().st_size} bytes)", flush=True)

    # One shot of each page the run touched, so the panels are seen doing their
    # job rather than described.
    for index, name in enumerate(("Chat", "Scene", "Tasks", "Projects", "Benchmark")):
        window.pages.setCurrentIndex(index)
        pump(0.3)
        page = shot.with_name(f"{shot.stem}-{name.lower()}{shot.suffix}")
        window.grab().save(str(page))
        print(f"  {name}: {page}", flush=True)
    window.pages.setCurrentIndex(0)
    pump(0.2)
    print("--- transcript ---", flush=True)
    print(window.chat.transcript_text(), flush=True)
    print("--- footer:", window.cost_status.text(), flush=True)
    print("--- status:", window.mcp_status.text(), "|", window.llm_status.text(), flush=True)

    core.stop()
    await context.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blender-mcp", required=True)
    parser.add_argument("--blender-mcp-env", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--prompt", default="Create a cube named ShotCube at location 1, 2, 0, then move it to 3, 0, 0."
    )
    parser.add_argument("--out", type=Path, default=Path("out/studio-shot.png"))
    parser.add_argument("--data-dir", type=Path, default=Path("out/data"))
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
