"""GUI tests, offscreen.

The window is not the hard part of this project and is deliberately not the
subject of many tests: what it must do is render the events it is given without
touching the network. So these tests drive it with fabricated events and assert
that what a person would see is there — and one test runs a real agent against a
fake bridge, to prove the event wiring end to end.

They need a display. ``QT_QPA_PLATFORM=offscreen`` is set below, before Qt is
imported, so they run in CI without one.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.context import AppContext  # noqa: E402
from app.core.events import Event, EventBus, EventType  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.gui.benchmark.panel import BenchmarkPanel  # noqa: E402
from app.gui.bridge import CoreThread, elide, format_duration, format_money  # noqa: E402
from app.gui.chat.widget import ChatView  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.gui.models.panel import ModelsPanel  # noqa: E402
from app.gui.scene.panel import ScenePanel  # noqa: E402
from app.gui.settings.panel import SettingsPanel  # noqa: E402
from app.gui.tasks.panel import TasksPanel  # noqa: E402
from app.llm.providers.mock import MockLLMProvider, ScriptedTurn  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_widget_left_behind(qapp: QApplication) -> None:
    """Close every top-level widget once the test is done.

    A widget that is still alive when the interpreter tears Qt down takes the
    process with it: the suite finishes green and then dumps core, which reads
    like a flaky test and is not one. This also turns a leaked window into
    something a test can see.
    """
    yield
    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


def pump(times: int = 20, step: float = 0.01) -> None:
    """Let queued signals arrive."""
    app = QApplication.instance()
    for _ in range(times):
        app.processEvents()  # type: ignore[union-attr]
        time.sleep(step)


# --- formatting helpers -----------------------------------------------------


def test_a_long_label_is_ellipsised() -> None:
    assert elide("a" * 300, 10) == "a" * 9 + "…"
    assert elide("short", 10) == "short"
    assert elide("a\n\n  b   c", 20) == "a b c"


def test_money_keeps_its_precision_at_the_small_end() -> None:
    assert format_money(0) == "$0"
    assert format_money(0.0004) == "$0.0004"
    assert format_money(1.5) == "$1.50"


def test_durations_read_the_way_a_person_says_them() -> None:
    assert format_duration(4.2) == "4.2s"
    assert format_duration(75) == "1m 15s"
    assert format_duration(3725) == "1h 02m"


# --- chat -------------------------------------------------------------------


def test_the_transcript_shows_messages_streaming_and_tool_calls(qapp) -> None:
    view = ChatView()
    view.add_message("user", "Create a table with four legs.")
    bubble = view.start_stream()
    bubble.append("I will build ")
    bubble.append("it now.")
    view.add_tool("c1", "blender.create_object")
    view.finish_tool("c1", is_error=False, text='{"object": {"name": "TableTop"}}', duration_ms=41.8)

    text = view.transcript_text()
    assert "Create a table with four legs." in text
    assert "I will build it now." in text, "the streamed text reassembles exactly"
    assert "✓ blender.create_object · 42 ms" in text
    assert view.summary() == {
        "messages": 2,
        "tools": 1,
        "text_chars": len("Create a table with four legs.") + len("I will build it now."),
    }


def test_a_bubble_hugs_its_text(qapp) -> None:
    """A one-line answer must not occupy a screen.

    Measured after a layout pass: an unshown widget reports a default height,
    which says nothing about how it will look.
    """
    view = ChatView()
    view.resize(900, 600)
    view.show()
    qapp.processEvents()

    bubble = view.add_message("assistant", "Done.")
    qapp.processEvents()
    assert bubble.height() < 80, f"a one-line bubble took {bubble.height()}px"

    bubble.append(" A considerably longer second sentence that wraps on a narrow width.")
    qapp.processEvents()
    assert bubble.height() < 240, "and the cap still holds"

    layout_items = [view._transcript_layout.itemAt(i) for i in range(view._transcript_layout.count())]  # noqa: SLF001
    assert layout_items[0].geometry().height() == bubble.height(), "the spacer takes the rest"


def test_a_sub_second_duration_keeps_its_precision(qapp) -> None:
    """Two calls at 213 and 217 ms must not both read "213 ms"."""
    view = ChatView()
    view.add_tool("a", "t1")
    view.finish_tool("a", is_error=False, text="{}", duration_ms=213.4)
    view.add_tool("b", "t2")
    view.finish_tool("b", is_error=False, text="{}", duration_ms=217.9)
    text = view.transcript_text()
    assert "t1 · 213 ms" in text
    assert "t2 · 218 ms" in text


def test_a_slow_tool_call_is_shown_in_seconds(qapp) -> None:
    view = ChatView()
    view.add_tool("a", "blender.render")
    view.finish_tool("a", is_error=False, text="{}", duration_ms=2450.0)
    assert "blender.render · 2.45 s" in view.transcript_text()


def test_a_failed_tool_call_is_marked_as_a_failure(qapp) -> None:
    view = ChatView()
    view.add_tool("c2", "blender.get_object")
    view.finish_tool("c2", is_error=True, text="not found (OBJECT_NOT_FOUND)", duration_ms=3.0)
    assert "✗ blender.get_object" in view.transcript_text()


def test_the_send_button_is_disabled_while_a_run_is_in_flight(qapp) -> None:
    view = ChatView()
    view.set_busy(True)
    assert view.send.isEnabled() is False
    assert view.stop.isEnabled() is True
    view.set_busy(False)
    assert view.send.isEnabled() is True


def test_a_submitted_message_empties_the_input(qapp) -> None:
    view = ChatView()
    sent: list[str] = []
    view.submitted.connect(sent.append)
    view.input.setText("  hello  ")
    view._submit()  # noqa: SLF001 - the return key path, which the test drives
    assert sent == ["hello"]
    assert view.input.text() == ""


def test_a_long_tool_result_is_ellipsised_in_the_card_but_kept_in_the_tooltip(qapp) -> None:
    view = ChatView()
    view.add_tool("c3", "blender.get_objects")
    payload = "x" * 5000
    view.finish_tool("c3", is_error=False, text=payload, duration_ms=12.0)
    card = view._cards["c3"]  # noqa: SLF001
    assert len(card.detail.text()) < 1000
    assert len(card.full_text()) == 5000


# --- scene ------------------------------------------------------------------


def test_the_scene_panel_reads_a_scene_payload(qapp) -> None:
    panel = ScenePanel()
    panel.show_scene(
        {
            "scene": "Scene",
            "objects": [{"name": "Cube", "type": "MESH", "location": [0, 0, 1], "dimensions": [2, 2, 2]}],
            "objects_count": 1,
            "objects_shown": 1,
            "objects_truncated": False,
            "active_camera": "Camera",
            "render_engine": "CYCLES",
            "frame": 1,
        }
    )
    assert panel.fields["scene"].text() == "Scene"
    assert panel.fields["objects"].text() == "1"
    assert panel.fields["engine"].text() == "CYCLES"
    assert panel.object_count() == 1
    assert panel.tree.topLevelItem(0).text(0) == "Cube"


def test_a_truncated_scene_says_so(qapp) -> None:
    panel = ScenePanel()
    panel.show_scene(
        {
            "scene": "Scene",
            "objects": [],
            "objects_count": 900,
            "objects_shown": 200,
            "objects_truncated": True,
        }
    )
    assert "truncated=True" in panel.fields["objects"].text()


# --- tasks ------------------------------------------------------------------


def test_the_tasks_panel_adds_updates_and_totals(qapp) -> None:
    panel = TasksPanel()
    panel.add_or_update(
        {
            "id": "t1",
            "name": "Tripo generation",
            "provider": "tripo",
            "state": "running",
            "progress": 0.4,
            "duration_s": 42.0,
            "credits": 100,
            "cost_usd": 0.0,
        }
    )
    assert panel.table.rowCount() == 1
    panel.add_or_update(
        {
            "id": "t1",
            "name": "Tripo generation",
            "state": "succeeded",
            "progress": 1.0,
            "duration_s": 90.0,
            "credits": 100,
            "cost_usd": 0.0,
        }
    )
    assert panel.table.rowCount() == 1, "the same task updates its row"
    assert panel.table.item(0, 2).text() == "succeeded"
    assert panel.totals()["credits"] == 100.0


# --- models and settings ----------------------------------------------------


def test_the_models_panel_lists_providers_and_models(qapp) -> None:
    panel = ModelsPanel(secrets_backend="keyring")
    panel.show_providers(
        [
            {
                "name": "space-bunny",
                "kind": "openai-compatible",
                "base_url": "https://x/v1",
                "key": "••••1234",
                "models": [1],
                "ready": True,
            }
        ]
    )
    panel.show_models(
        [
            {
                "id": "space-bunny-free",
                "provider": "space-bunny",
                "supports_tools": True,
                "supports_vision": False,
                "context_window": 128000,
                "input_price": 0,
                "output_price": 0,
            }
        ]
    )
    assert panel.providers_table.item(0, 0).text() == "space-bunny"
    assert panel.providers_table.item(0, 5).text() == "yes"
    assert panel.models_table.item(0, 0).text() == "space-bunny-free"
    assert panel.models_table.item(0, 3).text() == "no", "capabilities are data, and are shown"
    assert panel.provider_key.echoMode().name == "Password", "a key is never shown in the clear"


def test_a_model_selection_is_reported_as_provider_and_id(qapp) -> None:
    panel = ModelsPanel()
    chosen: list[str] = []
    panel.model_selected.connect(chosen.append)
    panel.show_models([{"id": "gpt-4o-mini", "provider": "openai", "supports_tools": True}])
    panel.models_table.selectRow(0)
    assert chosen == ["openai:gpt-4o-mini"]


def test_the_settings_panel_loads_and_reports_dotted_keys(qapp) -> None:
    settings = Settings()
    settings.agent.max_steps = 12
    panel = SettingsPanel()
    panel.load(settings)
    assert panel.max_steps.value() == 12
    changes: list[tuple[str, object]] = []
    panel.setting_changed.connect(lambda key, value: changes.append((key, value)))
    panel.max_steps.setValue(20)
    panel.max_steps.valueChanged.emit(20)
    assert ("agent.max_steps", 20) in changes


# --- benchmark --------------------------------------------------------------


def test_the_benchmark_table_says_when_runs_were_not_isolated(qapp) -> None:
    panel = BenchmarkPanel()
    panel.show_comparison(
        [
            {
                "id": "r1",
                "model": "a-model @ scripted",
                "status": "ok",
                "duration_s": 12.0,
                "tokens": 768,
                "llm_cost_usd": 0.003,
                "mcp_calls": 3,
                "tool_errors": 0,
                "scene_reset": "copy_only",
            },
            {
                "id": "r2",
                "model": "b-model @ scripted",
                "status": "ok",
                "duration_s": 20.0,
                "tokens": 900,
                "llm_cost_usd": 0.004,
                "mcp_calls": 4,
                "tool_errors": 0,
                "scene_reset": "verified",
            },
        ]
    )
    assert panel.table.rowCount() == 2
    assert "1 with a verified starting scene" in panel.summary.text()
    assert "does not rank models" in panel.summary.text()


def test_scoring_a_run_needs_a_selected_row(qapp) -> None:
    panel = BenchmarkPanel()
    saved: list[tuple[str, dict]] = []
    panel.review_requested.connect(lambda run_id, scores: saved.append((run_id, scores)))
    panel.show_comparison([{"id": "r1", "model": "a", "status": "ok", "scene_reset": "verified"}])
    panel.table.selectRow(0)
    panel.spins["Overall"].setText("4")
    panel._save_review()  # noqa: SLF001
    assert saved and saved[0][0] == "r1"
    assert saved[0][1]["Overall"] == 4


# --- the window, with a real agent -----------------------------------------


@pytest.fixture
async def window(qapp, database_path: Path):
    """A real window on a real context, with a fake bridge and a scripted model."""
    from app.core.settings import MCPServerConfig
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
            import json as _json

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


async def test_the_window_runs_a_turn_and_shows_every_step(window, qapp) -> None:
    window.model_selector.setCurrentIndex(0)
    window.send("what is in the scene?")
    for _ in range(60):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if not window.chat.input.isEnabled():
            break
    for _ in range(40):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "scene is empty" in window.chat.transcript_text():
            break

    text = window.chat.transcript_text()
    assert "what is in the scene?" in text
    assert "blender.get_scene" in text, "the tool call is visible, not hidden behind a spinner"
    assert "scene is empty" in text.lower()
    assert "tokens" in window.cost_status.text(), "the footer carries the usage"


async def test_a_run_can_be_stopped_from_the_window(window, qapp) -> None:
    assert window.chat.stop.isEnabled() is False
    window.chat.set_busy(True)
    assert window.chat.stop.isEnabled() is True
    window.stop_run()
    qapp.processEvents()
    assert window.chat.send.isEnabled() is True


async def test_stopping_reaches_the_agent_that_is_actually_running(window, qapp) -> None:
    """The button used to be decorative.

    It set the UI back to idle while the agent carried on to the end, and the
    window then said "Stopped" over a run that had not stopped. The provider
    here takes half a minute, so a run that ends at once can only have been
    cancelled.
    """
    never_finishes = asyncio.Event()

    class Slow(MockLLMProvider):
        async def chat(self, request):  # type: ignore[override]
            await never_finishes.wait()
            raise AssertionError("the model was allowed to finish")

        async def stream(self, request):  # type: ignore[override]
            await never_finishes.wait()
            raise AssertionError("the model was allowed to finish")
            yield  # pragma: no cover - makes this an async generator

    window.context.llm.set_provider("scripted", Slow([], model="scripted-model"))
    window.model_selector.setCurrentIndex(0)
    window.send("what is in the scene?")
    agent = None
    for _ in range(100):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        agent = agent or window._active_agent  # noqa: SLF001
        if agent is not None and window.current_run_id and agent.is_running(window.current_run_id):
            break
    assert agent is not None, "the run started"
    run_id = window.current_run_id

    window.stop_run()
    for _ in range(100):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if not agent.is_running(run_id):
            break
    assert not agent.is_running(run_id), "the agent was cancelled, not just forgotten by the window"
    assert window.chat.send.isEnabled() is True
    never_finishes.set()


async def test_events_from_the_bus_reach_the_window(window, qapp) -> None:
    """The window shows the event bus, not a private copy of the truth."""
    bus = EventBus()
    window.context.bus = bus
    window.core.attach_bus()

    window._on_event(Event(type=EventType.TOKEN, run_id="r", payload={"text": "hello "}))  # noqa: SLF001
    window._on_event(Event(type=EventType.TOKEN, run_id="r", payload={"text": "world"}))  # noqa: SLF001
    window._on_event(
        Event(type=EventType.TOOL_STARTED, run_id="r", payload={"call_id": "c1", "tool": "blender.get_scene"})
    )  # noqa: SLF001
    window._on_event(
        Event(
            type=EventType.TOOL_FINISHED,
            run_id="r",
            payload={"call_id": "c1", "text": '{"objects_count": 0}', "images": 0},
        )
    )  # noqa: SLF001
    qapp.processEvents()

    text = window.chat.transcript_text()
    assert "hello world" in text
    assert "blender.get_scene" in text
