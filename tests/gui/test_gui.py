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
import base64
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

#: One pixel, so a card can be asked to show a real image.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

from app.core.events import Event, EventBus, EventType  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.gui.benchmark.panel import BenchmarkPanel  # noqa: E402
from app.gui.bridge import elide, format_duration, format_money  # noqa: E402
from app.gui.chat.widget import (  # noqa: E402
    MAX_ATTACHMENTS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    ChatView,
    encode_image,
    image_suffix,
)
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
    sent: list[tuple[str, list[str]]] = []
    view.submitted.connect(lambda text, images: sent.append((text, list(images))))
    view.input.setText("  hello  ")
    view._submit()  # noqa: SLF001 - the return key path, which the test drives
    assert sent == [("hello", [])]
    assert view.input.text() == ""


def test_a_long_tool_result_is_ellipsised_in_the_card_but_kept_in_the_tooltip(qapp) -> None:
    view = ChatView()
    view.add_tool("c3", "blender.get_objects")
    payload = "x" * 5000
    view.finish_tool("c3", is_error=False, text=payload, duration_ms=12.0)
    card = view._cards["c3"]  # noqa: SLF001
    assert len(card.detail.text()) < 1000
    assert len(card.full_text()) == 5000


# --- attached pictures ------------------------------------------------------

#: A 2x2 PNG, built by Qt so the test needs no image library and no fixture file.
TINY_PNG = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFElEQVR4nGP8z4AATAxIYBQMEQAAWAQB9Ji0lGAAAAABJRU5ErkJggg=="
    )
).decode()


def _png(width: int, height: int) -> str:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    return encode_image(image)


def test_a_pasted_picture_is_staged_and_sent_with_the_message(qapp) -> None:
    view = ChatView()
    sent: list[tuple[str, list[str]]] = []
    view.submitted.connect(lambda text, images: sent.append((text, list(images))))
    data = _png(4, 4)
    assert view.add_attachment(data) is True
    assert view.attachments() == [data]
    view.input.setText("build this")
    view._submit()  # noqa: SLF001
    assert sent[0][0] == "build this"
    assert sent[0][1] == [data]


def test_a_picture_alone_is_still_a_message(qapp) -> None:
    """Paste a screenshot, hit Enter, say nothing: that is a turn, not an empty one."""
    view = ChatView()
    sent: list[tuple[str, list[str]]] = []
    view.submitted.connect(lambda text, images: sent.append((text, list(images))))
    view.add_attachment(_png(4, 4))
    view._submit()  # noqa: SLF001
    assert len(sent) == 1
    assert sent[0][0] == ""
    assert len(sent[0][1]) == 1


def test_an_empty_composer_sends_nothing(qapp) -> None:
    view = ChatView()
    sent: list[tuple[str, list[str]]] = []
    view.submitted.connect(lambda text, images: sent.append((text, list(images))))
    view._submit()  # noqa: SLF001
    assert sent == []


def test_attachments_are_taken_not_copied(qapp) -> None:
    """A staged picture goes out once. Left behind, it would ride into a later
    turn the user never attached it to."""
    view = ChatView()
    sent: list[tuple[str, list[str]]] = []
    view.submitted.connect(lambda text, images: sent.append((text, list(images))))
    data = _png(4, 4)
    view.add_attachment(data)
    view.input.setText("first")
    view._submit()  # noqa: SLF001
    view.input.setText("second")
    view._submit()  # noqa: SLF001
    assert sent[0][1] == [data]
    assert sent[1][1] == []
    assert view.attachments() == []


def test_the_attachment_limit_is_reported_not_silently_applied(qapp) -> None:
    view = ChatView()
    refusals: list[str] = []
    view.rejected_attachment.connect(refusals.append)
    data = _png(4, 4)
    for _ in range(MAX_ATTACHMENTS):
        assert view.add_attachment(data) is True
    assert view.add_attachment(data) is False
    assert len(view.attachments()) == MAX_ATTACHMENTS
    assert refusals and str(MAX_ATTACHMENTS) in refusals[0]


def test_a_removed_attachment_does_not_go_out(qapp) -> None:
    view = ChatView()
    view.add_attachment(_png(4, 4))
    view.add_attachment(_png(4, 4))
    view._remove_attachment(0)  # noqa: SLF001
    assert len(view.attachments()) == 1


def test_a_picture_in_the_transcript_is_the_one_that_was_sent(qapp) -> None:
    view = ChatView()
    data = _png(4, 4)
    bubble = view.add_message("user", "build this", images=[data])
    assert bubble.attached == [data]


def test_encoding_shrinks_a_picture_that_is_too_large_to_send(qapp) -> None:
    """A 4K screenshot is megabytes of base64 in a prompt. It is scaled down to
    the side limit, and the caller gets a payload within the byte budget."""
    data = _png(3000, 2000)
    assert len(base64.b64decode(data)) <= MAX_IMAGE_BYTES
    decoded = QImage()
    assert decoded.loadFromData(base64.b64decode(data), "PNG") is True
    assert max(decoded.width(), decoded.height()) <= MAX_IMAGE_SIDE


def test_encoding_refuses_something_that_is_not_an_image(qapp) -> None:
    """A drop of a text file has to leave the composer alone, not raise."""
    assert encode_image(QImage()) == ""
    assert encode_image(QImage(4, 4, QImage.Format.Format_RGB32)) != ""


def test_a_pasted_picture_never_lands_as_text_in_the_composer(qapp) -> None:
    """The failure this replaces: a QLineEdit drops a pasted image with no sign
    it was offered, so the screenshot simply never arrives."""
    from PySide6.QtGui import QClipboard, QGuiApplication

    QGuiApplication.clipboard().setImage(QImage(4, 4, QImage.Format.Format_RGB32), QClipboard.Mode.Clipboard)
    view = ChatView()
    view.input.paste()  # noqa: SLF001 - the Ctrl+V path
    assert view.input.text() == ""
    assert len(view.attachments()) == 1


def test_pasted_text_still_goes_in_as_text(qapp) -> None:
    from PySide6.QtGui import QClipboard, QGuiApplication

    QGuiApplication.clipboard().setText("just words", QClipboard.Mode.Clipboard)
    view = ChatView()
    view.input.paste()  # noqa: SLF001
    assert view.input.text() == "just words"
    assert view.attachments() == []


def test_the_attachment_strip_appears_only_when_something_is_attached(qapp) -> None:
    # isHidden, not isVisible: the window is never shown in a test, so a child of
    # an unshown parent is never "visible" whatever was asked of it.
    view = ChatView()
    assert view._attachment_bar.isHidden() is True  # noqa: SLF001
    view.add_attachment(_png(4, 4))
    assert view._attachment_bar.isHidden() is False  # noqa: SLF001
    view.take_attachments()
    assert view._attachment_bar.isHidden() is True  # noqa: SLF001


# --- saving a picture -------------------------------------------------------


def test_a_tool_render_can_be_written_out(qapp, tmp_path) -> None:
    """A render is the thing a person most often wants to keep, and until now
    the only copy was inside a card that nothing could reach."""
    from app.gui.chat.widget import ToolCallCard

    png = base64.b64encode(_TINY_PNG).decode()
    card = ToolCallCard("blender.render_preview")
    card.set_result(is_error=False, text="{}", duration_ms=10.0, images=1, image_data=[png])
    label = next(w for w in card.findChildren(QLabel) if w.objectName() == "chat-image")
    target = tmp_path / "render.png"
    assert label.save_to(str(target)) is True
    assert target.read_bytes() == base64.b64decode(png)


def test_an_attached_picture_can_be_written_out(qapp, tmp_path) -> None:
    view = ChatView()
    data = _png(4, 4)
    bubble = view.add_message("user", "build this", images=[data])
    label = next(w for w in bubble.findChildren(QLabel) if w.objectName() == "chat-image")
    target = tmp_path / "reference.png"
    assert label.save_to(str(target)) is True
    assert target.read_bytes() == base64.b64decode(data)


def test_a_saved_picture_gets_the_extension_its_bytes_actually_are(qapp) -> None:
    """An attachment is written as JPEG when PNG would be too large, so the name
    a save dialog offers has to come from the bytes and not from the path Qt
    happened to hand us. A .png that is really a JPEG opens in nothing."""
    assert image_suffix(b"\x89PNG\r\n\x1a\nstuff") == ".png"
    assert image_suffix(b"\xff\xd8\xff\xe0stuff") == ".jpg"
    assert image_suffix(b"RIFF\x00\x00\x00\x00WEBPmore") == ".webp"
    assert image_suffix(b"GIF89a") == ".gif"
    assert image_suffix(b"not a picture at all") == ".png"


def test_the_suggested_name_carries_the_right_extension(qapp) -> None:
    view = ChatView()
    bubble = view.add_message("user", "", images=[_png(4, 4)])
    label = next(w for w in bubble.findChildren(QLabel) if w.objectName() == "chat-image")
    assert label.suggested_name().endswith(".png")
    assert "blender-ai-studio-" in label.suggested_name()


def test_a_picture_that_cannot_be_written_says_so_rather_than_raising(qapp) -> None:
    view = ChatView()
    bubble = view.add_message("user", "", images=[_png(4, 4)])
    label = next(w for w in bubble.findChildren(QLabel) if w.objectName() == "chat-image")
    # A directory that cannot be created: the write fails, and the user gets a
    # false rather than a traceback out of a right-click.
    assert label.save_to("/proc/definitely/not/creatable/render.png") is False


def test_finishing_a_card_twice_does_not_stack_two_pictures(qapp) -> None:
    from app.gui.chat.widget import ToolCallCard

    png = base64.b64encode(_TINY_PNG).decode()
    card = ToolCallCard("blender.render_preview")
    card.set_result(is_error=False, text="{}", duration_ms=10.0, images=1, image_data=[png])
    card.set_result(is_error=False, text="{}", duration_ms=20.0, images=1, image_data=[png])
    assert len([w for w in card.findChildren(QLabel) if w.objectName() == "chat-image"]) == 1


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


async def test_a_key_can_be_given_to_a_provider_without_being_redefined(window, qapp) -> None:
    """The key is what was missing; the provider is what works.

    "Add provider" also sets a key, and also replaces the whole configuration
    with a placeholder model. So it was the only way to enable a configured
    provider, and the price was losing the model list it already had.
    """
    from app.storage.secrets import key_name

    before = window.context.llm.config("scripted")
    assert before is not None
    window.models.providers_table.selectRow(0)
    assert window.models._selected_provider() == "scripted", "the row is the target"
    window.models.existing_key.setText("sk-test")
    window.models.save_existing_key.click()
    for _ in range(40):
        qapp.processEvents()
        await asyncio.sleep(0.01)

    assert window.context.secrets.get(key_name("scripted")) == "sk-test"
    after = window.context.llm.config("scripted")
    assert after.models == before.models, "setting a key did not touch the model list"
    assert after.default_model == before.default_model


async def test_a_key_for_a_provider_that_does_not_exist_is_refused(window, qapp) -> None:
    """A key stored under an unknown name looks saved and enables nothing.

    The combo cannot produce that name, but a typed one can -- the 3D provider
    field is a free-text line, and a provider can be removed while its key is
    still on screen.
    """
    from app.storage.secrets import key_name

    window._set_api_key("nonesuch", "sk-test")
    for _ in range(20):
        qapp.processEvents()
        await asyncio.sleep(0.01)

    assert not window.context.secrets.get(key_name("nonesuch"))
    assert "nonesuch" in window.chat.transcript_text()


def test_a_key_can_be_pasted_without_the_keyboard(qapp) -> None:
    """A password field has no standard context menu, and a shortcut that gets
    swallowed once leaves a secret with no way in but typing it out in full."""
    from PySide6.QtGui import QGuiApplication

    from app.gui.models.panel import _paste_into

    panel = ModelsPanel(secrets_backend="file")
    panel.show_providers([{"name": "gemini", "kind": "gemini"}])
    QGuiApplication.clipboard().setText("  sk-from-clipboard\n")

    assert panel.paste_existing_key.isEnabled() is False, "no row, no paste"
    panel.providers_table.selectRow(0)
    panel.paste_existing_key.click()
    assert panel.existing_key.text() == "sk-from-clipboard", "trimmed, and not with a newline"

    _paste_into(panel.existing_key)
    assert panel.existing_key.text() == "sk-from-clipboard", "pasting twice is not two keys"


def test_every_field_that_holds_a_secret_can_be_pasted_into(qapp) -> None:
    """Three fields hold keys. All three have to accept one."""
    from PySide6.QtGui import QGuiApplication

    from app.gui.models.panel import _paste_into

    panel = ModelsPanel(secrets_backend="file")
    QGuiApplication.clipboard().setText("sk-shared")
    for field in (panel.existing_key, panel.provider_key, panel.three_d_key):
        _paste_into(field)
        assert field.text() == "sk-shared"
        assert field.echoMode().name == "Password", "and still does not show it"


def test_the_key_button_saves_for_the_selected_row_and_nowhere_else(qapp) -> None:
    panel = ModelsPanel(secrets_backend="file")
    panel.show_providers(
        [{"name": "gemini", "kind": "gemini"}, {"name": "groq", "kind": "openai-compatible"}]
    )
    assert panel.existing_key.echoMode().name == "Password", "a key is never shown in the clear"
    assert panel.save_existing_key.isEnabled() is False, "nothing to save for until a row is chosen"

    seen: list[tuple[str, str]] = []
    panel.set_api_key.connect(lambda provider, key: seen.append((provider, key)))

    panel.providers_table.selectRow(1)
    assert panel.key_target.text() == "groq", "the window says where the key will go"
    panel.existing_key.setText("sk-abc")
    panel.save_existing_key.click()
    assert seen == [("groq", "sk-abc")]

    panel.existing_key.setText("sk-second")
    panel.providers_table.selectRow(0)
    assert panel.existing_key.text() == "", "the old key is not carried to another provider"
    panel.providers_table.selectRow(1)
    assert panel.existing_key.text() == ""

    panel.clear_existing_key.click()
    assert seen[-1] == ("groq", ""), "an empty key clears it, which is the only way to"


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


async def test_a_benchmark_run_is_saved_so_it_can_be_scored(window, qapp) -> None:
    """The table used to be given empty ids.

    Scoring a row then wrote nothing, silently: the panel looked complete and
    the reviews table stayed empty. The whole point of a manual review is that
    the row it reviews is the row that was run.
    """
    window.benchmark.prompts.setPlainText("what is in the scene?")
    window.model_selector.setCurrentIndex(0)
    window._run_benchmark("", "scripted:scripted-model")  # noqa: SLF001
    for _ in range(120):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.benchmark.run_button.isEnabled() and window.benchmark.table.rowCount():
            break
    await asyncio.sleep(0.1)
    qapp.processEvents()

    assert window.benchmark.table.rowCount() == 1
    row_id = window.benchmark.table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert row_id, "the row carries the id of a run that exists"

    suites = await window._benchmarks.suites()  # noqa: SLF001
    assert suites and suites[0]["runs"] == 1
    runs = await window._benchmarks.runs(suites[0]["id"])  # noqa: SLF001
    assert runs[0]["id"] == row_id

    window._save_review(row_id, {"Overall": "4", "notes": "clear"})  # noqa: SLF001
    for _ in range(60):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if "Score saved" in window.benchmark.summary.text():
            break
    reviews = await window._benchmarks.reviews(row_id)  # noqa: SLF001
    assert reviews and reviews[0]["overall"] == 4, "the score reached the database"


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


async def test_saving_one_mcp_server_keeps_the_others(window, qapp) -> None:
    """Saving the Blender command used to delete every other server.

    A filesystem server configured in .env vanished -- from the settings, the
    database and the manager -- because the save assigned a one-element list.
    Someone setting up a second MCP server would have found it gone after
    opening Settings and pressing Save.
    """
    from app.core.settings import MCPServerConfig

    other = MCPServerConfig(name="files", command="/usr/bin/python3", args=["-m", "fs_server"])
    window.context.settings.mcp_servers = [window.context.settings.mcp_servers[0], other]
    window.settings.mcp_name.setText("Blender MCP")
    window.settings.mcp_command.setText("/usr/bin/python3.11")

    await window._save_mcp(  # noqa: SLF001
        {
            "name": "Blender MCP",
            "command": "/usr/bin/python3.11",
            "args": ["-m", "server.main"],
            "cwd": "/tmp",
            "blender_port": 8767,
            "connect_timeout": 30.0,
        }
    )

    names = [s.name for s in window.context.settings.mcp_servers]
    assert names == ["Blender MCP", "files"], "the other server is still configured"
    stored = await window.context.studio.settings.all()  # noqa: SLF001
    assert {s["name"] for s in stored["mcp_servers"]} == {"Blender MCP", "files"}, "and still in the database"


async def test_a_new_mcp_server_can_be_added_by_name(window, qapp) -> None:
    from app.core.settings import MCPServerConfig

    window.context.settings.mcp_servers = [
        MCPServerConfig(name="Blender MCP", command="/usr/bin/python3.11", args=["-m", "server.main"])
    ]
    await window._save_mcp(  # noqa: SLF001
        {
            "name": "files",
            "command": "/usr/bin/python3",
            "args": ["-m", "fs_server"],
            "cwd": None,
            "blender_port": 8765,
            "connect_timeout": 30.0,
        }
    )
    assert [s.name for s in window.context.settings.mcp_servers] == ["Blender MCP", "files"]


async def test_editing_a_server_replaces_it_rather_than_duplicating_it(window, qapp) -> None:
    from app.core.settings import MCPServerConfig

    window.context.settings.mcp_servers = [
        MCPServerConfig(name="Blender MCP", command="/old/python", args=["-m", "server.main"])
    ]
    await window._save_mcp(  # noqa: SLF001
        {
            "name": "Blender MCP",
            "command": "/new/python",
            "args": ["-m", "server.main"],
            "cwd": None,
            "blender_port": 8765,
            "connect_timeout": 30.0,
        }
    )
    servers = window.context.settings.mcp_servers
    assert len(servers) == 1
    assert servers[0].command == "/new/python"


async def test_a_window_keeps_one_conversation_across_turns(window, qapp) -> None:
    """Each turn used to mint a new conversation.

    The model could not remember anything, while the transcript looked
    continuous -- so the window showed a memory that was not there. Two turns
    have to land in one conversation, and be reloadable afterwards.
    """
    provider = MockLLMProvider(
        [
            ScriptedTurn(tool_calls=[("blender.get_scene", {})]),
            ScriptedTurn(text="First answer."),
            ScriptedTurn(text="Second answer."),
        ],
        model="scripted-model",
    )
    window.context.llm.set_provider("scripted", provider)
    window.model_selector.setCurrentIndex(0)

    window.send("what is in the scene?")
    for _ in range(80):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "First answer." in window.chat.transcript_text():
            break
    window.send("and now?")
    for _ in range(80):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "Second answer." in window.chat.transcript_text():
            break

    conversations = await window.context.studio.conversations.list()  # noqa: SLF001
    assert len(conversations) == 1, "one conversation, not one per turn"
    messages = await window.context.studio.messages.list(conversations[0].id)  # noqa: SLF001
    roles = [m.role for m in messages]
    # One row per model response, so the turn that called a tool contributes two
    # assistant messages; what matters is that both user turns are in here, in
    # one conversation, in order.
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert roles.count("user") == 2
    assert [m.content for m in messages].index("and now?") > [m.content for m in messages].index(
        "First answer."
    )
    assert "Second answer." in messages[-1].content


async def test_reopening_the_window_puts_the_transcript_back(window, qapp) -> None:
    """Every message was in the database and none of it came back."""
    provider = MockLLMProvider(
        [ScriptedTurn(tool_calls=[("blender.get_scene", {})]), ScriptedTurn(text="Remembered.")],
        model="scripted-model",
    )
    window.context.llm.set_provider("scripted", provider)
    window.model_selector.setCurrentIndex(0)
    window.send("what is in the scene?")
    for _ in range(80):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if window.chat.send.isEnabled() and "Remembered." in window.chat.transcript_text():
            break

    window.chat.clear()
    assert window.chat.transcript_text() == ""
    # Through the window's own path, not the coroutine: rebuilding the
    # transcript is a widget operation, and the window has to be the thread
    # that does it.
    window._restore_conversation()  # noqa: SLF001
    for _ in range(200):
        qapp.processEvents()
        await asyncio.sleep(0.01)
        if "Remembered." in window.chat.transcript_text():
            break
    text = window.chat.transcript_text()
    assert "what is in the scene?" in text
    assert "Remembered." in text, "and the window carries on where it left off"


def test_a_tool_card_shows_the_picture_it_returned(qapp) -> None:
    """A card that says "1 image" makes the user go and find it.

    A render is the reason the tool exists, so the render belongs in the card.
    """
    import base64

    from app.gui.chat.widget import ToolCallCard

    png = base64.b64encode(_TINY_PNG).decode()
    card = ToolCallCard("blender.render_preview")
    card.set_result(is_error=False, text="{}", duration_ms=2100, images=1, image_data=[png])
    shown = [w for w in card.findChildren(QLabel) if w.objectName() == "chat-image"]
    assert shown, "the picture is in the card"
    assert shown[0].pixmap().width() <= 360, "and scaled to fit the transcript"
    assert "2.10 s" in card.header.text()


def test_no_module_of_the_app_chooses_a_qt_platform() -> None:
    """The bug that made the window never appear on a real desktop.

    One panel set ``QT_QPA_PLATFORM=offscreen`` at import time -- a convenience
    for running GUI tests on a headless machine. But the window imports the
    panels before Qt is asked for a platform, so on a machine with a display the
    studio started *offscreen*: no window, an event loop that never returns, and
    a process that looked alive and did nothing. Every test passed, because the
    tests wanted offscreen.

    An import must not decide where the window goes. The test suite sets the
    variable itself, before Qt is imported; that is the right place for it.

    Checked in the source rather than by importing: a module already imported by
    an earlier test is cached, so the check would pass on the broken code.
    """
    app_dir = Path(__file__).resolve().parents[2] / "app"
    offenders = [
        path.relative_to(app_dir).as_posix()
        for path in app_dir.rglob("*.py")
        if "QT_QPA_PLATFORM" in path.read_text(encoding="utf-8")
        and "os.environ.get" not in _only_reading(path)
    ]
    assert not offenders, f"these choose a Qt platform on import: {offenders}"


def _only_reading(path: Path) -> str:
    """The lines of ``path`` that only read the variable, which main.py may do."""
    return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if "get(" in line)
