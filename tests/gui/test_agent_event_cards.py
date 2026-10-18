import pytest

from app.core.events import Event, EventType
from app.gui.chat.widget import ToolCallCard

pytestmark = pytest.mark.gui


async def test_tool_start_exposes_arguments_and_ends_prose_stream(window):
    window.chat.start_stream()
    window.chat.append_stream("Inspecting scene")
    window._on_event(Event(type=EventType.TOOL_STARTED, payload={
        "call_id": "inspect", "tool": "blender.get_object", "arguments": {"name": "Cube"},
    }))
    assert window.chat._streaming is None
    card = window.chat._cards["inspect"]
    assert isinstance(card, ToolCallCard)
    assert "Cube" in card.arguments.text()
