from app.llm.base import ChatRequest, ContentPart, Message, Role, ToolCall
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


def test_tool_images_follow_all_results_without_breaking_call_group():
    first = ToolCall(id="render", name="render", arguments={})
    second = ToolCall(id="scene", name="scene", arguments={})
    messages = [
        Message(role=Role.ASSISTANT, tool_calls=[first, second]),
        Message.tool_result(first, "rendered", [ContentPart.image_part("YWJj", "image/png")]),
        Message.tool_result(second, "scene info"),
        Message(role=Role.ASSISTANT, content="Verified"),
    ]
    payload = OpenAICompatibleProvider()._payload(ChatRequest(model="vision", messages=messages), stream=False)
    wire = payload["messages"]
    assert [message["role"] for message in wire] == ["assistant", "tool", "tool", "user", "assistant"]
    assert wire[1]["content"] == "rendered"
    assert wire[3]["content"][1]["image_url"]["url"] == "data:image/png;base64,YWJj"
    assert "render" in wire[3]["content"][0]["text"]


def test_tool_without_images_does_not_add_user_message():
    call = ToolCall(id="scene", name="scene", arguments={})
    payload = OpenAICompatibleProvider()._payload(
        ChatRequest(model="text", messages=[Message.tool_result(call, "ok")]), stream=False,
    )
    assert payload["messages"] == [{"role": "tool", "tool_call_id": "scene", "content": "ok"}]
