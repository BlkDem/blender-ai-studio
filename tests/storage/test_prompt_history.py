"""Prompt history: read back from the prompts themselves.

There is no history table, and that is the point. A prompt *is* a stored user
message, so a second list would be a second thing to keep in step with the
first, and would eventually disagree with the conversation it claims to remember.
"""

from __future__ import annotations

from app.storage.repositories import Studio


async def _ask(studio: Studio, conversation, text: str) -> None:
    await studio.messages.add(conversation.id, "user", text)


async def test_the_prompts_come_back_newest_first(studio: Studio) -> None:
    conversation = await studio.conversations.create(None, "")
    for text in ("first", "second", "third"):
        await _ask(studio, conversation, text)
    history = await studio.messages.recent_user_texts()
    assert [text for text, _when in history] == ["third", "second", "first"]


async def test_the_history_crosses_conversations(studio: Studio) -> None:
    """Yesterday's prompt is the one being looked for as often as today's."""
    first = await studio.conversations.create(None, "")
    await _ask(studio, first, "from yesterday")
    second = await studio.conversations.create(None, "")
    await _ask(studio, second, "from today")
    history = await studio.messages.recent_user_texts()
    assert [text for text, _when in history] == ["from today", "from yesterday"]


async def test_answers_are_not_prompts(studio: Studio) -> None:
    conversation = await studio.conversations.create(None, "")
    await _ask(studio, conversation, "a question")
    await studio.messages.add(conversation.id, "assistant", "an answer")
    history = await studio.messages.recent_user_texts()
    assert [text for text, _when in history] == ["a question"]


async def test_the_same_prompt_twice_in_a_row_is_one_row(studio: Studio) -> None:
    """Asking it again straight away means the first answer was not what was
    wanted, and a list showing the line twice hides that."""
    conversation = await studio.conversations.create(None, "")
    await _ask(studio, conversation, "make a table")
    await _ask(studio, conversation, "make a table")
    history = await studio.messages.recent_user_texts()
    assert [text for text, _when in history] == ["make a table"]


async def test_the_same_prompt_asked_again_later_is_two_rows(studio: Studio) -> None:
    """Two prompts apart, "again, but smaller" is genuinely a second thing."""
    conversation = await studio.conversations.create(None, "")
    await _ask(studio, conversation, "make a table")
    await _ask(studio, conversation, "add a chair")
    await _ask(studio, conversation, "make a table")
    history = await studio.messages.recent_user_texts()
    assert [text for text, _when in history] == ["make a table", "add a chair"]


async def test_an_empty_prompt_is_not_history(studio: Studio) -> None:
    conversation = await studio.conversations.create(None, "")
    await _ask(studio, conversation, "   ")
    assert await studio.messages.recent_user_texts() == []


async def test_the_limit_is_honoured_and_worth_honouring(studio: Studio) -> None:
    conversation = await studio.conversations.create(None, "")
    for index in range(12):
        await _ask(studio, conversation, f"prompt {index}")
    history = await studio.messages.recent_user_texts(limit=5)
    assert len(history) == 5
    assert history[0][0] == "prompt 11"


async def test_a_studio_with_no_prompts_has_no_history(studio: Studio) -> None:
    assert await studio.messages.recent_user_texts() == []
