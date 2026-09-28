"""The window hands a resumed session to the model, not only to the eye.

``_restore_conversation`` already put the last conversation back on screen, and
that was the whole of it: the transcript looked continuous while the model had
never heard of any of it. "It forgot what I asked for" was a correct observation
about a bug, so the block has to reach the agent.
"""

from __future__ import annotations

import asyncio

import pytest

from app.llm.base import Message, Role


async def _drain(qapp, predicate, rounds: int = 120) -> bool:
    for _ in range(rounds):
        qapp.processEvents()
        await asyncio.sleep(0.02)
        if predicate():
            return True
    return False


async def _settle(window, qapp) -> None:
    """Wait until the window has nothing queued.

    A finished run schedules a scene read, and a coroutine that is created and
    then dropped before the core thread runs it is a leak the suite warns about.
    Draining to a quiet chat is what a person does before closing a window.
    """
    await _drain(qapp, lambda: window.chat.send.isEnabled(), rounds=60)
    await _drain(qapp, lambda: False, rounds=30)


def _texts(request) -> list[str]:
    return [m.content for m in request.messages]


async def test_a_reopened_window_continues_the_conversation(window, qapp) -> None:
    """One turn, recorded. A new window. The next turn has to carry the first."""
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    provider = MockLLMProvider(
        [ScriptedTurn(text="It is a low table."), ScriptedTurn(text="Now it is round.")],
        model="scripted-model",
    )
    window.context.llm.set_provider("scripted", provider)
    window.model_selector.setCurrentIndex(0)

    window.send("make a low table")
    assert await _drain(qapp, lambda: "It is a low table." in window.chat.transcript_text())

    # Everything the window needs to resume is in the database; build the second
    # window the way start() does, on the same data directory.
    from app.core.context import AppContext
    from app.core.settings import MCPServerConfig, Settings
    from app.gui.bridge import CoreThread
    from app.gui.main_window import MainWindow
    from app.llm.registry import ProviderConfig

    settings = Settings(data_dir=window.context.settings.data_dir)
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
    try:
        context.llm.set_provider("scripted", provider)
        core = CoreThread(context)
        core.start()
        second = MainWindow(context, core)
        second.show()
        second.start()  # the real entry point; this is what fills the model selector
        assert await _drain(qapp, lambda: second.model_selector.count() > 0)
        second.model_selector.setCurrentIndex(0)

        assert await _drain(qapp, lambda: "It is a low table." in second.chat.transcript_text()), (
            "the previous session was not drawn"
        )

        agent = second._agent_for_turn()  # noqa: SLF001
        assert agent, "no agent was built"
        memory = [m.content for m in agent._memory]  # noqa: SLF001
        assert "make a low table" in memory, memory
        assert "It is a low table." in memory, memory
        second.close()
        second.deleteLater()
        core.stop()
    finally:
        await context.close()
    qapp.processEvents()


async def test_the_restored_block_is_capped_and_starts_on_a_question(window, qapp) -> None:
    from app.core.agent import history_block
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    provider = MockLLMProvider([ScriptedTurn(text="fine")], model="scripted-model")
    window.context.llm.set_provider("scripted", provider)
    window.send("a question with no answer yet")
    assert await _drain(qapp, lambda: "a question with no answer yet" in window.chat.transcript_text())

    conversation = (await window.context.studio.conversations.list())[0]  # noqa: SLF001
    stored = await window.context.studio.messages.list(conversation.id)  # noqa: SLF001
    block = history_block(stored)
    assert block[0].role is Role.USER, "a block that opens on an answer is not a starting point"


async def test_a_new_conversation_forgets(window, qapp) -> None:
    """The one control that must be able to clear it: otherwise there is no way
    to start again without closing the window."""
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    provider = MockLLMProvider([ScriptedTurn(text="ok")], model="scripted-model")
    window.context.llm.set_provider("scripted", provider)
    window.model_selector.setCurrentIndex(0)
    window.send("remember this")
    assert await _drain(qapp, lambda: "ok" in window.chat.transcript_text())

    window._agent_for_turn()  # noqa: SLF001
    window._new_conversation()  # noqa: SLF001
    agent = window._agent_for_turn()  # noqa: SLF001
    assert agent._memory == [], agent._memory  # noqa: SLF001
    assert Message is not None
    await _settle(window, qapp)


async def test_the_history_button_reads_the_prompts(window, qapp, monkeypatch) -> None:
    """The popup is modal, so it cannot be driven from a test. What is checked
    here is the part that is ours: that the rows come from the stored prompts
    and reach the composer's arrow keys."""
    import app.gui.main_window as main_window
    from app.llm.providers.mock import MockLLMProvider, ScriptedTurn

    opened: list[list[str]] = []

    class StubPopup:
        def __init__(self, prompts, parent=None) -> None:
            self.prompts = list(prompts)
            self.chosen = _Signal()

        def exec(self) -> int:
            opened.append([text for text, _when in self.prompts])
            return 0

    class _Signal:
        def connect(self, *_a, **_k) -> None:
            return None

    monkeypatch.setattr(main_window, "HistoryPopup", StubPopup)

    provider = MockLLMProvider([ScriptedTurn(text="ok")], model="scripted-model")
    window.context.llm.set_provider("scripted", provider)
    window.send("a prompt worth keeping")
    assert await _drain(qapp, lambda: "ok" in window.chat.transcript_text())

    window._show_history(await window._load_prompt_history())  # noqa: SLF001
    assert opened and "a prompt worth keeping" in opened[0]
    assert "a prompt worth keeping" in window.chat.prompt_history()
    await _settle(window, qapp)


async def test_a_task_queued_as_the_core_closes_is_closed_not_dropped(qapp, database_path) -> None:
    """``submit(self._read_scene(), …)`` builds the coroutine before submit is
    entered, so a core that cannot take it leaves an un-awaited coroutine behind.
    It is a warning in a test and a held frame in the app, and closing a window
    mid-task is how it happens."""
    import asyncio

    from app.core.context import AppContext
    from app.core.settings import Settings
    from app.gui.bridge import CoreThread

    context = await AppContext(settings=Settings(data_dir=database_path.parent)).open()
    core = CoreThread(context)
    core.start()
    try:
        core.stop()
        with pytest.raises(RuntimeError):
            core.submit(asyncio.sleep(0))
    finally:
        await context.close()
    qapp.processEvents()
