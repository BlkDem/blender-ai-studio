"""The history popup and the arrow keys that share its list.

Both offer the same prompts, and the test that matters is the one that says so:
two ways to reach the same answer, disagreeing, is worse than one way.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from app.gui.chat.history import HistoryPopup
from app.gui.chat.widget import ChatView

#: Newest first, the way they are stored and the way they are offered.
PROMPTS = [
    ("make a round table", 1_700_000_300.0),
    ("add four legs", 1_700_000_200.0),
    ("make it darker wood", 1_700_000_100.0),
]


def popup(qapp: QApplication) -> HistoryPopup:
    dialog = HistoryPopup(PROMPTS)
    dialog.show()
    return dialog


def test_the_prompts_are_listed_newest_first(qapp: QApplication) -> None:
    dialog = popup(qapp)
    assert dialog.list.count() == 3
    assert dialog.matching() == [text for text, _when in PROMPTS]
    assert dialog.list.item(0).text().endswith("make a round table")


def test_a_filter_narrows_the_list_and_the_count_says_so(qapp: QApplication) -> None:
    dialog = popup(qapp)
    dialog.filter.setText("legs")
    assert dialog.matching() == ["add four legs"]
    assert "1 of 3" in dialog.status.text()


def test_a_filter_that_matches_nothing_says_so_rather_than_looking_empty(qapp: QApplication) -> None:
    """An empty list with no explanation reads as "you have no history"."""
    dialog = popup(qapp)
    dialog.filter.setText("zzzz")
    assert dialog.matching() == []
    assert "0 of 3" in dialog.status.text()


def test_enter_takes_the_selected_prompt(qapp: QApplication) -> None:
    dialog = popup(qapp)
    taken: list[str] = []
    dialog.chosen.connect(taken.append)
    dialog.list.setCurrentRow(1)
    dialog._take()  # noqa: SLF001 - the Enter path
    assert taken == ["add four legs"]


def test_the_filter_and_the_arrow_keys_agree(qapp: QApplication) -> None:
    dialog = popup(qapp)
    dialog.filter.setText("wood")
    # Whatever Enter will take has to be what matching() reports, or the popup
    # offers one thing and the composer walks another.
    dialog.list.setCurrentRow(dialog.list.row(dialog.list.item(0)) if dialog.list.count() else 0)
    visible_row = next(r for r in range(dialog.list.count()) if not dialog.list.item(r).isHidden())
    dialog.list.setCurrentRow(visible_row)
    dialog._take()  # noqa: SLF001
    assert dialog.matching() == ["make it darker wood"]


def test_the_row_shows_a_short_prompt_and_the_tooltip_the_whole_one(qapp: QApplication) -> None:
    long_prompt = "x" * 400
    dialog = HistoryPopup([(long_prompt, 1_700_000_000.0)])
    dialog.show()
    assert len(dialog.list.item(0).text()) < 140
    assert dialog.list.item(0).toolTip() == long_prompt


def test_an_empty_history_opens_to_an_empty_list(qapp: QApplication) -> None:
    dialog = HistoryPopup([])
    dialog.show()
    assert dialog.list.count() == 0
    assert "0 prompts" in dialog.status.text()


# --- the composer's arrow keys ---------------------------------------------


def _key(view: ChatView, key: Qt.Key) -> str:
    """Press a key on the composer and report what ended up in it.

    The value returned is the input's text rather than a slot's return: what the
    person has to see is the text, and going through the real key event is the
    only way to know the shortcut is wired up at all.
    """
    event = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.ControlModifier, "")
    view.input.keyPressEvent(event)  # noqa: SLF001 - the shortcut path
    return view.input.text()


def ctrl_up(view: ChatView) -> str:
    return _key(view, Qt.Key.Key_Up)


def ctrl_down(view: ChatView) -> str:
    return _key(view, Qt.Key.Key_Down)


def test_ctrl_up_walks_backwards_through_the_prompts(qapp: QApplication) -> None:
    view = ChatView()
    view.set_prompt_history([text for text, _when in PROMPTS])
    assert ctrl_up(view) == "make a round table"
    assert ctrl_up(view) == "add four legs"
    assert view.input.text() == "add four legs"


def test_walking_forward_comes_back_to_the_draft(qapp: QApplication) -> None:
    """Walking past the newest prompt must not throw away what was being typed."""
    view = ChatView()
    view.set_prompt_history([text for text, _when in PROMPTS])
    view.input.setText("half-written thought")
    ctrl_up(view)
    ctrl_up(view)
    ctrl_down(view)
    assert view.input.text() == "make a round table"
    ctrl_down(view)
    assert view.input.text() == "half-written thought"


def test_walking_off_the_end_of_the_oldest_keeps_the_oldest(qapp: QApplication) -> None:
    view = ChatView()
    view.set_prompt_history(["only one"])
    ctrl_up(view)
    ctrl_up(view)
    assert view.input.text() == "only one"


def test_arrow_keys_do_nothing_without_history(qapp: QApplication) -> None:
    """Ctrl+Up on a fresh install must not eat what is being typed."""
    view = ChatView()
    view.input.setText("typed")
    ctrl_up(view)
    ctrl_down(view)
    assert view.input.text() == "typed"


def test_a_prompt_taken_from_the_popup_lands_in_the_input(qapp: QApplication) -> None:
    view = ChatView()
    view.put_prompt("make it darker wood")
    assert view.input.text() == "make it darker wood"


def test_sending_resets_the_walk(qapp: QApplication) -> None:
    """Otherwise the next Ctrl+Up lands two prompts back from where the person
    last was, for no reason they can see."""
    view = ChatView()
    view.set_prompt_history([text for text, _when in PROMPTS])
    ctrl_up(view)
    ctrl_up(view)
    view.input.setText("and now")
    view._submit()  # noqa: SLF001
    assert ctrl_up(view) == "make a round table"


def test_the_button_asks_for_the_history_rather_than_reading_the_database(qapp: QApplication) -> None:
    """The window owns the database; a widget that opened it would be a query on
    the GUI thread, which is a frozen window."""
    view = ChatView()
    asked: list[int] = []
    view.history_requested.connect(lambda: asked.append(1))
    view.history.click()
    assert asked == [1]
