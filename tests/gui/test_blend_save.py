"""Save workflow tests use the real window fixture and core thread."""
from unittest.mock import AsyncMock

from tests.gui.test_projects import create_through_the_panel, pump


async def test_save_updates_only_workspace(window, qapp, monkeypatch, tmp_path):
    project_id = await create_through_the_panel(window, qapp, "Save test")
    await pump(qapp, lambda: not window._workspace_busy)
    path = str(tmp_path / "scene.blend")
    await window.context.studio.projects.update(project_id, workspace=path)
    window.context.settings.agent.allow_execute_python = True
    saved = AsyncMock(return_value=path)
    monkeypatch.setattr("app.core.blend_save.save_blend", saved)
    window.save_action.trigger()
    await pump(qapp, lambda: not window._saving)
    saved.assert_awaited_once_with(window.context.mcp, path, pack=True)
    project = await window.context.studio.projects.get(project_id)
    assert project.workspace == path
    assert not project.initial_blend


async def test_save_error_does_not_update_database(window, qapp, monkeypatch):
    project_id = await create_through_the_panel(window, qapp, "Failure")
    window.context.settings.agent.allow_execute_python = True
    monkeypatch.setattr("app.core.blend_save.save_blend", AsyncMock(side_effect=RuntimeError("pack failed")))
    path, error = await window._do_save_blend(project_id, "/new.blend", True)
    assert not path and "pack failed" in error
    assert not (await window.context.studio.projects.get(project_id)).workspace


async def test_save_as_cancel_and_busy(window, monkeypatch):
    from app.gui.main_window import QFileDialog

    monkeypatch.setattr(QFileDialog, "exec", lambda self: 0)
    window._saving = True
    window._choose_save(None, True)
    assert not window._saving
    window._saving = True
    window._close_project()
    window._save_blend()
    assert window._saving
    window._saving = False
