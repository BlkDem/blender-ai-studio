"""Getting a generated asset into the scene.

``blender-mcp`` has no import tool, and embedding Blender's API in this client is
the boundary the whole project is built to avoid. So the import goes through the
one tool that can do it — ``blender.execute_python``, off by default — and this
module is the extension point the architecture anticipated: when
``blender.import_asset`` exists upstream, this file becomes a call to it and
nothing above it changes.

Two things are checked before the import rather than after a confusing failure,
because both are common and neither is the caller's fault:

* **whether the gate is open** — without it there is no way in, and the honest
  answer says which tool is missing;
* **what a failure means** — the studio and Blender are often on different sides
  of a filesystem boundary (a WSL-hosted server driving a Windows Blender, say),
  where a POSIX path is simply not a path. There is no way to ask Blender whether
  a file is readable from inside the sandbox — ``import os`` is blocked on
  purpose — so the import is attempted and the error is read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import ThreeDError
from app.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

#: The tool an import needs today, and the one that would replace it.
EXECUTE_PYTHON = "blender.execute_python"
FUTURE_IMPORT_TOOL = "blender.import_asset"

_IMPORT_CODE = """
import bpy
result = bpy.ops.import_scene.gltf(filepath={path!r})
bpy.context.view_layer.update()
imported = [obj.name for obj in bpy.context.scene.objects]
"""


@dataclass
class ImportReport:
    """What the import did."""

    imported: list[str] = field(default_factory=list)
    object_count: int = 0
    scene_total: int = 0
    path: str = ""
    via: str = EXECUTE_PYTHON

    def summary(self) -> dict[str, Any]:
        return {
            "imported": self.imported,
            "new_objects": len(self.imported),
            "scene_total": self.scene_total,
            "path": self.path,
            "via": self.via,
        }


def can_import(bridge: MCPManager) -> bool:
    """Whether there is any way in from here."""
    tools = bridge.tools()
    return FUTURE_IMPORT_TOOL in tools or EXECUTE_PYTHON in tools


async def import_asset(
    bridge: MCPManager,
    path: Path,
    *,
    timeout: float = 300.0,
) -> ImportReport:
    """Import a ``.glb``/``.gltf`` into the running Blender.

    Raises rather than returning a report with a failure in it: a caller that
    ignores the result would otherwise carry on believing the object is in the
    scene, which is the one belief that must never be wrong.
    """
    if not path.exists():
        raise ThreeDError(
            f"The asset at {path} is not on this machine",
            hint="The provider returned a URL; download it first.",
            path=str(path),
        )
    if bridge.tools().get(FUTURE_IMPORT_TOOL) is not None:
        return await _import_with_the_future_tool(bridge, path, timeout=timeout)
    if EXECUTE_PYTHON not in bridge.tools():
        raise ThreeDError(
            "This Blender cannot import an asset yet",
            hint=(
                f"{EXECUTE_PYTHON} is the only way in today and it is disabled. "
                f"Enable it in Settings → Agent, or wait for {FUTURE_IMPORT_TOOL} "
                "in blender-mcp."
            ),
            path=str(path),
        )
    # What arrived is the difference between the scene before and after. The
    # operator itself only says FINISHED, and a name it invents is worse than no
    # name at all -- "did the object actually show up" is the whole question.
    before = await _names(bridge, timeout)
    outcome = await _call(
        bridge, EXECUTE_PYTHON, {"code": _IMPORT_CODE.format(path=_windows_path(path))}, timeout
    )
    if outcome.get("is_error"):
        raise ThreeDError(f"Blender refused the import: {_short(outcome.get('text', ''))}", path=str(path))
    try:
        payload = json.loads(outcome.get("text") or "{}")
    except json.JSONDecodeError as exc:
        raise ThreeDError("Blender's answer to the import was not readable", path=str(path)) from exc

    after = await _names(bridge, timeout)
    arrived = [name for name in after if name not in before]
    if not arrived:
        # Blender accepted the file and nothing is in the scene: a GLB with no
        # geometry, or one that went somewhere the object list does not reach.
        # Saying so is more useful than reporting an empty success.
        raise ThreeDError(
            f"Blender read {path.name} but no object appeared",
            hint="The file may contain no geometry, or only cameras and lights.",
            path=str(path),
        )
    report = ImportReport(path=str(path))
    report.imported = arrived or _imported_names(payload)
    report.scene_total = len(after)
    report.object_count = len(report.imported)
    logger.info("imported %d object(s) from %s: %s", len(report.imported), path.name, arrived)
    return report


async def _import_with_the_future_tool(bridge: MCPManager, path: Path, *, timeout: float) -> ImportReport:
    """The path this project is written towards, exercised the day it exists."""
    outcome = await _call(bridge, FUTURE_IMPORT_TOOL, {"path": str(path)}, timeout)
    if outcome.get("is_error"):
        raise _explain(outcome.get("text", ""), path)
    try:
        payload = json.loads(outcome.get("text") or "{}")
    except json.JSONDecodeError:
        payload = {}
    return ImportReport(
        imported=list(payload.get("imported") or []),
        scene_total=int(payload.get("scene_total") or 0),
        path=str(path),
        via=FUTURE_IMPORT_TOOL,
    )


def _explain(text: str, path: Path) -> ThreeDError:
    """Say what a failed import actually means.

    The common failure is not a bad file but a path the far side cannot open: a
    WSL-hosted studio driving a Windows Blender gets ``/mnt/c/...``, which means
    nothing there. Reading that as "Blender refused" sends the user looking in
    the wrong place entirely.
    """
    reason = _short(text)
    lowered = reason.lower()
    unreadable = any(
        marker in lowered
        for marker in (
            "cannot open",
            "could not open",
            "no such file",
            "not found",
            "unable to open",
            "cannot read",
            "eof",
            "gltf loader",
        )
    )
    if unreadable and str(path).startswith("/mnt/"):
        return ThreeDError(
            f"Blender cannot open {path} — it cannot see this filesystem",
            hint=(
                "The studio and Blender are on different sides. The path was given to "
                f"Blender as {_windows_path(path)}; if that is wrong, put the asset "
                "somewhere Blender can read, or run the studio on the same side."
            ),
            path=str(path),
        )
    if unreadable:
        return ThreeDError(
            f"Blender could not read the asset: {reason}",
            hint="The file may be truncated, or not a GLB at all.",
            path=str(path),
        )
    return ThreeDError(
        f"Blender refused the import: {reason}",
        hint="The glTF add-on may be disabled, or the file may exceed its limits.",
        path=str(path),
    )


def _windows_path(path: Path) -> str:
    """A path Blender can open.

    Under WSL, ``/mnt/c/Users/x`` and ``C:/Users/x`` are the same file and only
    the second one means anything to a Windows Blender.
    """
    text = str(path)
    if not text.startswith("/mnt/"):
        return text
    parts = text[len("/mnt/") :].split("/")
    # Any drive, not just C: a D: or E: is just as unreadable to Windows.
    if len(parts) < 2 or len(parts[0]) != 1 or not parts[0].isalpha():
        return text
    return f"{parts[0].upper()}:/" + "/".join(parts[1:])


async def _call(bridge: MCPManager, tool: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
    """One tool call as a dict, so the helpers above stay about policy, not plumbing."""
    outcome = await bridge.call_tool(tool, arguments, timeout=timeout)
    return {"is_error": outcome.is_error, "text": outcome.text, "code": outcome.error_code}


async def _names(bridge: MCPManager, timeout: float) -> list[str]:
    """The names in the scene, best effort: a report is not worth failing over."""
    scene = await _call(bridge, "blender.get_objects", {"limit": 500}, timeout)
    if scene.get("is_error"):
        return []
    try:
        objects = json.loads(scene.get("text") or "{}").get("objects", [])
    except json.JSONDecodeError:
        return []
    return [str(obj.get("name")) for obj in objects if obj.get("name")]


def _imported_names(payload: dict[str, Any]) -> list[str]:
    """The names the import reported, from wherever the answer put them."""
    result = payload.get("result")
    if isinstance(result, list):
        return [str(name) for name in result]
    if isinstance(result, dict):
        for key in ("imported", "objects", "names"):
            if isinstance(result.get(key), list):
                return [str(name) for name in result[key]]
    return []


def _short(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit]
