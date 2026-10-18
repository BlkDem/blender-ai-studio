"""Explicit, verified saving of the running Blender scene; never opens a file."""
from __future__ import annotations

import json
import re
from typing import Any


def blender_path(path: str) -> str:
    path = path.replace("\\", "/")
    return re.sub(r"^/mnt/([a-zA-Z])/", lambda m: m[1].upper() + ":/", path)


async def save_blend(bridge: Any, path: str, *, pack: bool = True) -> str:
    """Return the target only after FINISHED and the matching Blender filepath."""
    if not path or not path.lower().endswith(".blend"):
        raise ValueError("Choose a .blend file.")
    if "blender.execute_python" not in bridge.tools():
        raise RuntimeError("Saving requires blender.execute_python; it is disabled or unavailable. No settings were changed.")
    target = blender_path(path)
    code = "import bpy\n"
    if pack:
        code += "packed = bpy.ops.file.pack_all()\nif 'FINISHED' not in packed:\n    raise RuntimeError('Pack resources did not finish')\n"
    code += (
        f"saved = bpy.ops.wm.save_as_mainfile(filepath={target!r})\n"
        "result = {'studio_save': True, 'finished': 'FINISHED' in saved, 'filepath': bpy.data.filepath}\n"
    )
    outcome = await bridge.call_tool("blender.execute_python", {"code": code})
    if outcome.is_error:
        raise RuntimeError(f"Blender save failed: {outcome.text}")
    try:
        payload = json.loads(outcome.text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Blender did not confirm the save.") from exc
    # The server and executor each wrap their result. Accept only the explicit
    # report produced by our fixed code, not a generic success flag.
    for _ in range(5):
        if not isinstance(payload, dict) or payload.get("studio_save") is True:
            break
        payload = payload.get("result")
    if not isinstance(payload, dict) or payload.get("studio_save") is not True or payload.get("finished") is not True:
        raise RuntimeError("Blender did not confirm FINISHED; workspace was not updated.")
    actual = blender_path(str(payload.get("filepath", "")))
    expected = target
    if re.match(r"^[A-Za-z]:/", target):
        actual, expected = actual.casefold(), expected.casefold()
    if actual != expected:
        raise RuntimeError(f"Blender reported a different filepath: {payload.get('filepath')!r}")
    return path
