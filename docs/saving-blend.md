# Saving Blender work

Use **File → Save .blend** (`Ctrl+S`) to save the running Blender scene to the
active project's **workspace**. A highlighted row in Projects is not the active
project. Without a workspace, Save opens the Save as dialog instead.

**File → Save as…** (`Ctrl+Shift+S`) always asks for a destination. The dialog
uses the `.blend` extension and asks before replacing an existing file. On
success the destination becomes the active project's workspace. The project's
`initial_blend` remains unchanged: it is a starting source, not a save target.
Without an active project a file can still be saved, but no project is updated.

**Pack resources (textures) before saving** in the File menu is checked by
default. Blender runs `bpy.ops.file.pack_all()` before saving, embedding supported
external resources such as image textures. Disable it to keep external links;
then those resources must remain available when moving the file. A packing error
aborts saving and is shown in the chat, rather than silently losing textures.
Packing may change the in-memory scene even if the later save fails.

Saving requires an available `blender.execute_python` MCP tool and the explicit
Studio permission in Settings. The application does not turn either on. It sends
fixed Blender API code, never opens another file, and accepts success only when
Blender reports both `FINISHED` and the expected `bpy.data.filepath`. Errors,
timeouts and unverified responses do not update the workspace. If a timeout
occurs, Blender may nevertheless have written the file: inspect it before retrying.
A database error after a verified disk write is also reported as a failure.

WSL paths such as `/mnt/c/Users/maxim/work/scene.blend` are sent to Blender as
`C:/Users/maxim/work/scene.blend`; native Windows paths are supported too. Other
paths must be accessible on the machine running Blender. Choose an existing
parent directory writable by Blender. The Studio does not copy files between
Linux and Windows installations.

Wait for agent runs and file operations to finish before saving. While a save
or its destination dialog is in progress, another save, a new agent turn and
project changes are blocked. After success the project list and scene refresh.
Projects with only a workspace (no starting file) also offer **Open working file**;
opening still depends on the existing MCP opening capabilities and security rules.
