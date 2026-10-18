# Saving .blend from chat

The context registers the typed local tool `save_blend_file` for every agent.
Provide `path` explicitly, ending in `.blend`; `pack_resources` is a boolean
and defaults to `true`. Ask the user for a destination when it is unknown.
Saving may overwrite an existing destination. The parent directory must already
exist and be writable on the Blender machine. WSL `/mnt/c/...` destinations are
translated to `C:/...`; no installation or files are copied between machines.

The tool reuses `app/core/blend_save.py`, also used by GUI saving. It checks
`agent.allow_execute_python` at execution time, then requires an available
`blender.execute_python`. It does not enable permissions. Packing must return
`FINISHED` before saving; saving must return `FINISHED` and the expected
`bpy.data.filepath` before `saved: true` is returned. Errors are returned to the
model as `Error:` and must not be reported as success. A timeout can occur after
a disk write; inspect before retrying. Packing can change in-memory resources
before a later save fails.

Chat destinations are exports: they never update a project's workspace, even
when a project is active. Use GUI **Save as…** when adopting a new workspace.
This avoids updating a project that changed while a chat operation was pending.

## Verified MCP policy (Linux and Windows checkouts)

The actual validator was inspected and tested in both:

- `/home/maksim/projects/blender-mcp/server/validation/python.py`
- `/mnt/c/Users/maxim/projects/blender-mcp/server/validation/python.py`

In these versions `visit_Call` rejects every target ending in `.read` or `.write`
(line 223–225), explaining the `bpy.data.libraries.write` file-access error.
`bpy.ops.wm.open_mainfile` is explicitly blocked too. The helper's fixed
`bpy.ops.file.pack_all()` and `bpy.ops.wm.save_as_mainfile(filepath=...)` code
passes both real validators, with and without packing. No general file-writing
policy change or exception is needed, and no validator was modified.

Both checkouts separately gate execution in `server/mcp/tools/python.py`
(`_require_permission`) using `ALLOW_PYTHON_EXECUTION`, default false in
`server/config.py`; registration also respects `ENABLED_TOOLS` and the permission
in `server/mcp/server.py`. An unavailable/disabled tool is an actionable refusal,
not a reason to bypass the policy. If the operator deliberately permits Python,
configure the actual launched MCP server's environment and restart it, keeping
its allowlist and AST screen. A stricter future server should instead expose a
narrow typed save endpoint with path validation, not permit arbitrary writes.

`tests/core/test_save_blend_tool.py` runs generated code through each external
validator when its checkout is present (otherwise these integration checks skip),
and confirms libraries.write, open, and open_mainfile remain blocked.
