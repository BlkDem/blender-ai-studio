# Troubleshooting

The failures that actually happen, with the message you will see. Search for the
text in quotes.

## Blender

**"No Blender instance is connected. Start Blender, enable the 'Blender MCP'
add-on and press Connect."**

The MCP server is running; the add-on is not attached to it. Open Blender, enable
the add-on, and press Connect in the sidebar. If you have just restarted the
studio, the add-on reconnects on its own within a second or two — the studio
starts the server, and the add-on is looking for it.

**The studio says the bridge port is 8765 and the add-on is on 8767.**

They are two different ports and both are correct. `blender_port` in the MCP
server configuration is the port the *add-on* connects to; the MCP server itself is
a child process over stdio and has no port. `--check` prints what the child was
told:

```bash
.venv/bin/python -m app.main --check
```

**`BLENDER_NOT_CONNECTED` in the log, but the tools list is full.**

That is normal before the add-on attaches: the catalogue comes from the MCP
server, which is up, and the individual calls answer until Blender is.

## The model

**"a tool that always fails is worse than no tool" — `generate_3d_asset` is
missing.**

A 3D provider is configured but has no key. The tool is only offered when a
provider is both enabled and keyed. Add the key in Models, or accept the scene
without 3D generation.

**"Authentication required" / "Authentication failed" from a provider.**

The key is missing, wrong, or in the wrong place. Keys live in the secret store
(`<data dir>/secrets.json`, or the system keyring when one is installed), not in
`.env`. The GUI writes them; for the environment, the fallback is
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or
`f"{PROVIDER}_API_KEY"` for a provider of any other kind.

**A local model answers in prose and never calls a tool.**

Small models are unreliable at tool use; this is a property of the model, not a
fault in the studio, and the benchmark exists to measure it. Two things help:
declare `supports_tools: true` on the model so it is offered tools at all, and
give it a system prompt that says which tool to reach for. Qwen2.5-3B drives
Blender end to end but often prefers `blender.create_object` for an object a 3D
service could model.

**A model never sees a render.**

Only models declared `supports_vision: true` are sent an image — a text-only model
rejects the request, and the run would die on a capability you never asked about.
The window still shows the render in the tool card either way.

**The context fills up after a few turns.**

A conversation is replayed every turn, so a long one grows. Press **New** for a
blank conversation in the same project, or close the project to step out of it
entirely. The cost of the replay is visible in the footer's token count.

**This model models/gemini-2.5-flash is no longer available to new users.**

A retired model, and the provider names its replacement in the error. Take the
name it gives you, not one from a remembered list — a list of model ids is what
went stale. The live catalogue is the authority, and for OpenAI-compatible
providers the cheapest way to read it is to ask: configure the model, run
`examples/model_check.py --provider <name>`, and the failure names the current
model. Gemini moved 2.5 to 3.x; 2.5 still answers for some accounts and not for
others, which is exactly the case a remembered list cannot describe.

**Agent stopped: agent steps limit reached (30 of 30).**

The model asked for tools thirty times without ever answering in words. One step
is one request plus everything it asked for, so the run could only have ended at
the limit. The cause is a loop, not a short budget: read the last tool cards in
the transcript and look for the same call with the same arguments twice. Small
models are the usual reason — they reach for a tool instead of answering from what
they were already told — and the fix is in the prompt, not in the limit.

**The model is "unavailable" but everything is configured correctly.**

A provider with no capacity for the moment, or a free tier that is out of shared
quota, answers 429 or 5xx. The studio reports that as retryable rather than as a
configuration fault, because there is nothing to edit: the request was right and
the provider was busy. `examples/model_check.py` separates the two for the same
reason, and calling it "broken" would send you to fix something already correct.

**A tool the model wants is missing.**

Two different causes with one symptom. The tool is not published: `agent.allow_execute_python`
is off by default, and `generate_3d_asset` needs a keyed 3D provider. Or the model
was never offered it: only tools published by a server the studio is connected to
are listed, so check the MCP server is connected in Settings and that the tool
appears under `blender.list_tools`.

## 3D

**"Tripo returned HTTP 403: You don't have enough credit to create this task"**

The account has no credit. Tripo's web app and its API are billed **separately**:
credits in one do not pay for the other, and each has its own top-up in the
platform dashboard. Nothing was charged — Tripo freezes credits at submission and
returns them if the task fails.

**"The model is ready, but there is no way into Blender"**

The generation succeeded and the download is on disk, but the import is gated.
Two gates exist, and both must be open: the MCP server must be started with
`ALLOW_PYTHON_EXECUTION=true` in its environment, and the studio's own
`agent.allow_execute_python` must be on. The second is the Settings checkbox, and
`--check` reports it.

**"Blender cannot open <path> — it cannot see this filesystem"**

The studio and Blender are on different sides of a filesystem. The usual case is a
WSL-hosted studio driving a Windows Blender: `/mnt/c/...` and `C:/...` are the
same file, and the studio translates, but an asset that is genuinely somewhere
else cannot be read from the other side. Point `three_d.download_dir` at a place
both can see, or run the studio on the same side as Blender.

**"Blender read <file> but no object appeared"**

The file was readable and Blender imported it, but nothing entered the scene —
usually a GLB with no geometry in it, or one that contains only cameras and
lights. Treated as a failure on purpose: a report that said "done" with an empty
scene would be worse.

**"Timed out after 900s"**

`three_d.poll_timeout` was reached. Raise it, or use `--resume TASK_ID` on
`asset_run.py` to pick the task up where the provider has it. Re-submitting pays
twice.

**The task says failed but the money is gone.**

Check the credits column: they are recorded when the provider reports them, before
the import, because the model was generated either way.

## The window

**"No display found."**

A desktop app on a machine with no display. Use `--check`, `--prompt`,
`--benchmark` or `--list-tools`, or set `QT_QPA_PLATFORM=offscreen` for a
screenshot run.

**The Tasks table is empty after a restart, on an older build.**

Tasks are stored from this version on. A build before that kept them in memory
only, which is why the panel said nothing about earlier sessions.

**Stop does nothing.**

It cancels the run the window started. A run already finished, or one started by a
script, is not something the window can stop.

**A tool error is marked in red but the run carried on.**

That is the loop working: the refusal went back to the model with a reason and it
tried something else. The transcript shows the recovery.

## The studio itself

**"No MCP server is configured"**

`STUDIO_MCP_SERVERS` is unset, or set to something that is not a JSON list of
objects. The Settings page edits servers by name and writes the whole list.

**A saved MCP server disappeared.**

Not any more: saving upserts one server by name. On a build before that fix,
saving the Blender server replaced the list and deleted the others.

**A setting in `.env` has no effect.**

Nested fields need either the delimiter form or JSON:

```bash
STUDIO_AGENT__MAX_STEPS=30          # STUDIO_ prefix, __ between levels
STUDIO_AGENT='{"max_steps": 30}'    # or the whole object as JSON
```

The database wins over both, because that is where the window writes what you
changed there. A stored value that will not parse is ignored and logged, not
applied.

**Where is my data?**

`--data-dir`, else `STUDIO_DATA_DIR`, else `~/.local/share/blender-ai-studio`:
`studio.db` and `secrets.json` in one directory, and `assets/` for downloads.
One directory to look in, one to delete.

**The suite ends in "dumped core" with every test green.**

A widget outlived Qt. If you added a GUI test, close what you create; there is an
autouse fixture in `tests/gui/conftest.py` that does it for you.

## Still stuck

`--check` answers "is it configured"; `--list-tools` answers "what does the server
publish". Between them, most problems are one of the two. If the answer looks
right and the behaviour does not, the logs are: `--log-level DEBUG` on the CLI, or
the Log level field in Settings.
