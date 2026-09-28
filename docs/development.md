# Development

Everything needed to work on the studio, and the shape a change is expected to
take.

## Getting set up

Python 3.11 or newer. The studio never writes into a `blender-mcp` checkout, and
nothing here needs Blender installed.

```bash
cd blender-ai-studio
python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m app.main --check
```

If your distribution has no `python3.11-venv`, `uv` works and is quicker:

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

`--check` builds the same application the window does and reports what is
configured and what is not. It exits non-zero when the core path is broken, which
makes it usable as a smoke test in CI.

## The daily loop

```bash
.venv/bin/python -m pytest                      # everything
.venv/bin/python -m pytest -m "not gui"         # no display needed
.venv/bin/python -m pytest tests/core -x        # one area, stop at the first failure
ruff check . && ruff format .                   # lint and format
mypy app                                        # types, with disallow_untyped_defs
```

The suite is 292 tests and takes about 35 seconds. `-p no:cacheprovider` is worth
adding when running under a file watcher. GUI tests set `QT_QPA_PLATFORM=offscreen`
themselves before importing Qt, so they need no display.

`mypy app` is deliberately not configured to be quiet: `disallow_untyped_defs` is
on, and a new function without annotations is a failure, not a suggestion.

## Running it for real

```bash
.venv/bin/python -m app.main                          # the window
.venv/bin/python -m app.main --prompt "add a lamp"    # one turn, headless
.venv/bin/python -m app.main --list-tools             # what the MCP server publishes
.venv/bin/python -m app.main --benchmark "p:model,q:model" --prompt "..."
```

`--benchmark` runs every model against every prompt in order, saves each run as it
finishes, and prints the table with the id of the run behind each row.

### The live runs

`examples/` holds the scripts that prove something against a real Blender and real
services. They are not fixtures and they are not smoke tests; each answers a
question unit tests cannot.

| Script | What it answers |
|---|---|
| `acceptance_run.py` | Does the whole chain work — a model, an agent, MCP, a real Blender? |
| `asset_run.py` | Does a 3D request reach the scene? Tripo, or a local GLB with `--import-only` |
| `pipeline_run.py` | The 3D path with a provider that always succeeds, so the pipeline is not confounded with a billing refusal |
| `gui_shot.py` | What does a person actually see, with real events |
| `live_run.py` | Everything at once, in seven sections |

All of them take `--blender-mcp`, `--python` and `--port`; run `--help` on any of
them. `asset_run.py` and `pipeline_run.py` open the import gate themselves, since
importing is the point of what they are testing. `live_run.py` does not: it leaves
the gate closed unless you pass `--allow-execute-python`, so a run can also prove
the refusal is a good one.

A local model is enough for most of it:

```bash
curl -L -o qwen2.5-3b-q4.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-q4_k_m.gguf
./llama-server -m qwen2.5-3b-q4.gguf --alias local-qwen --jinja --port 11400 --ctx-size 8192
```

## The shape of a change

**A bug fix starts with a test that fails for the right reason.** Not "the fix
works" — the old code, the wrong answer. Several fixes here were verified by
putting the bug back and watching the test fail; it is the only way to know the
test is about the bug.

**Comments explain why, not what.** The code says what it does. A comment earns
its place by saying why a reasonable person would otherwise change it — a
boundary, a failure mode that was measured, a decision that looks wrong and is
not. Several comments record a bug that happened here; that is what makes them
worth keeping.

**A refusal is a feature.** When a thing cannot be done, the code says so in
terms the user can act on: which tool, which setting, which file, which side of
the filesystem. "None", "" and a silent no-op are all worse.

**Verify against the real thing when you can.** Every claim in the README about
live behaviour was checked against a real Blender, a real model and a real Tripo.
Several bugs in this repository were invisible to unit tests and obvious in ten
seconds of real use: a settings save that deleted other servers, a switch that
controlled nothing, a render nobody could read.

**Push back on your own tests.** A test that passes both before and after a change
is not testing the change. If a bug fix does not break a test, either the test is
wrong or the bug was not what you thought.

## Adding to the suite

Tests mirror `app/` directory for directory. Use the real thing where it is cheap
and a double where it is not:

- `tests/llm/`, `tests/mcp/` build real requests and assert the wire shape.
- `tests/agent/` drives the loop with `MockLLMProvider` and `ScriptedTurn`, which
  is a scripted model: turns in, turns out, no network.
- `tests/providers3d/` uses `MockThreeDProvider`, which can be told to fail, to
  take three polls, or to serve a real GLB.
- `tests/gui/` builds a real `MainWindow` on a real `AppContext` with a fake
  bridge. A widget left alive when the interpreter tears Qt down takes the process
  with it, so there is an autouse fixture that closes them.
- `app/mcp/support.py` is a real MCP server over a real pipe, so the client is
  tested against a handshake rather than a mock of one.

Anything that costs money or needs credentials does not belong in the suite. It
belongs in `examples/`, where a person decides to run it.

## Before you push

```bash
ruff check . && ruff format . && mypy app
.venv/bin/python -m pytest
.venv/bin/python -m app.main --check
git status --short          # nothing untracked that matters
```

Then check the things that do not fail automatically: is there a key in a tracked
file (`git grep -n "tsk_\|sk-"`), a path from your own machine, a screenshot in
the repository?
