# Architecture

How the pieces fit, and why they are cut where they are. For configuration see
the README; for day-to-day work see [development.md](development.md).

## The one-sentence version

The studio is an LLM client. Blender is reached only over MCP, through
`blender-mcp`, and every model-specific or service-specific detail is hidden
behind a provider interface so that nothing above it knows which one is in use.

## The shape

```text
              ┌─────────────────────────── GUI (PySide6) ───────────────────────────┐
              │  Chat · Scene · Tasks · Projects · Benchmark · Models · Settings      │
              └───────────────▲──────────────────────────┬──────────────────────────┘
                              │ events (queue + QTimer)  │ coroutines (one at a time)
              ┌───────────────┴──────────────────────────▼──────────────────────────┐
              │                            CoreThread                                 │
              │            one asyncio loop, one thread, no Qt inside                 │
              └──┬──────────────┬──────────────┬───────────────┬─────────────────────┘
                 │              │              │               │
            ┌────▼────┐   ┌─────▼─────┐   ┌────▼─────┐   ┌─────▼──────┐
            │  Agent  │   │TaskManager│   │ Studio   │   │ Settings  │
            └──┬───┬──┘   └────┬──────┘   │ (SQLite) │   │  + .env   │
               │   │           │          └──────────┘   └────────────┘
     MCP tools │   │ local     │ task events
               │   │ tools     │
     ┌─────────▼───▼────┐  ┌───▼──────────────┐
     │    MCPManager    │  │ generate_3d_asset│
     │  N servers, one  │  └────┬─────────────┘
     │  merged catalogue│       │ submit · poll · download
     └─────────┬────────┘  ┌────▼──────┐
               │           │ThreeDReg. │
     ┌─────────▼────────┐  └───────────┘
     │  blender-mcp     │  ┌──────────────┐
     │  (child process) │  │ LLMRegistry  │  OpenAI · Anthropic · Gemini
     └─────────┬────────┘  │              │  any OpenAI-compatible server
               │ WebSocket └──────────────┘
     ┌─────────▼────────┐
     │ Blender add-on   │
     └──────────────────┘
```

## The boundaries, and what each one buys

### 1. Blender is reached only over MCP

The studio never imports `bpy`, never speaks the add-on's WebSocket protocol, and
never embeds Blender's API. It configures a child process and talks MCP to it.

*What it costs:* the studio can only do what `blender-mcp` exposes. Importing an
asset and reading a render both needed a tool that did not exist.

*What it buys:* the backend stays a separate project with its own tests, the
studio runs against any MCP server (there is a second one in the test suite), and
a Blender upgrade cannot reach into the client.

### 2. The GUI never touches the network

`CoreThread` (`app/gui/bridge.py`) runs the asyncio core on one thread with its own
loop. The GUI reaches it only through `CoreThread.submit(coroutine, on_done)`, and
hears about it only through the event bus, which is drained by a 20 ms `QTimer`
into the GUI thread. No coroutine runs in the Qt thread, and no Qt object is
touched from the core.

*What it costs:* every call is a round trip through a queue, so the UI code is
chattier than a direct call would be.

*What it buys:* a window stays responsive while a 43 MB model downloads, and the
core can be driven headlessly by the CLI and the tests with no Qt at all — which
is why the same code path is testable on a machine with no display.

### 3. Providers are behind interfaces

`LLMProvider` (`app/llm/base.py`) and `ThreeDProvider` (`app/providers3d/base.py`)
are abstract bases with a small required surface; the registries build them from
configuration by `kind`. Nothing above the registry knows whether the model is
GPT, Claude, Gemini or something local, and nothing above the 3D registry knows
that Tripo's statuses are spelled `success` and `banned`.

*What it costs:* a new service is a new class plus a table entry, not a config
line.

*What it buys:* models are equal citizens. A local llama.cpp server and a hosted
API are both just a base URL, and the same is true of a 3D provider.

### 4. The core never imports Qt

`app/core`, `app/llm`, `app/mcp`, `app/providers3d`, `app/storage`,
`app/benchmark` have no Qt import. Only `app/gui` does. A test can therefore
build the entire application, minus the window, in a headless process — which is
what `--check` does, and a `--check` that assembled something simpler would be a
check that passes while the app is broken.

### 5. The database is not optional, and neither is it the truth

Everything the studio does is recorded: projects, conversations, messages, tool
calls, LLM requests, 3D tasks, settings, benchmark runs and reviews. The Tasks
panel, the cost figures and the benchmark table are all views of what was stored,
so a number in the window and a number in the database come from the same place.

The one deliberate exception: the studio is not the source of truth for Blender.
`blender-mcp` is, and the studio asks it.

## A turn, end to end

1. The user types. `MainWindow.send` adds a bubble and submits
   `_run_agent(text)` to the core.
2. `AppContext.agent()` (`app/core/context.py`) builds an `Agent`
   (`app/core/agent.py`): an LLM provider resolved by name, the merged MCP
   catalogue, the local tools, the budget, and the system prompt.
3. `Agent.run` creates a run id, records the user message, and builds the request:
   system prompt, history, the new user message, and the tool list. The tool list
   is where the `execute_python` gate applies.
4. The model answers, either with tool calls or with text.
5. Each tool call runs on the core. An MCP tool goes through `MCPManager`; a local
   tool is awaited directly. Either way the result is published as
   `TOOL_FINISHED` or `TOOL_FAILED` with a duration, and a rendered image is
   attached to the next request if the model can see.
6. The result goes back to the model as a tool message, and the loop continues.
7. When the model answers with text, the run finishes: the assistant message is
   stored, usage is priced and recorded, totals are published, and the window
   refreshes the scene.

Every step is cancellable, and every step is bounded: steps, tool calls, wall
clock, session cost, 3D credits, and one request's cost.

## 3D, end to end

1. The model calls `generate_3d_asset`. The tool is only offered when a provider
   is both enabled and keyed — a tool that always fails wastes a step and teaches
   the model that the tool is noise.
2. `provider.create()` is awaited inline. A refusal — no credit, a bad model name
   — comes back in the same breath, because that is something the model can still
   act on.
3. The slow half becomes a tracked task: poll with a progress callback, download,
   import, and record the credits **as soon as the provider reports them**, not on
   the way to a green result.
4. The task publishes `TASK_*` events, which is what the panel renders, and is
   written to `three_d_tasks` twice: once queued, once settled.
5. The import goes through `app/providers3d/importer.py`, which prefers a future
   `blender.import_asset` MCP tool and otherwise uses `blender.execute_python`.
   What arrived is a **scene diff**, not the operator's word for it.

## Where the files live

```text
app/
  main.py            the entry point: --check, --prompt, --benchmark, or the window
  core/
    context.py       AppContext: assembles everything, and builds agents
    agent.py         the run loop, the tool gate, budgets, cancellation
    task_manager.py  background work, tracked and stored
    settings.py      every setting, from defaults to .env to the database
    events.py        the event bus the GUI renders
    cost.py          pricing and the limits it is checked against
    errors.py        the errors that carry a hint
  llm/               the LLMProvider interface, four providers, the registry
                     openai · anthropic · gemini · openai-compatible
  mcp/               the manager, one session per server, image fetching
  providers3d/       the ThreeDProvider interface, Tripo, a mock, the importer
  storage/           SQLite, migrations, repositories, the secret store
  benchmark/         the runner, the models, the storage
  gui/               the window and its seven panels, and the core thread
  mcp/support.py     a real MCP server used by the tests and as a second server
examples/            the live runs, described in the README's testing section
tests/               292 tests, mirroring app/ directory for directory
```

## Things that will look like mistakes

- **`blender.execute_python` is off by default, and the studio also refuses it.**
  Two gates on purpose: the server can be started open, and a client that cannot
  close it is a client whose switch is a promise it does not keep. The studio
  withholds the tool from the model and refuses a call with
  `error_code="TOOL_DISABLED"` and a message naming the setting.
- **The catalog is not in the system prompt.** Sending 15 tool schemas to a small
  model cost 30 000 input tokens per turn; the model can already see the tool
  list, and the prompt says where to look.
- **`--benchmark` produces a table and no winner.** A heuristic that crowned a
  best model would be a claim about taste dressed as a metric.
- **The importer crosses a filesystem boundary on purpose.** The studio and
  Blender are often on different sides — a WSL-hosted server driving a Windows
  Blender — and a path is not a file. The same reasoning is why the studio reads
  a render itself instead of relying on `blender://render/latest`.
- **A `mock` provider exists in both registries under slightly different names**
  (`mock` vs `mock-3d`). It is a test double, and the mismatch is untidy rather
  than meaningful.

## Reading further

- [providers.md](providers.md) — adding an LLM provider, a 3D provider, a tool.
- [development.md](development.md) — running it, testing it, the shape of a change.
- [troubleshooting.md](troubleshooting.md) — the failures that actually happen.
