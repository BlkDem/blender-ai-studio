# Blender AI Studio

A desktop client for driving Blender with an AI model. You write a sentence, the
model decides which tools to call, and Blender changes. The model is a setting:
GPT, Claude, Gemini, Space Bunny, JEV, or anything that speaks the OpenAI wire
format, including one you run yourself.

The studio does not talk to Blender. It speaks MCP to
[`blender-mcp`](../blender-mcp), which does, and the tool list it works from is
whatever that server publishes — there is no `blender.create_object` written
anywhere in this project.

```text
you:  Create a table with four legs.

LLM           decides: get_scene, then create objects
agent         runs the tools, reads each result, corrects if needed
MCP client    stdio -> blender-mcp
blender-mcp   WebSocket -> a running Blender
you:          watch the tool cards appear, and the viewport change
```

## Contents

**Start here:** this page. Then, depending on why you are here:

| | |
|---|---|
| [Install](#install) · [Connect Blender](#connect-blender) · [Connect a model](#connect-a-model) · [Check what you configured](#check-what-you-configured) | getting it running |
| [Your first task](#your-first-task) · [What it looks like](#what-it-looks-like) | using it |
| [The panels](#the-panels) · [3D generation](#3d-generation) · [The vision loop](#the-vision-loop) · [Projects](#projects) · [Benchmark](#benchmark) | what each part does |
| [Costs and limits](#costs-and-limits) · [Configuration](#configuration) | tuning it |
| [Testing](#testing) · [Development](#development) · [Roadmap](#roadmap) | working on it |
| [docs/architecture.md](docs/architecture.md) | how it fits together, and why |
| [docs/providers.md](docs/providers.md) | adding a model provider, a 3D provider, a tool |
| [docs/development.md](docs/development.md) | the daily loop, and the shape of a change |
| [docs/troubleshooting.md](docs/troubleshooting.md) | the failures that actually happen |
| [CONTRIBUTING.md](CONTRIBUTING.md) | before you open a pull request |

## Install

Python 3.11 or newer. The studio does not touch your Blender installation and does
not write into your `blender-mcp` checkout.

```bash
cd blender-ai-studio
python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
```

`PySide6-Essentials` is enough for the window; a system keyring is used for API
keys when `pip install ".[keyring]"` finds one.

Check the install before starting Blender:

```bash
.venv/bin/python -m app.main --check
```

## Connect Blender

`blender-mcp` runs as a child process over stdio, and its own bridge listens for
the Blender add-on. Point the studio at your checkout in the GUI
(**Settings → Blender**), or in the environment:

```bash
export STUDIO_MCP_SERVERS='[{
  "name": "Blender MCP",
  "command": "/usr/bin/python3.11",
  "args": ["-m", "server.main"],
  "cwd": "/path/to/blender-mcp",
  "env": {"PYTHONPATH": "/path/to/blender-mcp"},
  "blender_port": 8765
}]'
```

Then start Blender with the add-on enabled and press **Connect** in the Blender MCP
sidebar. The add-on reconnects on its own, so you can start it either way round.
The studio waits for it: a Blender that has not attached yet is not the same as
one that is not there.

## Connect a model

**Settings → Models** (or the Models panel) takes a provider, a base URL and a
key. A model is a row of metadata — capabilities and prices — so a model the
studio has never heard of is a row, not a code change:

```json
{
  "id": "space-bunny-free",
  "display_name": "Space Bunny",
  "supports_tools": true,
  "supports_vision": false,
  "context_window": 128000,
  "input_price": 0,
  "output_price": 0
}
```

The **OpenAI-compatible** provider is the one to reach for first: it is OpenAI
itself, and also any gateway, vLLM or Ollama that copies the wire format. Enter
the base URL and the model id, and the studio talks to it.

Keys go to the system keyring when there is one, otherwise to a `0600` file in
the data directory. They are never written to a project, never logged, never sent
to a model, and never included in a benchmark log.

A provider that ships configured but without a key — gemini, cerebras, mistral,
github — is given one without being redefined: click its row in the Models table,
**Paste** the key, **Save key**. "Add provider" also sets a key, and also replaces
the whole configuration, so it is the wrong tool for a provider you already have.
Pasting has its own button because Qt withholds the standard context menu from a
password field, and typing a key out in full is not an answer.

## Check what you configured

A model that is listed is not a model that works. Ask it, one model at a time:

```bash
.venv/bin/python examples/model_check.py            # everything configured
.venv/bin/python examples/model_check.py --provider openai
```

Four questions per model, because each fails differently:

| Column | Question |
|---|---|
| `answers` | does it reply at all? |
| `tools` | does it call a tool it was offered, by the name it was given? |
| `goes on` | can it use the tool's answer? |
| `sees` | does an image survive the round trip? |

`goes on` is the one that is easy to skip and expensive to miss: a provider can
attach something to a call that has to come back with the result — Gemini signs
each function call and refuses a result whose signature did not — and a model can
call tools perfectly while being unable to continue.

`unavailable` is not `broken`. A provider with no capacity, or one holding your
account to a quota, is not a configuration to go and fix, and the script says
which of the two it found. A model the provider has retired says so and names
its replacement; check the live catalogue rather than a remembered list, because
that list is what went stale in the first place.

## Your first task

Start Blender, connect the add-on, pick a model, and type:

```text
Create a cube named TestCube at location 2, 0, 1.
```

You will see the tool call as a card in the transcript, with its arguments, how
long it took and whether it worked. A second call verifies it:

```text
tool: ✓ blender.create_object · 41 ms
tool: ✓ blender.get_object · 3 ms
     cube TestCube at [2.0, 0.0, 1.0]
```

To do the same without a window:

```bash
.venv/bin/python -m app.main --prompt "Create a cube named TestCube at location 2, 0, 1."
```

## What it looks like

Seven pages on the left, a transcript on the right, and a status line that always
answers three questions: is Blender attached, is the model ready, is 3D ready.

```text
┌────────────────────────────────────────────────────────────────┐
│ Blender AI Studio    ● Blender            Model: space-bunny ▾  │
├──────────┬─────────────────────────────────────────────────────┤
│ Chat     │  user: Create a table with four legs.               │
│ Scene    │                                                     │
│ Tasks    │  assistant: I'll build it from primitives.           │
│ Benchmark│                                                     │
│ Models   │  🔧 blender.get_scene            ✓ 124 objects 8ms │
│ Settings │  🔧 blender.create_object        ✓ TableTop 41ms     │
│          │  🔧 blender.update_object        ✓ location 12ms     │
│          │                                                     │
│          │  Built a table: a top at 0.75 m and four legs.      │
│          │  3 tool calls · 1 204 tokens · $0.0041 · 8.2s       │
├──────────┴─────────────────────────────────────────────────────┤
│ MCP: connected · 15 tools    LLM: space-bunny    3D: tripo      │
└────────────────────────────────────────────────────────────────┘
```

A tool call is visually separate from prose, because watching the model work is
one of the point of the thing.

## Architecture

```text
                       Blender AI Studio
                              │
                  ┌───────────┴───────────┐
                  │      GUI (PySide6)    │  Chat, Scene, Tasks,
                  │   queue + drain timer │  Benchmark, Models, Settings
                  └───────────┬───────────┘
                              │ coroutines
                       AI Agent ──────────── TaskManager
                       │    │    │
              LLM       │  3D     │  MCP client (stdio)
              │         │         │        │
       OpenAI-compatible  │    blender-mcp (its own process)
       Anthropic          │         │  WebSocket
       Gemini        Tripo 3D        │
       Mock                   └──── Blender
```

Four abstractions carry the whole design, and the rules around them are short:

| | |
|---|---|
| `LLMProvider` | `chat`, `stream`, `models`, `close`. Nothing above it sees a vendor shape. |
| `MCPManager` | `connect`, `tools`, `call_tool`, `read_resource`. Nothing above it names a Blender tool. |
| `ThreeDProvider` | `create`, `status`, `download`, `wait_for`. Tripo is one implementation. |
| `Agent` | asks a model, runs what it asked for, stops under a budget. |

The layering rules that keep it true:

- **The GUI never blocks.** Every network call is a coroutine on the core thread;
  results come back as queued Qt signals.
- **The core never imports Qt.** `--check`, the acceptance run and the tests drive
  the same application the window does.
- **No model is special-cased.** There is no `if model == "space-bunny"` in this
  project. Capabilities are metadata; prices are metadata.
- **Secrets are not settings.** They live in the keyring or a `0600` file, and
  the in-memory cache refuses to print itself.

```text
app/
├── main.py            --check, --prompt, --benchmark, --list-tools, or the window
├── core/              agent, events, cost, settings, task manager, context
├── llm/               base (the conversation format), models, registry, providers/
├── mcp/               client (one server), manager (all of them), models
├── providers3d/       base, registry, tripo, mock
├── benchmark/         runner, storage, models
├── storage/           database, migrations, repositories, secrets
└── gui/               bridge, main_window, chat/, scene/, tasks/, models/, benchmark/, settings/
```

## The panels

| Panel | What it is for |
|---|---|
| **Chat** | The conversation, streamed, with a card per tool call: arguments, duration, success, images. Stop any time. |
| **Scene** | What Blender is showing — connection, scene, object count, active object, camera, engine, frame — read over MCP like everything else. |
| **Tasks** | Background work: provider, status, progress, duration, credits, cost. Cancel from here, and it is restored from the database on the next launch. |
| **Projects** | Named pieces of work with a starting `.blend`. Turns are filed under the open one, and opening it brings the transcript back. |
| **Benchmark** | Build a suite, run it, read the table, score the results yourself. |
| **Models** | Providers, keys, capabilities, and the model selector's contents. |
| **Settings** | Blender's server, the agent's limits, and the budgets. |

## 3D generation

The agent has one tool of its own, `generate_3d_asset`, and it is offered only
when a 3D provider is enabled *and* has a key — a tool that always fails teaches
a model that the tool is noise.

```text
you:      Create a medieval wooden chest.
agent:    generate_3d_asset {"prompt": "medieval wooden chest", "texture": true}
tasks:    Tripo generation · queued · est. 100 credits
you:      …later
tasks:    succeeded · 100 credits · ~/.local/share/blender-ai-studio/assets/chest.glb
```

Tripo is implemented against the real v3 API: submit, poll, download, and
verified against the live service — a generated PBR model (43 MB) imported into
a running Blender, 20 credits for one text-to-3D task with texture. Its
`code`/`data` envelope and its `success`/`banned` statuses are translated inside
the provider, so nothing above it knows they exist. `banned` in particular is
reported as "Tripo refused this prompt under its content policy", because that
is the user's prompt to fix rather than a failure to retry.

A generated asset is **not** in the scene the moment it is submitted — that takes
minutes and a provider's own state machine. So the tool submits, answers the
model immediately, and hands the rest to a tracked task: poll, download, import.
The Tasks panel, the database and the agent's costs all come from that one task,
so the numbers in the table are the numbers the user watched appear. A submission
that is *refused* — an account with no credit, a prompt the provider will not take
— comes back in the same breath as the call, because that is something to go and
fix rather than wait for.

Importing needs `blender.execute_python`, which is off by default; when it is
off, the task says so, names the tool, and **keeps the download**. The extension
point for a future `blender.import_asset` MCP tool is the same place, and adding
it there would not touch the agent or the task.

A generation outlives the client that asked for it, so `asset_run.py --resume
TASK_ID` picks a task up where the provider already has it instead of paying for
it twice. The import report says where the asset landed, because a GLB arrives
where its file says — usually the world origin, which is often inside whatever
else is standing there.

Where assets are downloaded matters and is a setting
(`three_d.download_dir`, defaulting to `data_dir/assets`): the studio and Blender
are often on different sides of a filesystem, and a model that lands somewhere
Blender cannot read is a model that cannot be imported. The importer says so in
those terms, and reports the path it actually handed over.

## The vision loop

Rendering is the one tool result a model can act on without parsing text, so the
loop is: render, look, correct.

Closing it needed three things that all existed and none of which was connected.
A render comes back as a *path on Blender's side of the filesystem*, and
`blender://render/latest` returns the bytes — read from the MCP server's own
filesystem, which in the usual arrangement (server in WSL, Blender on Windows) is
a different machine. So the studio reads the render itself, translating the path
across the boundary; the importer already had to do exactly this for generated
models. The image then goes into the next request as a content part, encoded per
provider, and the chat card shows it rather than saying how many there were.

Whether a model is *sent* the picture is asked, not assumed: the model catalogue
declares `supports_vision`, and a text-only model gets the text alone, because
sending an image to a model that cannot see is a request it rejects.

Verified live against Qwen2.5-VL-3B: a real render of the real scene, read across
the boundary, and the model described what was in it.

## Projects

A project is a named piece of work with a starting `.blend`, and every turn,
generation and review belongs to one. Opening a project brings its transcript
back; closing it stops filing; deleting it takes its conversations with it. A
project's file is opened in Blender **from a copy** — the file a project starts
from is never the file being edited, or the next run would not start where the
last one did.

## Benchmark

The point of a benchmark here is that runs are *comparable*, so every run gets
its own copy of the starting `.blend` and the table carries a **Scene reset**
column: `verified` (the file was opened), `copy_only` (a copy was staged; open it
yourself) or `unverified`. A comparison that mixes them is not a comparison, and
the table says so at the bottom.

```text
Model           Status      Time      Tokens   Cost      Tools  Errors  Scene reset
a-model @ gpt   ok          18.4s     3 210     $0.0121   7      0       verified
b-model @ claude ok        24.1s     4 880     $0.0187   9      1       verified

2 run(s); 2 with a verified starting scene. Scores are yours — the studio does not rank models.
```

Metrics stored per run: duration, tokens, cost, MCP calls, tool errors, 3D
calls, credits, the final scene, and the transcript. Scores are a person's:
geometry, materials, instruction following, composition, overall, and notes.
There is no automatic winner anywhere in this project — a heuristic that crowned
a best model would be a claim about taste dressed as a metric.

A comparison belongs somewhere repeatable, so it runs headless:

```bash
.venv/bin/python -m app.main \
    --benchmark "qwen3b:local-qwen,qwen15b:local-qwen15" \
    --prompt "How many objects are in the scene? Use blender.get_objects." \
    --blend ~/scenes/room.blend
```

Several models, several prompts (`--prompt` repeated, or `--prompts file.txt`),
one run each, sequential — two models driving one Blender at once would
interleave their tool calls and the scenes would belong to neither. Each run is
saved as it finishes, so a comparison stopped half-way is still a record, and the
id printed next to each row is the one a review attaches to. The window has the
same panel; the command is the repeatable half of it.

## Costs and limits

Two currencies, tracked apart from the first line of code: token bills and 3D
credits are both "cost" and neither is the other, so a benchmark table sums them
only where both are real money, and always shows the split.

| Limit | Default | What it stops |
|---|---|---|
| `agent.max_steps` | 30 | a model that keeps asking the same thing |
| `agent.max_tool_calls` | 50 | checked per call, so six tools in one turn stop at the fifth |
| `agent.max_seconds` | 900 | a run that cannot end |
| `agent.max_request_cost` | $2 | one enormous response |
| `agent.max_session_cost` | $5 | a long conversation |
| `agent.max_3d_credits` | 200 | a plan that would spend real credits |
| `agent.allow_execute_python` | off | the one tool that can do anything |

A run that stops between `begin_transaction` and `commit` leaves the transaction
open, and the *next* run's `begin_transaction` is then refused with
`TRANSACTION_ACTIVE` — a failure with no obvious cause. The agent therefore tracks
its own transactions and says so when a run ends with one open, and the Scene
panel has a **Commit transaction** / **Roll back transaction** button. Both outcomes
destroy something, so neither is done for you.

A run that reaches a limit stops with a reason in the transcript and a
`RUN_FAILED` event, and the usage that was already spent is still recorded — a
cost report that omits the expensive request is the one report that cannot be
trusted.

A **step** is one request to the model plus everything it asked for in that
reply, so `agent steps limit reached (30 of 30)` means the model asked for tools
thirty times and never answered in words. The limit is a backstop, not a
diagnosis: raising it buys more of a loop. What to look for is the last few tool
cards in the transcript — a model that keeps asking the same question with the
same arguments is not working, and the window should say so rather than count to
thirty.

## Configuration

Defaults, then `.env`/environment (`STUDIO_*`), then the database where the GUI
writes what you changed. Dotted keys, so changing one number does not rewrite its
neighbours.

| Variable | Meaning |
|---|---|
| `STUDIO_DATA_DIR` | database, secrets and settings live here |
| `STUDIO_MCP_SERVERS` | the MCP servers, as JSON |
| `STUDIO_LLM_PROVIDERS` | providers, models and prices, as JSON |
| `STUDIO_THREE_D__PROVIDER` | which 3D provider |
| `STUDIO_AGENT__MAX_STEPS` | one of the limits above |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | fallbacks for headless runs |
| `STUDIO_LOG_LEVEL` | `DEBUG` for a full transcript of every request |

See `.env.example`. Secrets do not belong in it: put keys in the GUI, or in the
environment for a headless run.

## Development

```bash
.venv/bin/python -m pytest                 # everything
.venv/bin/python -m pytest -m "not gui"    # without a display
.venv/bin/python -m app.main --check       # configuration, live
.venv/bin/python -m app.main --list-tools  # what the MCP server publishes
```

The live acceptance run — a real agent, a real MCP server, a real Blender:

```bash
.venv/bin/python examples/acceptance_run.py \
    --blender-mcp /path/to/blender-mcp --python /path/to/python
```

It creates a cube at 2,0,1, moves it, and verifies both with second tool calls.
With no API key it drives a scripted planner, so everything except a model's
judgement is exercised; with `--provider` it uses a real model.

Verified against a real model — Qwen2.5-3B-Instruct served by `llama.cpp` on
`http://127.0.0.1:11400/v1`, talking to a live Blender 5.2 GUI over MCP. The model
chose its own sequence (`get_scene` → `begin_transaction` → `create_object` →
`commit_transaction`), and the run passed all 23 checks. Note what the second
check asserts: the object ends up where *the model asked*, not where the prompt
said. A 3B model reading "3, 3, 1" as (3, 0, 1) is a finding about the model, and
measuring prompt-following is the benchmark's job.

To run the same thing locally, for free:

```bash
# 1. a model server (16 MB binary, no key needed)
curl -L -o llama.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11217/llama-b11217-bin-ubuntu-x64.tar.gz
tar xzf llama.tar.gz
./llama-b11217/llama-server -m qwen2.5-3b-q4.gguf --alias local-qwen \
  --jinja --port 11400 --ctx-size 8192

# 2. the acceptance run against it
.venv/bin/python examples/acceptance_run.py --data-dir /tmp/studio \
  --provider localqwen --base-url http://127.0.0.1:11400/v1 --model local-qwen \
  --blender-mcp /path/to/blender-mcp --port 8767

# 3. or photograph the window mid-run
.venv/bin/python examples/gui_shot.py --blender-mcp /path/to/blender-mcp \
  --provider localqwen --base-url http://127.0.0.1:11400/v1 --model local-qwen
```

`--jinja` matters: it is what makes the server use the model's own chat template,
and therefore what makes tool calling work.

## Testing

292 tests, no network and no Blender needed for the majority.

| Area | What is covered |
|---|---|
| LLM | request shape per provider, streaming, tool calls, reasoning fields, image parts, cost |
| MCP | real server over a real pipe: handshake, discovery, errors, reconnection, reading an image a tool left on disk |
| Agent | the loop, budgets, cancellation, persistence, local tools, the `execute_python` gate |
| 3D | request construction, envelope unwrapping, status mapping, polling, download, import, credits |
| Tasks | stored as they start and as they settle, restored by the panel, one row per task |
| Projects | listing, filing turns under one, reopening the transcript, cascade on delete |
| Benchmark | isolation, metrics, storage, the absence of a verdict, the headless command |
| Storage | migrations, foreign keys, concurrency, secrets |
| GUI | panels, the transcript, a render in a tool card, and full runs through the window, offscreen |

The bugs these found are in the commit messages, and the live runs found more that
unit tests cannot: a render nobody could read across a filesystem, a settings save
that deleted every other MCP server, a switch that controlled nothing, and credits
recorded only on the way to a green result.

`examples/live_run.py` is the other half: seven sections against a real Blender, a
real model and a real 3D provider, each answering something only a live run can.

```bash
python examples/live_run.py \
    --blender-mcp ../blender-mcp --python python3 --port 8767 \
    --provider localqwen --provider-spec localqwen:local-qwen \
    --base-url http://127.0.0.1:11400/v1 --allow-execute-python
```

## Roadmap

- [x] MCP client, agent loop, streaming, tool calls
- [x] OpenAI-compatible, Anthropic, Gemini, mock providers
- [x] Tripo: text→3D, image→3D, polling, download
- [x] Cost tracking, budgets, cancellation
- [x] Benchmark with isolated runs and manual review
- [x] Chat, Scene, Tasks, Projects, Benchmark, Models, Settings
- [x] Vision loop: render, read it across the filesystem, send it, act on it
- [x] Projects: named work with a starting `.blend`, and turns filed under it
- [x] Tasks that outlive the window: stored, and restored on the next launch
- [x] Multiple MCP servers, editable by name in the window
- [ ] `blender.import_asset`, so a generated asset lands in the scene without
  `execute_python` — the importer prefers it the moment it exists
- [ ] Say "you already asked for that" when a model repeats a tool call with the
  same arguments, instead of counting to the step limit and stopping there
- [ ] More 3D providers behind the same interface
- [ ] Project files: a project remembers its scene, but does not yet save one

## Licence

MIT.
