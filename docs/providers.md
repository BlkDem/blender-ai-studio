# Extending the studio

Three things can be added without touching anything else: an LLM provider, a 3D
provider, and a tool. Each is a class plus a table entry.

## An LLM provider

Subclass `LLMProvider` (`app/llm/base.py`) and implement four methods. The
registry builds providers from configuration by `kind`.

```python
from app.llm.base import ChatRequest, ChatResponse, LLMProvider, StreamChunk
from app.llm.models import ModelInfo


class MyProvider(LLMProvider):
    name = "mine"

    def default_model(self) -> str:
        return "some-model"

    def models(self) -> list[str]:
        return ["some-model", "another-model"]

    async def chat(self, request: ChatRequest) -> ChatResponse: ...  # one request, one response

    async def stream(self, request: ChatRequest):  # an async generator
        ...
        yield StreamChunk(text="...")
```

Then register the kind in `app/llm/registry.py`:

```python
PROVIDER_TYPES: dict[str, tuple[type[LLMProvider], str, str]] = {
    # name: (class, default base url, environment variable holding the key)
    "openai": (OpenAIProvider, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "mine": (MyProvider, "https://api.example/v1", "MY_API_KEY"),
}
```

…and add a branch in `LLMRegistry._build()` where the other kinds are built
(`registry.py:135`). The third element of the tuple is the environment variable
that holds the key; an entry of `""` means only the secret store is consulted.
A provider whose `kind` is not in the table is still looked up as
`f"{NAME.upper()}_API_KEY"`.

### What a provider must get right

- **Capabilities come from data, not from the name.** `ModelInfo.supports_tools`,
  `supports_vision` and `context_window` are read by the agent. Nothing anywhere
  branches on a model id, which is why a local model and a hosted one are equal.
- **Return usage.** `ChatResponse.usage` is what the cost is computed from. A
  provider that returns none is reported as costing nothing, which is worse than
  being expensive.
- **Images.** If the service is multimodal, encode `Message.parts` — see
  `_to_wire` in `openai_compatible.py` and `anthropic.py` for the two shapes, and
  `gemini.py` for the third. A part of type `image` carries base64 `data` and a
  `mime_type`.
- **Errors.** Raise `app.core.errors.StudioError` subclasses, or return a
  response the agent will stop on. A provider that raises `ValueError` from a
  malformed answer takes the run down rather than telling the model what was
  wrong with it.
- **Tell a busy provider from a broken one.** 429 and 5xx are `RateLimitError` and
  a `retryable=True` `ProviderError`. A report that calls capacity exhaustion a
  broken configuration sends the user to edit something that is already right.
- **A parameter name is not universal.** OpenAI's newer models reject
  `max_tokens` and name `max_completion_tokens` in the error, while Groq,
  OpenRouter, llama.cpp and vLLM know only `max_tokens`. The adapter asks: it
  sends, reads the refusal, and retries once with the spelling that model asked
  for. Branching on a model id instead would put the knowledge where it goes
  stale, and a gateway that changes its mind would be caught out by it.
- **Whatever the provider attaches to a call has to come back with the result.**
  Gemini signs each function call and refuses a result whose signature did not
  return, which is the second request failing, not the first. Two things break it
  if you are careless: reading the field from the wrong level of the response
  (an empty signature looks exactly like a model that sent none), and an
  intermediary that rebuilds the call from what it was given. The agent reassembles
  a call out of `StreamChunk`s, so anything the chunk does not carry is gone by the
  next turn — carry it there too.

### Testing it

`tests/llm/test_providers.py` builds a request and asserts the wire shape per
provider. Add yours there, or use `MockLLMProvider` with `ScriptedTurn` when you
only need the agent's behaviour.

## A 3D provider

Subclass `ThreeDProvider` (`app/providers3d/base.py`). Four abstract methods; the
polling loop, the credit estimate and the task vocabulary come from the base
class.

```python
from app.providers3d.base import ThreeDProvider, failed_task, succeeded_task
from app.providers3d.models import AssetRequest, AssetResult, ProviderTask, TaskStatus


class My3D(ThreeDProvider):
    name = "mine-3d"
    display_name = "Mine 3D"
    enabled = True

    def supported_kinds(self) -> set[str]:
        return {"text_to_3d", "image_to_3d"}

    async def create(
        self, request: AssetRequest, *, run_id: str = ""
    ) -> ProviderTask: ...  # submit, and return a task carrying the provider's own id

    async def status(
        self, task: ProviderTask
    ) -> ProviderTask: ...  # poll once; set task.status, task.progress, task.error

    async def download(
        self, task: ProviderTask, destination: Path
    ) -> Path: ...  # write the file and return its path

    async def close(self) -> None:
        return None
```

Register it in `app/providers3d/registry.py`:

```python
PROVIDER_TYPES: dict[str, type[ThreeDProvider]] = {
    "tripo": TripoProvider,
    "mine-3d": My3D,
}
```

### What a provider must get right

- **Translate the service's vocabulary.** Tripo's `success` and `banned` become
  `TaskStatus.SUCCEEDED` and a refusal whose message names the user's prompt.
  Nothing above the provider should know the service's spellings.
- **A refusal is not an exception, unless it is one.** `create` returning a failed
  task reaches the model as data. Raising reaches the caller as an error. Choose
  deliberately: an empty account is a refusal the model should hear; a
  programming mistake is an exception.
- **Set `task.credits` when the service reports them**, not when your run
  succeeds. A task whose import failed still cost money, and that is exactly the
  cost a person wants to see.
- **`estimate_credits` is a guess, and is labelled as one** in the tool result the
  model reads.

## A tool

A tool the studio owns is a `LocalTool` (`app/core/agent.py`): a name, a
description, a JSON schema, and an async handler.

```python
from app.core.agent import LocalTool


def my_tool() -> LocalTool:
    async def handler(arguments: dict, *, run_id: str = "") -> dict:
        return {"ok": True, "detail": "..."}

    return LocalTool(
        name="do_the_thing",
        description="What it does, and when the model should reach for it.",
        schema={
            "type": "object",
            "properties": {"what": {"type": "string"}},
            "required": ["what"],
        },
        handler=handler,
    )
```

Pass it to the agent:

```python
context.agent(provider, model, extra_tools=[my_tool()])
```

### Rules a tool follows here

- **The description is the model's only guidance.** Name the boundary: what the
  tool is for, what it is not for, and an example. A model asked for "a medieval
  wooden chest" reaches for `blender.create_object` unless something tells it
  that a modelled object and an arrangement of primitives are different requests.
- **Return a dict.** The agent reads `images`, `credits` and `cost_usd` out of it
  and hands the rest to the model. Anything else is stringified, with no cost.
- **Raise, or return an error field.** `{"error": "..."}` reaches the model as
  something it can act on; an exception becomes a failed run.
- **Anything slow belongs in `TaskManager`, not in the handler.** The handler is
  awaited inside the agent's step, and a two-minute generation there blocks the
  run, the window and the budget check. `generate_3d_asset` submits, returns a
  task id, and the minutes happen on a task.

## An MCP server

Nothing to implement — a server is configuration. `MCPManager` discovers tools
at runtime, and a name that appears on two servers is an error rather than a
silent choice.

```json
STUDIO_MCP_SERVERS='[
  {"name": "blender", "command": "python3", "args": ["-m", "server.main"],
   "cwd": "/path/to/blender-mcp", "env": {"PYTHONPATH": "/path/to/blender-mcp"},
   "blender_port": 8767},
  {"name": "files", "command": "python3", "args": ["-m", "fs_server"]}
]'
```

In the window, the Settings page edits one server at a time, identified by name,
and saves without disturbing the others.

`app/mcp/support.py` is a real, runnable MCP server used by the tests — the
quickest template for a new one, and the way to see a second server's tools join
the merged catalogue.
