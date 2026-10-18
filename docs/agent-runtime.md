# Agent runtime

Studio runs an iterative agent, not a one-shot chat completion. Each user turn
contains the conversation, server instructions, and the available tool schemas.
The model can return text and several tool calls. Studio executes calls in order,
returns each result under its original call ID, and asks the model to continue.
The turn finishes when the model replies without tool calls, or a limit/error
stops execution.

## Tools and results

MCP tools are discovered from connected servers. Local tools add document reads,
explicit Blender saves, optional 3D generation, and background-task management.
Arguments are validated against the advertised JSON Schema before dispatch.
Unknown, disabled, invalid, and failed calls return errors to the model, allowing
it to correct its request. These outcomes also appear as failed tool cards.
The Studio Python-execution switch is enforced at dispatch, not just in the prompt.

Tool cards show arguments and results. Long results can be expanded into a
selectable text view. Render images are displayed and, for vision-capable models,
returned to the model. OpenAI-compatible endpoints receive tool images as an
additional user message after the complete group of tool results, because image
content is not supported in their tool-role messages.

## Background tasks

`generate_3d_asset` can return before generation/download/import completes. The
model can then use:

- `list_tasks`: discover tracked tasks, optionally only active ones.
- `get_task`: inspect status, progress, result, costs, and errors.
- `wait_task`: wait up to 300 seconds without cancelling work on a wait timeout.
- `cancel_task`: request cancellation and inspect the resulting state.

Use Studio task IDs, not remote provider IDs. A submitted task is not a completed
asset; the model should inspect completion and import results before claiming
success. Task management currently covers tasks tracked by the running Studio
process; it is not a durable job scheduler across application restarts.

## Limits and continuation

Execution is bounded by step, tool-call, cost, and wall-clock limits. The
wall-clock deadline interrupts in-flight awaits as well as the loop. Cancellation
or a timeout does not guarantee rollback of external side effects: inspect the
scene or destination before repeating a mutating operation.

Between turns, memory keeps complete tool-call/result groups. Interrupted groups
and orphaned results are excluded from replay; memory trimming drops whole groups
rather than producing an invalid provider conversation. Unsupported model tool
and streaming capabilities are respected.

## Security boundary

The agent does not provide a general approval dialog or filesystem sandbox.
Document reads remain bounded local reads, and arbitrary Blender Python remains
an explicitly enabled powerful capability. Only connect trusted MCP servers and
models and enable Python when appropriate. Prompt instructions to request
confirmation are not a substitute for an enforced permission boundary.
