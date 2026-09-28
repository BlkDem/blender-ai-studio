# Contributing

The project is a desktop AI client for Blender, and the two rules that shape most
of its code are these: **Blender is reached only over MCP**, and **a refusal the
user can act on is better than a silent failure**.

## Getting set up

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m app.main --check
```

Then read [docs/development.md](docs/development.md) for the daily loop, and
[docs/architecture.md](docs/architecture.md) for why the pieces are cut where they
are.

## A change, start to finish

1. **Reproduce it, then write the test that fails for that reason.** Not "the fix
   works" — the old behaviour, the wrong answer. If the bug does not break a test,
   either the test is wrong or the diagnosis was.
2. **Fix it.**
3. **Run the whole thing:**
   ```bash
   ruff check . && ruff format . && mypy app && .venv/bin/python -m pytest
   ```
4. **Check it for real** if it touches Blender, a model, 3D or the window. The
   scripts in `examples/` exist for this; `live_run.py` covers the most ground in
   one go.
5. **Write the commit message.** What was broken, why it was broken, and what the
   fix is. Every bug this project has had is described in its commit message,
   which is the only place some of them are recorded.

## What is expected of a change

- **Comments explain why.** What the code does is the code's job. A comment earns
  its place by explaining a boundary, a measured failure, or a decision that looks
  wrong and is not.
- **No silent no-ops.** If something cannot be done, say so in terms the user can
  act on: which tool, which setting, which file, which side of a filesystem.
- **Capabilities are data.** Nothing may branch on a model's name; declare them in
  the model catalog.
- **No keys, no machine-specific paths, no scratch directories** in tracked
  files. Paths from the machine you happened to develop on are the most common
  thing to get in by accident.
- **New behaviour comes with a test**, and anything that costs money or needs
  credentials belongs in `examples/`, not in the suite.

## Reporting a bug

What you did, what you expected, what happened instead, and the output of
`.venv/bin/python -m app.main --check`. The exact text of an error message is
worth more than a description of it.

## Licence

MIT.
