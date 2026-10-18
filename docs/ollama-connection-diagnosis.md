# Ollama connection diagnosis (2026-10-17)

The Desktop launcher starts Windows Python in `C:\Users\maxim\projects\blender-ai-studio`.
The effective settings database is **not AppData**: with the current settings it is
`C:\Users\maxim\.local\share\blender-ai-studio\studio.db`.
It was queried with Windows Python (opening the live SQLite WAL database through
WSL's mounted filesystem returned `disk I/O error`). No secrets were read or changed.

## Observed facts

- Windows `ollama` PID 5324 listens on `127.0.0.1:11434`; `ollama app` PID 12028
  also has its own listener on port 62730 (not the inference endpoint).
- Windows `Invoke-RestMethod /api/tags` and HTTPX `/v1/models` both succeed on
  port 11434, listing `qwen3-coder:latest`. Port 11400 refuses connections.
- Stored `localqwen` uses `openai-compatible`, `http://127.0.0.1:11434/v1`,
  default model `qwen3-coder`, no options. The model alias works; a 4-token-capped
  chat returned `OK` (2 completion tokens). No URL or model correction was needed.
- Windows Python had no HTTP_PROXY, HTTPS_PROXY, ALL_PROXY or NO_PROXY variables;
  HTTPX worked with both `trust_env=True` and `False`. Proxy routing was not an
  observed cause in that invocation. An already-running GUI may have a different
  inherited environment, which this probe cannot retrospectively inspect.
- WSL has no Ollama process/listener. WSL localhost ports 11434 and 11400 refuse
  connections; the default-route Windows host `172.21.160.1:11434` times out.
  Thus the same localhost URL does not work from this WSL installation.
- After the patch, a Windows smoke test using `Settings`, database overrides,
  `registry_from_settings`, and the actual `localqwen` provider returned `OK`
  with a 4-token cap and a 45-second outer timeout.

## Scope and conclusion

The Windows failure did **not** reproduce: its persisted endpoint is already
correct and the model works. The confirmed networking failure is WSL attempting
to reach a Windows loopback-only listener. This is not proof that an earlier
Windows GUI error had the same cause. No firewall, service binding, database,
API key, or user UI work was changed. Restart the Windows GUI to load the patch.
Running the Linux UI against this Windows-only listener still requires an
explicitly chosen networking setup; no such security-sensitive change is made
implicitly.

The transport now defaults to bypassing environment proxies for exact loopback
hosts (explicit `trust_env` can override it), normalizes root or `/api` URLs on
local Ollama conventional ports to `/v1`, and presents safe connection diagnostics
including Windows/WSL and Ollama checks. Remote gateway paths remain unchanged.

Only these patch files were copied to Windows after verifying that its existing
provider file matched Git HEAD and that new files did not already exist:

- `app/llm/connection.py`
- `app/llm/providers/openai_compatible.py`
- `tests/llm/test_local_connection.py`

Linux verification: `pytest tests/llm -q` — 91 passed.
