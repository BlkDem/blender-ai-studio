# Models and providers editor

The Models page has three scrollable tabs:

- **Models**: select a row to edit its exact ID, tool/image capabilities, context size and token prices. **New model** starts a separate entry. Saving selects that model; the default checkbox controls the provider's default independently. Renaming preserves other model metadata. Removing a default chooses the first remaining model, or clears the default if none remain. Disabled providers' models remain editable but are not offered in the chat selector.
- **Providers**: select a row to edit its kind, URL, enabled state, default model and JSON options. Existing names are stable identifiers; use **New provider** for a new name. An initial model ID is optional; models can also be added on the Models tab. Provider updates retain the model catalogue. Removing a provider removes its catalogue, but deliberately retains its saved credentials.
- **3D providers**: edit the configured provider, enabled flag and options as a JSON object. The available implementations are those registered in `app.providers3d.registry.PROVIDER_TYPES` (currently `mock` and `tripo`); the editor does not force a provider change. Options correspond to `ThreeDConfig`, including model, quality, texture, credit limit, polling and download directory.

Provider/model deletions and credential deletion require confirmation. Secret fields never display stored credentials. A blank key on Save means **keep the existing key**, not delete it. Use **Delete saved key** explicitly to remove a credential. Environment credentials may still be used after deleting a stored LLM key. Keys live in the configured secret store, not in the settings JSON/database entries. Normal configuration is persisted through the application settings store, and cached provider clients are invalidated when their configuration or key changes.

JSON options must be objects. Keep API keys out of options: use the dedicated password fields. Existing provider names cannot be renamed because projects and saved selections refer to those stable identifiers.
