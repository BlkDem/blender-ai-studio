"""Is everything I configured actually working?

Configuration drifts. A model is renamed, a key is missing, a base URL moves, a
model answers in prose instead of calling the tool it was offered. Nothing about
those failures looks like a failure in the window: the provider simply never
appears, or a run ends without touching the scene.

So this asks each configured model the three questions that matter, one at a
time, and prints a table:

1. **keyed** — is there a key for this provider at all;
2. **answers** — does a plain request come back;
3. **tools** — does it call a tool when asked, by its dotted MCP name;
4. **goes on** — after calling a tool, can it use the answer? Some providers
   attach something to a call that has to come back with the result, and the
   second request is refused when it does not.
5. **sees** — for models declared multimodal, does an image survive a round
   trip (only asked when the catalog claims it).

Anything keyed and broken is a mistake worth fixing before a session does, so
this exits non-zero when it finds one.

    python examples/model_check.py
    python examples/model_check.py --provider openai --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.errors import RateLimitError  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.llm.base import (  # noqa: E402
    ChatRequest,
    ContentPart,
    Message,
    Role,
    ToolSpec,
)
from app.llm.registry import registry_from_settings  # noqa: E402

#: A 16x16 fully opaque red square. Opaque, because a translucent pixel has no
#: single right answer -- it is whatever it is composited over -- and a check
#: that cannot be failed is not a check.
RED_SQUARE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGUlEQVR4nGP4z8DwnxLMMGrAqAGjBgwXAwAwxP4QHCfkAAAAAABJRU5ErkJggg=="

#: Dotted on purpose: this is the name an MCP tool actually has, and the one that
#: two of the providers refuse to accept without translating.
THE_TOOL = ToolSpec(
    name="blender.create_object",
    description="Create an object in the Blender scene",
    parameters={
        "type": "object",
        "properties": {"type": {"type": "string"}, "name": {"type": "string"}},
        "required": ["type", "name"],
    },
)

PLAIN = "Reply with the single word: ready"
#: A reasoning model spends its first tokens thinking, and answers nothing. The
#: budget has to cover the thinking, or "empty response" is the probe's fault
#: rather than the model's. It is a ceiling, not a target: the call still stops
#: as soon as the answer is finished.
PLAIN_MAX_TOKENS = 400

#: Asking for a tool call takes more thought than asking for one word, and a
#: reasoning model can spend a small budget entirely before it writes the call.
TOOL_MAX_TOKENS = 2000

#: One word of image description, plus the thinking that precedes it. gpt-5-mini
#: spends about 190 tokens reasoning about a single pixel before answering.
#: 800 was enough for one-word answers to a 16x16 swatch and not much else: a
#: reasoning model asked about a real render runs well past it, and a budget
#: spent entirely on thinking comes back as finish_reason "length" with no
#: content at all -- reported as an empty answer, which is a limit and not a
#: failure to see. Measured on space-bunny-free against a 640x400 Blender render:
#: 232 to 1893 completion tokens, one truncation in six at 600, none in eight at
#: 2000. So this is the same budget TOOL_MAX_TOKENS uses, for the same reason.
VISION_MAX_TOKENS = 2000
ASK_FOR_TOOL = (
    "Use blender.create_object to create a cylinder named Probe. Call the tool; do not describe it."
)
ASK_ABOUT_IMAGE = "This is an image. Reply with one word: what colour is it?"

#: What a tool answered, handed back for the second turn. A model can call a
#: tool and still be unable to continue: the second request has to carry
#: whatever the provider attached to the call it is answering, and Gemini
#: refuses one whose thought signature did not come back.
TOOL_RESULT = json.dumps({"name": "Probe", "type": "CYLINDER", "created": True})


@dataclass
class Verdict:
    provider: str
    model: str
    vision: bool
    keyed: bool
    answers: str = ""
    tools: str = ""
    sees: str = ""
    continues: str = ""
    limited: bool = False

    @property
    def broken(self) -> bool:
        """A model is broken by its own account, not by a shared quota.

        A rate limit says the free pool is empty right now; calling the model
        broken sends the user to fix a configuration that is already correct.
        """
        return self.keyed and not self.limited and ("error" in self.answers or "error" in self.tools)

    def row(self) -> str:
        def mark(value: str) -> str:
            return {"ok": "ok", "no": "NO", "skipped": "-"}.get(value, value[:34])

        return (
            f"{self.provider:<11} {self.model:<40} "
            f"{mark(self.answers):<10} {mark(self.tools):<10} "
            f"{mark(self.continues):<10} {mark(self.sees):<10}"
        )


async def check_one(
    provider, provider_name: str, model: str, vision: bool, keyed: bool, timeout: float
) -> Verdict:
    verdict = Verdict(provider=provider_name, model=model, vision=vision, keyed=keyed)
    if not keyed:
        verdict.answers = verdict.tools = verdict.continues = verdict.sees = "skipped"
        return verdict

    try:
        reply = await provider.chat(
            ChatRequest(model=model, messages=[Message.user(PLAIN)], max_tokens=PLAIN_MAX_TOKENS)
        )
        text = (reply.text or "").strip()
        # A reasoning model can spend its whole budget thinking and answer
        # nothing. That is a budget too small, not a model that is broken.
        verdict.answers = "ok" if text else ("reasoning only" if reply.reasoning else "empty")
    except Exception as exc:  # noqa: BLE001 - the point of the script is to report these
        verdict.limited = _transient(exc)
        verdict.answers = "unavailable" if verdict.limited else f"error: {_short(exc)}"
        verdict.tools = verdict.continues = verdict.sees = "skipped"
        return verdict

    try:
        reply = await provider.chat(
            ChatRequest(
                model=model,
                messages=[Message.user(ASK_FOR_TOOL)],
                tools=[THE_TOOL],
                max_tokens=TOOL_MAX_TOKENS,
            )
        )
        called = [call for call in reply.tool_calls if call.name == THE_TOOL.name]
        if called:
            verdict.tools = "ok"
        elif reply.tool_calls:
            verdict.tools = f"wrong name: {reply.tool_calls[0].name[:20]}"
        else:
            verdict.tools = "no"
    except Exception as exc:  # noqa: BLE001
        verdict.limited = verdict.limited or _transient(exc)
        verdict.tools = "unavailable" if verdict.limited else f"error: {_short(exc)}"
        verdict.continues = verdict.sees = "skipped"
        return verdict

    if verdict.tools == "ok":
        try:
            follow = await provider.chat(
                ChatRequest(
                    model=model,
                    messages=[
                        Message.user(ASK_FOR_TOOL),
                        Message.assistant("", reply.tool_calls),
                        Message.tool_result(reply.tool_calls[0], TOOL_RESULT),
                    ],
                    tools=[THE_TOOL],
                    max_tokens=PLAIN_MAX_TOKENS,
                )
            )
            verdict.continues = "ok" if (follow.text or "").strip() else "empty"
        except Exception as exc:  # noqa: BLE001
            verdict.limited = verdict.limited or _transient(exc)
            verdict.continues = "unavailable" if verdict.limited else f"error: {_short(exc)}"
            verdict.sees = "skipped"
    else:
        verdict.continues = "skipped"

    if vision:
        message = Message(
            role=Role.USER,
            content=ASK_ABOUT_IMAGE,
            parts=[
                ContentPart.text_part(ASK_ABOUT_IMAGE),
                ContentPart.image_part(RED_SQUARE_BASE64, "image/png"),
            ],
        )
        try:
            reply = await provider.chat(
                ChatRequest(model=model, messages=[message], max_tokens=VISION_MAX_TOKENS)
            )
            answer = (reply.text or "").strip()
            if not answer:
                verdict.sees = "empty"
            elif "red" in answer.lower():
                verdict.sees = "ok"
            else:
                # It answered, but not from the image. Saying "I cannot see
                # images" passes as an answer, and would pass this check too.
                verdict.sees = f"wrong: {answer[:20]}"
        except Exception as exc:  # noqa: BLE001
            verdict.sees = f"error: {_short(exc)}"
    else:
        verdict.sees = "skipped"
    return verdict


def _transient(exc: object) -> bool:
    """Broken, or busy?

    A provider with no capacity for the moment, or one holding this account to
    a quota, is not a configuration that needs fixing. A report that calls that
    broken sends the user to edit something which is already right.
    """
    return isinstance(exc, RateLimitError) or bool(getattr(exc, "retryable", False))


def _short(exc: object, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text[:limit]


async def main(options: argparse.Namespace) -> int:
    settings = Settings()
    from app.storage.secrets import SecretStore

    secrets = SecretStore(settings.resolved_secrets_path)
    registry = registry_from_settings(settings, secrets)

    wanted: list[tuple[str, str]] = []
    for config in registry.configs():
        if options.provider and config.name != options.provider:
            continue
        if options.model:
            wanted.append((config.name, options.model))
            continue
        for entry in config.models:
            wanted.append((config.name, str(entry.get("id", ""))))
    if not wanted:
        print("No providers are configured. Check STUDIO_LLM_PROVIDERS in .env.", file=sys.stderr)
        return 1

    print(f"{'provider':<11} {'model':<40} {'answers':<10} {'tools':<10} {'goes on':<10} {'sees':<10}")
    print("-" * 84)
    verdicts: list[Verdict] = []
    for provider_name, model_id in wanted:
        provider, resolved = registry.resolve(provider_name, model_id or "")
        config = registry.config(provider_name)
        catalog = next((m for m in (config.models if config else []) if m.get("id") == resolved), {})
        keyed = registry.is_configured(provider_name)
        verdict = await check_one(
            provider,
            provider_name,
            resolved,
            bool(catalog.get("supports_vision")),
            keyed,
            options.timeout,
        )
        verdicts.append(verdict)
        print(verdict.row(), flush=True)

    print()
    problems = [v for v in verdicts if v.broken]
    no_tools = [v for v in verdicts if v.tools == "no"]
    for verdict in no_tools:
        print(f"note: {verdict.provider}/{verdict.model} answers, but does not call tools.")
        print("      It will not drive Blender. Keep it for questions, not for scenes.")
    for verdict in problems:
        print(
            f"broken: {verdict.provider}/{verdict.model} -- answers={verdict.answers} "
            f"tools={verdict.tools} goes_on={verdict.continues}"
        )
    throttled = [v for v in verdicts if v.limited]
    for verdict in throttled:
        print(
            f"unavailable: {verdict.provider}/{verdict.model} -- the account is fine, "
            f"the provider is busy. Try again later."
        )
    if problems:
        print(f"\n{len(problems)} configured model(s) are broken. Fix those before a session.")
        return 1
    usable = len(verdicts) - len(no_tools) - len(throttled)
    print(f"\nAll {usable} keyed model(s) are usable.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", default="", help="check one provider only")
    parser.add_argument("--model", default="", help="check one model only")
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="seconds to wait for one model before calling it unanswered",
    )
    parser.epilog = "This check never talks to Blender. It answers from the studio's own config, so a model that fails here will fail in a session for the same reason."
    raise SystemExit(asyncio.run(main(parser.parse_args())))
