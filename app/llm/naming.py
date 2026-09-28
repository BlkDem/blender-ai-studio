"""Tool names on the wire, and the names the model gets back.

An MCP tool is named ``blender.create_object``: the server name, a dot, the tool
name. That is the right name here -- it is how the studio finds the tool again,
and how a person reads the transcript.

Several providers will not accept it. OpenAI's function names must match
``^[a-zA-Z0-9_-]+$``, and Anthropic's is the same shape; both answer a dotted
name with a 400 and no explanation of what they wanted. A studio that only works
with a local server is a studio that only works on one machine.

So the name is translated on the way out and translated back on the way in, per
request, by replacing what a provider will not take. The model still sees a
descriptive name -- ``blender_create_object`` rather than ``tool_3`` -- because a
name that says nothing is a name the model cannot choose well.
"""

from __future__ import annotations

import re

#: What OpenAI, Anthropic and Gemini all agree on.
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
#: The maximum a provider documents, and the one that matters: a longer name is
#: rejected outright, and the usual cause is a long MCP name.
MAX_LENGTH = 64


class ToolNameMap:
    """One request's translation between the studio's names and the wire's."""

    def __init__(self, names: list[str] | None = None) -> None:
        self._to_wire: dict[str, str] = {}
        self._to_model: dict[str, str] = {}
        self._used: set[str] = set()
        for name in names or []:
            self.to_wire(name)

    def to_wire(self, name: str) -> str:
        """The name to send, unchanged when the provider will take it."""
        existing = self._to_wire.get(name)
        if existing is not None:
            return existing
        candidate = _sanitise(name)
        # Two MCP tools can differ only by a character the provider forbids, so
        # a collision is possible and silently sending the wrong tool is worse
        # than an ugly name.
        if candidate in self._used:
            stem = candidate[: MAX_LENGTH - 4]
            for index in range(2, 100):
                attempt = f"{stem}_{index}"
                if attempt not in self._used:
                    candidate = attempt
                    break
        self._to_wire[name] = candidate
        self._to_model[candidate] = name
        self._used.add(candidate)
        return candidate

    def to_model(self, name: str) -> str:
        """What the studio should dispatch, given what came back."""
        return self._to_model.get(name, name)


def _sanitise(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in name)
    cleaned = cleaned.strip("_") or "tool"
    if len(cleaned) <= MAX_LENGTH and SAFE_NAME.match(cleaned):
        return cleaned
    if len(cleaned) > MAX_LENGTH:
        # Truncated names collide, so the tail is kept: the interesting part of
        # "blender.something.a_very_long_tool_name" is the end of it.
        cleaned = f"{cleaned[: MAX_LENGTH - 9]}_{abs(hash(name)) % 100000:05d}"
    return cleaned


def needs_translation(name: str) -> bool:
    return not SAFE_NAME.match(name)
