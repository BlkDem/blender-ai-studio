"""Read-only access to user-supplied local text files.

The model receives a path as text, not the file behind it. This module is the
small, bounded bridge between those two things. It deliberately does not read
Blender scene binaries: those belong to Blender/MCP tools, not the studio's
local text reader.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 1 * 1024 * 1024
MAX_RETURN_CHARS = 200_000

TEXT_EXTENSIONS = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".cfg", ".css", ".csv", ".glsl", ".h", ".hpp",
        ".html", ".ini", ".java", ".js", ".jsx", ".json", ".md", ".mjs", ".py",
        ".ps1", ".rst", ".rs", ".sh", ".sql", ".svg", ".toml", ".ts", ".tsx",
        ".txt", ".vue", ".xml", ".yaml", ".yml",
    }
)
BINARY_EXTENSIONS = frozenset({".blend", ".glb", ".gltf", ".fbx", ".obj", ".stl", ".dae"})
_WINDOWS_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


@dataclass(frozen=True, slots=True)
class TextFile:
    """A bounded file payload suitable for returning to an LLM."""

    path: Path
    filename: str
    text: str
    byte_count: int
    truncated: bool


def path_candidates(path: str | os.PathLike[str]) -> list[Path]:
    """Return native and WSL/Windows equivalents, without touching the disk."""
    text = os.fspath(path).strip().replace("\\", "/")
    if not text:
        return []
    candidates = [Path(text)]
    match = _WINDOWS_PATH.match(text)
    if match:
        drive, rest = match.groups()
        candidates.append(Path(f"/mnt/{drive.lower()}/{rest}"))
    elif text.startswith("/mnt/"):
        parts = text[5:].split("/", 1)
        if len(parts) == 2 and len(parts[0]) == 1 and parts[0].isalpha():
            candidates.append(Path(f"{parts[0].upper()}:/{parts[1]}"))
    return candidates


def resolve_file(path: str | os.PathLike[str]) -> Path | None:
    """Find the first regular file visible to this process."""
    for candidate in path_candidates(path):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def read_text_file(
    path: str | os.PathLike[str], *, max_chars: int = MAX_RETURN_CHARS
) -> TextFile:
    """Read a supported UTF-8 document with strict size and type limits."""
    raw_path = os.fspath(path).strip()
    if not raw_path or "\x00" in raw_path:
        raise ValueError("a non-empty path without NUL bytes is required")
    suffix = Path(raw_path.replace("\\", "/")).suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        raise ValueError(f"{suffix} is a binary Blender/3D file; inspect it through Blender/MCP tools")
    if suffix not in TEXT_EXTENSIONS:
        raise ValueError("only supported UTF-8 text files can be read")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or not 1 <= max_chars <= MAX_RETURN_CHARS:
        raise ValueError(f"max_chars must be an integer between 1 and {MAX_RETURN_CHARS}")
    resolved = resolve_file(raw_path)
    if resolved is None:
        raise FileNotFoundError(f"file not found or not visible to the studio: {raw_path}")
    try:
        size = resolved.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"file is larger than the {MAX_FILE_BYTES} byte limit")
        data = resolved.read_bytes()
    except OSError as exc:
        raise OSError(f"could not read {raw_path}: {exc.strerror or exc}") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("file is not valid UTF-8 text") from exc
    truncated = len(text) > max_chars
    return TextFile(
        path=resolved,
        filename=resolved.name,
        text=text[:max_chars],
        byte_count=len(data),
        truncated=truncated,
    )
