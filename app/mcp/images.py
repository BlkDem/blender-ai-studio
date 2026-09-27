"""Getting a picture the bridge wrote on the other side of a filesystem.

``blender.render_preview`` answers with a path, and ``blender://render/latest``
with the bytes -- but the resource reads the file *from the server's own
filesystem*. When the server is in WSL and Blender is on Windows, the render
exists and the server cannot open it, so the honest-looking answer is "no
rendered image is available" for a picture that is sitting right there.

This is the same boundary the asset import has to cross, met from the other
direction, and it is solved the same way: translate the path, then read it. The
alternative -- changing the render tool's contract -- would move a problem into
the backend that every client would have to solve anyway.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from app.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

#: A guard, not a style choice: a render is small, and something that is not
#: probably not a picture.
MAX_IMAGE_BYTES = 32 * 1024 * 1024

#: ``output_path`` is the field the render tools answer with. Matching the field
#: rather than the tool name means a new render tool works without a change here.
_PATH_FIELD = re.compile(r'"(?:output_path|path)"\s*:\s*"([^"]+)"')


def paths_in(text: str) -> list[str]:
    """Every file path a JSON-ish tool result names."""
    return [match.group(1) for match in _PATH_FIELD.finditer(text or "")]


def candidates(path: str) -> list[Path]:
    """Where this path might live, most likely first.

    A Windows path handed to a WSL process means nothing, and
    ``C:/Users/x/a.png`` and ``/mnt/c/Users/x/a.png`` are one file. The path as
    given comes first: when both ends share a filesystem it is already right, and
    a translation that is not needed should not be attempted.
    """
    found = [Path(path)]
    text = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        drive, rest = text[0].lower(), text[2:].replace("\\", "/")
        found.append(Path(f"/mnt/{drive}/{rest.lstrip('/')}"))
    elif text.startswith("/mnt/"):
        parts = text[len("/mnt/") :].split("/", 1)
        if len(parts) == 2 and len(parts[0]) == 1:
            found.append(Path(f"{parts[0].upper()}:/{parts[1]}"))
    return found


def readable(path: str) -> Path | None:
    """The path as this machine sees it, or None."""
    for candidate in candidates(path):
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - a path the OS cannot even ask about
            continue
    return None


def is_image(path: Path) -> bool:
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return True
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:  # pragma: no cover - unreadable file
        return False
    return (
        head.startswith(b"\x89PNG")
        or head.startswith(b"\xff\xd8\xff")
        or head[:6]
        in (
            b"GIF87a",
            b"GIF89a",
        )
    )


def fetch(bridge: MCPManager, text: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[str, str] | None:
    """The first picture a tool result points at, as ``(base64, mime type)``.

    ``None`` when there is nothing to read, which is the ordinary case: most
    tools return no path at all, and a render on a shared filesystem is already
    inline.
    """
    from app.providers3d.importer import EXECUTE_PYTHON  # noqa: F401 - documents the sibling path

    for candidate in paths_in(text):
        found = readable(candidate)
        if found is None or not is_image(found):
            continue
        try:
            if found.stat().st_size > max_bytes:
                logger.info("%s is larger than the image limit; not reading it", found)
                continue
            payload = found.read_bytes()
        except OSError as exc:  # pragma: no cover - raced or vanished
            logger.info("could not read %s: %s", found, exc)
            continue
        mime = "image/png" if found.suffix.lower() == ".png" else f"image/{found.suffix.lstrip('.').lower()}"
        logger.info("read %s from %s", len(payload), found)
        return base64.b64encode(payload).decode(), mime
    return None
