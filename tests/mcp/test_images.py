"""A render the studio has to go and read.

``blender.render_preview`` answers with a path, and the ``blender://render/latest``
resource with the bytes -- read from the *server's* filesystem. Put the MCP
server in WSL and Blender on Windows and the render exists, is a real PNG, and
the server cannot open it, so the answer is "no rendered image is available".

A vision model that never receives the picture is the whole feature quietly not
working, so this is worth its own tests: path translation both ways, refusing
things that are not pictures, and stopping at a size limit.
"""

from __future__ import annotations

import base64
from pathlib import Path

from app.mcp.images import candidates, fetch, is_image, paths_in, readable

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 8


def test_a_path_is_found_in_a_json_result() -> None:
    text = '{"success": true, "output_path": "/home/x/shot.png", "render_time": 2.1}'
    assert paths_in(text) == ["/home/x/shot.png"]


def test_a_windows_path_is_translated_for_wsl() -> None:
    """The case that broke: the render is on the Windows side of the boundary.

    Checked as a translation rather than through the filesystem, so the test says
    the same thing on WSL, on macOS and in CI.
    """
    assert candidates("C:/Users/x/AppData/Local/Temp/shot.png") == [
        Path("C:/Users/x/AppData/Local/Temp/shot.png"),
        Path("/mnt/c/Users/x/AppData/Local/Temp/shot.png"),
    ]
    assert candidates("C:\\Users\\x\\shot.png")[1] == Path("/mnt/c/Users/x/shot.png")
    assert candidates("/mnt/c/Users/x/shot.png")[-1] == Path("C:/Users/x/shot.png")
    assert candidates("/home/x/shot.png") == [Path("/home/x/shot.png")], "a Linux path is left alone"
    assert readable("C:/nope/missing.png") is None


def test_a_posix_path_is_used_as_it_is(tmp_path: Path) -> None:
    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG)
    assert readable(str(shot)) == shot


def test_a_directory_is_not_a_picture(tmp_path: Path) -> None:
    (tmp_path / "render.png").mkdir()
    assert readable(str(tmp_path / "render.png")) is None


def test_only_images_are_recognised(tmp_path: Path) -> None:
    assert is_image(_write(tmp_path / "a.png")) is True
    assert is_image(_write(tmp_path / "a.blend", b"BLENDER-v300")) is False
    extensionless = _write(tmp_path / "noext", PNG)
    assert is_image(extensionless) is True, "a PNG is a PNG whatever it is called"
    jpeg = _write(tmp_path / "a.jpg", b"\xff\xd8\xff\xe0rest")
    assert is_image(jpeg) is True


def test_a_result_with_no_path_yields_nothing() -> None:
    assert fetch(None, '{"objects": []}') is None  # type: ignore[arg-type]
    assert fetch(None, "just prose") is None  # type: ignore[arg-type]


def test_the_picture_is_returned_as_base64_with_its_type(tmp_path: Path) -> None:
    shot = _write(tmp_path / "shot.png")
    text = f'{{"output_path": "{shot}", "render_time": 1.2}}'
    found = fetch(None, text)  # type: ignore[arg-type]
    assert found is not None
    data, mime = found
    assert base64.b64decode(data) == shot.read_bytes()
    assert mime == "image/png"


def test_a_path_that_does_not_exist_is_not_an_error(tmp_path: Path) -> None:
    """Blender cleans up its own temporary renders; a missing file is ordinary."""
    assert fetch(None, f'{{"output_path": "{tmp_path / "gone.png"}"}}') is None  # type: ignore[arg-type]


def test_a_huge_file_is_left_alone(tmp_path: Path) -> None:
    big = _write(tmp_path / "big.png", b"\x89PNG\r\n\x1a\n" + b"0" * 2048)
    assert fetch(None, f'{{"output_path": "{big}"}}', max_bytes=1024) is None  # type: ignore[arg-type]


def test_a_non_image_at_the_path_is_ignored(tmp_path: Path) -> None:
    blend = _write(tmp_path / "scene.blend", b"BLENDER-v300")
    assert fetch(None, f'{{"output_path": "{blend}"}}') is None  # type: ignore[arg-type]


def _write(path: Path, payload: bytes = PNG) -> Path:
    path.write_bytes(payload)
    return path
