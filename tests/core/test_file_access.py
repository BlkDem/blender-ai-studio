from pathlib import Path

import pytest

from app.core.file_access import (
    BINARY_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_RETURN_CHARS,
    path_candidates,
    read_text_file,
    resolve_file,
)


def test_windows_and_wsl_paths_are_candidates_without_touching_disk() -> None:
    assert path_candidates(r"C:\Users\max\brief.md") == [
        Path("C:/Users/max/brief.md"), Path("/mnt/c/Users/max/brief.md")
    ]
    assert path_candidates("/mnt/d/work/brief.md") == [
        Path("/mnt/d/work/brief.md"), Path("D:/work/brief.md")
    ]
    assert path_candidates("/home/max/brief.md") == [Path("/home/max/brief.md")]


def test_reads_utf8_and_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "brief.md"
    path.write_bytes("\ufeffПривет\r\nостров".encode("utf-8"))
    result = read_text_file(path)
    assert result.text == "Привет\r\nостров"
    assert result.filename == "brief.md"
    assert result.byte_count == path.stat().st_size
    assert result.truncated is False


def test_truncates_returned_text_without_reading_unbounded_content(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    result = read_text_file(path, max_chars=4)
    assert result.text == "abcd"
    assert result.truncated is True


@pytest.mark.parametrize("suffix", sorted(BINARY_EXTENSIONS))
def test_binary_blender_and_3d_files_are_refused(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"scene{suffix}"
    path.write_bytes(b"binary")
    with pytest.raises(ValueError, match="binary Blender/3D"):
        read_text_file(path)


def test_invalid_utf8_and_unknown_extensions_are_refused(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        read_text_file(bad)
    unknown = tmp_path / "notes.bin"
    unknown.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="supported UTF-8"):
        read_text_file(unknown)


def test_missing_directory_and_oversize_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text_file(tmp_path / "missing.md")
    directory = tmp_path / "folder.md"
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        read_text_file(directory)
    large = tmp_path / "large.md"
    large.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="larger"):
        read_text_file(large)


def test_resolve_file_returns_only_regular_files(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("ok", encoding="utf-8")
    assert resolve_file(path) == path
    assert resolve_file(tmp_path) is None


def test_max_chars_is_bounded() -> None:
    with pytest.raises(ValueError, match=str(MAX_RETURN_CHARS)):
        read_text_file("x.md", max_chars=MAX_RETURN_CHARS + 1)
