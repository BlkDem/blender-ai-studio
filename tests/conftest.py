"""Shared fixtures.

The database fixture is the important one: storage and benchmark tests get a real
SQLite file in a temporary directory rather than a mock, because the thing being
tested *is* the SQL.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "studio-data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def database_path(data_dir: Path) -> Path:
    return data_dir / "studio.db"


@pytest.fixture
def secrets_file(data_dir: Path) -> Path:
    return data_dir / "secrets.json"


@pytest.fixture
async def studio(database_path: Path) -> AsyncIterator["object"]:
    """A migrated, empty studio database, closed at the end of the test."""
    from app.storage.repositories import Studio

    instance = await Studio.open(database_path)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def raw_database(database_path: Path) -> Iterator["object"]:
    """The bare connection, for tests about migrations and pragmas."""
    from app.storage.database import Database

    database = Database(database_path)
    database.connect()
    try:
        yield database
    finally:
        database.close()
