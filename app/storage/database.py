"""SQLite, without an async driver in the way.

The studio's database work is small and short: a chat transcript, a cost line, a
benchmark row. A driver per connection and a thread per query would cost more
than it saves, so this is stdlib ``sqlite3`` with every call pushed to a worker
thread and one lock around the connection. WAL is on because the GUI writes a
usage row while an agent is streaming, and a reader blocking a writer is exactly
the wrong trade for that.

Foreign keys are off by default in SQLite and are a silent source of orphaned
rows, so they are switched on here rather than in every schema.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.core.errors import StorageError

#: Applied to every connection the studio opens.
PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
)

Row = dict[str, Any]


class Database:
    """A single-file SQLite database, used from async code.

    One connection per thread, not one connection shared between them. The
    studio does its database work in worker threads (``asyncio.to_thread``) while
    the GUI thread reads, and sharing a ``sqlite3`` connection across threads is
    a documented no-no that segfaults rather than complaining. Thread-local
    connections plus WAL are the boring, correct answer: readers never block, and
    a writer waits at most ``busy_timeout``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._closed = False

    # --- lifecycle ---------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path, check_same_thread=False, isolation_level=None, timeout=5.0
            )
        except (sqlite3.Error, OSError) as exc:
            # OSError matters: a path under a *file* fails in mkdir, not in
            # sqlite, and the person needs to be told which path was at fault.
            raise StorageError(f"Could not open the database at {self.path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        for pragma in PRAGMAS:
            connection.execute(pragma)
        with self._connections_lock:
            self._connections.append(connection)
        return connection

    def connect(self) -> None:
        """Open this thread's connection. Cheap, and idempotent."""
        self._closed = False
        if getattr(self._local, "connection", None) is None:
            self._local.connection = self._open()

    def close(self) -> None:
        """Close every connection this object opened, from any thread."""
        with self._connections_lock:
            connections, self._connections = self._connections, []
        for connection in connections:
            with contextlib.suppress(sqlite3.Error):  # pragma: no cover - already closed
                connection.close()
        self._local = threading.local()
        self._closed = True

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- sync core ---------------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        if self._closed:
            raise StorageError("This database is closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._open()
            self._local.connection = connection
        return connection

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.connection.execute(sql, tuple(params))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        return [dict(row) for row in self.connection.execute(sql, tuple(params)).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> list[Row] | Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # --- async surface -----------------------------------------------------

    async def run(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a write in a worker thread. Returns ``lastrowid``."""

        def work() -> int:
            cursor = self.connection.execute(sql, tuple(params))
            return int(cursor.lastrowid or 0)

        return await asyncio.to_thread(work)

    async def all(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        return await asyncio.to_thread(self.query, sql, params)

    async def one(self, sql: str, params: Sequence[Any] = ()) -> Row | None:
        rows = await self.all(sql, params)
        return rows[0] if rows else None

    async def script(self, statements: Iterable[str]) -> None:
        """Run several statements, for migrations."""

        def work() -> None:
            connection = self.connection
            for statement in statements:
                connection.execute(statement)

        await asyncio.to_thread(work)

    async def backup_to(self, destination: Path) -> Path:
        """Copy the database, WAL and all, for a user to keep."""

        def work() -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            target = sqlite3.connect(destination)
            with target:
                self.connection.backup(target)
            target.close()
            return destination

        return await asyncio.to_thread(work)


def json_dumps(value: Any) -> str:
    """Stable JSON for columns. Sorted keys make diffs readable in the DB."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def json_loads(raw: str | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default
