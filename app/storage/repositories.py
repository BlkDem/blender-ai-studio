"""Repositories: the only SQL in the application lives here.

Everything above this layer works with plain dataclasses, so the storage shape can
change without touching the agent or the GUI, and a test can hold a database in a
temporary directory and assert on rows rather than on SQL.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import StorageError
from app.storage.database import Database, Row, json_dumps, json_loads


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --- dataclasses ------------------------------------------------------------


@dataclass(slots=True)
class Project:
    id: str
    name: str
    created_at: float
    updated_at: float
    blender_mcp: str | None = None
    initial_blend: str | None = None
    default_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Conversation:
    id: str
    project_id: str | None
    title: str
    created_at: float
    updated_at: float


@dataclass(slots=True)
class Message:
    id: str
    conversation_id: str
    run_id: str
    role: str
    content: str
    reasoning: str | None
    created_at: float


@dataclass(slots=True)
class ToolCallRecord:
    id: str
    run_id: str
    tool: str
    arguments: dict[str, Any]
    result: str | None
    is_error: bool
    error_code: str | None
    started_at: float
    finished_at: float | None
    duration_ms: float | None = None


@dataclass(slots=True)
class LLMRequestRecord:
    id: str
    run_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    is_error: bool
    error: str | None
    created_at: float
    cached_tokens: int = 0
    messages: int = 0


@dataclass(slots=True)
class ThreeDTaskRecord:
    id: str
    run_id: str
    provider: str
    kind: str
    prompt: str
    model: str
    status: str
    started_at: float
    provider_task_id: str | None = None
    progress: float = 0.0
    credits: int = 0
    cost_usd: float = 0.0
    result_url: str | None = None
    local_path: str | None = None
    error: str | None = None
    finished_at: float | None = None


# --- repositories -----------------------------------------------------------


class ProjectRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(
        self,
        name: str,
        *,
        blender_mcp: str | None = None,
        initial_blend: str | None = None,
        default_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        now = time.time()
        project = Project(
            id=_id("prj"),
            name=name,
            created_at=now,
            updated_at=now,
            blender_mcp=blender_mcp,
            initial_blend=initial_blend,
            default_model=default_model,
            metadata=metadata or {},
        )
        await self._db.run(
            """INSERT INTO projects
               (id, name, created_at, updated_at, blender_mcp, initial_blend, default_model, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project.id,
                project.name,
                project.created_at,
                project.updated_at,
                project.blender_mcp,
                project.initial_blend,
                project.default_model,
                json_dumps(project.metadata),
            ),
        )
        return project

    async def get(self, project_id: str) -> Project | None:
        row = await self._db.one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return _to_project(row) if row else None

    async def list(self) -> list[Project]:
        rows = await self._db.all("SELECT * FROM projects ORDER BY updated_at DESC")
        return [_to_project(row) for row in rows]

    async def update(self, project_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"name", "blender_mcp", "initial_blend", "default_model"}
        assignments, values = [], []
        for key, value in fields.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        if "metadata" in fields:
            assignments.append("metadata = ?")
            values.append(json_dumps(fields["metadata"]))
        assignments.append("updated_at = ?")
        values.extend([time.time(), project_id])
        await self._db.run(f"UPDATE projects SET {', '.join(assignments)} WHERE id = ?", values)

    async def delete(self, project_id: str) -> None:
        await self._db.run("DELETE FROM projects WHERE id = ?", (project_id,))


def _to_project(row: Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        blender_mcp=row.get("blender_mcp"),
        initial_blend=row.get("initial_blend"),
        default_model=row.get("default_model"),
        metadata=json_loads(row.get("metadata"), {}) or {},
    )


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, project_id: str | None, title: str = "") -> Conversation:
        now = time.time()
        conversation = Conversation(
            id=_id("cnv"), project_id=project_id, title=title, created_at=now, updated_at=now
        )
        await self._db.run(
            """INSERT INTO conversations (id, project_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation.id, project_id, title, now, now),
        )
        return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        row = await self._db.one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        if not row:
            return None
        return Conversation(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def list(self, project_id: str | None = None) -> list[Conversation]:
        if project_id is None:
            rows = await self._db.all("SELECT * FROM conversations ORDER BY updated_at DESC")
        else:
            rows = await self._db.all(
                "SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            )
        return [
            Conversation(
                id=row["id"],
                project_id=row["project_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def rename(self, conversation_id: str, title: str) -> None:
        await self._db.run(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, time.time(), conversation_id),
        )


class MessageRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        run_id: str = "",
        reasoning: str | None = None,
    ) -> Message:
        message = Message(
            id=_id("msg"),
            conversation_id=conversation_id,
            run_id=run_id,
            role=role,
            content=content,
            reasoning=reasoning,
            created_at=time.time(),
        )
        await self._db.run(
            """INSERT INTO messages (id, conversation_id, run_id, role, content, reasoning, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.conversation_id,
                message.run_id,
                message.role,
                message.content,
                message.reasoning,
                message.created_at,
            ),
        )
        await self._db.run(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (time.time(), conversation_id),
        )
        return message

    async def list(self, conversation_id: str) -> Sequence[Message]:
        rows = await self._db.all(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
            (conversation_id,),
        )
        return [
            Message(
                id=row["id"],
                conversation_id=row["conversation_id"],
                run_id=row["run_id"],
                role=row["role"],
                content=row["content"],
                reasoning=row["reasoning"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def recent_user_texts(self, limit: int = 200) -> list[tuple[str, float]]:
        """What the person has asked, newest first, across every conversation.

        The history of prompts is not a separate thing that has to be kept in
        step with the conversation: a prompt *is* a stored user message, so
        reading them back is a question, not a second record that can drift from
        the first.

        Consecutive repeats are collapsed, because a person who asks the same
        thing twice in a row did it because the first answer was not what they
        wanted, and a list that shows the same line twice hides that. Older
        repeats survive, because "again, but smaller" two prompts apart is
        genuinely two things.
        """
        rows = await self._db.all(
            """SELECT content, created_at FROM messages
               WHERE role = 'user' AND TRIM(content) != ''
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?""",
            (max(1, limit * 2),),
        )
        seen: set[str] = set()
        history: list[tuple[str, float]] = []
        for row in rows:
            text = str(row["content"] or "").strip()
            if text in seen:
                continue
            seen.add(text)
            history.append((text, float(row["created_at"] or 0.0)))
            if len(history) >= limit:
                break
        return history

    async def tool_calls_for_run(self, run_id: str) -> Sequence[ToolCallRecord]:
        rows = await self._db.all(
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY started_at, rowid", (run_id,)
        )
        return [_to_tool_call(row) for row in rows]


class ToolCallRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def start(self, run_id: str, tool: str, arguments: dict[str, Any]) -> ToolCallRecord:
        record = ToolCallRecord(
            id=_id("tc"),
            run_id=run_id,
            tool=tool,
            arguments=arguments,
            result=None,
            is_error=False,
            error_code=None,
            started_at=time.time(),
            finished_at=None,
        )
        await self._db.run(
            """INSERT INTO tool_calls
               (id, run_id, tool, arguments, result, is_error, error_code, started_at, finished_at)
               VALUES (?, ?, ?, ?, NULL, 0, NULL, ?, NULL)""",
            (record.id, run_id, tool, json_dumps(arguments), record.started_at),
        )
        return record

    async def finish(
        self,
        call_id: str,
        *,
        result: str | None = None,
        error_code: str | None = None,
        is_error: bool = False,
    ) -> None:
        await self._db.run(
            "UPDATE tool_calls SET result = ?, error_code = ?, is_error = ?, finished_at = ? WHERE id = ?",
            (result, error_code, int(is_error), time.time(), call_id),
        )


class UsageRepository:
    """LLM usage, kept apart from 3D usage so a cost report cannot confuse them."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def record(self, record: LLMRequestRecord) -> None:
        await self._db.run(
            """INSERT INTO llm_requests
               (id, run_id, provider, model, messages, input_tokens, output_tokens, cached_tokens,
                cost_usd, latency_ms, is_error, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.run_id,
                record.provider,
                record.model,
                record.messages,
                record.input_tokens,
                record.output_tokens,
                record.cached_tokens,
                record.cost_usd,
                record.latency_ms,
                int(record.is_error),
                record.error,
                record.created_at,
            ),
        )

    async def for_run(self, run_id: str) -> list[LLMRequestRecord]:
        rows = await self._db.all(
            "SELECT * FROM llm_requests WHERE run_id = ? ORDER BY created_at", (run_id,)
        )
        return [
            LLMRequestRecord(
                id=row["id"],
                run_id=row["run_id"],
                provider=row["provider"],
                model=row["model"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cached_tokens=row["cached_tokens"],
                cost_usd=row["cost_usd"],
                latency_ms=row["latency_ms"],
                is_error=bool(row["is_error"]),
                error=row["error"],
                created_at=row["created_at"],
                messages=row["messages"],
            )
            for row in rows
        ]

    async def session_total(self, run_id: str) -> tuple[int, int, float]:
        row = await self._db.one(
            """SELECT COALESCE(SUM(input_tokens), 0) AS input,
                      COALESCE(SUM(output_tokens), 0) AS output,
                      COALESCE(SUM(cost_usd), 0) AS cost
               FROM llm_requests WHERE run_id = ?""",
            (run_id,),
        )
        if not row:
            return 0, 0, 0.0
        return int(row["input"]), int(row["output"]), float(row["cost"])


class ThreeDTaskRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, record: ThreeDTaskRecord) -> None:
        await self._db.run(
            """INSERT INTO three_d_tasks
               (id, run_id, provider, provider_task_id, kind, prompt, model, status, progress,
                credits, cost_usd, result_url, local_path, error, started_at, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.run_id,
                record.provider,
                record.provider_task_id,
                record.kind,
                record.prompt,
                record.model,
                record.status,
                record.progress,
                record.credits,
                record.cost_usd,
                record.result_url,
                record.local_path,
                record.error,
                record.started_at,
                record.finished_at,
            ),
        )

    async def update(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "provider_task_id",
            "status",
            "progress",
            "credits",
            "cost_usd",
            "result_url",
            "local_path",
            "error",
            # A task that has finished and cannot record when it finished is a
            # task whose duration -- the number a person compares models by --
            # is unrecoverable after the fact.
            "finished_at",
        }
        assignments, values = [], []
        for key, value in fields.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(task_id)
        await self._db.run(f"UPDATE three_d_tasks SET {', '.join(assignments)} WHERE id = ?", values)

    async def list(self, run_id: str | None = None, limit: int = 100) -> list[ThreeDTaskRecord]:
        if run_id:
            rows = await self._db.all(
                "SELECT * FROM three_d_tasks WHERE run_id = ? ORDER BY started_at DESC LIMIT ?",
                (run_id, limit),
            )
        else:
            rows = await self._db.all(
                "SELECT * FROM three_d_tasks ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        return [
            ThreeDTaskRecord(
                id=row["id"],
                run_id=row["run_id"],
                provider=row["provider"],
                provider_task_id=row["provider_task_id"],
                kind=row["kind"],
                prompt=row["prompt"],
                model=row["model"],
                status=row["status"],
                progress=row["progress"],
                credits=row["credits"],
                cost_usd=row["cost_usd"],
                result_url=row["result_url"],
                local_path=row["local_path"],
                error=row["error"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]


class SettingsRepository:
    """The GUI's own settings, as overrides on top of the environment."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def all(self) -> dict[str, Any]:
        rows = await self._db.all("SELECT key, value FROM settings")
        return {row["key"]: json_loads(row["value"], row["value"]) for row in rows}

    async def get(self, key: str, default: Any = None) -> Any:
        """One value, or the default. Reading the whole table to pick one key out
        of it is a query shaped by the caller's convenience."""
        row = await self._db.one("SELECT value FROM settings WHERE key = ?", (key,))
        if row is None:
            return default
        return json_loads(row["value"], row["value"])

    async def set(self, key: str, value: Any) -> None:
        await self._db.run(
            """INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (key, json_dumps(value), time.time()),
        )

    async def delete(self, key: str) -> None:
        await self._db.run("DELETE FROM settings WHERE key = ?", (key,))

    async def set_many(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            await self.set(key, value)


def _to_tool_call(row: Row) -> ToolCallRecord:
    finished = row.get("finished_at")
    return ToolCallRecord(
        id=row["id"],
        run_id=row["run_id"],
        tool=row["tool"],
        arguments=json_loads(row["arguments"], {}),
        result=row.get("result"),
        is_error=bool(row.get("is_error")),
        error_code=row.get("error_code"),
        started_at=row["started_at"],
        finished_at=finished,
        duration_ms=((finished - row["started_at"]) * 1000.0) if finished else None,
    )


class Studio:
    """The repositories, assembled. One object for the app to hold."""

    def __init__(self, database: Database) -> None:
        self.db = database
        self.projects = ProjectRepository(database)
        self.conversations = ConversationRepository(database)
        self.messages = MessageRepository(database)
        self.tool_calls = ToolCallRepository(database)
        self.usage = UsageRepository(database)
        self.three_d = ThreeDTaskRepository(database)
        self.settings = SettingsRepository(database)

    @classmethod
    async def open(cls, path: Path | str) -> Studio:
        from app.storage.migrations import migrate

        database = Database(path)
        database.connect()
        version = await migrate(database)
        if version == 0:
            raise StorageError("Migrations did not apply; the database is unusable")
        return cls(database)

    def close(self) -> None:
        self.db.close()
