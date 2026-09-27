"""Schema, as an ordered list of migrations.

Migrations are append-only and identified by number. There is no ``down``: a
desktop app that can throw away its database is a desktop app that can throw
away a benchmark, and a benchmark is the one thing here that is expensive to
reproduce. ``user_version`` records what has been applied.
"""

from __future__ import annotations

from app.storage.database import Database

#: Each entry is (version, description, statements).
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "projects, conversations, messages and settings",
        [
            """
            CREATE TABLE projects (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                blender_mcp   TEXT,
                initial_blend TEXT,
                default_model TEXT,
                metadata      TEXT NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE conversations (
                id          TEXT PRIMARY KEY,
                project_id  TEXT REFERENCES projects(id) ON DELETE CASCADE,
                title       TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """,
            """
            CREATE TABLE messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                run_id          TEXT,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL DEFAULT '',
                reasoning       TEXT,
                created_at      REAL NOT NULL
            )
            """,
            "CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at)",
            """
            CREATE TABLE tool_calls (
                id         TEXT PRIMARY KEY,
                run_id     TEXT NOT NULL,
                message_id TEXT,
                tool       TEXT NOT NULL,
                arguments  TEXT NOT NULL DEFAULT '{}',
                result     TEXT,
                is_error   INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                started_at REAL NOT NULL,
                finished_at REAL
            )
            """,
            "CREATE INDEX idx_tool_calls_run ON tool_calls(run_id, started_at)",
            """
            CREATE TABLE settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
        ],
    ),
    (
        2,
        "usage, three-d tasks and asset outputs",
        [
            """
            CREATE TABLE llm_requests (
                id            TEXT PRIMARY KEY,
                run_id        TEXT NOT NULL,
                provider      TEXT NOT NULL,
                model         TEXT NOT NULL,
                messages      INTEGER NOT NULL DEFAULT 0,
                input_tokens  INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cached_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd      REAL NOT NULL DEFAULT 0,
                latency_ms    REAL NOT NULL DEFAULT 0,
                is_error      INTEGER NOT NULL DEFAULT 0,
                error         TEXT,
                created_at    REAL NOT NULL
            )
            """,
            "CREATE INDEX idx_llm_requests_run ON llm_requests(run_id)",
            """
            CREATE TABLE three_d_tasks (
                id           TEXT PRIMARY KEY,
                run_id       TEXT NOT NULL,
                provider     TEXT NOT NULL,
                provider_task_id TEXT,
                kind         TEXT NOT NULL,
                prompt       TEXT NOT NULL DEFAULT '',
                model        TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL,
                progress     REAL NOT NULL DEFAULT 0,
                credits      INTEGER NOT NULL DEFAULT 0,
                cost_usd     REAL NOT NULL DEFAULT 0,
                result_url   TEXT,
                local_path   TEXT,
                error        TEXT,
                started_at   REAL NOT NULL,
                finished_at  REAL
            )
            """,
            "CREATE INDEX idx_three_d_tasks_run ON three_d_tasks(run_id, started_at)",
        ],
    ),
    (
        3,
        "benchmark suites, runs and manual review",
        [
            """
            CREATE TABLE benchmark_suites (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            )
            """,
            """
            CREATE TABLE benchmark_runs (
                id              TEXT PRIMARY KEY,
                suite_id        TEXT NOT NULL REFERENCES benchmark_suites(id) ON DELETE CASCADE,
                task_index      INTEGER NOT NULL DEFAULT 0,
                task_prompt     TEXT NOT NULL,
                run_id          TEXT,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                blend_file      TEXT,
                scene_reset     TEXT NOT NULL DEFAULT 'unverified',
                status          TEXT NOT NULL,
                started_at      REAL NOT NULL,
                finished_at     REAL,
                duration_s      REAL NOT NULL DEFAULT 0,
                input_tokens    INTEGER NOT NULL DEFAULT 0,
                output_tokens   INTEGER NOT NULL DEFAULT 0,
                llm_cost_usd    REAL NOT NULL DEFAULT 0,
                mcp_calls       INTEGER NOT NULL DEFAULT 0,
                tool_errors     INTEGER NOT NULL DEFAULT 0,
                three_d_calls   INTEGER NOT NULL DEFAULT 0,
                three_d_credits INTEGER NOT NULL DEFAULT 0,
                three_d_cost_usd REAL NOT NULL DEFAULT 0,
                total_cost_usd  REAL NOT NULL DEFAULT 0,
                final_scene     TEXT,
                transcript      TEXT,
                error           TEXT
            )
            """,
            "CREATE INDEX idx_benchmark_runs_suite ON benchmark_runs(suite_id, task_index, model)",
            """
            CREATE TABLE benchmark_reviews (
                id                     TEXT PRIMARY KEY,
                run_id                 TEXT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                geometry               INTEGER,
                materials              INTEGER,
                instruction_following  INTEGER,
                composition            INTEGER,
                overall                INTEGER,
                notes                  TEXT NOT NULL DEFAULT '',
                created_at             REAL NOT NULL
            )
            """,
        ],
    ),
]

LATEST_VERSION = max(version for version, _, _ in MIGRATIONS)


async def migrate(database: Database) -> int:
    """Apply anything not yet applied. Returns the resulting version."""
    row = await database.one("PRAGMA user_version")
    current = int(row["user_version"]) if row and "user_version" in row else 0
    if current == 0:
        current = _read_user_version(database)
    for version, description, statements in MIGRATIONS:
        if version <= current:
            continue
        await database.script(statements)
        await database.run(f"PRAGMA user_version = {version}")
        current = version
    return current


def _read_user_version(database: Database) -> int:
    row = database.query_one("PRAGMA user_version")
    if not row:
        return 0
    return int(next(iter(row.values()), 0))
