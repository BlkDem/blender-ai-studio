"""Benchmark results, in the database.

A run is expensive to reproduce: minutes of a model's time, a real Blender, and a
scene nobody can rebuild exactly. So the row keeps what happened (metrics) *and*
what was said (transcript, final scene), and a person's scores live beside it
rather than in a comment.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.benchmark.runner import Comparison, RunOutcome
from app.storage.database import Database, json_dumps, json_loads


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class BenchmarkStorage:
    """Suites, runs and manual reviews."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # --- suites ------------------------------------------------------------

    async def create_suite(self, name: str, description: str = "") -> str:
        suite_id = _id("suite")
        await self._db.run(
            "INSERT INTO benchmark_suites (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (suite_id, name, description, time.time()),
        )
        return suite_id

    async def suites(self) -> list[dict[str, Any]]:
        rows = await self._db.all(
            """SELECT s.*, COUNT(r.id) AS runs FROM benchmark_suites s
               LEFT JOIN benchmark_runs r ON r.suite_id = s.id
               GROUP BY s.id ORDER BY s.created_at DESC"""
        )
        return rows

    async def runs(self, suite_id: str) -> list[dict[str, Any]]:
        return await self._db.all(
            "SELECT * FROM benchmark_runs WHERE suite_id = ? ORDER BY task_index, started_at", (suite_id,)
        )

    async def run(self, run_id: str) -> dict[str, Any] | None:
        return await self._db.one("SELECT * FROM benchmark_runs WHERE id = ?", (run_id,))

    # --- writing runs ------------------------------------------------------

    async def save_run(
        self,
        suite_id: str,
        outcome: RunOutcome,
        *,
        task_index: int = 0,
        started_at: float = 0.0,
    ) -> str:
        """Persist one outcome, transcript and final scene included."""
        run_id = _id("brun")
        totals = outcome.totals
        await self._db.run(
            """INSERT INTO benchmark_runs
               (id, suite_id, task_index, task_prompt, run_id, provider, model, blend_file,
                scene_reset, status, started_at, finished_at, duration_s, input_tokens,
                output_tokens, llm_cost_usd, mcp_calls, tool_errors, three_d_calls,
                three_d_credits, three_d_cost_usd, total_cost_usd, final_scene, transcript, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                suite_id,
                task_index,
                outcome.task.prompt,
                outcome.run_id,
                outcome.model.provider,
                outcome.model.label(),
                outcome.blend_file,
                outcome.scene_reset,
                outcome.status,
                started_at or time.time(),
                time.time(),
                outcome.duration_s,
                totals.input_tokens,
                totals.output_tokens,
                totals.llm_usd,
                outcome.mcp_calls,
                outcome.tool_errors,
                0,
                totals.three_d_credits,
                totals.three_d_usd,
                totals.llm_usd + totals.three_d_usd,
                json_dumps(outcome.final_scene),
                json_dumps(outcome.transcript),
                outcome.error,
            ),
        )
        return run_id

    async def save_comparison(self, suite_id: str, comparison: Comparison) -> list[str]:
        return [
            await self.save_run(suite_id, outcome, task_index=index)
            for index, outcome in enumerate(comparison.outcomes)
        ]

    # --- a person's judgement ---------------------------------------------

    async def review(
        self,
        run_id: str,
        *,
        geometry: int | None = None,
        materials: int | None = None,
        instruction_following: int | None = None,
        composition: int | None = None,
        overall: int | None = None,
        notes: str = "",
    ) -> str:
        """Record a human score.

        There is no automatic "best model" anywhere in this project. A heuristic
        that crowned a winner would be a claim about taste dressed as a metric,
        and the person who has to live with the result is the one who scores it.
        """
        review_id = _id("brev")
        await self._db.run(
            """INSERT INTO benchmark_reviews
               (id, run_id, geometry, materials, instruction_following, composition, overall, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review_id,
                run_id,
                geometry,
                materials,
                instruction_following,
                composition,
                overall,
                notes,
                time.time(),
            ),
        )
        return review_id

    async def reviews(self, run_id: str) -> list[dict[str, Any]]:
        return await self._db.all(
            "SELECT * FROM benchmark_reviews WHERE run_id = ? ORDER BY created_at", (run_id,)
        )

    async def table(self, suite_id: str) -> list[dict[str, Any]]:
        """The comparison view: one row per run, with any human scores joined in."""
        rows = await self._db.all(
            """SELECT r.id, r.model, r.provider, r.status, r.duration_s, r.input_tokens,
                      r.output_tokens, r.llm_cost_usd, r.mcp_calls, r.tool_errors,
                      r.scene_reset, r.task_prompt,
                      v.geometry, v.materials, v.instruction_following, v.composition,
                      v.overall, v.notes
               FROM benchmark_runs r
               LEFT JOIN benchmark_reviews v ON v.run_id = r.id
               WHERE r.suite_id = ?
               ORDER BY r.task_index, r.model""",
            (suite_id,),
        )
        for row in rows:
            row["tokens"] = int(row["input_tokens"]) + int(row["output_tokens"])
            row["final_scene"] = json_loads(row.get("final_scene"), {})
        return rows

    async def transcripts(self, suite_id: str) -> dict[str, list[dict[str, Any]]]:
        """Every stored transcript, so a session can be read back after the fact."""
        rows = await self._db.all(
            "SELECT id, model, transcript FROM benchmark_runs WHERE suite_id = ?", (suite_id,)
        )
        return {row["id"]: json_loads(row["transcript"], []) for row in rows}

    async def delete_suite(self, suite_id: str) -> None:
        await self._db.run("DELETE FROM benchmark_suites WHERE id = ?", (suite_id,))
