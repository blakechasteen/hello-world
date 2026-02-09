"""
Postgres adapter — 2026-02-09

Manages: memory_audit, configs, plans, entity_aliases tables.
Uses asyncpg for async Postgres access.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

from .models import (
    AuditAction,
    AuditRow,
    ConfigRow,
    EntityAlias,
    PlanRow,
    PlanState,
    PressureTier,
    ResolutionPath,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS memory_audit (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    loop_id TEXT,
    loom_id TEXT,
    action TEXT NOT NULL,
    resolution_path TEXT,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    pressure_tier TEXT NOT NULL DEFAULT 'tier0',
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS configs (
    key TEXT NOT NULL,
    value_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    loom_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (key, loom_id)
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    neo4j_node_id TEXT,
    name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'draft',
    progress REAL NOT NULL DEFAULT 0.0,
    loom_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    loom_id TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (alias, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_loom ON memory_audit (loom_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON memory_audit (action);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON memory_audit (timestamp);
CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases (entity_id);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases (alias);
CREATE INDEX IF NOT EXISTS idx_plans_loom ON plans (loom_id);
"""


class PostgresAdapter:
    """Async Postgres adapter for audit, configs, plans, entity_aliases."""

    def __init__(self, dsn: str = "postgresql://hololoom:hololoom@localhost:5432/hololoom"):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None  # type: ignore[name-defined]

    # ---- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        if asyncpg is None:
            raise ImportError("asyncpg is required: pip install asyncpg")
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(_DDL)
        logger.info("PostgresAdapter connected and schema ensured")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PostgresAdapter:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        if self._pool is None:
            raise RuntimeError("PostgresAdapter not connected. Call connect() first.")
        return self._pool

    # ---- audit log --------------------------------------------------------

    async def write_audit(self, row: AuditRow) -> None:
        await self.pool.execute(
            """
            INSERT INTO memory_audit (id, timestamp, loop_id, loom_id, action,
                                      resolution_path, tokens_used, pressure_tier, details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            row.id,
            row.timestamp,
            row.loop_id,
            row.loom_id,
            row.action.value,
            row.resolution_path.value if row.resolution_path else None,
            row.tokens_used,
            row.pressure_tier.value,
            json.dumps(row.details),
        )

    async def get_audit_rows(
        self,
        *,
        loom_id: Optional[str] = None,
        loop_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        limit: int = 50,
    ) -> list[AuditRow]:
        clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if loom_id:
            clauses.append(f"loom_id = ${idx}")
            params.append(loom_id)
            idx += 1
        if loop_id:
            clauses.append(f"loop_id = ${idx}")
            params.append(loop_id)
            idx += 1
        if action:
            clauses.append(f"action = ${idx}")
            params.append(action.value)
            idx += 1

        where = " AND ".join(clauses) if clauses else "TRUE"
        sql = f"SELECT * FROM memory_audit WHERE {where} ORDER BY timestamp DESC LIMIT ${idx}"
        params.append(limit)

        rows = await self.pool.fetch(sql, *params)
        return [self._row_to_audit(r) for r in rows]

    async def get_resolution_path_distribution(
        self, *, loom_id: Optional[str] = None, limit: int = 1000
    ) -> dict[str, int]:
        if loom_id:
            rows = await self.pool.fetch(
                """
                SELECT resolution_path, count(*) as cnt
                FROM memory_audit
                WHERE loom_id = $1 AND resolution_path IS NOT NULL
                GROUP BY resolution_path
                ORDER BY cnt DESC
                LIMIT $2
                """,
                loom_id,
                limit,
            )
        else:
            rows = await self.pool.fetch(
                """
                SELECT resolution_path, count(*) as cnt
                FROM memory_audit
                WHERE resolution_path IS NOT NULL
                GROUP BY resolution_path
                ORDER BY cnt DESC
                LIMIT $1
                """,
                limit,
            )
        return {r["resolution_path"]: r["cnt"] for r in rows}

    # ---- entity aliases ---------------------------------------------------

    async def upsert_alias(self, alias: EntityAlias) -> None:
        await self.pool.execute(
            """
            INSERT INTO entity_aliases (alias, entity_id, loom_id, confidence)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (alias, entity_id) DO UPDATE
                SET confidence = EXCLUDED.confidence,
                    loom_id = COALESCE(EXCLUDED.loom_id, entity_aliases.loom_id)
            """,
            alias.alias.lower(),
            alias.entity_id,
            alias.loom_id,
            alias.confidence,
        )

    async def find_aliases(
        self,
        name: str,
        *,
        loom_id: Optional[str] = None,
    ) -> list[EntityAlias]:
        name_lower = name.lower()
        if loom_id:
            rows = await self.pool.fetch(
                """
                SELECT alias, entity_id, loom_id, confidence
                FROM entity_aliases
                WHERE alias = $1 AND (loom_id = $2 OR loom_id IS NULL)
                ORDER BY confidence DESC
                """,
                name_lower,
                loom_id,
            )
        else:
            rows = await self.pool.fetch(
                """
                SELECT alias, entity_id, loom_id, confidence
                FROM entity_aliases
                WHERE alias = $1
                ORDER BY confidence DESC
                """,
                name_lower,
            )
        return [
            EntityAlias(
                alias=r["alias"],
                entity_id=r["entity_id"],
                loom_id=r["loom_id"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    # ---- plans ------------------------------------------------------------

    async def upsert_plan(self, plan: PlanRow) -> None:
        await self.pool.execute(
            """
            INSERT INTO plans (id, neo4j_node_id, name, state, progress, loom_id, updated_at, details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    state = EXCLUDED.state,
                    progress = EXCLUDED.progress,
                    updated_at = EXCLUDED.updated_at,
                    details = EXCLUDED.details
            """,
            plan.id,
            plan.neo4j_node_id,
            plan.name,
            plan.state.value,
            plan.progress,
            plan.loom_id,
            plan.updated_at,
            json.dumps(plan.details),
        )

    async def get_plan(self, plan_id: str) -> Optional[PlanRow]:
        row = await self.pool.fetchrow("SELECT * FROM plans WHERE id = $1", plan_id)
        if not row:
            return None
        return PlanRow(
            id=row["id"],
            neo4j_node_id=row["neo4j_node_id"],
            name=row["name"],
            state=PlanState(row["state"]),
            progress=row["progress"],
            loom_id=row["loom_id"],
            updated_at=row["updated_at"],
            details=json.loads(row["details"]) if isinstance(row["details"], str) else row["details"],
        )

    # ---- configs ----------------------------------------------------------

    async def set_config(self, cfg: ConfigRow) -> None:
        await self.pool.execute(
            """
            INSERT INTO configs (key, value_jsonb, loom_id, updated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (key, loom_id) DO UPDATE
                SET value_jsonb = EXCLUDED.value_jsonb,
                    updated_at = EXCLUDED.updated_at
            """,
            cfg.key,
            json.dumps(cfg.value),
            cfg.loom_id or "",
            cfg.updated_at,
        )

    async def get_config(self, key: str, loom_id: Optional[str] = None) -> Optional[ConfigRow]:
        row = await self.pool.fetchrow(
            "SELECT * FROM configs WHERE key = $1 AND loom_id = $2",
            key,
            loom_id or "",
        )
        if not row:
            return None
        return ConfigRow(
            key=row["key"],
            value=json.loads(row["value_jsonb"]) if isinstance(row["value_jsonb"], str) else row["value_jsonb"],
            loom_id=row["loom_id"] or None,
            updated_at=row["updated_at"],
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _row_to_audit(r: Any) -> AuditRow:
        details = r["details"]
        if isinstance(details, str):
            details = json.loads(details)
        return AuditRow(
            id=r["id"],
            timestamp=r["timestamp"],
            loop_id=r["loop_id"],
            loom_id=r["loom_id"],
            action=AuditAction(r["action"]),
            resolution_path=ResolutionPath(r["resolution_path"]) if r["resolution_path"] else None,
            tokens_used=r["tokens_used"],
            pressure_tier=PressureTier(r["pressure_tier"]),
            details=details,
        )
