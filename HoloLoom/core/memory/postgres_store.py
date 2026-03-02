"""
Direct Postgres Access for Memory Bus — spec §10.3.

Five tables: facts, plans, configs, memory_audit, entity_aliases.
Uses psycopg 3 with native async support.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    logger.warning("psycopg not installed — PostgresStore unavailable. pip install 'psycopg[binary]'")


# =============================================================================
# DDL — matches scripts/init_memory_bus.sql
# =============================================================================

DDL_STATEMENTS = [
    # Facts — flat fast lookup
    """
    CREATE TABLE IF NOT EXISTS facts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        neo4j_node_id TEXT NOT NULL,
        content TEXT NOT NULL,
        confidence FLOAT DEFAULT 1.0,
        valid_from TIMESTAMPTZ,
        valid_to TIMESTAMPTZ,
        domain TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # Plans — state machine
    """
    CREATE TABLE IF NOT EXISTS plans (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        neo4j_node_id TEXT NOT NULL,
        name TEXT NOT NULL,
        state TEXT CHECK (state IN ('active', 'paused', 'completed', 'abandoned')),
        progress FLOAT DEFAULT 0.0,
        loom_id TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # Configs — key-value store
    """
    CREATE TABLE IF NOT EXISTS configs (
        key TEXT PRIMARY KEY,
        value JSONB NOT NULL,
        loom_id TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # Audit log — append-only, never deleted
    """
    CREATE TABLE IF NOT EXISTS memory_audit (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        loop_id TEXT,
        loom_id TEXT,
        action TEXT,
        resolution_path TEXT,
        tokens_used INT,
        pressure_tier INT,
        details JSONB
    )
    """,

    # Entity aliases — supports resolve_entity
    """
    CREATE TABLE IF NOT EXISTS entity_aliases (
        alias TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        loom_id TEXT,
        confidence FLOAT DEFAULT 1.0,
        PRIMARY KEY (alias, entity_id)
    )
    """,
]

# Index for case-insensitive alias lookup
DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_alias_lower ON entity_aliases (LOWER(alias))",
    "CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases (entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action ON memory_audit (action)",
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON memory_audit (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts (domain)",
    "CREATE INDEX IF NOT EXISTS idx_plans_state ON plans (state)",
]


class PostgresStore:
    """Direct Postgres access for Memory Bus flat/fast data.

    Manages 5 tables: facts, plans, configs, memory_audit, entity_aliases.
    Uses psycopg 3 with native async connections.
    """

    def __init__(self, dsn: str):
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg is required. Install with: pip install 'psycopg[binary]'")
        self.dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None

    async def initialize(self) -> dict:
        """Connect and create tables if they don't exist.

        Returns:
            Dict with initialization status.
        """
        self._conn = await psycopg.AsyncConnection.connect(
            self.dsn, row_factory=dict_row, autocommit=True
        )

        results = {"tables_created": 0, "indexes_created": 0, "errors": []}

        for stmt in DDL_STATEMENTS:
            try:
                await self._conn.execute(stmt)
                results["tables_created"] += 1
            except Exception as e:
                results["errors"].append(str(e))
                logger.error("DDL failed: %s", e)

        for stmt in DDL_INDEXES:
            try:
                await self._conn.execute(stmt)
                results["indexes_created"] += 1
            except Exception as e:
                results["errors"].append(str(e))
                logger.warning("Index creation failed: %s", e)

        logger.info("PostgresStore initialized: %d tables, %d indexes",
                     results["tables_created"], results["indexes_created"])
        return results

    async def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            await self._conn.close()
            self._conn = None

    def _check_conn(self):
        if self._conn is None or self._conn.closed:
            raise RuntimeError("PostgresStore not initialized. Call initialize() first.")

    # =========================================================================
    # Entity Aliases (spec §10.3)
    # =========================================================================

    async def add_alias(
        self, alias: str, entity_id: str,
        loom_id: Optional[str] = None, confidence: float = 1.0
    ) -> None:
        """Add an entity alias. Upserts on (alias, entity_id)."""
        self._check_conn()
        await self._conn.execute(
            """
            INSERT INTO entity_aliases (alias, entity_id, loom_id, confidence)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (alias, entity_id)
            DO UPDATE SET confidence = EXCLUDED.confidence, loom_id = EXCLUDED.loom_id
            """,
            (alias, entity_id, loom_id, confidence),
        )

    async def add_aliases_batch(self, aliases: List[tuple]) -> int:
        """Batch insert aliases. Each tuple: (alias, entity_id, loom_id, confidence)."""
        self._check_conn()
        count = 0
        async with self._conn.transaction():
            for alias, entity_id, loom_id, confidence in aliases:
                await self._conn.execute(
                    """
                    INSERT INTO entity_aliases (alias, entity_id, loom_id, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (alias, entity_id)
                    DO UPDATE SET confidence = EXCLUDED.confidence, loom_id = EXCLUDED.loom_id
                    """,
                    (alias, entity_id, loom_id, confidence),
                )
                count += 1
        return count

    async def resolve_alias(self, alias: str, loom_id: Optional[str] = None) -> List[Dict]:
        """Resolve an alias to entity IDs. Case-insensitive with confidence ranking.

        Returns list of dicts: [{"entity_id": ..., "alias": ..., "confidence": ..., "loom_id": ...}]
        """
        self._check_conn()
        query = """
            SELECT alias, entity_id, loom_id, confidence
            FROM entity_aliases
            WHERE LOWER(alias) = LOWER(%s)
        """
        params: list = [alias]

        if loom_id:
            query += " AND loom_id = %s"
            params.append(loom_id)

        query += " ORDER BY confidence DESC"

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

    # =========================================================================
    # Audit Log (spec §10.3) — append-only
    # =========================================================================

    async def log_audit(
        self,
        action: str,
        loop_id: Optional[str] = None,
        loom_id: Optional[str] = None,
        resolution_path: Optional[str] = None,
        tokens_used: int = 0,
        pressure_tier: int = 0,
        details: Optional[Dict] = None,
    ) -> int:
        """Append an audit log entry. Returns the row ID."""
        self._check_conn()
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO memory_audit (loop_id, loom_id, action, resolution_path, tokens_used, pressure_tier, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (loop_id, loom_id, action, resolution_path, tokens_used, pressure_tier,
                 json.dumps(details) if details else None),
            )
            row = await cur.fetchone()
            return row[0]

    async def query_audit(
        self, action: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Query the audit log."""
        self._check_conn()
        query = "SELECT * FROM memory_audit"
        params: list = []

        if action:
            query += " WHERE action = %s"
            params.append(action)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

    # =========================================================================
    # Config Store (spec §10.3)
    # =========================================================================

    async def get_config(self, key: str) -> Optional[Dict]:
        """Get a config value by key."""
        self._check_conn()
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM configs WHERE key = %s", (key,))
            return await cur.fetchone()

    async def set_config(
        self, key: str, value: Dict, loom_id: Optional[str] = None
    ) -> None:
        """Set a config value. Upserts on key."""
        self._check_conn()
        await self._conn.execute(
            """
            INSERT INTO configs (key, value, loom_id, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (key)
            DO UPDATE SET value = EXCLUDED.value, loom_id = EXCLUDED.loom_id, updated_at = NOW()
            """,
            (key, json.dumps(value), loom_id),
        )

    # =========================================================================
    # Facts (spec §10.3)
    # =========================================================================

    async def store_fact(
        self,
        neo4j_node_id: str,
        content: str,
        confidence: float = 1.0,
        domain: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ) -> str:
        """Store a fact. Returns the UUID."""
        self._check_conn()
        fact_id = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT INTO facts (id, neo4j_node_id, content, confidence, domain, valid_from, valid_to)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (fact_id, neo4j_node_id, content, confidence, domain, valid_from, valid_to),
        )
        return fact_id

    async def query_facts(
        self,
        domain: Optional[str] = None,
        content_like: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Query facts with optional filters."""
        self._check_conn()
        query = "SELECT * FROM facts WHERE 1=1"
        params: list = []

        if domain:
            query += " AND domain = %s"
            params.append(domain)
        if content_like:
            query += " AND content ILIKE %s"
            params.append(f"%{content_like}%")

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

    # =========================================================================
    # Plans (spec §10.3)
    # =========================================================================

    async def store_plan(
        self,
        neo4j_node_id: str,
        name: str,
        state: str = "active",
        loom_id: Optional[str] = None,
    ) -> str:
        """Store a plan. Returns the UUID."""
        self._check_conn()
        plan_id = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT INTO plans (id, neo4j_node_id, name, state, loom_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (plan_id, neo4j_node_id, name, state, loom_id),
        )
        return plan_id

    async def update_plan_state(
        self, plan_id: str, state: str, progress: float = 0.0
    ) -> None:
        """Update plan state and progress."""
        self._check_conn()
        await self._conn.execute(
            """
            UPDATE plans SET state = %s, progress = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (state, progress, plan_id),
        )

    async def get_active_plans(self, loom_id: Optional[str] = None) -> List[Dict]:
        """Get all active plans."""
        self._check_conn()
        query = "SELECT * FROM plans WHERE state = 'active'"
        params: list = []

        if loom_id:
            query += " AND loom_id = %s"
            params.append(loom_id)

        query += " ORDER BY updated_at DESC"

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

    # =========================================================================
    # Health
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check Postgres connectivity and table status."""
        if self._conn is None or self._conn.closed:
            return {"status": "disconnected", "tables": {}}

        tables = {}
        for table in ["facts", "plans", "configs", "memory_audit", "entity_aliases"]:
            try:
                async with self._conn.cursor() as cur:
                    await cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                    row = await cur.fetchone()
                    tables[table] = {"exists": True, "count": row[0]}
            except Exception as e:
                tables[table] = {"exists": False, "error": str(e)}

        return {
            "status": "connected",
            "dsn": self.dsn.split("@")[-1],  # hide credentials
            "tables": tables,
        }
