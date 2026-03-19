#!/usr/bin/env python3
"""
SQL Backend for Hybrid Query Routing

Implements precision data storage with PostgreSQL connection pooling.
Part 2: Foundation Infrastructure (Days 1-5)

Features:
- PostgreSQL connection pooling
- Async context manager lifecycle
- CRUD operations for 4 precision tables
- Query performance tracking
- Graceful degradation to SQLite
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional PostgreSQL support (graceful degradation)
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logging.warning("asyncpg not available - falling back to SQLite")


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """SQL query result with metadata"""
    rows: list[dict[str, Any]]
    row_count: int
    latency_ms: float
    success: bool
    error: str | None = None
    backend: str = "postgresql"  # or "sqlite"


@dataclass
class SQLConfig:
    """SQL backend configuration"""
    # PostgreSQL settings
    host: str = "localhost"
    port: int = 5432
    database: str = "hololoom"
    user: str = "hololoom"
    password: str = "hololoom"

    # Connection pool settings
    min_pool_size: int = 2
    max_pool_size: int = 10
    pool_timeout: float = 30.0

    # SQLite fallback
    sqlite_path: str = "./data/hololoom.db"
    fallback_to_sqlite: bool = True

    # Schema
    schema_path: str = "HoloLoom/infrastructure/sql/schema.sql"


class SQLBackend:
    """
    SQL backend for precision data storage

    Supports PostgreSQL (production) and SQLite (development/fallback)
    """

    def __init__(self, config: SQLConfig):
        self.config = config
        self.pool: Any | None = None  # asyncpg pool or None
        self.sqlite_conn: sqlite3.Connection | None = None
        self.backend_type: str = "uninitialized"
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self):
        """Connect to database (PostgreSQL or SQLite fallback)"""
        if HAS_ASYNCPG:
            try:
                await self._connect_postgresql()
                return
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
                if not self.config.fallback_to_sqlite:
                    raise

        # Fallback to SQLite
        await self._connect_sqlite()

    async def _connect_postgresql(self):
        """Connect to PostgreSQL with connection pooling"""
        logger.info(f"Connecting to PostgreSQL at {self.config.host}:{self.config.port}")

        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            timeout=self.config.pool_timeout
        )

        self.backend_type = "postgresql"
        logger.info("PostgreSQL connection pool established")

        # Initialize schema
        await self._initialize_schema_postgresql()

    async def _connect_sqlite(self):
        """Connect to SQLite (fallback or development)"""
        logger.info(f"Using SQLite backend at {self.config.sqlite_path}")

        # Create directory if needed
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # SQLite is synchronous, so we use a thread pool executor
        # Disable same-thread check since we manage thread safety via executor
        self.sqlite_conn = sqlite3.connect(
            self.config.sqlite_path,
            check_same_thread=False  # Allow cross-thread access (safe with executor)
        )
        self.sqlite_conn.row_factory = sqlite3.Row  # Dict-like rows

        self.backend_type = "sqlite"
        logger.info("SQLite connection established")

        # Initialize schema
        await self._initialize_schema_sqlite()

    async def _initialize_schema_postgresql(self):
        """Initialize PostgreSQL schema"""
        schema_sql = self._read_schema()

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("PostgreSQL schema initialized")

    async def _initialize_schema_sqlite(self):
        """Initialize SQLite schema"""
        schema_sql = self._read_schema()

        # SQLite executescript doesn't support parameter binding
        cursor = self.sqlite_conn.cursor()
        cursor.executescript(schema_sql)
        self.sqlite_conn.commit()

        logger.info("SQLite schema initialized")

    def _read_schema(self) -> str:
        """Read schema SQL file"""
        # Try multiple path strategies for robustness
        schema_path = Path(self.config.schema_path)

        # Strategy 1: Absolute path or relative to CWD
        if schema_path.exists():
            return schema_path.read_text()

        # Strategy 2: Relative to this module's directory
        module_dir = Path(__file__).parent
        schema_path = module_dir / "schema.sql"
        if schema_path.exists():
            return schema_path.read_text()

        # Strategy 3: Relative to project root
        project_root = module_dir.parent.parent
        schema_path = project_root / self.config.schema_path
        if schema_path.exists():
            return schema_path.read_text()

        raise FileNotFoundError(
            f"Schema file not found. Tried:\n"
            f"  - {self.config.schema_path}\n"
            f"  - {module_dir / 'schema.sql'}\n"
            f"  - {project_root / self.config.schema_path}"
        )

    async def execute_query(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> QueryResult:
        """
        Execute SQL query and return results

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            QueryResult with rows and metadata
        """
        if self._closed:
            raise RuntimeError("Backend is closed")

        start_time = asyncio.get_event_loop().time()

        try:
            if self.backend_type == "postgresql":
                result = await self._execute_postgresql(query, params)
            elif self.backend_type == "sqlite":
                result = await self._execute_sqlite(query, params)
            else:
                raise RuntimeError("Backend not initialized")

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return QueryResult(
                rows=result,
                row_count=len(result),
                latency_ms=latency_ms,
                success=True,
                backend=self.backend_type
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(f"Query failed: {e}")

            return QueryResult(
                rows=[],
                row_count=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                backend=self.backend_type
            )

    async def _execute_postgresql(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query on PostgreSQL"""
        async with self.pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)

            return [dict(row) for row in rows]

    async def _execute_sqlite(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query on SQLite (wrapped in asyncio)"""
        # SQLite is synchronous - run in thread pool
        loop = asyncio.get_event_loop()

        def _sync_execute():
            cursor = self.sqlite_conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        return await loop.run_in_executor(None, _sync_execute)

    # ========================================================================
    # CRUD Operations for Precision Tables
    # ========================================================================

    async def insert_policy_rule(
        self,
        rule_id: str,
        rule_name: str,
        rule_type: str,
        rule_logic: dict[str, Any],
        confidence: float = 1.0,
        domain: str = "beekeeping",
        neo4j_node_id: str | None = None
    ) -> QueryResult:
        """Insert policy rule"""
        query = """
            INSERT INTO policy_rules
            (rule_id, rule_name, rule_type, rule_logic, confidence, domain, neo4j_node_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """ if self.backend_type == "postgresql" else """
            INSERT INTO policy_rules
            (rule_id, rule_name, rule_type, rule_logic, confidence, domain, neo4j_node_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            rule_id, rule_name, rule_type,
            json.dumps(rule_logic), confidence, domain, neo4j_node_id
        ]

        return await self.execute_query(query, params)

    async def get_policy_rule(self, rule_id: str) -> QueryResult:
        """Get policy rule by ID"""
        query = """
            SELECT * FROM policy_rules WHERE rule_id = $1
        """ if self.backend_type == "postgresql" else """
            SELECT * FROM policy_rules WHERE rule_id = ?
        """

        return await self.execute_query(query, [rule_id])

    async def insert_transaction_log(
        self,
        transaction_id: str,
        transaction_type: str,
        entity_type: str,
        entity_id: str,
        user_id: str,
        action_data: dict[str, Any],
        neo4j_node_id: str | None = None
    ) -> QueryResult:
        """Insert transaction log"""
        query = """
            INSERT INTO transaction_logs
            (transaction_id, transaction_type, entity_type, entity_id, user_id, action_data, neo4j_node_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """ if self.backend_type == "postgresql" else """
            INSERT INTO transaction_logs
            (transaction_id, transaction_type, entity_type, entity_id, user_id, action_data, neo4j_node_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            transaction_id, transaction_type, entity_type, entity_id, user_id,
            json.dumps(action_data), neo4j_node_id
        ]

        return await self.execute_query(query, params)

    async def insert_audit_trail(
        self,
        audit_id: str,
        audit_type: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        compliance_flag: bool = False
    ) -> QueryResult:
        """Insert audit trail"""
        query = """
            INSERT INTO audit_trails
            (audit_id, audit_type, resource_type, resource_id, user_id, before_state, after_state, compliance_flag)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """ if self.backend_type == "postgresql" else """
            INSERT INTO audit_trails
            (audit_id, audit_type, resource_type, resource_id, user_id, before_state, after_state, compliance_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            audit_id, audit_type, resource_type, resource_id, user_id,
            json.dumps(before_state) if before_state else None,
            json.dumps(after_state) if after_state else None,
            compliance_flag
        ]

        return await self.execute_query(query, params)

    async def insert_user_permission(
        self,
        permission_id: str,
        user_id: str,
        resource_type: str,
        permission_level: str,
        neo4j_user_node: str | None = None,
        expires_at: datetime | None = None
    ) -> QueryResult:
        """Insert user permission"""
        query = """
            INSERT INTO user_permissions
            (permission_id, user_id, resource_type, permission_level, neo4j_user_node, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """ if self.backend_type == "postgresql" else """
            INSERT INTO user_permissions
            (permission_id, user_id, resource_type, permission_level, neo4j_user_node, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        params = [
            permission_id, user_id, resource_type, permission_level,
            neo4j_user_node, expires_at
        ]

        return await self.execute_query(query, params)

    async def close(self):
        """Close database connections"""
        if self._closed:
            return

        logger.info(f"Closing {self.backend_type} connection")

        if self.pool:
            await self.pool.close()
            self.pool = None

        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None

        self._closed = True
        logger.info("SQL backend closed")


# ============================================================================
# Factory Function
# ============================================================================

def create_sql_backend(config: SQLConfig | None = None) -> SQLBackend:
    """
    Create SQL backend with default or custom configuration

    Args:
        config: SQL configuration (uses defaults if None)

    Returns:
        SQLBackend instance (not yet connected - use async context manager)
    """
    if config is None:
        config = SQLConfig()

    return SQLBackend(config)
