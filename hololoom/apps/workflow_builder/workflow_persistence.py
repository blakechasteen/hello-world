"""SQLite persistence for workflows following ThoughtPersistence pattern.

Created: 2025-12-23
Part of HoloLoom Workflow Builder production integration.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


@dataclass
class WorkflowRecord:
    """Persistent workflow record."""
    id: str
    name: str
    description: str
    nodes: list[dict]
    connections: list[dict]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    owner_id: str = "default"


@dataclass
class VersionRecord:
    """Workflow version for history tracking."""
    id: str
    workflow_id: str
    version_number: int
    parent_version: str | None
    changes: dict[str, Any]
    created_at: datetime
    created_by: str


@dataclass
class ExecutionRecord:
    """Execution log entry."""
    id: str
    workflow_id: str
    version_id: str | None
    status: str  # pending, running, completed, failed, cancelled
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    node_results: dict[str, Any]
    started_at: datetime
    completed_at: datetime | None
    error: str | None


# ==================== Marketplace Records ====================

@dataclass
class TemplateCategoryRecord:
    """Template category for marketplace organization."""
    id: str
    name: str
    parent_id: str | None
    description: str
    icon: str
    display_order: int = 0


@dataclass
class MarketplaceTemplateRecord:
    """Template listing in the marketplace."""
    id: str
    workflow_id: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    author: str
    license: str
    is_featured: bool
    usage_count: int
    avg_rating: float
    rating_count: int
    tags: list[str]
    published_at: datetime
    updated_at: datetime


@dataclass
class TemplateRatingRecord:
    """User rating for a template."""
    id: str
    template_id: str
    user_id: str
    rating: int  # 1-5
    comment: str
    created_at: datetime


class WorkflowPersistence:
    """SQLite-backed workflow persistence.

    Provides CRUD operations for workflows, version history tracking,
    and execution logging. Uses async SQLite for non-blocking I/O.

    Usage:
        async with WorkflowPersistence("./workflows.db") as persistence:
            await persistence.save_workflow(workflow_record)
            workflow = await persistence.get_workflow(workflow_id)
    """

    SCHEMA = """
    -- Workflows table
    CREATE TABLE IF NOT EXISTS workflows (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        nodes TEXT NOT NULL,  -- JSON array of node definitions
        connections TEXT NOT NULL,  -- JSON array of connection definitions
        metadata TEXT DEFAULT '{}',  -- JSON object for additional data
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        owner_id TEXT DEFAULT 'default'
    );

    -- Version history table
    CREATE TABLE IF NOT EXISTS versions (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        parent_version TEXT,
        changes TEXT NOT NULL,  -- JSON object describing changes
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE,
        UNIQUE(workflow_id, version_number)
    );

    -- Execution logs table
    CREATE TABLE IF NOT EXISTS executions (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        version_id TEXT,
        status TEXT NOT NULL,
        input_data TEXT NOT NULL,  -- JSON object
        output_data TEXT,  -- JSON object (nullable for pending/running)
        node_results TEXT DEFAULT '{}',  -- JSON object with per-node results
        started_at TEXT NOT NULL,
        completed_at TEXT,
        error TEXT,
        FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner_id);
    CREATE INDEX IF NOT EXISTS idx_workflows_updated ON workflows(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_versions_workflow ON versions(workflow_id);
    CREATE INDEX IF NOT EXISTS idx_versions_number ON versions(workflow_id, version_number DESC);
    CREATE INDEX IF NOT EXISTS idx_executions_workflow ON executions(workflow_id);
    CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
    CREATE INDEX IF NOT EXISTS idx_executions_started ON executions(started_at DESC);

    -- ==================== Marketplace Tables ====================

    -- Template categories (hierarchical)
    CREATE TABLE IF NOT EXISTS template_categories (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        parent_id TEXT REFERENCES template_categories(id),
        description TEXT DEFAULT '',
        icon TEXT DEFAULT '',
        display_order INTEGER DEFAULT 0
    );

    -- Marketplace template listings
    CREATE TABLE IF NOT EXISTS marketplace_templates (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        category TEXT NOT NULL,
        difficulty TEXT CHECK(difficulty IN ('beginner', 'intermediate', 'advanced')) DEFAULT 'intermediate',
        author TEXT DEFAULT 'HoloLoom Team',
        license TEXT DEFAULT 'MIT',
        is_featured INTEGER DEFAULT 0,
        usage_count INTEGER DEFAULT 0,
        avg_rating REAL DEFAULT 0.0,
        rating_count INTEGER DEFAULT 0,
        tags TEXT DEFAULT '[]',  -- JSON array
        published_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
    );

    -- User ratings for templates
    CREATE TABLE IF NOT EXISTS template_ratings (
        id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        rating INTEGER CHECK(rating BETWEEN 1 AND 5) NOT NULL,
        comment TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY (template_id) REFERENCES marketplace_templates(id) ON DELETE CASCADE,
        UNIQUE(template_id, user_id)  -- One rating per user per template
    );

    -- Marketplace indexes
    CREATE INDEX IF NOT EXISTS idx_categories_parent ON template_categories(parent_id);
    CREATE INDEX IF NOT EXISTS idx_categories_order ON template_categories(display_order);
    CREATE INDEX IF NOT EXISTS idx_marketplace_category ON marketplace_templates(category);
    CREATE INDEX IF NOT EXISTS idx_marketplace_difficulty ON marketplace_templates(difficulty);
    CREATE INDEX IF NOT EXISTS idx_marketplace_featured ON marketplace_templates(is_featured DESC);
    CREATE INDEX IF NOT EXISTS idx_marketplace_usage ON marketplace_templates(usage_count DESC);
    CREATE INDEX IF NOT EXISTS idx_marketplace_rating ON marketplace_templates(avg_rating DESC);
    CREATE INDEX IF NOT EXISTS idx_ratings_template ON template_ratings(template_id);
    """

    def __init__(self, db_path: str = "./workflows.db"):
        """Initialize persistence with database path.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.
        """
        self.db_path = Path(db_path)
        self._connection: Any = None

    async def initialize(self):
        """Initialize database and create tables if needed."""
        if aiosqlite is None:
            raise ImportError(
                "aiosqlite is required for WorkflowPersistence. "
                "Install it with: pip install aiosqlite"
            )
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._connection.executescript(self.SCHEMA)
        await self._connection.commit()

    async def close(self):
        """Close database connection gracefully."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    # ==================== Workflow CRUD ====================

    async def save_workflow(self, workflow: WorkflowRecord) -> str:
        """Save or update a workflow.

        Uses INSERT OR REPLACE for upsert behavior.

        Args:
            workflow: WorkflowRecord to save

        Returns:
            Workflow ID
        """
        now = datetime.utcnow().isoformat()
        created_at = workflow.created_at.isoformat() if workflow.created_at else now

        await self._connection.execute(
            """INSERT OR REPLACE INTO workflows
               (id, name, description, nodes, connections, metadata, created_at, updated_at, owner_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workflow.id,
                workflow.name,
                workflow.description,
                json.dumps(workflow.nodes),
                json.dumps(workflow.connections),
                json.dumps(workflow.metadata),
                created_at,
                now,
                workflow.owner_id
            )
        )
        await self._connection.commit()
        return workflow.id

    async def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        """Get a workflow by ID.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            WorkflowRecord if found, None otherwise
        """
        async with self._connection.execute(
            "SELECT id, name, description, nodes, connections, metadata, created_at, updated_at, owner_id FROM workflows WHERE id = ?",
            (workflow_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return WorkflowRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    nodes=json.loads(row[3]),
                    connections=json.loads(row[4]),
                    metadata=json.loads(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    owner_id=row[8]
                )
        return None

    async def list_workflows(
        self,
        owner_id: str = "default",
        limit: int = 100,
        offset: int = 0
    ) -> list[WorkflowRecord]:
        """List workflows for an owner.

        Args:
            owner_id: Owner identifier (default for single-user mode)
            limit: Maximum number of workflows to return
            offset: Pagination offset

        Returns:
            List of WorkflowRecords ordered by updated_at DESC
        """
        workflows = []
        async with self._connection.execute(
            """SELECT id, name, description, nodes, connections, metadata, created_at, updated_at, owner_id
               FROM workflows
               WHERE owner_id = ?
               ORDER BY updated_at DESC
               LIMIT ? OFFSET ?""",
            (owner_id, limit, offset)
        ) as cursor:
            async for row in cursor:
                workflows.append(WorkflowRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    nodes=json.loads(row[3]),
                    connections=json.loads(row[4]),
                    metadata=json.loads(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    owner_id=row[8]
                ))
        return workflows

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and all associated data.

        Cascading delete removes versions and executions.

        Args:
            workflow_id: Workflow to delete

        Returns:
            True if workflow was deleted, False if not found
        """
        cursor = await self._connection.execute(
            "DELETE FROM workflows WHERE id = ?",
            (workflow_id,)
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def workflow_exists(self, workflow_id: str) -> bool:
        """Check if a workflow exists.

        Args:
            workflow_id: Workflow to check

        Returns:
            True if exists, False otherwise
        """
        async with self._connection.execute(
            "SELECT 1 FROM workflows WHERE id = ? LIMIT 1",
            (workflow_id,)
        ) as cursor:
            return await cursor.fetchone() is not None

    # ==================== Version Management ====================

    async def save_version(self, version: VersionRecord) -> str:
        """Save a workflow version.

        Args:
            version: VersionRecord to save

        Returns:
            Version ID
        """
        await self._connection.execute(
            """INSERT INTO versions
               (id, workflow_id, version_number, parent_version, changes, created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                version.id,
                version.workflow_id,
                version.version_number,
                version.parent_version,
                json.dumps(version.changes),
                version.created_at.isoformat(),
                version.created_by
            )
        )
        await self._connection.commit()
        return version.id

    async def get_versions(self, workflow_id: str, limit: int = 50) -> list[VersionRecord]:
        """Get version history for a workflow.

        Args:
            workflow_id: Workflow to get versions for
            limit: Maximum versions to return

        Returns:
            List of VersionRecords ordered by version_number DESC
        """
        versions = []
        async with self._connection.execute(
            """SELECT id, workflow_id, version_number, parent_version, changes, created_at, created_by
               FROM versions
               WHERE workflow_id = ?
               ORDER BY version_number DESC
               LIMIT ?""",
            (workflow_id, limit)
        ) as cursor:
            async for row in cursor:
                versions.append(VersionRecord(
                    id=row[0],
                    workflow_id=row[1],
                    version_number=row[2],
                    parent_version=row[3],
                    changes=json.loads(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    created_by=row[6]
                ))
        return versions

    async def get_latest_version_number(self, workflow_id: str) -> int:
        """Get the latest version number for a workflow.

        Args:
            workflow_id: Workflow to check

        Returns:
            Latest version number, or 0 if no versions exist
        """
        async with self._connection.execute(
            "SELECT MAX(version_number) FROM versions WHERE workflow_id = ?",
            (workflow_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row[0] is not None else 0

    async def create_version(
        self,
        workflow_id: str,
        changes: dict[str, Any],
        created_by: str = "default"
    ) -> VersionRecord:
        """Create a new version for a workflow.

        Automatically increments version number and links to parent.

        Args:
            workflow_id: Workflow to version
            changes: Description of changes made
            created_by: User who created this version

        Returns:
            Created VersionRecord
        """
        versions = await self.get_versions(workflow_id, limit=1)
        parent_version = versions[0].id if versions else None
        latest_number = await self.get_latest_version_number(workflow_id)

        version = VersionRecord(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            version_number=latest_number + 1,
            parent_version=parent_version,
            changes=changes,
            created_at=datetime.utcnow(),
            created_by=created_by
        )

        await self.save_version(version)
        return version

    # ==================== Execution Logging ====================

    async def save_execution(self, execution: ExecutionRecord) -> str:
        """Save or update an execution record.

        Args:
            execution: ExecutionRecord to save

        Returns:
            Execution ID
        """
        await self._connection.execute(
            """INSERT OR REPLACE INTO executions
               (id, workflow_id, version_id, status, input_data, output_data, node_results, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                execution.id,
                execution.workflow_id,
                execution.version_id,
                execution.status,
                json.dumps(execution.input_data),
                json.dumps(execution.output_data) if execution.output_data else None,
                json.dumps(execution.node_results),
                execution.started_at.isoformat(),
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.error
            )
        )
        await self._connection.commit()
        return execution.id

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution by ID.

        Args:
            execution_id: Execution to retrieve

        Returns:
            ExecutionRecord if found, None otherwise
        """
        async with self._connection.execute(
            """SELECT id, workflow_id, version_id, status, input_data, output_data,
                      node_results, started_at, completed_at, error
               FROM executions WHERE id = ?""",
            (execution_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return ExecutionRecord(
                    id=row[0],
                    workflow_id=row[1],
                    version_id=row[2],
                    status=row[3],
                    input_data=json.loads(row[4]),
                    output_data=json.loads(row[5]) if row[5] else None,
                    node_results=json.loads(row[6]),
                    started_at=datetime.fromisoformat(row[7]),
                    completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    error=row[9]
                )
        return None

    async def get_executions(
        self,
        workflow_id: str,
        limit: int = 50,
        status: str | None = None
    ) -> list[ExecutionRecord]:
        """Get execution history for a workflow.

        Args:
            workflow_id: Workflow to get executions for
            limit: Maximum executions to return
            status: Optional status filter (pending, running, completed, failed, cancelled)

        Returns:
            List of ExecutionRecords ordered by started_at DESC
        """
        executions = []

        if status:
            query = """SELECT id, workflow_id, version_id, status, input_data, output_data,
                              node_results, started_at, completed_at, error
                       FROM executions
                       WHERE workflow_id = ? AND status = ?
                       ORDER BY started_at DESC LIMIT ?"""
            params = (workflow_id, status, limit)
        else:
            query = """SELECT id, workflow_id, version_id, status, input_data, output_data,
                              node_results, started_at, completed_at, error
                       FROM executions
                       WHERE workflow_id = ?
                       ORDER BY started_at DESC LIMIT ?"""
            params = (workflow_id, limit)

        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                executions.append(ExecutionRecord(
                    id=row[0],
                    workflow_id=row[1],
                    version_id=row[2],
                    status=row[3],
                    input_data=json.loads(row[4]),
                    output_data=json.loads(row[5]) if row[5] else None,
                    node_results=json.loads(row[6]),
                    started_at=datetime.fromisoformat(row[7]),
                    completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    error=row[9]
                ))
        return executions

    async def update_execution_status(
        self,
        execution_id: str,
        status: str,
        output_data: dict[str, Any] | None = None,
        node_results: dict[str, Any] | None = None,
        error: str | None = None
    ) -> bool:
        """Update execution status and optionally results.

        Args:
            execution_id: Execution to update
            status: New status
            output_data: Optional final output data
            node_results: Optional per-node results
            error: Optional error message (for failed status)

        Returns:
            True if updated, False if not found
        """
        now = datetime.utcnow().isoformat()
        completed_at = now if status in ("completed", "failed", "cancelled") else None

        # Build dynamic update
        updates = ["status = ?", "updated_at = ?"]
        params = [status, now]

        if completed_at:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if output_data is not None:
            updates.append("output_data = ?")
            params.append(json.dumps(output_data))

        if node_results is not None:
            updates.append("node_results = ?")
            params.append(json.dumps(node_results))

        if error is not None:
            updates.append("error = ?")
            params.append(error)

        params.append(execution_id)

        cursor = await self._connection.execute(
            f"UPDATE executions SET {', '.join(updates)} WHERE id = ?",
            params
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def create_execution(
        self,
        workflow_id: str,
        input_data: dict[str, Any],
        version_id: str | None = None
    ) -> ExecutionRecord:
        """Create a new execution record in pending status.

        Args:
            workflow_id: Workflow to execute
            input_data: Input data for execution
            version_id: Optional specific version to execute

        Returns:
            Created ExecutionRecord
        """
        execution = ExecutionRecord(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            version_id=version_id,
            status="pending",
            input_data=input_data,
            output_data=None,
            node_results={},
            started_at=datetime.utcnow(),
            completed_at=None,
            error=None
        )

        await self.save_execution(execution)
        return execution

    # ==================== Statistics ====================

    async def get_workflow_stats(self, workflow_id: str) -> dict[str, Any]:
        """Get statistics for a workflow.

        Args:
            workflow_id: Workflow to get stats for

        Returns:
            Dict with version_count, execution_count, success_rate, etc.
        """
        stats = {
            "workflow_id": workflow_id,
            "version_count": 0,
            "execution_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
            "avg_duration_ms": None
        }

        # Version count
        async with self._connection.execute(
            "SELECT COUNT(*) FROM versions WHERE workflow_id = ?",
            (workflow_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["version_count"] = row[0]

        # Execution stats
        async with self._connection.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
               FROM executions WHERE workflow_id = ?""",
            (workflow_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["execution_count"] = row[0]
            stats["completed_count"] = row[1] or 0
            stats["failed_count"] = row[2] or 0

            if stats["execution_count"] > 0:
                stats["success_rate"] = stats["completed_count"] / stats["execution_count"]

        return stats

    # ==================== Marketplace Categories ====================

    async def save_category(self, category: TemplateCategoryRecord) -> str:
        """Save or update a template category."""
        await self._connection.execute(
            """INSERT OR REPLACE INTO template_categories
               (id, name, parent_id, description, icon, display_order)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                category.id,
                category.name,
                category.parent_id,
                category.description,
                category.icon,
                category.display_order
            )
        )
        await self._connection.commit()
        return category.id

    async def get_categories(self, parent_id: str | None = None) -> list[TemplateCategoryRecord]:
        """Get categories, optionally filtered by parent."""
        categories = []
        if parent_id is None:
            query = "SELECT id, name, parent_id, description, icon, display_order FROM template_categories WHERE parent_id IS NULL ORDER BY display_order"
            params = ()
        else:
            query = "SELECT id, name, parent_id, description, icon, display_order FROM template_categories WHERE parent_id = ? ORDER BY display_order"
            params = (parent_id,)

        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                categories.append(TemplateCategoryRecord(
                    id=row[0],
                    name=row[1],
                    parent_id=row[2],
                    description=row[3],
                    icon=row[4],
                    display_order=row[5]
                ))
        return categories

    async def get_all_categories(self) -> list[TemplateCategoryRecord]:
        """Get all categories."""
        categories = []
        async with self._connection.execute(
            "SELECT id, name, parent_id, description, icon, display_order FROM template_categories ORDER BY display_order"
        ) as cursor:
            async for row in cursor:
                categories.append(TemplateCategoryRecord(
                    id=row[0],
                    name=row[1],
                    parent_id=row[2],
                    description=row[3],
                    icon=row[4],
                    display_order=row[5]
                ))
        return categories

    # ==================== Marketplace Templates ====================

    async def save_marketplace_template(self, template: MarketplaceTemplateRecord) -> str:
        """Save or update a marketplace template."""
        await self._connection.execute(
            """INSERT OR REPLACE INTO marketplace_templates
               (id, workflow_id, category, difficulty, author, license, is_featured,
                usage_count, avg_rating, rating_count, tags, published_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                template.id,
                template.workflow_id,
                template.category,
                template.difficulty,
                template.author,
                template.license,
                1 if template.is_featured else 0,
                template.usage_count,
                template.avg_rating,
                template.rating_count,
                json.dumps(template.tags),
                template.published_at.isoformat(),
                template.updated_at.isoformat()
            )
        )
        await self._connection.commit()
        return template.id

    async def get_marketplace_template(self, template_id: str) -> MarketplaceTemplateRecord | None:
        """Get a marketplace template by ID."""
        async with self._connection.execute(
            """SELECT id, workflow_id, category, difficulty, author, license, is_featured,
                      usage_count, avg_rating, rating_count, tags, published_at, updated_at
               FROM marketplace_templates WHERE id = ?""",
            (template_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return MarketplaceTemplateRecord(
                    id=row[0],
                    workflow_id=row[1],
                    category=row[2],
                    difficulty=row[3],
                    author=row[4],
                    license=row[5],
                    is_featured=bool(row[6]),
                    usage_count=row[7],
                    avg_rating=row[8],
                    rating_count=row[9],
                    tags=json.loads(row[10]),
                    published_at=datetime.fromisoformat(row[11]),
                    updated_at=datetime.fromisoformat(row[12])
                )
        return None

    async def list_marketplace_templates(
        self,
        category: str | None = None,
        difficulty: str | None = None,
        featured_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "usage_count"
    ) -> list[MarketplaceTemplateRecord]:
        """List marketplace templates with filtering."""
        templates = []
        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if difficulty:
            conditions.append("difficulty = ?")
            params.append(difficulty)
        if featured_only:
            conditions.append("is_featured = 1")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = f"ORDER BY {order_by} DESC"

        query = f"""SELECT id, workflow_id, category, difficulty, author, license, is_featured,
                           usage_count, avg_rating, rating_count, tags, published_at, updated_at
                    FROM marketplace_templates {where_clause} {order_clause} LIMIT ? OFFSET ?"""
        params.extend([limit, offset])

        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                templates.append(MarketplaceTemplateRecord(
                    id=row[0],
                    workflow_id=row[1],
                    category=row[2],
                    difficulty=row[3],
                    author=row[4],
                    license=row[5],
                    is_featured=bool(row[6]),
                    usage_count=row[7],
                    avg_rating=row[8],
                    rating_count=row[9],
                    tags=json.loads(row[10]),
                    published_at=datetime.fromisoformat(row[11]),
                    updated_at=datetime.fromisoformat(row[12])
                ))
        return templates

    async def search_marketplace_templates(
        self,
        query: str,
        category: str | None = None,
        difficulty: str | None = None,
        limit: int = 20
    ) -> list[MarketplaceTemplateRecord]:
        """Search templates by name/description from workflow."""
        templates = []
        conditions = ["(w.name LIKE ? OR w.description LIKE ?)"]
        params = [f"%{query}%", f"%{query}%"]

        if category:
            conditions.append("m.category = ?")
            params.append(category)
        if difficulty:
            conditions.append("m.difficulty = ?")
            params.append(difficulty)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        sql = f"""SELECT m.id, m.workflow_id, m.category, m.difficulty, m.author, m.license,
                         m.is_featured, m.usage_count, m.avg_rating, m.rating_count, m.tags,
                         m.published_at, m.updated_at
                  FROM marketplace_templates m
                  JOIN workflows w ON m.workflow_id = w.id
                  WHERE {where_clause}
                  ORDER BY m.usage_count DESC LIMIT ?"""

        async with self._connection.execute(sql, params) as cursor:
            async for row in cursor:
                templates.append(MarketplaceTemplateRecord(
                    id=row[0],
                    workflow_id=row[1],
                    category=row[2],
                    difficulty=row[3],
                    author=row[4],
                    license=row[5],
                    is_featured=bool(row[6]),
                    usage_count=row[7],
                    avg_rating=row[8],
                    rating_count=row[9],
                    tags=json.loads(row[10]),
                    published_at=datetime.fromisoformat(row[11]),
                    updated_at=datetime.fromisoformat(row[12])
                ))
        return templates

    async def increment_usage_count(self, template_id: str) -> bool:
        """Increment usage count for a template."""
        cursor = await self._connection.execute(
            "UPDATE marketplace_templates SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), template_id)
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    # ==================== Template Ratings ====================

    async def save_rating(self, rating: TemplateRatingRecord) -> str:
        """Save or update a template rating."""
        await self._connection.execute(
            """INSERT OR REPLACE INTO template_ratings
               (id, template_id, user_id, rating, comment, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rating.id,
                rating.template_id,
                rating.user_id,
                rating.rating,
                rating.comment,
                rating.created_at.isoformat()
            )
        )
        await self._connection.commit()

        # Update template's average rating
        await self._update_template_rating_stats(rating.template_id)
        return rating.id

    async def _update_template_rating_stats(self, template_id: str):
        """Update template's avg_rating and rating_count."""
        async with self._connection.execute(
            "SELECT AVG(rating), COUNT(*) FROM template_ratings WHERE template_id = ?",
            (template_id,)
        ) as cursor:
            row = await cursor.fetchone()
            avg_rating = row[0] or 0.0
            rating_count = row[1] or 0

        await self._connection.execute(
            "UPDATE marketplace_templates SET avg_rating = ?, rating_count = ? WHERE id = ?",
            (avg_rating, rating_count, template_id)
        )
        await self._connection.commit()

    async def get_ratings(self, template_id: str, limit: int = 50) -> list[TemplateRatingRecord]:
        """Get ratings for a template."""
        ratings = []
        async with self._connection.execute(
            """SELECT id, template_id, user_id, rating, comment, created_at
               FROM template_ratings WHERE template_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (template_id, limit)
        ) as cursor:
            async for row in cursor:
                ratings.append(TemplateRatingRecord(
                    id=row[0],
                    template_id=row[1],
                    user_id=row[2],
                    rating=row[3],
                    comment=row[4],
                    created_at=datetime.fromisoformat(row[5])
                ))
        return ratings
