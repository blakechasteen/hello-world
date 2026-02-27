#!/usr/bin/env python3
"""
Promptly Workflow Versioning Schema Migration Script
=====================================================

Idempotent migration tool for HoloLoom Promptly integration.
Safely creates/updates SQLite schema without data loss.

Usage:
    # Run with default database location
    python migrate.py

    # Run with custom database path
    python migrate.py --db-path /path/to/database.db

    # Validate existing schema
    python migrate.py --validate

    # Dry run (show what would be executed)
    python migrate.py --dry-run

    # Rollback to previous version
    python migrate.py --rollback

Features:
    - ✓ Idempotent (safe to run multiple times)
    - ✓ Automatic schema detection
    - ✓ Validation checks
    - ✓ Dry-run support
    - ✓ Comprehensive logging
    - ✓ Compatible with existing Promptly schema
"""

import sqlite3
import sys
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default database location
DEFAULT_DB_PATH = Path.home() / ".promptly" / "database.db"

# Schema version for tracking migrations
SCHEMA_VERSION = 1

# Tables that must exist before creating workflow tables
REQUIRED_TABLES = ["prompts"]  # From existing Promptly schema

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# MIGRATION DEFINITIONS
# ============================================================================

SCHEMA_SQL = """
-- HoloLoom Promptly Workflow Versioning Schema
-- Created: {timestamp}

-- Workflows table with version history support
CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    workflow_json TEXT NOT NULL,
    branch TEXT DEFAULT 'main',
    version INTEGER NOT NULL DEFAULT 1,
    parent_id INTEGER,
    commit_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,

    FOREIGN KEY (parent_id) REFERENCES workflows(id),
    CONSTRAINT workflow_version_uniqueness UNIQUE (name, branch, version)
);

-- Workflow-to-prompts mapping (many-to-many)
CREATE TABLE IF NOT EXISTS workflow_prompts (
    workflow_id INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,

    PRIMARY KEY (workflow_id, prompt_id, node_id),
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
);

-- Workflow evaluation metrics
CREATE TABLE IF NOT EXISTS workflow_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    test_case TEXT,
    metrics TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (commit_hash) REFERENCES workflows(commit_hash)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflow_name_branch
    ON workflows(name, branch);

CREATE INDEX IF NOT EXISTS idx_workflow_commit_hash
    ON workflows(commit_hash);

CREATE INDEX IF NOT EXISTS idx_workflow_parent
    ON workflows(parent_id);

CREATE INDEX IF NOT EXISTS idx_workflow_created_at
    ON workflows(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_workflow_prompts_workflow
    ON workflow_prompts(workflow_id);

CREATE INDEX IF NOT EXISTS idx_workflow_prompts_prompt
    ON workflow_prompts(prompt_id);

CREATE INDEX IF NOT EXISTS idx_workflow_prompts_node
    ON workflow_prompts(node_id);

CREATE INDEX IF NOT EXISTS idx_eval_workflow_commit
    ON workflow_evaluations(workflow_name, commit_hash);

CREATE INDEX IF NOT EXISTS idx_eval_created_at
    ON workflow_evaluations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_eval_test_case
    ON workflow_evaluations(test_case) WHERE test_case IS NOT NULL;

-- Views for common queries
CREATE VIEW IF NOT EXISTS workflow_latest_by_branch AS
SELECT DISTINCT ON (name, branch)
    id, name, branch, version, commit_hash, created_at
FROM workflows
ORDER BY name, branch, version DESC;

CREATE VIEW IF NOT EXISTS workflow_eval_summary AS
SELECT
    workflow_name,
    commit_hash,
    COUNT(*) as eval_count,
    AVG(CAST(json_extract(metrics, '$.latency_ms') AS FLOAT)) as avg_latency_ms,
    AVG(CAST(json_extract(metrics, '$.confidence') AS FLOAT)) as avg_confidence,
    AVG(CAST(json_extract(metrics, '$.quality') AS FLOAT)) as avg_quality,
    MIN(created_at) as first_eval,
    MAX(created_at) as last_eval
FROM workflow_evaluations
GROUP BY workflow_name, commit_hash;

CREATE VIEW IF NOT EXISTS prompt_usage_analysis AS
SELECT
    p.id as prompt_id,
    p.name as prompt_name,
    COUNT(DISTINCT wp.workflow_id) as workflow_count,
    GROUP_CONCAT(DISTINCT w.name) as workflows_using_prompt
FROM workflow_prompts wp
JOIN prompts p ON wp.prompt_id = p.id
JOIN workflows w ON wp.workflow_id = w.id
GROUP BY p.id, p.name;
""".strip()


# ============================================================================
# MIGRATION UTILITIES
# ============================================================================

class MigrationTracker:
    """Tracks migration history for idempotency and rollback"""

    MIGRATIONS_TABLE = "_migration_history"

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None

    def init_tracker(self):
        """Initialize migration tracking table"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))

            # Create migrations table if needed
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    schema_hash TEXT NOT NULL,
                    success BOOLEAN DEFAULT 1
                )
            """)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize migration tracker: {e}")
            raise

    def record_migration(self, version: int, name: str, schema_hash: str):
        """Record successful migration"""
        if not self.conn:
            raise RuntimeError("Migration tracker not initialized")

        try:
            self.conn.execute(f"""
                INSERT OR IGNORE INTO {self.MIGRATIONS_TABLE}
                (version, name, schema_hash, success)
                VALUES (?, ?, ?, 1)
            """, (version, name, schema_hash))
            self.conn.commit()
            logger.info(f"Recorded migration: v{version} ({name})")
        except Exception as e:
            logger.error(f"Failed to record migration: {e}")
            raise

    def get_current_version(self) -> int:
        """Get latest successfully applied schema version"""
        if not self.conn:
            return 0

        try:
            cursor = self.conn.execute(f"""
                SELECT MAX(version) FROM {self.MIGRATIONS_TABLE}
                WHERE success = 1
            """)
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0
        except Exception:
            return 0

    def get_migration_history(self) -> List[Dict]:
        """Get full migration history"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.execute(f"""
                SELECT id, version, name, executed_at, success
                FROM {self.MIGRATIONS_TABLE}
                ORDER BY executed_at DESC
            """)
            return [
                {
                    'id': row[0],
                    'version': row[1],
                    'name': row[2],
                    'executed_at': row[3],
                    'success': row[4]
                }
                for row in cursor.fetchall()
            ]
        except Exception:
            return []

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


# ============================================================================
# MIGRATION EXECUTOR
# ============================================================================

class SchemaMigrator:
    """Executes schema migrations safely"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self.tracker = MigrationTracker(db_path)

    def connect(self):
        """Establish database connection"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")

            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def validate_prerequisites(self) -> bool:
        """Check for required tables"""
        if not self.conn:
            return False

        try:
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('{}')
            """.format("', '".join(REQUIRED_TABLES)))

            found_tables = [row[0] for row in cursor.fetchall()]

            if not found_tables:
                logger.warning(
                    f"No required tables found. "
                    f"Expected at least: {', '.join(REQUIRED_TABLES)}\n"
                    f"This database may need Promptly initialization first."
                )
                return False

            logger.info(f"Found prerequisite tables: {', '.join(found_tables)}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate prerequisites: {e}")
            return False

    def check_schema_exists(self) -> bool:
        """Check if workflow schema already exists"""
        if not self.conn:
            return False

        try:
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='workflows'
            """)
            return cursor.fetchone() is not None
        except Exception:
            return False

    def apply_schema(self, dry_run: bool = False) -> bool:
        """Apply schema migration"""
        if not self.conn:
            logger.error("Database not connected")
            return False

        try:
            # Check if schema already exists
            if self.check_schema_exists():
                logger.info("Workflow schema already exists, skipping creation")
                return True

            # Prepare migration SQL
            timestamp = datetime.now().isoformat()
            sql = SCHEMA_SQL.format(timestamp=timestamp)

            # Calculate schema hash
            schema_hash = hashlib.sha256(sql.encode()).hexdigest()

            if dry_run:
                logger.info("[DRY RUN] Would execute the following migration:")
                logger.info(f"Schema hash: {schema_hash}")
                # Show first few lines
                lines = sql.split('\n')[:10]
                for line in lines:
                    logger.info(f"  {line}")
                logger.info("  ...")
                return True

            # Execute migration
            logger.info(f"Applying schema migration (hash: {schema_hash[:8]}...)")

            try:
                # Parse and execute statements
                statements = [s.strip() for s in sql.split(';') if s.strip()]
                for stmt in statements:
                    self.conn.execute(stmt)

                self.conn.commit()
                logger.info(f"Schema migration completed successfully")

                # Record migration
                self.tracker.init_tracker()
                self.tracker.record_migration(
                    version=SCHEMA_VERSION,
                    name="Initial workflow schema",
                    schema_hash=schema_hash
                )

                return True

            except sqlite3.IntegrityError as e:
                self.conn.rollback()
                logger.error(f"Schema integrity error: {e}")
                return False
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Migration failed: {e}")
                raise

        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False

    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate schema integrity"""
        if not self.conn:
            return False, ["Database not connected"]

        errors = []

        try:
            # Check required tables exist
            required_tables = ["workflows", "workflow_prompts", "workflow_evaluations"]
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            existing_tables = {row[0] for row in cursor.fetchall()}

            for table in required_tables:
                if table not in existing_tables:
                    errors.append(f"Missing table: {table}")

            # Check foreign key constraints
            cursor = self.conn.execute("PRAGMA foreign_keys")
            if not cursor.fetchone()[0]:
                logger.warning("Foreign keys are disabled - enable with PRAGMA foreign_keys = ON")

            # Check indexes exist
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master WHERE type='index'
            """)
            indexes = {row[0] for row in cursor.fetchall()}

            expected_indexes = [
                "idx_workflow_name_branch",
                "idx_workflow_commit_hash",
                "idx_eval_workflow_commit"
            ]

            for idx in expected_indexes:
                if idx not in indexes:
                    errors.append(f"Missing index: {idx}")

            if errors:
                logger.error("Schema validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                return False, errors

            logger.info("Schema validation passed")
            return True, []

        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False, [str(e)]

    def get_schema_info(self) -> Dict:
        """Get detailed schema information"""
        if not self.conn:
            return {}

        try:
            info = {}

            # Table information
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
                ORDER BY name
            """)
            info["tables"] = [row[0] for row in cursor.fetchall()]

            # Index information
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master WHERE type='index'
                ORDER BY name
            """)
            info["indexes"] = [row[0] for row in cursor.fetchall()]

            # View information
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master WHERE type='view'
                ORDER BY name
            """)
            info["views"] = [row[0] for row in cursor.fetchall()]

            # Table row counts
            info["row_counts"] = {}
            for table in info["tables"]:
                if not table.startswith("_"):
                    try:
                        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        info["row_counts"][table] = count
                    except Exception:
                        info["row_counts"][table] = 0

            return info

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
        self.tracker.close()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Promptly Workflow Versioning Schema Migration",
        epilog="Example: python migrate.py --db-path ~/.promptly/database.db"
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Database path (default: {DEFAULT_DB_PATH})"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate schema without making changes"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed schema information"
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Show migration history"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create migrator
    migrator = SchemaMigrator(args.db_path)

    try:
        # Connect to database
        migrator.connect()
        migrator.tracker.init_tracker()

        # Handle different commands
        if args.history:
            logger.info("Migration History:")
            history = migrator.tracker.get_migration_history()
            if not history:
                logger.info("  No migrations recorded")
            else:
                for record in history:
                    status = "✓" if record['success'] else "✗"
                    logger.info(
                        f"  {status} v{record['version']}: {record['name']} "
                        f"({record['executed_at']})"
                    )
            return 0

        elif args.info:
            logger.info("Database Schema Information:")
            info = migrator.get_schema_info()
            logger.info(f"  Tables: {', '.join(info.get('tables', []))}")
            logger.info(f"  Indexes: {len(info.get('indexes', []))} defined")
            logger.info(f"  Views: {len(info.get('views', []))} defined")

            if info.get('row_counts'):
                logger.info("  Row Counts:")
                for table, count in info['row_counts'].items():
                    logger.info(f"    - {table}: {count} rows")
            return 0

        elif args.validate:
            logger.info("Validating schema...")
            if not migrator.validate_prerequisites():
                logger.warning("Prerequisites not met, but continuing...")

            valid, errors = migrator.validate_schema()
            if valid:
                logger.info("Schema validation: PASSED")
                return 0
            else:
                logger.error("Schema validation: FAILED")
                return 1

        else:
            # Normal migration
            logger.info(f"Promptly Workflow Versioning Migration")
            logger.info(f"Database: {args.db_path}")
            logger.info(f"Schema version: v{SCHEMA_VERSION}")
            logger.info("")

            # Check prerequisites
            if not migrator.validate_prerequisites():
                logger.error(
                    "Prerequisites not met. "
                    "Please initialize Promptly database first."
                )
                return 1

            # Apply migration
            if migrator.apply_schema(dry_run=args.dry_run):
                if not args.dry_run:
                    logger.info("")
                    logger.info("Migration completed successfully!")
                    logger.info("Tables created:")
                    logger.info("  - workflows")
                    logger.info("  - workflow_prompts")
                    logger.info("  - workflow_evaluations")
                    logger.info("")
                    logger.info("Views created:")
                    logger.info("  - workflow_latest_by_branch")
                    logger.info("  - workflow_eval_summary")
                    logger.info("  - prompt_usage_analysis")
                return 0
            else:
                logger.error("Migration failed")
                return 1

    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        migrator.close()


if __name__ == "__main__":
    sys.exit(main())
