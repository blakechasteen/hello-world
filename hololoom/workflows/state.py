"""
State Management for Workflow Execution
========================================

Provides state persistence backends for workflow checkpointing and recovery.

Backends:
- InMemoryState: Fast, no persistence (development)
- SQLiteState: File-based persistence (single-node production)
- RedisState: Distributed persistence (multi-node production)

Author: HoloLoom Team
Date: November 2025
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path

from hololoom.utils.security import sanitize_uri

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    """Abstract state backend interface."""

    @abstractmethod
    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> None:
        """Save execution state."""
        pass

    @abstractmethod
    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Load execution state."""
        pass

    @abstractmethod
    async def list_executions(self) -> List[str]:
        """List all execution IDs."""
        pass

    @abstractmethod
    async def delete_state(self, execution_id: str) -> None:
        """Delete execution state."""
        pass


class InMemoryState(StateBackend):
    """In-memory state backend (no persistence)."""

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        logger.info("InMemoryState initialized")

    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> None:
        self._storage[execution_id] = state
        logger.debug(f"State saved: {execution_id}")

    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(execution_id)

    async def list_executions(self) -> List[str]:
        return list(self._storage.keys())

    async def delete_state(self, execution_id: str) -> None:
        self._storage.pop(execution_id, None)


class SQLiteState(StateBackend):
    """SQLite-backed state persistence."""

    def __init__(self, db_path: str = "./workflow_state.db"):
        self.db_path = Path(db_path)
        self._init_db()
        logger.info(f"SQLiteState initialized: {db_path}")

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_state (
                    execution_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> None:
        state_json = json.dumps(state)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_state (execution_id, state) VALUES (?, ?)",
                (execution_id, state_json)
            )
            conn.commit()
        logger.debug(f"State saved to SQLite: {execution_id}")

    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT state FROM workflow_state WHERE execution_id = ?",
                (execution_id,)
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    async def list_executions(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT execution_id FROM workflow_state ORDER BY timestamp DESC")
            return [row[0] for row in cursor.fetchall()]

    async def delete_state(self, execution_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM workflow_state WHERE execution_id = ?", (execution_id,))
            conn.commit()


class RedisState(StateBackend):
    """Redis-backed distributed state."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis backend."""
        try:
            import redis.asyncio as redis
            self.redis_url = redis_url
            self.redis = redis.from_url(redis_url)
            logger.info(f"RedisState initialized: {sanitize_uri(redis_url)}")
        except ImportError:
            logger.error("Redis not installed. Install with: pip install redis")
            raise

    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> None:
        state_json = json.dumps(state)
        await self.redis.set(f"workflow:{execution_id}", state_json)
        logger.debug(f"State saved to Redis: {execution_id}")

    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        state_json = await self.redis.get(f"workflow:{execution_id}")
        return json.loads(state_json) if state_json else None

    async def list_executions(self) -> List[str]:
        keys = await self.redis.keys("workflow:*")
        return [key.decode().replace("workflow:", "") for key in keys]

    async def delete_state(self, execution_id: str) -> None:
        await self.redis.delete(f"workflow:{execution_id}")


class CheckpointManager:
    """Manage workflow checkpoints for recovery."""

    def __init__(self, state_backend: StateBackend):
        self.state_backend = state_backend
        logger.info("CheckpointManager initialized")

    async def save_checkpoint(
        self,
        execution_id: str,
        node_id: str,
        state: Dict[str, Any]
    ) -> str:
        """Save checkpoint at specific node."""
        checkpoint_id = f"{execution_id}:{node_id}"
        checkpoint_data = {
            "execution_id": execution_id,
            "node_id": node_id,
            "state": state,
            "timestamp": str(datetime.now()),
        }

        await self.state_backend.save_state(checkpoint_id, checkpoint_data)
        logger.info(f"✓ Checkpoint saved: {checkpoint_id}")
        return checkpoint_id

    async def restore_checkpoint(
        self,
        execution_id: str,
        node_id: str
    ) -> Dict[str, Any]:
        """Restore state from checkpoint."""
        checkpoint_id = f"{execution_id}:{node_id}"
        checkpoint_data = await self.state_backend.load_state(checkpoint_id)

        if not checkpoint_data:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        logger.info(f"✓ Checkpoint restored: {checkpoint_id}")
        return checkpoint_data["state"]

    async def list_checkpoints(self, execution_id: str) -> List[str]:
        """List all checkpoints for execution."""
        all_executions = await self.state_backend.list_executions()
        return [e for e in all_executions if e.startswith(f"{execution_id}:")]


# Import datetime for CheckpointManager
from datetime import datetime
