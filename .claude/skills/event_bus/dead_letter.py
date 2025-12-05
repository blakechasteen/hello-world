"""
Dead Letter Queue - Failed Event Handling
==========================================

**Created**: December 3, 2025
**Purpose**: Handle events that failed delivery to subscribers

## Features

- Persistent dead letter storage
- Failure reason tracking
- Configurable retry policies
- Manual reprocessing capability
- Age-based cleanup

## Use Cases

1. **Transient Failures**: Retry events that failed due to temporary issues
2. **Debugging**: Inspect failed events to diagnose issues
3. **Monitoring**: Track failure patterns across subscribers
4. **Recovery**: Manually reprocess events after fixing handlers
"""

import asyncio
import aiosqlite
import json
import uuid
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging

from skills.protocol import SkillEvent, EventType

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class RetryStrategy(Enum):
    """How to handle retries."""
    NONE = "none"              # No automatic retry
    FIXED = "fixed"            # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff


@dataclass
class DeadLetterConfig:
    """Dead letter queue configuration."""
    db_path: str = "dead_letters.db"
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_delay_seconds: float = 60.0
    retry_multiplier: float = 2.0      # For exponential backoff
    max_retry_delay_seconds: float = 3600.0  # 1 hour max
    retention_days: int = 7
    enable_auto_retry: bool = True
    auto_retry_interval_seconds: float = 300.0  # 5 minutes


@dataclass
class DeadLetter:
    """A failed event in the dead letter queue."""
    id: str
    event: SkillEvent
    subscription_id: str
    error_message: str
    error_type: str
    failure_count: int
    first_failure: str
    last_failure: str
    next_retry: Optional[str]
    status: str  # "pending", "retrying", "exhausted", "resolved"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Schema
# ============================================================================

CREATE_DEAD_LETTERS_TABLE = """
CREATE TABLE IF NOT EXISTS dead_letters (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    event_data TEXT NOT NULL,
    subscription_id TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_type TEXT NOT NULL,
    failure_count INTEGER DEFAULT 1,
    first_failure TEXT NOT NULL,
    last_failure TEXT NOT NULL,
    next_retry TEXT,
    status TEXT DEFAULT 'pending',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_dl_status ON dead_letters(status);",
    "CREATE INDEX IF NOT EXISTS idx_dl_next_retry ON dead_letters(next_retry);",
    "CREATE INDEX IF NOT EXISTS idx_dl_subscription ON dead_letters(subscription_id);",
    "CREATE INDEX IF NOT EXISTS idx_dl_event ON dead_letters(event_id);",
]


# ============================================================================
# Dead Letter Queue
# ============================================================================

class DeadLetterQueue:
    """
    Persistent queue for failed events.

    Provides:
    - Storage of failed events with failure details
    - Configurable retry policies
    - Manual and automatic reprocessing
    - Cleanup of old entries
    """

    def __init__(self, config: Optional[DeadLetterConfig] = None):
        """
        Initialize dead letter queue.

        Args:
            config: Queue configuration
        """
        self.config = config or DeadLetterConfig()
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._retry_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            'events_added': 0,
            'events_retried': 0,
            'events_resolved': 0,
            'events_exhausted': 0
        }

        logger.info(f"DeadLetterQueue created (db={self.config.db_path})")

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        if self._initialized:
            return

        # Ensure directory exists
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._db = await aiosqlite.connect(str(db_path))
        await self._db.execute("PRAGMA journal_mode=WAL;")

        # Create schema
        await self._db.execute(CREATE_DEAD_LETTERS_TABLE)
        for index_sql in CREATE_INDEXES:
            await self._db.execute(index_sql)

        await self._db.commit()

        self._initialized = True
        logger.info("DeadLetterQueue initialized")

    async def add(
        self,
        event: SkillEvent,
        subscription_id: str,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a failed event to the dead letter queue.

        Args:
            event: The event that failed
            subscription_id: ID of the subscription that failed
            error: The exception that occurred
            metadata: Additional context

        Returns:
            Dead letter ID
        """
        if not self._initialized:
            await self.initialize()

        dl_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Calculate next retry time
        next_retry = None
        if self.config.retry_strategy != RetryStrategy.NONE:
            next_retry = self._calculate_next_retry(1)

        # Serialize event
        event_data = json.dumps({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'skill_name': event.skill_name,
            'topic': event.topic,
            'timestamp': event.timestamp,
            'correlation_id': event.correlation_id,
            'causation_id': event.causation_id,
            'sequence': event.sequence,
            'payload': event.payload,
            'metadata': event.metadata
        })

        await self._db.execute(
            """
            INSERT INTO dead_letters
            (id, event_id, event_data, subscription_id, error_message, error_type,
             failure_count, first_failure, last_failure, next_retry, status, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dl_id,
                event.event_id,
                event_data,
                subscription_id,
                str(error),
                type(error).__name__,
                1,
                now,
                now,
                next_retry,
                'pending',
                json.dumps(metadata or {}),
                now
            )
        )

        await self._db.commit()

        self.stats['events_added'] += 1
        logger.warning(f"Added to dead letter queue: {event.event_id} (subscription={subscription_id})")

        return dl_id

    async def record_retry_failure(
        self,
        dl_id: str,
        error: Exception
    ) -> bool:
        """
        Record a retry attempt failure.

        Args:
            dl_id: Dead letter ID
            error: The new exception

        Returns:
            True if retries remain, False if exhausted
        """
        if not self._initialized:
            await self.initialize()

        # Get current state
        cursor = await self._db.execute(
            "SELECT failure_count FROM dead_letters WHERE id = ?",
            (dl_id,)
        )
        row = await cursor.fetchone()

        if not row:
            return False

        failure_count = row[0] + 1
        now = datetime.now().isoformat()

        if failure_count >= self.config.max_retries:
            # Exhausted retries
            await self._db.execute(
                """
                UPDATE dead_letters
                SET failure_count = ?, last_failure = ?, error_message = ?,
                    status = 'exhausted', next_retry = NULL
                WHERE id = ?
                """,
                (failure_count, now, str(error), dl_id)
            )
            self.stats['events_exhausted'] += 1
            logger.error(f"Dead letter exhausted: {dl_id}")
            retries_remain = False
        else:
            # Schedule next retry
            next_retry = self._calculate_next_retry(failure_count)
            await self._db.execute(
                """
                UPDATE dead_letters
                SET failure_count = ?, last_failure = ?, error_message = ?,
                    next_retry = ?, status = 'pending'
                WHERE id = ?
                """,
                (failure_count, now, str(error), next_retry, dl_id)
            )
            retries_remain = True

        await self._db.commit()
        return retries_remain

    async def mark_resolved(self, dl_id: str) -> bool:
        """
        Mark a dead letter as resolved (successfully reprocessed).

        Args:
            dl_id: Dead letter ID

        Returns:
            True if found and updated
        """
        if not self._initialized:
            await self.initialize()

        cursor = await self._db.execute(
            """
            UPDATE dead_letters
            SET status = 'resolved', next_retry = NULL
            WHERE id = ?
            """,
            (dl_id,)
        )

        await self._db.commit()

        if cursor.rowcount > 0:
            self.stats['events_resolved'] += 1
            logger.info(f"Dead letter resolved: {dl_id}")
            return True
        return False

    async def get_pending(
        self,
        limit: int = 100,
        ready_only: bool = True
    ) -> List[DeadLetter]:
        """
        Get pending dead letters.

        Args:
            limit: Maximum entries to return
            ready_only: Only return entries ready for retry

        Returns:
            List of DeadLetter objects
        """
        if not self._initialized:
            await self.initialize()

        if ready_only:
            now = datetime.now().isoformat()
            cursor = await self._db.execute(
                """
                SELECT * FROM dead_letters
                WHERE status = 'pending' AND (next_retry IS NULL OR next_retry <= ?)
                ORDER BY first_failure ASC
                LIMIT ?
                """,
                (now, limit)
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM dead_letters
                WHERE status = 'pending'
                ORDER BY first_failure ASC
                LIMIT ?
                """,
                (limit,)
            )

        rows = await cursor.fetchall()
        return [self._row_to_dead_letter(row) for row in rows]

    async def get_by_subscription(
        self,
        subscription_id: str,
        include_resolved: bool = False
    ) -> List[DeadLetter]:
        """
        Get dead letters for a specific subscription.

        Args:
            subscription_id: Subscription ID
            include_resolved: Include resolved entries

        Returns:
            List of DeadLetter objects
        """
        if not self._initialized:
            await self.initialize()

        if include_resolved:
            cursor = await self._db.execute(
                "SELECT * FROM dead_letters WHERE subscription_id = ? ORDER BY first_failure DESC",
                (subscription_id,)
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM dead_letters
                WHERE subscription_id = ? AND status != 'resolved'
                ORDER BY first_failure DESC
                """,
                (subscription_id,)
            )

        rows = await cursor.fetchall()
        return [self._row_to_dead_letter(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self._initialized:
            await self.initialize()

        cursor = await self._db.execute(
            """
            SELECT status, COUNT(*) FROM dead_letters GROUP BY status
            """
        )
        rows = await cursor.fetchall()
        status_counts = dict(rows)

        return {
            **self.stats,
            'by_status': status_counts,
            'total_in_queue': sum(status_counts.values()),
            'config': {
                'max_retries': self.config.max_retries,
                'retry_strategy': self.config.retry_strategy.value,
                'retention_days': self.config.retention_days
            }
        }

    async def prune_old_entries(self, retention_days: Optional[int] = None) -> int:
        """
        Remove old resolved/exhausted entries.

        Args:
            retention_days: Days to retain (uses config default if not specified)

        Returns:
            Number of entries removed
        """
        if not self._initialized:
            await self.initialize()

        days = retention_days or self.config.retention_days
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = await self._db.execute(
            """
            DELETE FROM dead_letters
            WHERE status IN ('resolved', 'exhausted') AND created_at < ?
            """,
            (cutoff,)
        )

        await self._db.commit()

        deleted = cursor.rowcount
        logger.info(f"Pruned {deleted} old dead letters")
        return deleted

    async def start_auto_retry(
        self,
        retry_handler: Callable[[DeadLetter], Awaitable[bool]]
    ) -> None:
        """
        Start automatic retry background task.

        Args:
            retry_handler: Async function to retry an event, returns True on success
        """
        if not self.config.enable_auto_retry:
            logger.info("Auto-retry disabled")
            return

        if self._retry_task is not None:
            logger.warning("Auto-retry already running")
            return

        async def retry_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.auto_retry_interval_seconds)
                    await self._process_retries(retry_handler)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Auto-retry error: {e}")

        self._retry_task = asyncio.create_task(retry_loop())
        logger.info("Auto-retry started")

    async def stop_auto_retry(self) -> None:
        """Stop automatic retry background task."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
            self._retry_task = None
            logger.info("Auto-retry stopped")

    async def _process_retries(
        self,
        retry_handler: Callable[[DeadLetter], Awaitable[bool]]
    ) -> None:
        """Process pending retries."""
        pending = await self.get_pending(limit=50, ready_only=True)

        for dl in pending:
            try:
                success = await retry_handler(dl)
                if success:
                    await self.mark_resolved(dl.id)
                    self.stats['events_retried'] += 1
                else:
                    await self.record_retry_failure(dl.id, Exception("Retry returned False"))
            except Exception as e:
                await self.record_retry_failure(dl.id, e)

    def _calculate_next_retry(self, failure_count: int) -> str:
        """Calculate next retry timestamp."""
        if self.config.retry_strategy == RetryStrategy.FIXED:
            delay = self.config.retry_delay_seconds
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.retry_delay_seconds * (
                self.config.retry_multiplier ** (failure_count - 1)
            )
            delay = min(delay, self.config.max_retry_delay_seconds)
        else:
            return None

        next_retry = datetime.now() + timedelta(seconds=delay)
        return next_retry.isoformat()

    def _row_to_dead_letter(self, row: tuple) -> DeadLetter:
        """Convert database row to DeadLetter object."""
        event_data = json.loads(row[2])
        event = SkillEvent(
            event_id=event_data['event_id'],
            event_type=EventType(event_data['event_type']),
            skill_name=event_data['skill_name'],
            topic=event_data['topic'],
            timestamp=event_data['timestamp'],
            correlation_id=event_data.get('correlation_id'),
            causation_id=event_data.get('causation_id'),
            sequence=event_data.get('sequence', 0),
            payload=event_data.get('payload', {}),
            metadata=event_data.get('metadata', {})
        )

        return DeadLetter(
            id=row[0],
            event=event,
            subscription_id=row[3],
            error_message=row[4],
            error_type=row[5],
            failure_count=row[6],
            first_failure=row[7],
            last_failure=row[8],
            next_retry=row[9],
            status=row[10],
            metadata=json.loads(row[11])
        )

    async def close(self) -> None:
        """Close database connection."""
        await self.stop_auto_retry()

        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
            logger.info("DeadLetterQueue closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ============================================================================
# Factory Function
# ============================================================================

def create_dead_letter_queue(
    db_path: str = "dead_letters.db",
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
) -> DeadLetterQueue:
    """
    Create a dead letter queue instance.

    Args:
        db_path: Database file path
        max_retries: Maximum retry attempts
        retry_strategy: Retry timing strategy

    Returns:
        DeadLetterQueue instance
    """
    config = DeadLetterConfig(
        db_path=db_path,
        max_retries=max_retries,
        retry_strategy=retry_strategy
    )
    return DeadLetterQueue(config)
