"""
HoloLoom Feedback Store
=======================
Persistent storage for user feedback to enable reflection learning.

The Feedback Store captures user ratings and explicit feedback on responses,
enabling the system to learn what works and adapt over time through:
- Thompson Sampling prior updates
- Policy weight adjustments
- Pattern quality tracking
- A/B testing foundation

Architecture:
- SQLite backend (lightweight, zero dependencies)
- Async/await throughout
- Bayesian learning signals
- Complete interaction history

Author: Claude Code (with HoloLoom by Blake)
Date: 2025-11-20
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FeedbackRecord:
    """
    A single feedback record from a user interaction.

    Captures the complete context of a query-response interaction
    and the user's rating/feedback.
    """
    # Primary key
    feedback_id: str

    # Query context
    query_text: str
    response_text: str
    tool_used: str
    confidence: float

    # User feedback
    user_rating: float  # 0.0 (bad) to 1.0 (excellent)
    feedback_type: str  # "helpful", "not_helpful", "rating", "explicit"
    explicit_feedback: str | None = None  # Optional text feedback

    # Context
    user_id: str = "@unknown:matrix.org"
    timestamp: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'feedback_id': self.feedback_id,
            'query_text': self.query_text,
            'response_text': self.response_text,
            'tool_used': self.tool_used,
            'confidence': self.confidence,
            'user_rating': self.user_rating,
            'feedback_type': self.feedback_type,
            'explicit_feedback': self.explicit_feedback,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata)
        }


@dataclass
class LearningSignals:
    """
    Aggregated learning signals for a tool or pattern.

    Used to update Thompson Sampling priors and policy weights.
    """
    tool_name: str

    # Success statistics
    total_samples: int
    successful_samples: int  # rating >= 0.7
    failed_samples: int  # rating < 0.7

    # Quality metrics
    avg_rating: float
    avg_confidence: float
    success_rate: float

    # Thompson Sampling priors
    alpha: float  # Successes + 1 (prior)
    beta: float   # Failures + 1 (prior)
    expected_reward: float  # E[X] = α / (α + β)

    # Time window
    window_start: datetime
    window_end: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'tool_name': self.tool_name,
            'total_samples': self.total_samples,
            'successful_samples': self.successful_samples,
            'failed_samples': self.failed_samples,
            'avg_rating': self.avg_rating,
            'avg_confidence': self.avg_confidence,
            'success_rate': self.success_rate,
            'alpha': self.alpha,
            'beta': self.beta,
            'expected_reward': self.expected_reward,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat()
        }


# ============================================================================
# Feedback Store
# ============================================================================

class FeedbackStore:
    """
    Persistent storage for user feedback with learning signal extraction.

    Stores feedback in SQLite and provides methods to:
    - Record user feedback on responses
    - Extract learning signals per tool
    - Update Thompson Sampling priors
    - Query feedback history

    Usage:
        async with FeedbackStore() as store:
            feedback_id = await store.store_feedback(
                query="What is Thompson Sampling?",
                response="Thompson Sampling is...",
                tool_used="answer",
                confidence=0.92,
                user_rating=1.0,
                feedback_type="helpful",
                user_id="@alice:matrix.org"
            )

            signals = await store.get_learning_signals("answer")
            alpha, beta = await store.update_thompson_priors("answer", feedback_id)
    """

    def __init__(self, db_path: str = "./data/feedback.db"):
        """
        Initialize feedback store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db: aiosqlite.Connection | None = None

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize database connection and schema."""
        self.db = await aiosqlite.connect(str(self.db_path))
        await self._create_schema()
        logger.info(f"FeedbackStore initialized at {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()
            logger.info("FeedbackStore closed")

    async def _create_schema(self):
        """Create database schema if not exists."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                tool_used TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_rating REAL NOT NULL,
                feedback_type TEXT NOT NULL,
                explicit_feedback TEXT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Create index on tool_used for fast lookups
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_used ON feedback(tool_used)
        """)

        # Create index on timestamp for time-based queries
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
        """)

        # Create index on user_id for per-user analytics
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON feedback(user_id)
        """)

        await self.db.commit()

    async def store_feedback(
        self,
        query: str,
        response: str,
        tool_used: str,
        confidence: float,
        user_rating: float,
        feedback_type: str,
        user_id: str = "@unknown:matrix.org",
        explicit_feedback: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Store user feedback on a response.

        Args:
            query: Original query text
            response: Generated response text
            tool_used: Tool that generated the response
            confidence: System confidence (0.0-1.0)
            user_rating: User rating (0.0-1.0, where 0.7+ is success)
            feedback_type: Type of feedback (helpful, not_helpful, rating, explicit)
            user_id: User who provided feedback
            explicit_feedback: Optional text feedback from user
            metadata: Optional additional metadata

        Returns:
            feedback_id: Unique ID for this feedback record
        """
        # Generate unique feedback ID
        import hashlib
        feedback_id = hashlib.md5(
            f"{query}{response}{user_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Create record
        record = FeedbackRecord(
            feedback_id=feedback_id,
            query_text=query,
            response_text=response,
            tool_used=tool_used,
            confidence=confidence,
            user_rating=user_rating,
            feedback_type=feedback_type,
            explicit_feedback=explicit_feedback,
            user_id=user_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Insert into database
        await self.db.execute("""
            INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.feedback_id,
            record.query_text,
            record.response_text,
            record.tool_used,
            record.confidence,
            record.user_rating,
            record.feedback_type,
            record.explicit_feedback,
            record.user_id,
            record.timestamp.isoformat(),
            json.dumps(record.metadata)
        ))

        await self.db.commit()

        logger.info(
            f"Feedback stored: {feedback_id} "
            f"(tool={tool_used}, rating={user_rating:.2f})"
        )

        return feedback_id

    async def get_learning_signals(
        self,
        tool: str | None = None,
        min_samples: int = 10,
        time_window: timedelta | None = None
    ) -> dict[str, LearningSignals]:
        """
        Extract learning signals for tools.

        Args:
            tool: Specific tool to analyze (None = all tools)
            min_samples: Minimum samples required to return signals
            time_window: Only consider feedback within this window (None = all time)

        Returns:
            Dict mapping tool name to LearningSignals
        """
        # Build query
        query = "SELECT tool_used, confidence, user_rating, timestamp FROM feedback"
        params = []

        if tool:
            query += " WHERE tool_used = ?"
            params.append(tool)

        if time_window:
            cutoff = (datetime.now() - time_window).isoformat()
            if tool:
                query += " AND timestamp >= ?"
            else:
                query += " WHERE timestamp >= ?"
            params.append(cutoff)

        # Fetch all feedback
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        # Aggregate by tool
        tool_data = defaultdict(list)
        for row in rows:
            tool_name, conf, rating, ts = row
            tool_data[tool_name].append({
                'confidence': conf,
                'rating': rating,
                'timestamp': datetime.fromisoformat(ts)
            })

        # Calculate signals per tool
        signals = {}
        for tool_name, data in tool_data.items():
            if len(data) < min_samples:
                continue

            # Extract ratings and confidence
            ratings = np.array([d['rating'] for d in data])
            confidences = np.array([d['confidence'] for d in data])
            timestamps = [d['timestamp'] for d in data]

            # Calculate statistics
            total = len(ratings)
            successes = int(np.sum(ratings >= 0.7))
            failures = total - successes

            avg_rating = float(np.mean(ratings))
            avg_confidence = float(np.mean(confidences))
            success_rate = successes / total

            # Thompson Sampling priors (Beta distribution)
            # Start with uniform prior (1, 1)
            alpha = successes + 1
            beta = failures + 1
            expected_reward = alpha / (alpha + beta)

            signals[tool_name] = LearningSignals(
                tool_name=tool_name,
                total_samples=total,
                successful_samples=successes,
                failed_samples=failures,
                avg_rating=avg_rating,
                avg_confidence=avg_confidence,
                success_rate=success_rate,
                alpha=alpha,
                beta=beta,
                expected_reward=expected_reward,
                window_start=min(timestamps),
                window_end=max(timestamps)
            )

        logger.info(f"Extracted learning signals for {len(signals)} tools")
        return signals

    async def update_thompson_priors(
        self,
        tool: str,
        feedback_id: str
    ) -> tuple[float, float]:
        """
        Update Thompson Sampling priors based on feedback.

        Args:
            tool: Tool name
            feedback_id: Feedback record ID

        Returns:
            Tuple of (alpha, beta) priors after update
        """
        # Get feedback record
        cursor = await self.db.execute(
            "SELECT user_rating FROM feedback WHERE feedback_id = ?",
            (feedback_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise ValueError(f"Feedback record not found: {feedback_id}")

        rating = row[0]

        # Get current signals
        signals = await self.get_learning_signals(tool=tool, min_samples=1)

        if tool not in signals:
            # New tool, start with uniform prior
            alpha, beta = 1.0, 1.0
        else:
            alpha = signals[tool].alpha
            beta = signals[tool].beta

        # Update based on rating
        if rating >= 0.7:
            # Success: increment alpha
            alpha += rating
        else:
            # Failure: increment beta
            beta += (1.0 - rating)

        logger.info(
            f"Thompson priors updated for {tool}: "
            f"α={alpha:.2f}, β={beta:.2f}, E[X]={alpha/(alpha+beta):.3f}"
        )

        return alpha, beta

    async def get_feedback_history(
        self,
        tool: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[FeedbackRecord]:
        """
        Query feedback history with filtering.

        Args:
            tool: Filter by tool name
            user_id: Filter by user
            limit: Maximum records to return
            offset: Skip first N records

        Returns:
            List of FeedbackRecord objects
        """
        query = "SELECT * FROM feedback"
        params = []
        conditions = []

        if tool:
            conditions.append("tool_used = ?")
            params.append(tool)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        # Convert to FeedbackRecord objects
        records = []
        for row in rows:
            records.append(FeedbackRecord(
                feedback_id=row[0],
                query_text=row[1],
                response_text=row[2],
                tool_used=row[3],
                confidence=row[4],
                user_rating=row[5],
                feedback_type=row[6],
                explicit_feedback=row[7],
                user_id=row[8],
                timestamp=datetime.fromisoformat(row[9]),
                metadata=json.loads(row[10]) if row[10] else {}
            ))

        return records

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get overall feedback statistics.

        Returns:
            Dict with comprehensive statistics
        """
        # Total feedback count
        cursor = await self.db.execute("SELECT COUNT(*) FROM feedback")
        total_count = (await cursor.fetchone())[0]

        # Per-tool statistics
        cursor = await self.db.execute("""
            SELECT tool_used, COUNT(*), AVG(user_rating), AVG(confidence)
            FROM feedback
            GROUP BY tool_used
        """)
        tool_stats = {}
        for row in await cursor.fetchall():
            tool_name, count, avg_rating, avg_conf = row
            tool_stats[tool_name] = {
                'count': count,
                'avg_rating': avg_rating,
                'avg_confidence': avg_conf
            }

        # Feedback type distribution
        cursor = await self.db.execute("""
            SELECT feedback_type, COUNT(*)
            FROM feedback
            GROUP BY feedback_type
        """)
        feedback_types = {row[0]: row[1] for row in await cursor.fetchall()}

        # Recent activity (last 24 hours)
        cutoff = (datetime.now() - timedelta(days=1)).isoformat()
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM feedback WHERE timestamp >= ?",
            (cutoff,)
        )
        recent_count = (await cursor.fetchone())[0]

        # Get learning signals
        signals = await self.get_learning_signals(min_samples=1)

        return {
            'total_feedback_count': total_count,
            'recent_feedback_24h': recent_count,
            'tool_statistics': tool_stats,
            'feedback_types': feedback_types,
            'learning_signals': {
                tool: signals_obj.to_dict()
                for tool, signals_obj in signals.items()
            }
        }
