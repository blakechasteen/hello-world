"""
Phase 3: Production Validation - Data Collection Framework

This module provides infrastructure for collecting production query data, quality metrics,
and user feedback for MRF validation.

Created: November 2025
Status: Production validation framework
"""

import asyncio
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class QuerySource(Enum):
    """Source of query data."""
    PRODUCTION = "production"  # Live production system
    SYNTHETIC = "synthetic"  # Generated test queries
    HISTORICAL = "historical"  # Historical production logs


class QualityMetric(Enum):
    """Quality metrics to collect."""
    ACCURACY = "accuracy"  # Factual correctness (0.0-1.0)
    COMPLETENESS = "completeness"  # Answer completeness (0.0-1.0)
    CLARITY = "clarity"  # Response clarity (0.0-1.0)
    RELEVANCE = "relevance"  # Relevance to query (0.0-1.0)
    CONCISENESS = "conciseness"  # Not overly verbose (0.0-1.0)


@dataclass
class ProductionQuery:
    """A production query with context and metadata."""

    query_id: str  # Unique identifier
    text: str  # Query text
    source: QuerySource
    timestamp: datetime = field(default_factory=datetime.now)

    # Context (optional)
    user_id: str | None = None
    session_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    # System used (for baseline collection)
    system_variant: str | None = None  # "traditional" or "unified_mrf"

    # Response
    response: str | None = None
    latency_ms: float | None = None

    # Quality metrics (auto-computed or human-rated)
    quality_metrics: dict[str, float] = field(default_factory=dict)

    # User feedback
    user_rating: int | None = None  # 1-5 stars
    user_feedback: str | None = None
    thumbs_up: bool | None = None  # Simple +1/-1 feedback

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['source'] = self.source.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProductionQuery':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['source'] = QuerySource(data['source'])
        return cls(**data)


class ProductionDataCollector:
    """
    Collects production query data and stores in SQLite database.

    Usage:
        collector = ProductionDataCollector("production_data.db")

        # Log query
        await collector.log_query(
            query_id="q_12345",
            text="What is Thompson Sampling?",
            source=QuerySource.PRODUCTION,
            system_variant="unified_mrf"
        )

        # Log response
        await collector.log_response(
            query_id="q_12345",
            response="Thompson Sampling is...",
            latency_ms=150.5
        )

        # Log quality metrics
        await collector.log_quality_metrics(
            query_id="q_12345",
            metrics={"accuracy": 0.95, "completeness": 0.90}
        )

        # Get queries for validation
        queries = await collector.get_queries(
            source=QuerySource.PRODUCTION,
            start_date=datetime.now() - timedelta(days=7)
        )
    """

    def __init__(self, db_path: str = "production_data.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Queries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                context TEXT,
                system_variant TEXT,
                response TEXT,
                latency_ms REAL,
                user_rating INTEGER,
                user_feedback TEXT,
                thumbs_up INTEGER
            )
        """)

        # Quality metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries(query_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_source ON queries(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_variant ON queries(system_variant)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_query ON quality_metrics(query_id)")

        conn.commit()
        conn.close()

    async def log_query(
        self,
        query_id: str,
        text: str,
        source: QuerySource,
        system_variant: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None
    ):
        """
        Log a production query.

        Args:
            query_id: Unique query identifier
            text: Query text
            source: QuerySource enum
            system_variant: "traditional" or "unified_mrf"
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional context dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO queries
            (query_id, text, source, timestamp, user_id, session_id, context, system_variant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_id,
            text,
            source.value,
            datetime.now().isoformat(),
            user_id,
            session_id,
            json.dumps(context) if context else None,
            system_variant
        ))

        conn.commit()
        conn.close()

    async def log_response(
        self,
        query_id: str,
        response: str,
        latency_ms: float
    ):
        """
        Log response for a query.

        Args:
            query_id: Query identifier
            response: Generated response
            latency_ms: Response latency in milliseconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE queries
            SET response = ?, latency_ms = ?
            WHERE query_id = ?
        """, (response, latency_ms, query_id))

        conn.commit()
        conn.close()

    async def log_quality_metrics(
        self,
        query_id: str,
        metrics: dict[str, float]
    ):
        """
        Log quality metrics for a query.

        Args:
            query_id: Query identifier
            metrics: Dictionary of metric_name -> value (0.0-1.0)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO quality_metrics
                (query_id, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (query_id, metric_name, metric_value, timestamp))

        conn.commit()
        conn.close()

    async def log_user_feedback(
        self,
        query_id: str,
        rating: int | None = None,
        feedback: str | None = None,
        thumbs_up: bool | None = None
    ):
        """
        Log user feedback for a query.

        Args:
            query_id: Query identifier
            rating: 1-5 star rating
            feedback: Text feedback
            thumbs_up: Simple thumbs up/down
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE queries
            SET user_rating = ?, user_feedback = ?, thumbs_up = ?
            WHERE query_id = ?
        """, (rating, feedback, 1 if thumbs_up else 0 if thumbs_up is not None else None, query_id))

        conn.commit()
        conn.close()

    async def get_queries(
        self,
        source: QuerySource | None = None,
        system_variant: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None
    ) -> list[ProductionQuery]:
        """
        Retrieve queries from database.

        Args:
            source: Filter by QuerySource
            system_variant: Filter by system variant
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of queries to return

        Returns:
            List of ProductionQuery objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)

        if system_variant:
            conditions.append("system_variant = ?")
            params.append(system_variant)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT query_id, text, source, timestamp, user_id, session_id,
                   context, system_variant, response, latency_ms,
                   user_rating, user_feedback, thumbs_up
            FROM queries
            WHERE {where_clause}
            ORDER BY timestamp DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to ProductionQuery objects
        queries = []
        for row in rows:
            query_obj = ProductionQuery(
                query_id=row[0],
                text=row[1],
                source=QuerySource(row[2]),
                timestamp=datetime.fromisoformat(row[3]),
                user_id=row[4],
                session_id=row[5],
                context=json.loads(row[6]) if row[6] else {},
                system_variant=row[7],
                response=row[8],
                latency_ms=row[9],
                user_rating=row[10],
                user_feedback=row[11],
                thumbs_up=bool(row[12]) if row[12] is not None else None
            )

            # Fetch quality metrics
            cursor.execute("""
                SELECT metric_name, metric_value
                FROM quality_metrics
                WHERE query_id = ?
            """, (query_obj.query_id,))

            metrics_rows = cursor.fetchall()
            query_obj.quality_metrics = {name: value for name, value in metrics_rows}

            queries.append(query_obj)

        conn.close()
        return queries

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total queries
        cursor.execute("SELECT COUNT(*) FROM queries")
        total_queries = cursor.fetchone()[0]

        # By source
        cursor.execute("""
            SELECT source, COUNT(*)
            FROM queries
            GROUP BY source
        """)
        by_source = {source: count for source, count in cursor.fetchall()}

        # By variant
        cursor.execute("""
            SELECT system_variant, COUNT(*)
            FROM queries
            WHERE system_variant IS NOT NULL
            GROUP BY system_variant
        """)
        by_variant = {variant: count for variant, count in cursor.fetchall()}

        # Average latency
        cursor.execute("""
            SELECT system_variant, AVG(latency_ms)
            FROM queries
            WHERE latency_ms IS NOT NULL
            GROUP BY system_variant
        """)
        avg_latency = {variant: latency for variant, latency in cursor.fetchall()}

        # Average rating
        cursor.execute("""
            SELECT system_variant, AVG(user_rating)
            FROM queries
            WHERE user_rating IS NOT NULL
            GROUP BY system_variant
        """)
        avg_rating = {variant: rating for variant, rating in cursor.fetchall()}

        conn.close()

        return {
            "total_queries": total_queries,
            "by_source": by_source,
            "by_variant": by_variant,
            "avg_latency_ms": avg_latency,
            "avg_rating": avg_rating
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Create collector
        collector = ProductionDataCollector("production_data.db")

        # Simulate logging some queries
        queries = [
            ("q_001", "What is Thompson Sampling?", "unified_mrf"),
            ("q_002", "Explain Bayesian optimization", "traditional"),
            ("q_003", "How does UCB work?", "unified_mrf"),
            ("q_004", "Compare multi-armed bandits", "traditional"),
        ]

        for query_id, text, variant in queries:
            await collector.log_query(
                query_id=query_id,
                text=text,
                source=QuerySource.PRODUCTION,
                system_variant=variant
            )

            # Simulate response
            response = f"Response to: {text}"
            latency = 150.0 if variant == "unified_mrf" else 140.0

            await collector.log_response(query_id, response, latency)

            # Simulate quality metrics
            accuracy = 0.92 if variant == "unified_mrf" else 0.75
            completeness = 0.88 if variant == "unified_mrf" else 0.72

            await collector.log_quality_metrics(query_id, {
                "accuracy": accuracy,
                "completeness": completeness
            })

            # Simulate user feedback
            rating = 5 if variant == "unified_mrf" else 3
            await collector.log_user_feedback(query_id, rating=rating, thumbs_up=(rating >= 4))

        # Get statistics
        stats = await collector.get_statistics()
        print("\nDatabase Statistics:")
        print(json.dumps(stats, indent=2))

        # Get queries for validation
        print("\nUnifiedMRF queries:")
        mrf_queries = await collector.get_queries(system_variant="unified_mrf")
        for q in mrf_queries:
            print(f"  {q.query_id}: {q.text[:50]}... (latency: {q.latency_ms:.1f}ms, rating: {q.user_rating}/5)")

    asyncio.run(main())
