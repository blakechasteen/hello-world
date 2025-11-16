#!/usr/bin/env python3
"""
Recursive Reasoning Analytics
==============================

Tracks performance metrics for recursive reasoning strategies.
Integrates with Promptly's analytics system to provide comprehensive
insights into strategy effectiveness, quality improvements, and costs.

Created: 2025-11-15
Integration: Promptly Analytics → HoloLoom

Metrics Tracked:
- Strategy usage frequency
- Average iterations per strategy
- Quality improvements (Δconfidence)
- Token usage and cost
- Success rates
- Latency per strategy
- Convergence rates

Usage:
    from HoloLoom.analytics.recursive_analytics import RecursiveAnalytics

    analytics = RecursiveAnalytics()

    # Track execution
    await analytics.track_execution(
        strategy=ReasoningStrategy.REFINE,
        query_text="Explain Thompson Sampling",
        iterations=3,
        initial_quality=0.75,
        final_quality=0.87,
        duration_ms=400,
        tokens_used=1500
    )

    # Get statistics
    stats = analytics.get_summary()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Avg quality gain: {stats['avg_quality_gain']:.2%}")
"""

import sqlite3
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

from HoloLoom.protocols.recursive_reasoning import ReasoningStrategy


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy"""
    strategy: str
    total_executions: int
    avg_iterations: float
    avg_quality_gain: float
    avg_duration_ms: float
    avg_tokens: int
    avg_cost: float
    success_rate: float  # % of executions that improved quality
    convergence_rate: float  # % that converged before max iterations


@dataclass
class ExecutionRecord:
    """Single execution record"""
    id: Optional[int]
    strategy: str
    query_text: str
    iterations: int
    initial_quality: float
    final_quality: float
    quality_gain: float
    duration_ms: float
    tokens_used: int
    cost: float
    converged: bool
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class RecursiveAnalytics:
    """
    Analytics tracker for recursive reasoning.

    Maintains SQLite database of all executions and provides
    aggregated statistics, trends, and insights.
    """

    def __init__(self, db_path: str = ".hololoom/recursive_analytics.db"):
        """
        Initialize analytics tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # In-memory cache for fast lookups
        self._cache = {
            "total_queries": 0,
            "strategy_counts": defaultdict(int),
            "last_refresh": 0
        }

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                query_text TEXT,
                iterations INTEGER,
                initial_quality REAL,
                final_quality REAL,
                quality_gain REAL,
                duration_ms REAL,
                tokens_used INTEGER,
                cost REAL,
                converged BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy
            ON executions(strategy)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON executions(timestamp)
        """)

        conn.commit()
        conn.close()

    async def track_execution(
        self,
        strategy: ReasoningStrategy,
        query_text: str,
        iterations: int,
        initial_quality: float,
        final_quality: float,
        duration_ms: float,
        tokens_used: int,
        converged: bool = False,
        cost_per_1k_tokens: float = 0.003,  # Claude Sonnet pricing
        metadata: Optional[Dict] = None
    ):
        """
        Track a single recursive reasoning execution.

        Args:
            strategy: Reasoning strategy used
            query_text: Query text
            iterations: Number of iterations executed
            initial_quality: Initial confidence score
            final_quality: Final confidence score after refinement
            duration_ms: Total duration in milliseconds
            tokens_used: Total tokens consumed
            converged: Whether reasoning converged before max iterations
            cost_per_1k_tokens: Cost per 1000 tokens
            metadata: Additional metadata (optional)
        """
        quality_gain = final_quality - initial_quality
        cost = (tokens_used / 1000) * cost_per_1k_tokens

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO executions (
                strategy, query_text, iterations,
                initial_quality, final_quality, quality_gain,
                duration_ms, tokens_used, cost, converged, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.value,
            query_text,
            iterations,
            initial_quality,
            final_quality,
            quality_gain,
            duration_ms,
            tokens_used,
            cost,
            converged,
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()
        conn.close()

        # Invalidate cache
        self._cache["last_refresh"] = 0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall analytics summary.

        Returns:
            Dictionary with overall statistics:
            - total_queries: Total number of queries
            - total_cost: Total cost across all queries
            - avg_quality_gain: Average quality improvement
            - avg_iterations: Average iterations per query
            - avg_duration_ms: Average latency
            - strategies: Per-strategy breakdown
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(quality_gain) as avg_gain,
                AVG(iterations) as avg_iterations,
                AVG(duration_ms) as avg_duration,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens
            FROM executions
        """)

        row = cursor.fetchone()
        total, avg_gain, avg_iterations, avg_duration, total_cost, total_tokens = row

        # Per-strategy breakdown
        cursor.execute("""
            SELECT
                strategy,
                COUNT(*) as count,
                AVG(iterations) as avg_iterations,
                AVG(quality_gain) as avg_gain,
                AVG(duration_ms) as avg_duration,
                AVG(tokens_used) as avg_tokens,
                AVG(cost) as avg_cost,
                SUM(CASE WHEN quality_gain > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as convergence_rate
            FROM executions
            GROUP BY strategy
        """)

        strategies = {}
        for row in cursor.fetchall():
            strat, count, avg_iter, avg_gain_s, avg_dur, avg_tok, avg_cost_s, success, convergence = row
            strategies[strat] = {
                "count": count,
                "avg_iterations": round(avg_iter, 1),
                "avg_quality_gain": round(avg_gain_s, 3),
                "avg_duration_ms": round(avg_dur, 1),
                "avg_tokens": int(avg_tok),
                "avg_cost": round(avg_cost_s, 4),
                "success_rate": round(success, 1),
                "convergence_rate": round(convergence, 1)
            }

        conn.close()

        return {
            "total_queries": total or 0,
            "avg_quality_gain": round(avg_gain or 0, 3),
            "avg_iterations": round(avg_iterations or 0, 1),
            "avg_duration_ms": round(avg_duration or 0, 1),
            "total_cost": round(total_cost or 0, 2),
            "total_tokens": total_tokens or 0,
            "strategies": strategies
        }

    def get_strategy_metrics(self, strategy: ReasoningStrategy) -> StrategyMetrics:
        """
        Get detailed metrics for a specific strategy.

        Args:
            strategy: Strategy to analyze

        Returns:
            StrategyMetrics with detailed statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(iterations) as avg_iterations,
                AVG(quality_gain) as avg_gain,
                AVG(duration_ms) as avg_duration,
                AVG(tokens_used) as avg_tokens,
                AVG(cost) as avg_cost,
                SUM(CASE WHEN quality_gain > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as convergence_rate
            FROM executions
            WHERE strategy = ?
        """, (strategy.value,))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return StrategyMetrics(
                strategy=strategy.value,
                total_executions=0,
                avg_iterations=0,
                avg_quality_gain=0,
                avg_duration_ms=0,
                avg_tokens=0,
                avg_cost=0,
                success_rate=0,
                convergence_rate=0
            )

        total, avg_iter, avg_gain, avg_dur, avg_tok, avg_cost, success, convergence = row

        return StrategyMetrics(
            strategy=strategy.value,
            total_executions=total,
            avg_iterations=round(avg_iter, 1),
            avg_quality_gain=round(avg_gain, 3),
            avg_duration_ms=round(avg_dur, 1),
            avg_tokens=int(avg_tok),
            avg_cost=round(avg_cost, 4),
            success_rate=round(success, 1),
            convergence_rate=round(convergence, 1)
        )

    def get_quality_trends(
        self,
        strategy: Optional[ReasoningStrategy] = None,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get quality improvement trends over time.

        Args:
            strategy: Optional specific strategy (None = all strategies)
            days: Number of days to look back

        Returns:
            List of daily quality statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        if strategy:
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    AVG(quality_gain) as avg_gain,
                    AVG(final_quality) as avg_quality,
                    COUNT(*) as count
                FROM executions
                WHERE strategy = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (strategy.value, cutoff))
        else:
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    AVG(quality_gain) as avg_gain,
                    AVG(final_quality) as avg_quality,
                    COUNT(*) as count
                FROM executions
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff,))

        trends = []
        for row in cursor.fetchall():
            date, avg_gain, avg_quality, count = row
            trends.append({
                "date": date,
                "avg_quality_gain": round(avg_gain, 3),
                "avg_final_quality": round(avg_quality, 3),
                "count": count
            })

        conn.close()
        return trends

    def get_total_executions_count(self, strategy: Optional[ReasoningStrategy] = None) -> int:
        """
        Get total count of executions.

        Args:
            strategy: Optional filter by strategy

        Returns:
            Total number of execution records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if strategy:
            cursor.execute("""
                SELECT COUNT(*) FROM executions WHERE strategy = ?
            """, (strategy.value,))
        else:
            cursor.execute("SELECT COUNT(*) FROM executions")

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_recent_executions(
        self,
        limit: int = 10,
        skip: int = 0,
        strategy: Optional[ReasoningStrategy] = None
    ) -> List[ExecutionRecord]:
        """
        Get recent execution records with pagination support.

        Args:
            limit: Maximum number of records per page
            skip: Number of records to skip (offset)
            strategy: Optional filter by strategy

        Returns:
            List of ExecutionRecords
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if strategy:
            cursor.execute("""
                SELECT * FROM executions
                WHERE strategy = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (strategy.value, limit, skip))
        else:
            cursor.execute("""
                SELECT * FROM executions
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, skip))

        records = []
        for row in cursor.fetchall():
            records.append(ExecutionRecord(
                id=row["id"],
                strategy=row["strategy"],
                query_text=row["query_text"],
                iterations=row["iterations"],
                initial_quality=row["initial_quality"],
                final_quality=row["final_quality"],
                quality_gain=row["quality_gain"],
                duration_ms=row["duration_ms"],
                tokens_used=row["tokens_used"],
                cost=row["cost"],
                converged=bool(row["converged"]),
                timestamp=row["timestamp"]
            ))

        conn.close()
        return records

    def get_recommendations(self) -> List[str]:
        """
        Generate AI-powered recommendations based on analytics.

        Returns:
            List of actionable recommendations
        """
        summary = self.get_summary()
        recommendations = []

        # Check overall usage
        if summary["total_queries"] < 10:
            recommendations.append(
                "Limited data available. Run more queries to get better insights."
            )
            return recommendations

        strategies = summary["strategies"]

        # Find best performing strategy
        if strategies:
            best_strategy = max(
                strategies.items(),
                key=lambda x: x[1]["avg_quality_gain"]
            )
            recommendations.append(
                f"Best performing strategy: {best_strategy[0]} "
                f"(avg gain: {best_strategy[1]['avg_quality_gain']:.1%})"
            )

            # Find most cost-effective
            cost_effective = min(
                strategies.items(),
                key=lambda x: x[1]["avg_cost"] / (x[1]["avg_quality_gain"] + 0.01)
            )
            recommendations.append(
                f"Most cost-effective: {cost_effective[0]} "
                f"(${cost_effective[1]['avg_cost']:.4f} per query)"
            )

            # Find fastest
            fastest = min(
                strategies.items(),
                key=lambda x: x[1]["avg_duration_ms"]
            )
            recommendations.append(
                f"Fastest strategy: {fastest[0]} "
                f"({fastest[1]['avg_duration_ms']:.0f}ms avg)"
            )

        # Cost warnings
        if summary["total_cost"] > 10:
            recommendations.append(
                f"⚠️ High cost alert: ${summary['total_cost']:.2f} total. "
                f"Consider using quality_threshold to reduce unnecessary refinement."
            )

        # Quality insights
        if summary["avg_quality_gain"] < 0.05:
            recommendations.append(
                "⚠️ Low quality improvements. Consider adjusting quality_threshold "
                "or trying different strategies."
            )

        return recommendations

    def export_to_csv(self, output_path: str):
        """
        Export all executions to CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM executions ORDER BY timestamp")

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "id", "strategy", "query_text", "iterations",
                "initial_quality", "final_quality", "quality_gain",
                "duration_ms", "tokens_used", "cost", "converged", "timestamp"
            ])

            # Data
            for row in cursor.fetchall():
                writer.writerow(row)

        conn.close()

    def get_comparison_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get comparison matrix of all strategies.

        Returns:
            Dictionary mapping strategy pairs to relative performance
        """
        strategies = self.get_summary()["strategies"]

        if not strategies:
            return {}

        matrix = {}
        for strat1 in strategies:
            matrix[strat1] = {}
            for strat2 in strategies:
                if strat1 == strat2:
                    matrix[strat1][strat2] = 1.0
                else:
                    # Compare quality gain
                    gain1 = strategies[strat1]["avg_quality_gain"]
                    gain2 = strategies[strat2]["avg_quality_gain"]

                    if gain2 > 0:
                        matrix[strat1][strat2] = gain1 / gain2
                    else:
                        matrix[strat1][strat2] = 0.0

        return matrix

    def get_quality_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get quality improvement trend over time (hourly buckets).

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            List of hourly quality statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                AVG(quality_gain) as avg_quality_gain,
                COUNT(*) as count,
                MIN(quality_gain) as min_gain,
                MAX(quality_gain) as max_gain
            FROM executions
            WHERE timestamp >= datetime('now', ?)
            GROUP BY hour
            ORDER BY hour ASC
        """, (f'-{hours} hours',))

        trends = []
        for row in cursor.fetchall():
            hour, avg_gain, count, min_gain, max_gain = row
            trends.append({
                'timestamp': hour,
                'avg_quality_gain': round(avg_gain, 3) if avg_gain else 0,
                'count': count,
                'min_gain': round(min_gain, 3) if min_gain else 0,
                'max_gain': round(max_gain, 3) if max_gain else 0
            })

        conn.close()
        return trends

    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by strategy (all-time).

        Returns:
            List of strategy performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                strategy,
                AVG(quality_gain) as avg_quality_gain,
                AVG(iterations) as avg_iterations,
                COUNT(*) as count,
                SUM(cost) as total_cost,
                AVG(duration_ms) as avg_duration,
                SUM(CASE WHEN quality_gain > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM executions
            GROUP BY strategy
            ORDER BY avg_quality_gain DESC
        """)

        performances = []
        for row in cursor.fetchall():
            strategy, avg_gain, avg_iter, count, total_cost, avg_duration, success_rate = row
            performances.append({
                'strategy': strategy,
                'avg_quality_gain': round(avg_gain, 3) if avg_gain else 0,
                'avg_iterations': round(avg_iter, 1) if avg_iter else 0,
                'count': count,
                'total_cost': round(total_cost, 2) if total_cost else 0,
                'avg_duration_ms': round(avg_duration, 1) if avg_duration else 0,
                'success_rate': round(success_rate, 1) if success_rate else 0
            })

        conn.close()
        return performances

    def get_query_volume_trend(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get query volume trend by day.

        Args:
            days: Number of days to look back (default: 7)

        Returns:
            List of daily query counts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                DATE(timestamp) as day,
                COUNT(*) as count,
                AVG(quality_gain) as avg_quality_gain,
                AVG(duration_ms) as avg_duration
            FROM executions
            WHERE timestamp >= datetime('now', ?)
            GROUP BY day
            ORDER BY day ASC
        """, (f'-{days} days',))

        trends = []
        for row in cursor.fetchall():
            day, count, avg_quality_gain, avg_duration = row
            trends.append({
                'date': day,
                'count': count,
                'avg_quality_gain': round(avg_quality_gain, 3) if avg_quality_gain else 0,
                'avg_duration_ms': round(avg_duration, 1) if avg_duration else 0
            })

        conn.close()
        return trends

    def get_iteration_distribution(self) -> Dict[str, int]:
        """
        Get distribution of iteration counts.

        Returns:
            Dictionary with iteration count buckets and frequencies
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                CASE
                    WHEN iterations = 1 THEN '1 iteration'
                    WHEN iterations = 2 THEN '2 iterations'
                    WHEN iterations = 3 THEN '3 iterations'
                    ELSE '4+ iterations'
                END as bucket,
                COUNT(*) as count
            FROM executions
            GROUP BY bucket
            ORDER BY
                CASE
                    WHEN iterations = 1 THEN 1
                    WHEN iterations = 2 THEN 2
                    WHEN iterations = 3 THEN 3
                    ELSE 4
                END
        """)

        distribution = {}
        for row in cursor.fetchall():
            bucket, count = row
            distribution[bucket] = count

        conn.close()
        return distribution

    def get_cost_by_strategy(self) -> List[Dict[str, Any]]:
        """
        Get cost breakdown by strategy.

        Returns:
            List of strategies with token and cost metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Token costs: input ~$0.003/1M, output ~$0.006/1M tokens
        # Using simplified model: avg cost per query
        cursor.execute("""
            SELECT
                strategy,
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                COUNT(*) as count,
                AVG(tokens_used) as avg_tokens,
                AVG(cost) as avg_cost
            FROM executions
            GROUP BY strategy
            ORDER BY total_cost DESC
        """)

        costs = []
        for row in cursor.fetchall():
            strategy, total_tokens, total_cost, count, avg_tokens, avg_cost = row
            # Estimate input vs output (typically 30% input, 70% output)
            input_cost = (total_cost or 0) * 0.3
            output_cost = (total_cost or 0) * 0.7

            costs.append({
                'strategy': strategy,
                'total_tokens': int(total_tokens) if total_tokens else 0,
                'total_cost': round(total_cost, 2) if total_cost else 0,
                'input_cost': round(input_cost, 2),
                'output_cost': round(output_cost, 2),
                'count': count,
                'avg_tokens': int(avg_tokens) if avg_tokens else 0,
                'avg_cost': round(avg_cost, 4) if avg_cost else 0
            })

        conn.close()
        return costs
