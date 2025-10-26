#!/usr/bin/env python3
"""
Prompt Analytics System
=======================
Track prompt performance, costs, and quality over time.

Features:
- Execution time tracking
- Quality score trends
- Cost per prompt
- Success/failure rates
- Best performing prompts
- Recommendations based on history
"""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict


@dataclass
class PromptExecution:
    """Single prompt execution record"""
    prompt_id: str  # Unique ID for the prompt
    prompt_name: str
    execution_time: float  # seconds
    quality_score: Optional[float] = None  # 0.0-1.0
    success: bool = True
    model: str = "unknown"
    backend: str = "unknown"
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptStats:
    """Aggregated statistics for a prompt"""
    prompt_name: str
    total_executions: int
    success_rate: float
    avg_execution_time: float
    avg_quality_score: Optional[float]
    total_cost: float
    total_tokens: int
    best_execution_time: float
    worst_execution_time: float
    trend: str  # "improving", "stable", "degrading"


class PromptAnalytics:
    """
    Track and analyze prompt performance.

    Uses SQLite for efficient storage and querying.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize analytics.

        Args:
            db_path: Path to SQLite database (default: ~/.promptly/analytics.db)
        """
        if db_path is None:
            promptly_dir = Path.home() / ".promptly"
            promptly_dir.mkdir(exist_ok=True)
            db_path = str(promptly_dir / "analytics.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                prompt_name TEXT NOT NULL,
                execution_time REAL NOT NULL,
                quality_score REAL,
                success INTEGER NOT NULL,
                model TEXT,
                backend TEXT,
                tokens_used INTEGER,
                cost REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')

        # Index for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prompt_name
            ON executions(prompt_name)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON executions(timestamp)
        ''')

        conn.commit()
        conn.close()

    def record_execution(self, execution: PromptExecution) -> int:
        """
        Record a prompt execution.

        Args:
            execution: PromptExecution object

        Returns:
            Record ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO executions (
                prompt_id, prompt_name, execution_time, quality_score,
                success, model, backend, tokens_used, cost, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.prompt_id,
            execution.prompt_name,
            execution.execution_time,
            execution.quality_score,
            1 if execution.success else 0,
            execution.model,
            execution.backend,
            execution.tokens_used,
            execution.cost,
            execution.timestamp,
            json.dumps(execution.metadata)
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def get_prompt_stats(self, prompt_name: str) -> Optional[PromptStats]:
        """
        Get aggregated statistics for a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            PromptStats or None if no executions found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get basic stats
        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(success) as successes,
                AVG(execution_time) as avg_time,
                AVG(quality_score) as avg_quality,
                SUM(COALESCE(cost, 0)) as total_cost,
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                MIN(execution_time) as best_time,
                MAX(execution_time) as worst_time
            FROM executions
            WHERE prompt_name = ?
        ''', (prompt_name,))

        row = cursor.fetchone()

        if not row or row[0] == 0:
            conn.close()
            return None

        total, successes, avg_time, avg_quality, total_cost, total_tokens, best_time, worst_time = row

        # Calculate trend (last 5 vs previous 5 quality scores)
        cursor.execute('''
            SELECT quality_score
            FROM executions
            WHERE prompt_name = ? AND quality_score IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (prompt_name,))

        quality_scores = [r[0] for r in cursor.fetchall()]
        trend = self._calculate_trend(quality_scores)

        conn.close()

        return PromptStats(
            prompt_name=prompt_name,
            total_executions=total,
            success_rate=successes / total if total > 0 else 0,
            avg_execution_time=avg_time or 0,
            avg_quality_score=avg_quality,
            total_cost=total_cost or 0,
            total_tokens=int(total_tokens or 0),
            best_execution_time=best_time or 0,
            worst_execution_time=worst_time or 0,
            trend=trend
        )

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from recent quality scores"""
        if len(scores) < 4:
            return "insufficient_data"

        recent = scores[:len(scores)//2]
        older = scores[len(scores)//2:]

        if not recent or not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"

    def get_top_prompts(self, metric: str = "quality", limit: int = 10) -> List[PromptStats]:
        """
        Get top performing prompts.

        Args:
            metric: "quality", "speed", "cost_efficiency", "success_rate"
            limit: Number of prompts to return

        Returns:
            List of PromptStats ordered by metric
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all prompt names
        cursor.execute('SELECT DISTINCT prompt_name FROM executions')
        prompt_names = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Get stats for each
        stats = []
        for name in prompt_names:
            prompt_stats = self.get_prompt_stats(name)
            if prompt_stats:
                stats.append(prompt_stats)

        # Sort by metric
        if metric == "quality":
            stats.sort(key=lambda x: x.avg_quality_score or 0, reverse=True)
        elif metric == "speed":
            stats.sort(key=lambda x: x.avg_execution_time)
        elif metric == "cost_efficiency":
            stats.sort(key=lambda x: (x.total_cost / x.total_executions) if x.total_executions > 0 else float('inf'))
        elif metric == "success_rate":
            stats.sort(key=lambda x: x.success_rate, reverse=True)

        return stats[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get overall analytics summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total_executions,
                COUNT(DISTINCT prompt_name) as unique_prompts,
                SUM(success) as total_successes,
                AVG(execution_time) as avg_time,
                AVG(quality_score) as avg_quality,
                SUM(COALESCE(cost, 0)) as total_cost,
                SUM(COALESCE(tokens_used, 0)) as total_tokens
            FROM executions
        ''')

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"executions": 0}

        total_exec, unique_prompts, total_success, avg_time, avg_quality, total_cost, total_tokens = row

        return {
            "total_executions": total_exec,
            "unique_prompts": unique_prompts,
            "success_rate": total_success / total_exec if total_exec > 0 else 0,
            "avg_execution_time": avg_time or 0,
            "avg_quality_score": avg_quality,
            "total_cost": total_cost or 0,
            "total_tokens": int(total_tokens or 0)
        }

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on analytics"""
        recommendations = []
        summary = self.get_summary()

        if summary.get("executions", 0) == 0:
            return ["No executions recorded yet. Start using prompts to get recommendations!"]

        # Success rate recommendation
        if summary["success_rate"] < 0.8:
            recommendations.append(
                f"Success rate is {summary['success_rate']:.1%}. Consider reviewing failed prompts."
            )

        # Cost optimization
        if summary["total_cost"] > 10:
            top_costly = self.get_top_prompts(metric="cost_efficiency", limit=3)
            if top_costly:
                recommendations.append(
                    f"High API costs detected (${summary['total_cost']:.2f}). "
                    f"Most expensive: '{top_costly[-1].prompt_name}'"
                )

        # Quality degradation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT prompt_name FROM executions')
        for (name,) in cursor.fetchall():
            stats = self.get_prompt_stats(name)
            if stats and stats.trend == "degrading":
                recommendations.append(
                    f"Quality degrading for '{name}'. Consider A/B testing a new version."
                )
        conn.close()

        # Performance optimization
        if summary["avg_execution_time"] > 30:
            recommendations.append(
                f"Average execution time is {summary['avg_execution_time']:.1f}s. "
                "Consider using smaller models or caching."
            )

        if not recommendations:
            recommendations.append("Everything looks good! Keep optimizing.")

        return recommendations


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Prompt Analytics")

    analytics = PromptAnalytics()

    # Get summary
    summary = analytics.get_summary()
    print(f"\nSummary:")
    print(f"  Total executions: {summary.get('total_executions', 0)}")
    print(f"  Unique prompts: {summary.get('unique_prompts', 0)}")
    print(f"  Success rate: {summary.get('success_rate', 0):.1%}")

    if summary.get("total_executions", 0) > 0:
        print(f"  Avg execution time: {summary.get('avg_execution_time', 0):.2f}s")
        print(f"  Avg quality score: {summary.get('avg_quality_score', 0):.2f}")
        print(f"  Total cost: ${summary.get('total_cost', 0):.2f}")

        # Get recommendations
        print("\nRecommendations:")
        for rec in analytics.get_recommendations():
            print(f"  - {rec}")
    else:
        print("\nNo data yet. Start executing prompts!")
