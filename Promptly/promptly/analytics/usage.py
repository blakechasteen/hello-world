"""
Usage Analytics - Track access patterns, popular prompts, and user activity

Provides:
- Prompt access patterns
- Most-used prompts
- Branch activity
- User activity (if multi-user)
- Evaluation frequency
- Chain execution statistics
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import json
from pathlib import Path


@dataclass
class UsageEvent:
    """A single usage event"""
    event_type: str  # 'prompt_get', 'prompt_add', 'eval', 'chain_exec', etc.
    resource_name: str  # prompt name, chain name, etc.
    branch: str
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AccessPattern:
    """Access pattern for a resource"""
    resource_name: str
    resource_type: str  # 'prompt', 'chain', etc.
    total_accesses: int
    unique_days: int
    first_access: datetime
    last_access: datetime
    access_frequency: float  # accesses per day
    peak_hour: int
    common_branches: List[str]


class UsageTracker:
    """Tracks individual usage events"""

    def __init__(self, analytics: 'UsageAnalytics'):
        self.analytics = analytics

    def track_prompt_access(self, name: str, branch: str, operation: str,
                          user: str = None, metadata: Dict = None):
        """Track prompt access"""
        event = UsageEvent(
            event_type=f'prompt_{operation}',
            resource_name=name,
            branch=branch,
            user=user,
            metadata=metadata or {}
        )
        self.analytics.record_event(event)

    def track_evaluation(self, prompt_name: str, branch: str,
                        test_count: int, avg_score: float = None,
                        user: str = None, metadata: Dict = None):
        """Track evaluation execution"""
        event = UsageEvent(
            event_type='evaluation',
            resource_name=prompt_name,
            branch=branch,
            user=user,
            metadata={
                'test_count': test_count,
                'avg_score': avg_score,
                **(metadata or {})
            }
        )
        self.analytics.record_event(event)

    def track_chain_execution(self, chain_name: str, step_count: int,
                            success: bool, user: str = None, metadata: Dict = None):
        """Track chain execution"""
        event = UsageEvent(
            event_type='chain_execution',
            resource_name=chain_name,
            branch='',  # Chains are not branch-specific
            user=user,
            metadata={
                'step_count': step_count,
                'success': success,
                **(metadata or {})
            }
        )
        self.analytics.record_event(event)

    def track_branch_operation(self, branch: str, operation: str,
                              user: str = None, metadata: Dict = None):
        """Track branch operations"""
        event = UsageEvent(
            event_type=f'branch_{operation}',
            resource_name=branch,
            branch=branch,
            user=user,
            metadata=metadata or {}
        )
        self.analytics.record_event(event)


class UsageAnalytics:
    """Analyzes and tracks usage patterns for Promptly"""

    def __init__(self, db_path: str, retention_days: int = 90):
        """
        Initialize usage analytics

        Args:
            db_path: Path to analytics database
            retention_days: How long to keep usage data
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.tracker = UsageTracker(self)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize analytics database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Usage events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                branch TEXT,
                user TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_events_type
            ON usage_events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_events_resource
            ON usage_events(resource_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_events_branch
            ON usage_events(branch)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp
            ON usage_events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_events_user
            ON usage_events(user)
        """)

        # Aggregated stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                resource_name TEXT,
                resource_type TEXT,
                stat_date DATE,
                access_count INTEGER,
                unique_users INTEGER,
                PRIMARY KEY (resource_name, resource_type, stat_date)
            )
        """)

        conn.commit()
        conn.close()

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def record_event(self, event: UsageEvent):
        """Record a usage event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO usage_events
            (event_type, resource_name, branch, user, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.event_type,
            event.resource_name,
            event.branch,
            event.user,
            json.dumps(event.metadata),
            event.timestamp.isoformat()
        ))

        conn.commit()
        conn.close()

    def get_most_used_prompts(self, limit: int = 10, days: int = 30,
                              branch: str = None) -> List[Dict]:
        """Get most frequently accessed prompts"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT resource_name, branch, COUNT(*) as access_count
            FROM usage_events
            WHERE event_type LIKE 'prompt_%'
            AND timestamp > ?
        """
        params = [cutoff.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        query += """
            GROUP BY resource_name, branch
            ORDER BY access_count DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'prompt_name': row[0],
                'branch': row[1],
                'access_count': row[2]
            })

        conn.close()
        return results

    def get_access_pattern(self, resource_name: str, resource_type: str = 'prompt') -> Optional[AccessPattern]:
        """Get detailed access pattern for a resource"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Determine event type pattern
        if resource_type == 'prompt':
            event_pattern = 'prompt_%'
        elif resource_type == 'chain':
            event_pattern = 'chain_%'
        else:
            event_pattern = '%'

        # Get access statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_accesses,
                COUNT(DISTINCT DATE(timestamp)) as unique_days,
                MIN(timestamp) as first_access,
                MAX(timestamp) as last_access,
                branch
            FROM usage_events
            WHERE resource_name = ?
            AND event_type LIKE ?
            GROUP BY branch
            ORDER BY COUNT(*) DESC
        """, (resource_name, event_pattern))

        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return None

        # Aggregate across branches
        total_accesses = sum(row[0] for row in rows)
        unique_days = max(row[1] for row in rows)
        first_access = min(datetime.fromisoformat(row[2]) for row in rows)
        last_access = max(datetime.fromisoformat(row[3]) for row in rows)
        common_branches = [row[4] for row in rows[:3]]

        # Calculate access frequency
        days_span = (last_access - first_access).days or 1
        access_frequency = total_accesses / days_span

        # Get peak hour
        cursor.execute("""
            SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, COUNT(*) as count
            FROM usage_events
            WHERE resource_name = ?
            AND event_type LIKE ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        """, (resource_name, event_pattern))

        peak_row = cursor.fetchone()
        peak_hour = peak_row[0] if peak_row else 0

        conn.close()

        return AccessPattern(
            resource_name=resource_name,
            resource_type=resource_type,
            total_accesses=total_accesses,
            unique_days=unique_days,
            first_access=first_access,
            last_access=last_access,
            access_frequency=access_frequency,
            peak_hour=peak_hour,
            common_branches=common_branches
        )

    def get_branch_activity(self, days: int = 30) -> List[Dict]:
        """Get activity statistics per branch"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                branch,
                COUNT(*) as total_events,
                COUNT(DISTINCT resource_name) as unique_prompts,
                COUNT(DISTINCT user) as unique_users,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM usage_events
            WHERE timestamp > ?
            AND branch IS NOT NULL
            AND branch != ''
            GROUP BY branch
            ORDER BY total_events DESC
        """, (cutoff.isoformat(),))

        results = []
        for row in cursor.fetchall():
            results.append({
                'branch': row[0],
                'total_events': row[1],
                'unique_prompts': row[2],
                'unique_users': row[3],
                'active_days': row[4],
                'avg_events_per_day': row[1] / days
            })

        conn.close()
        return results

    def get_user_activity(self, days: int = 30, limit: int = 20) -> List[Dict]:
        """Get activity statistics per user"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                user,
                COUNT(*) as total_events,
                COUNT(DISTINCT resource_name) as unique_resources,
                COUNT(DISTINCT branch) as unique_branches,
                MAX(timestamp) as last_activity
            FROM usage_events
            WHERE timestamp > ?
            AND user IS NOT NULL
            GROUP BY user
            ORDER BY total_events DESC
            LIMIT ?
        """, (cutoff.isoformat(), limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'user': row[0],
                'total_events': row[1],
                'unique_resources': row[2],
                'unique_branches': row[3],
                'last_activity': row[4],
                'avg_events_per_day': row[1] / days
            })

        conn.close()
        return results

    def get_evaluation_stats(self, days: int = 30) -> Dict:
        """Get evaluation execution statistics"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Total evaluations
        cursor.execute("""
            SELECT COUNT(*) FROM usage_events
            WHERE event_type = 'evaluation'
            AND timestamp > ?
        """, (cutoff.isoformat(),))

        total_evals = cursor.fetchone()[0]

        # Average score
        cursor.execute("""
            SELECT AVG(json_extract(metadata, '$.avg_score'))
            FROM usage_events
            WHERE event_type = 'evaluation'
            AND timestamp > ?
            AND json_extract(metadata, '$.avg_score') IS NOT NULL
        """, (cutoff.isoformat(),))

        avg_score = cursor.fetchone()[0]

        # Most evaluated prompts
        cursor.execute("""
            SELECT resource_name, COUNT(*) as eval_count
            FROM usage_events
            WHERE event_type = 'evaluation'
            AND timestamp > ?
            GROUP BY resource_name
            ORDER BY eval_count DESC
            LIMIT 5
        """, (cutoff.isoformat(),))

        most_evaluated = [
            {'prompt': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            'total_evaluations': total_evals,
            'average_score': avg_score,
            'most_evaluated_prompts': most_evaluated,
            'evaluations_per_day': total_evals / days
        }

    def get_chain_stats(self, days: int = 30) -> Dict:
        """Get chain execution statistics"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Total executions
        cursor.execute("""
            SELECT COUNT(*) FROM usage_events
            WHERE event_type = 'chain_execution'
            AND timestamp > ?
        """, (cutoff.isoformat(),))

        total_execs = cursor.fetchone()[0]

        # Success rate
        cursor.execute("""
            SELECT
                SUM(CASE WHEN json_extract(metadata, '$.success') = 1 THEN 1 ELSE 0 END) as successes,
                COUNT(*) as total
            FROM usage_events
            WHERE event_type = 'chain_execution'
            AND timestamp > ?
        """, (cutoff.isoformat(),))

        row = cursor.fetchone()
        success_rate = (row[0] / row[1] * 100) if row[1] > 0 else 0

        # Most executed chains
        cursor.execute("""
            SELECT resource_name, COUNT(*) as exec_count
            FROM usage_events
            WHERE event_type = 'chain_execution'
            AND timestamp > ?
            GROUP BY resource_name
            ORDER BY exec_count DESC
            LIMIT 5
        """, (cutoff.isoformat(),))

        most_executed = [
            {'chain': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            'total_executions': total_execs,
            'success_rate': success_rate,
            'most_executed_chains': most_executed,
            'executions_per_day': total_execs / days
        }

    def get_hourly_distribution(self, days: int = 7) -> List[Dict]:
        """Get hourly distribution of activity"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                COUNT(*) as event_count
            FROM usage_events
            WHERE timestamp > ?
            GROUP BY hour
            ORDER BY hour
        """, (cutoff.isoformat(),))

        results = []
        for row in cursor.fetchall():
            results.append({
                'hour': row[0],
                'event_count': row[1]
            })

        conn.close()
        return results

    def get_daily_activity(self, days: int = 30) -> List[Dict]:
        """Get daily activity counts"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as event_count,
                COUNT(DISTINCT resource_name) as unique_resources,
                COUNT(DISTINCT user) as unique_users
            FROM usage_events
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (cutoff.isoformat(),))

        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0],
                'event_count': row[1],
                'unique_resources': row[2],
                'unique_users': row[3]
            })

        conn.close()
        return results

    def get_event_type_distribution(self, days: int = 30) -> List[Dict]:
        """Get distribution of event types"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM usage_events
            WHERE timestamp > ?
            GROUP BY event_type
            ORDER BY count DESC
        """, (cutoff.isoformat(),))

        results = []
        total = 0
        for row in cursor.fetchall():
            count = row[1]
            total += count
            results.append({
                'event_type': row[0],
                'count': count
            })

        # Add percentages
        for item in results:
            item['percentage'] = (item['count'] / total * 100) if total > 0 else 0

        conn.close()
        return results

    def get_resource_timeline(self, resource_name: str, days: int = 30) -> List[Dict]:
        """Get timeline of events for a specific resource"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_type, branch, user, metadata, timestamp
            FROM usage_events
            WHERE resource_name = ?
            AND timestamp > ?
            ORDER BY timestamp DESC
        """, (resource_name, cutoff.isoformat()))

        results = []
        for row in cursor.fetchall():
            results.append({
                'event_type': row[0],
                'branch': row[1],
                'user': row[2],
                'metadata': json.loads(row[3]) if row[3] else {},
                'timestamp': row[4]
            })

        conn.close()
        return results

    def cleanup_old_events(self):
        """Remove events older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM usage_events
            WHERE timestamp < ?
        """, (cutoff.isoformat(),))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def aggregate_daily_stats(self, date: datetime = None):
        """Aggregate daily statistics for faster querying"""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Aggregate prompt stats
        cursor.execute("""
            INSERT OR REPLACE INTO usage_stats
            (resource_name, resource_type, stat_date, access_count, unique_users)
            SELECT
                resource_name,
                'prompt' as resource_type,
                DATE(timestamp) as stat_date,
                COUNT(*) as access_count,
                COUNT(DISTINCT user) as unique_users
            FROM usage_events
            WHERE event_type LIKE 'prompt_%'
            AND timestamp >= ?
            AND timestamp < ?
            GROUP BY resource_name, DATE(timestamp)
        """, (start_of_day.isoformat(), end_of_day.isoformat()))

        conn.commit()
        conn.close()

    def export_usage_data(self, output_path: str, days: int = 30):
        """Export usage data to JSON"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_type, resource_name, branch, user, metadata, timestamp
            FROM usage_events
            WHERE timestamp > ?
            ORDER BY timestamp
        """, (cutoff.isoformat(),))

        events = []
        for row in cursor.fetchall():
            events.append({
                'event_type': row[0],
                'resource_name': row[1],
                'branch': row[2],
                'user': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'timestamp': row[5]
            })

        conn.close()

        export_data = {
            'export_date': datetime.now().isoformat(),
            'days': days,
            'event_count': len(events),
            'events': events
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_path
