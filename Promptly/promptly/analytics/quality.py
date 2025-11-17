"""
Quality Metrics - Track prompt quality, evaluation scores, and trends

Provides:
- Prompt quality trends over time
- Evaluation score distributions
- Version quality comparison
- Success/failure rates
- A/B test results
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import statistics
from pathlib import Path


@dataclass
class QualityMeasurement:
    """A single quality measurement"""
    prompt_name: str
    version: int
    branch: str
    score: float
    evaluator_type: str
    test_case_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class QualityTrend:
    """Quality trend for a prompt over time"""
    prompt_name: str
    branch: str
    measurements: int
    avg_score: float
    min_score: float
    max_score: float
    std_dev: float
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_slope: float
    first_measurement: datetime
    last_measurement: datetime


@dataclass
class ABTestResult:
    """Result of an A/B test between two prompt versions"""
    prompt_name: str
    version_a: int
    version_b: int
    branch: str
    samples_a: int
    samples_b: int
    avg_score_a: float
    avg_score_b: float
    score_diff: float
    score_diff_percent: float
    statistical_significance: float  # p-value
    winner: Optional[int]  # version number or None if tie
    confidence: float  # confidence level (0-1)


class QualityTracker:
    """Tracks quality measurements"""

    def __init__(self, metrics: 'QualityMetrics'):
        self.metrics = metrics

    def record_evaluation(self, prompt_name: str, version: int, branch: str,
                         score: float, evaluator_type: str, test_case_id: str,
                         metadata: Dict = None):
        """Record an evaluation result"""
        measurement = QualityMeasurement(
            prompt_name=prompt_name,
            version=version,
            branch=branch,
            score=score,
            evaluator_type=evaluator_type,
            test_case_id=test_case_id,
            metadata=metadata or {}
        )
        self.metrics.record_measurement(measurement)

    def record_batch_evaluation(self, prompt_name: str, version: int, branch: str,
                                scores: List[float], evaluator_type: str,
                                metadata: Dict = None):
        """Record multiple evaluation scores at once"""
        for i, score in enumerate(scores):
            self.record_evaluation(
                prompt_name, version, branch, score,
                evaluator_type, f"batch_{i}",
                metadata
            )


class QualityMetrics:
    """Analyzes and tracks quality metrics for Promptly"""

    def __init__(self, db_path: str, retention_days: int = 180):
        """
        Initialize quality metrics

        Args:
            db_path: Path to metrics database
            retention_days: How long to keep quality data
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.tracker = QualityTracker(self)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize metrics database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Quality measurements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                branch TEXT NOT NULL,
                score REAL NOT NULL,
                evaluator_type TEXT NOT NULL,
                test_case_id TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_prompt
            ON quality_measurements(prompt_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_version
            ON quality_measurements(prompt_name, version)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_branch
            ON quality_measurements(branch)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_timestamp
            ON quality_measurements(timestamp)
        """)

        # A/B test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                version_a INTEGER NOT NULL,
                version_b INTEGER NOT NULL,
                branch TEXT NOT NULL,
                samples_a INTEGER,
                samples_b INTEGER,
                avg_score_a REAL,
                avg_score_b REAL,
                score_diff REAL,
                statistical_significance REAL,
                winner INTEGER,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Quality alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def record_measurement(self, measurement: QualityMeasurement):
        """Record a quality measurement"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO quality_measurements
            (prompt_name, version, branch, score, evaluator_type, test_case_id, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            measurement.prompt_name,
            measurement.version,
            measurement.branch,
            measurement.score,
            measurement.evaluator_type,
            measurement.test_case_id,
            json.dumps(measurement.metadata),
            measurement.timestamp.isoformat()
        ))

        conn.commit()
        conn.close()

        # Check for quality alerts
        self._check_quality_alerts(measurement)

    def get_prompt_quality_stats(self, prompt_name: str, branch: str = None,
                                 days: int = 30) -> Optional[Dict]:
        """Get quality statistics for a prompt"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                COUNT(*) as measurements,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                COUNT(DISTINCT version) as versions_tested
            FROM quality_measurements
            WHERE prompt_name = ?
            AND timestamp > ?
        """
        params = [prompt_name, cutoff.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        cursor.execute(query, params)
        row = cursor.fetchone()

        if not row or row[0] == 0:
            conn.close()
            return None

        # Get score distribution
        cursor.execute(f"""
            SELECT score FROM quality_measurements
            WHERE prompt_name = ?
            AND timestamp > ?
            {" AND branch = ?" if branch else ""}
            ORDER BY score
        """, params)

        scores = [row[0] for row in cursor.fetchall()]
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Calculate percentiles
        p25 = scores[len(scores) // 4] if scores else 0
        p50 = scores[len(scores) // 2] if scores else 0
        p75 = scores[3 * len(scores) // 4] if scores else 0

        conn.close()

        return {
            'prompt_name': prompt_name,
            'branch': branch,
            'measurements': row[0],
            'avg_score': row[1],
            'min_score': row[2],
            'max_score': row[3],
            'std_dev': std_dev,
            'versions_tested': row[4],
            'percentiles': {
                'p25': p25,
                'p50': p50,
                'p75': p75
            }
        }

    def get_quality_trend(self, prompt_name: str, branch: str = None,
                         days: int = 30) -> Optional[QualityTrend]:
        """Get quality trend for a prompt over time"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT score, timestamp
            FROM quality_measurements
            WHERE prompt_name = ?
            AND timestamp > ?
        """
        params = [prompt_name, cutoff.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        query += " ORDER BY timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return None

        scores = [row[0] for row in rows]
        timestamps = [datetime.fromisoformat(row[1]) for row in rows]

        # Calculate basic stats
        avg_score = statistics.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Calculate trend (simple linear regression)
        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = avg_score

        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'improving'
        else:
            trend_direction = 'declining'

        conn.close()

        return QualityTrend(
            prompt_name=prompt_name,
            branch=branch or 'all',
            measurements=len(scores),
            avg_score=avg_score,
            min_score=min_score,
            max_score=max_score,
            std_dev=std_dev,
            trend_direction=trend_direction,
            trend_slope=slope,
            first_measurement=timestamps[0],
            last_measurement=timestamps[-1]
        )

    def compare_versions(self, prompt_name: str, version_a: int, version_b: int,
                        branch: str = None, days: int = 30) -> Optional[Dict]:
        """Compare quality between two versions"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query_base = """
            SELECT AVG(score), MIN(score), MAX(score), COUNT(*)
            FROM quality_measurements
            WHERE prompt_name = ?
            AND version = ?
            AND timestamp > ?
        """
        params_a = [prompt_name, version_a, cutoff.isoformat()]
        params_b = [prompt_name, version_b, cutoff.isoformat()]

        if branch:
            query_base += " AND branch = ?"
            params_a.append(branch)
            params_b.append(branch)

        cursor.execute(query_base, params_a)
        row_a = cursor.fetchone()

        cursor.execute(query_base, params_b)
        row_b = cursor.fetchone()

        if not row_a or not row_b or row_a[3] == 0 or row_b[3] == 0:
            conn.close()
            return None

        avg_a, min_a, max_a, count_a = row_a
        avg_b, min_b, max_b, count_b = row_b

        diff = avg_b - avg_a
        diff_percent = (diff / avg_a * 100) if avg_a != 0 else 0

        winner = version_b if avg_b > avg_a else version_a if avg_a > avg_b else None

        conn.close()

        return {
            'prompt_name': prompt_name,
            'branch': branch,
            'version_a': {
                'version': version_a,
                'avg_score': avg_a,
                'min_score': min_a,
                'max_score': max_a,
                'samples': count_a
            },
            'version_b': {
                'version': version_b,
                'avg_score': avg_b,
                'min_score': min_b,
                'max_score': max_b,
                'samples': count_b
            },
            'comparison': {
                'score_diff': diff,
                'score_diff_percent': diff_percent,
                'winner': winner
            }
        }

    def run_ab_test(self, prompt_name: str, version_a: int, version_b: int,
                    branch: str = None, days: int = 30,
                    confidence_level: float = 0.95) -> Optional[ABTestResult]:
        """Run A/B test between two versions with statistical analysis"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get scores for both versions
        query = """
            SELECT score FROM quality_measurements
            WHERE prompt_name = ? AND version = ? AND timestamp > ?
        """
        params_base = [prompt_name, None, cutoff.isoformat()]

        if branch:
            query += " AND branch = ?"

        # Version A scores
        params_a = params_base.copy()
        params_a[1] = version_a
        if branch:
            params_a.append(branch)

        cursor.execute(query, params_a)
        scores_a = [row[0] for row in cursor.fetchall()]

        # Version B scores
        params_b = params_base.copy()
        params_b[1] = version_b
        if branch:
            params_b.append(branch)

        cursor.execute(query, params_b)
        scores_b = [row[0] for row in cursor.fetchall()]

        if not scores_a or not scores_b:
            conn.close()
            return None

        # Calculate statistics
        avg_a = statistics.mean(scores_a)
        avg_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0

        diff = avg_b - avg_a
        diff_percent = (diff / avg_a * 100) if avg_a != 0 else 0

        # Simplified t-test calculation
        n_a = len(scores_a)
        n_b = len(scores_b)

        pooled_std = ((std_a ** 2 / n_a) + (std_b ** 2 / n_b)) ** 0.5
        t_stat = diff / pooled_std if pooled_std != 0 else 0

        # Simplified p-value estimation (would use scipy.stats in production)
        # For now, use a rough approximation
        p_value = max(0.0, 1.0 - abs(t_stat) / 3.0)

        # Determine winner
        if p_value < (1 - confidence_level):
            winner = version_b if avg_b > avg_a else version_a
            confidence = 1 - p_value
        else:
            winner = None
            confidence = 0.5

        result = ABTestResult(
            prompt_name=prompt_name,
            version_a=version_a,
            version_b=version_b,
            branch=branch or 'all',
            samples_a=n_a,
            samples_b=n_b,
            avg_score_a=avg_a,
            avg_score_b=avg_b,
            score_diff=diff,
            score_diff_percent=diff_percent,
            statistical_significance=p_value,
            winner=winner,
            confidence=confidence
        )

        # Save result
        cursor.execute("""
            INSERT INTO ab_test_results
            (prompt_name, version_a, version_b, branch, samples_a, samples_b,
             avg_score_a, avg_score_b, score_diff, statistical_significance, winner, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_name, version_a, version_b, branch, n_a, n_b,
            avg_a, avg_b, diff, p_value, winner, confidence
        ))

        conn.commit()
        conn.close()

        return result

    def get_score_distribution(self, prompt_name: str = None, branch: str = None,
                              days: int = 30, bins: int = 10) -> Dict:
        """Get score distribution histogram"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT score FROM quality_measurements WHERE timestamp > ?"
        params = [cutoff.isoformat()]

        if prompt_name:
            query += " AND prompt_name = ?"
            params.append(prompt_name)

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        cursor.execute(query, params)
        scores = [row[0] for row in cursor.fetchall()]

        conn.close()

        if not scores:
            return {'bins': [], 'counts': []}

        # Create histogram
        min_score = min(scores)
        max_score = max(scores)
        bin_width = (max_score - min_score) / bins

        histogram = [0] * bins
        bin_edges = [min_score + i * bin_width for i in range(bins + 1)]

        for score in scores:
            if score == max_score:
                bin_idx = bins - 1
            else:
                bin_idx = min(int((score - min_score) / bin_width), bins - 1)
            histogram[bin_idx] += 1

        return {
            'bins': bin_edges,
            'counts': histogram,
            'total_samples': len(scores),
            'avg_score': statistics.mean(scores),
            'median_score': statistics.median(scores)
        }

    def get_quality_by_evaluator(self, days: int = 30) -> List[Dict]:
        """Get quality statistics grouped by evaluator type"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                evaluator_type,
                COUNT(*) as measurements,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score
            FROM quality_measurements
            WHERE timestamp > ?
            GROUP BY evaluator_type
            ORDER BY measurements DESC
        """, (cutoff.isoformat(),))

        results = []
        for row in cursor.fetchall():
            results.append({
                'evaluator_type': row[0],
                'measurements': row[1],
                'avg_score': row[2],
                'min_score': row[3],
                'max_score': row[4]
            })

        conn.close()
        return results

    def get_top_performing_prompts(self, limit: int = 10, branch: str = None,
                                   days: int = 30, min_samples: int = 5) -> List[Dict]:
        """Get top performing prompts by average score"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                prompt_name,
                branch,
                COUNT(*) as measurements,
                AVG(score) as avg_score,
                MAX(version) as latest_version
            FROM quality_measurements
            WHERE timestamp > ?
        """
        params = [cutoff.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        query += """
            GROUP BY prompt_name, branch
            HAVING COUNT(*) >= ?
            ORDER BY avg_score DESC
            LIMIT ?
        """
        params.extend([min_samples, limit])

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'prompt_name': row[0],
                'branch': row[1],
                'measurements': row[2],
                'avg_score': row[3],
                'latest_version': row[4]
            })

        conn.close()
        return results

    def get_declining_prompts(self, threshold_slope: float = -0.05,
                             days: int = 30, min_samples: int = 10) -> List[Dict]:
        """Get prompts with declining quality"""
        cutoff = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get prompts with enough samples
        cursor.execute("""
            SELECT DISTINCT prompt_name, branch
            FROM quality_measurements
            WHERE timestamp > ?
            GROUP BY prompt_name, branch
            HAVING COUNT(*) >= ?
        """, (cutoff.isoformat(), min_samples))

        declining = []

        for row in cursor.fetchall():
            trend = self.get_quality_trend(row[0], row[1], days)
            if trend and trend.trend_slope < threshold_slope:
                declining.append({
                    'prompt_name': trend.prompt_name,
                    'branch': trend.branch,
                    'trend_slope': trend.trend_slope,
                    'avg_score': trend.avg_score,
                    'measurements': trend.measurements
                })

        conn.close()

        # Sort by slope (most declining first)
        declining.sort(key=lambda x: x['trend_slope'])

        return declining

    def _check_quality_alerts(self, measurement: QualityMeasurement):
        """Check if measurement triggers quality alerts"""
        # Get recent average for this prompt
        recent_stats = self.get_prompt_quality_stats(
            measurement.prompt_name,
            measurement.branch,
            days=7
        )

        if not recent_stats:
            return

        # Alert if score is significantly below average
        if measurement.score < recent_stats['avg_score'] - 2 * recent_stats['std_dev']:
            self._create_alert(
                measurement.prompt_name,
                'low_score',
                'warning',
                f"Score {measurement.score:.3f} is significantly below average {recent_stats['avg_score']:.3f}",
                {'measurement_id': measurement.test_case_id}
            )

    def _create_alert(self, prompt_name: str, alert_type: str, severity: str,
                     message: str, metadata: Dict = None):
        """Create a quality alert"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO quality_alerts
            (prompt_name, alert_type, severity, message, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            prompt_name,
            alert_type,
            severity,
            message,
            json.dumps(metadata or {})
        ))

        conn.commit()
        conn.close()

    def get_active_alerts(self, severity: str = None) -> List[Dict]:
        """Get active quality alerts"""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT prompt_name, alert_type, severity, message, metadata, created_at
            FROM quality_alerts
            WHERE resolved_at IS NULL
        """
        params = []

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'prompt_name': row[0],
                'alert_type': row[1],
                'severity': row[2],
                'message': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'created_at': row[5]
            })

        conn.close()
        return results

    def cleanup_old_measurements(self):
        """Remove measurements older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM quality_measurements
            WHERE timestamp < ?
        """, (cutoff.isoformat(),))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted
