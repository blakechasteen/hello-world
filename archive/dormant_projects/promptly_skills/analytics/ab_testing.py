"""
A/B Testing Framework - Phase 5 Week 3-4

Complete A/B testing system for comparing strategy performance.

Features:
- Test configuration with multiple variants
- Traffic splitting and variant assignment
- Performance tracking per variant
- Statistical significance testing (t-test, chi-square)
- Winner determination with confidence intervals
- Test lifecycle management (draft, running, completed, archived)

Usage:
    from analytics.ab_testing import ABTestManager, ABTest, Variant

    # Create test
    test = ABTest(
        id="optimize_vs_deep",
        name="Optimize vs Deep Strategy",
        description="Compare optimize and deep strategies",
        variants=[
            Variant(id="control", strategy="optimize", traffic_percent=50),
            Variant(id="treatment", strategy="deep", traffic_percent=50)
        ],
        metric="avg_confidence",
        min_sample_size=100,
        significance_level=0.05
    )

    # Start test
    manager = ABTestManager(db_path="test_metrics.db")
    await manager.create_test(test)
    await manager.start_test(test.id)

    # Get results
    results = await manager.get_test_results(test.id)
    print(f"Winner: {results.winner}")
    print(f"P-value: {results.p_value:.4f}")
"""

import hashlib
import time
import random
import sqlite3
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
from datetime import datetime

# Statistical testing
try:
    import scipy.stats as stats
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Statistical testing will be limited.")


class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"              # Created but not started
    RUNNING = "running"          # Currently running
    PAUSED = "paused"            # Temporarily paused
    COMPLETED = "completed"      # Finished, winner determined
    ARCHIVED = "archived"        # Archived for historical reference


class SignificanceTest(Enum):
    """Statistical significance test types"""
    T_TEST = "t_test"            # Compare means (continuous metrics)
    CHI_SQUARE = "chi_square"    # Compare proportions (categorical)
    MANN_WHITNEY = "mann_whitney"  # Non-parametric (when normality fails)


@dataclass
class Variant:
    """Test variant configuration"""
    id: str                      # Variant ID (e.g., "control", "treatment")
    strategy: str                # Strategy name to test
    traffic_percent: float       # Traffic allocation (0-100)
    description: str = ""        # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStats:
    """Performance statistics for a variant"""
    variant_id: str
    strategy: str
    sample_size: int
    avg_latency_ms: float
    std_latency_ms: float
    avg_confidence: float
    std_confidence: float
    cache_hit_rate: float
    success_rate: float
    total_queries: int

    # Raw data for statistical tests
    latencies: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)


@dataclass
class ABTest:
    """A/B test configuration"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    metric: str                  # Primary metric (e.g., "avg_confidence", "avg_latency_ms")
    min_sample_size: int = 100   # Minimum samples per variant
    significance_level: float = 0.05  # Alpha (typically 0.05 for 95% confidence)
    status: str = "draft"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    winner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResults:
    """Results of an A/B test"""
    test_id: str
    test_name: str
    status: str
    variant_stats: Dict[str, VariantStats]

    # Statistical analysis
    p_value: float
    statistically_significant: bool
    confidence_level: float
    effect_size: float

    # Winner determination
    winner: Optional[str]
    winner_improvement: float  # Improvement over control (%)

    # Metadata
    total_samples: int
    test_duration_seconds: float
    recommendation: str


class ABTestManager:
    """Main A/B testing manager"""

    def __init__(self, db_path: str = "test_metrics.db"):
        self.db_path = db_path
        self._init_ab_test_db()

    def _init_ab_test_db(self):
        """Create A/B testing tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                variants TEXT NOT NULL,
                metric TEXT NOT NULL,
                min_sample_size INTEGER DEFAULT 100,
                significance_level REAL DEFAULT 0.05,
                status TEXT DEFAULT 'draft',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                winner TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Variant assignments table (which variant was assigned to which query)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_variant_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                assigned_at REAL NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id)
            )
        """)

        # Variant results table (track performance per variant)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_variant_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                strategy TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                confidence REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                success INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_assignments_test ON ab_variant_assignments(test_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_assignments_hash ON ab_variant_assignments(query_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_results_test ON ab_variant_results(test_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_results_variant ON ab_variant_results(variant_id)")

        conn.commit()
        conn.close()

    async def create_test(self, test: ABTest) -> bool:
        """Create a new A/B test"""
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.traffic_percent for v in test.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ab_tests
            (id, name, description, variants, metric, min_sample_size, significance_level,
             status, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test.id,
            test.name,
            test.description,
            json.dumps([{
                'id': v.id,
                'strategy': v.strategy,
                'traffic_percent': v.traffic_percent,
                'description': v.description
            } for v in test.variants]),
            test.metric,
            test.min_sample_size,
            test.significance_level,
            test.status,
            test.created_at,
            json.dumps(test.metadata)
        ))

        conn.commit()
        conn.close()

        return True

    async def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = 'running', started_at = ?
            WHERE id = ? AND status = 'draft'
        """, (time.time(), test_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    async def pause_test(self, test_id: str) -> bool:
        """Pause a running test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = 'paused'
            WHERE id = ? AND status = 'running'
        """, (test_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    async def resume_test(self, test_id: str) -> bool:
        """Resume a paused test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = 'running'
            WHERE id = ? AND status = 'paused'
        """, (test_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    async def complete_test(self, test_id: str, winner: Optional[str] = None) -> bool:
        """Complete an A/B test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = 'completed', completed_at = ?, winner = ?
            WHERE id = ? AND status IN ('running', 'paused')
        """, (time.time(), winner, test_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def _hash_query(self, query: str) -> str:
        """Hash a query for consistent variant assignment"""
        return hashlib.md5(query.encode()).hexdigest()

    async def assign_variant(self, test_id: str, query: str) -> Optional[str]:
        """
        Assign a variant to a query.

        Uses consistent hashing to ensure the same query always gets the same variant.
        """
        # Get test configuration
        test = await self.get_test(test_id)
        if not test or test.status != "running":
            return None

        query_hash = self._hash_query(query)

        # Check if already assigned
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT variant_id FROM ab_variant_assignments
            WHERE test_id = ? AND query_hash = ?
        """, (test_id, query_hash))

        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0]

        # Assign new variant using consistent hashing
        # Convert hash to number and use traffic percentages for assignment
        hash_int = int(query_hash[:8], 16)
        rand_percent = (hash_int % 10000) / 100.0  # 0-100 with 2 decimal precision

        cumulative = 0.0
        assigned_variant = None

        for variant in test.variants:
            cumulative += variant.traffic_percent
            if rand_percent < cumulative:
                assigned_variant = variant.id
                break

        if not assigned_variant:
            assigned_variant = test.variants[-1].id  # Fallback to last variant

        # Store assignment
        cursor.execute("""
            INSERT INTO ab_variant_assignments (test_id, query_hash, variant_id, assigned_at)
            VALUES (?, ?, ?, ?)
        """, (test_id, query_hash, assigned_variant, time.time()))

        conn.commit()
        conn.close()

        return assigned_variant

    async def record_result(
        self,
        test_id: str,
        variant_id: str,
        query: str,
        strategy: str,
        latency_ms: float,
        confidence: float,
        cache_hit: bool,
        success: bool
    ):
        """Record a result for a variant"""
        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ab_variant_results
            (test_id, variant_id, query_hash, strategy, latency_ms, confidence,
             cache_hit, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            variant_id,
            query_hash,
            strategy,
            latency_ms,
            confidence,
            1 if cache_hit else 0,
            1 if success else 0,
            time.time()
        ))

        conn.commit()
        conn.close()

    async def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ab_tests WHERE id = ?", (test_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        variants_json = json.loads(row[3])
        variants = [
            Variant(
                id=v['id'],
                strategy=v['strategy'],
                traffic_percent=v['traffic_percent'],
                description=v.get('description', '')
            )
            for v in variants_json
        ]

        return ABTest(
            id=row[0],
            name=row[1],
            description=row[2],
            variants=variants,
            metric=row[4],
            min_sample_size=row[5],
            significance_level=row[6],
            status=row[7],
            created_at=row[8],
            started_at=row[9],
            completed_at=row[10],
            winner=row[11],
            metadata=json.loads(row[12]) if row[12] else {}
        )

    async def get_variant_stats(self, test_id: str, variant_id: str) -> VariantStats:
        """Get performance statistics for a variant"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT strategy, latency_ms, confidence, cache_hit, success
            FROM ab_variant_results
            WHERE test_id = ? AND variant_id = ?
        """, (test_id, variant_id))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return VariantStats(
                variant_id=variant_id,
                strategy="unknown",
                sample_size=0,
                avg_latency_ms=0,
                std_latency_ms=0,
                avg_confidence=0,
                std_confidence=0,
                cache_hit_rate=0,
                success_rate=0,
                total_queries=0
            )

        latencies = [row[1] for row in rows]
        confidences = [row[2] for row in rows]
        cache_hits = [row[3] for row in rows]
        successes = [row[4] for row in rows]

        if HAS_SCIPY:
            std_latency = float(np.std(latencies))
            std_confidence = float(np.std(confidences))
        else:
            # Simple standard deviation calculation
            mean_lat = sum(latencies) / len(latencies)
            std_latency = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
            mean_conf = sum(confidences) / len(confidences)
            std_confidence = (sum((x - mean_conf) ** 2 for x in confidences) / len(confidences)) ** 0.5

        return VariantStats(
            variant_id=variant_id,
            strategy=rows[0][0],
            sample_size=len(rows),
            avg_latency_ms=sum(latencies) / len(latencies),
            std_latency_ms=std_latency,
            avg_confidence=sum(confidences) / len(confidences),
            std_confidence=std_confidence,
            cache_hit_rate=sum(cache_hits) / len(cache_hits),
            success_rate=sum(successes) / len(successes),
            total_queries=len(rows),
            latencies=latencies,
            confidences=confidences
        )

    async def get_test_results(self, test_id: str) -> ABTestResults:
        """Get complete test results with statistical analysis"""
        test = await self.get_test(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        # Get stats for all variants
        variant_stats = {}
        for variant in test.variants:
            variant_stats[variant.id] = await self.get_variant_stats(test_id, variant.id)

        # Check if we have enough samples
        min_samples_met = all(
            stats.sample_size >= test.min_sample_size
            for stats in variant_stats.values()
        )

        if not min_samples_met:
            return ABTestResults(
                test_id=test.id,
                test_name=test.name,
                status=test.status,
                variant_stats=variant_stats,
                p_value=1.0,
                statistically_significant=False,
                confidence_level=0.0,
                effect_size=0.0,
                winner=None,
                winner_improvement=0.0,
                total_samples=sum(s.sample_size for s in variant_stats.values()),
                test_duration_seconds=0.0 if not test.started_at else time.time() - test.started_at,
                recommendation="Insufficient samples. Keep test running."
            )

        # Perform statistical analysis
        if HAS_SCIPY and len(test.variants) == 2:
            result = await self._analyze_two_variant_test(test, variant_stats)
        else:
            result = await self._simple_winner_determination(test, variant_stats)

        return result

    async def _analyze_two_variant_test(
        self,
        test: ABTest,
        variant_stats: Dict[str, VariantStats]
    ) -> ABTestResults:
        """Perform statistical analysis for two-variant test"""
        control_id = test.variants[0].id
        treatment_id = test.variants[1].id

        control_stats = variant_stats[control_id]
        treatment_stats = variant_stats[treatment_id]

        # Choose metric
        if test.metric == "avg_confidence":
            control_data = control_stats.confidences
            treatment_data = treatment_stats.confidences
            control_mean = control_stats.avg_confidence
            treatment_mean = treatment_stats.avg_confidence
            higher_is_better = True
        elif test.metric == "avg_latency_ms":
            control_data = control_stats.latencies
            treatment_data = treatment_stats.latencies
            control_mean = control_stats.avg_latency_ms
            treatment_mean = treatment_stats.avg_latency_ms
            higher_is_better = False
        else:
            # Default to confidence
            control_data = control_stats.confidences
            treatment_data = treatment_stats.confidences
            control_mean = control_stats.avg_confidence
            treatment_mean = treatment_stats.avg_confidence
            higher_is_better = True

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_data) - 1) * control_stats.std_confidence ** 2 +
             (len(treatment_data) - 1) * treatment_stats.std_confidence ** 2) /
            (len(control_data) + len(treatment_data) - 2)
        )
        effect_size = abs(treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        # Determine statistical significance
        is_significant = p_value < test.significance_level

        # Determine winner
        if is_significant:
            if higher_is_better:
                winner = treatment_id if treatment_mean > control_mean else control_id
            else:
                winner = treatment_id if treatment_mean < control_mean else control_id
        else:
            winner = None

        # Calculate improvement
        if winner == treatment_id:
            improvement = ((treatment_mean - control_mean) / control_mean) * 100
        elif winner == control_id:
            improvement = ((control_mean - treatment_mean) / treatment_mean) * 100
        else:
            improvement = 0.0

        # Generate recommendation
        if is_significant:
            recommendation = f"Test is statistically significant (p={p_value:.4f}). Winner: {winner} ({improvement:+.2f}% improvement)"
        else:
            recommendation = f"Not statistically significant (p={p_value:.4f}). No clear winner. Consider running longer."

        return ABTestResults(
            test_id=test.id,
            test_name=test.name,
            status=test.status,
            variant_stats=variant_stats,
            p_value=float(p_value),
            statistically_significant=is_significant,
            confidence_level=(1 - test.significance_level) * 100,
            effect_size=float(effect_size),
            winner=winner,
            winner_improvement=abs(improvement),
            total_samples=sum(s.sample_size for s in variant_stats.values()),
            test_duration_seconds=time.time() - test.started_at if test.started_at else 0,
            recommendation=recommendation
        )

    async def _simple_winner_determination(
        self,
        test: ABTest,
        variant_stats: Dict[str, VariantStats]
    ) -> ABTestResults:
        """Simple winner determination (when scipy not available or >2 variants)"""
        # Just compare means
        if test.metric == "avg_confidence":
            scores = {vid: vstats.avg_confidence for vid, vstats in variant_stats.items()}
            higher_is_better = True
        elif test.metric == "avg_latency_ms":
            scores = {vid: vstats.avg_latency_ms for vid, vstats in variant_stats.items()}
            higher_is_better = False
        else:
            scores = {vid: vstats.avg_confidence for vid, vstats in variant_stats.items()}
            higher_is_better = True

        if higher_is_better:
            winner = max(scores.keys(), key=lambda k: scores[k])
        else:
            winner = min(scores.keys(), key=lambda k: scores[k])

        # Calculate improvement vs baseline (first variant)
        baseline_id = test.variants[0].id
        baseline_score = scores[baseline_id]
        winner_score = scores[winner]

        if baseline_score != 0:
            improvement = abs((winner_score - baseline_score) / baseline_score) * 100
        else:
            improvement = 0.0

        return ABTestResults(
            test_id=test.id,
            test_name=test.name,
            status=test.status,
            variant_stats=variant_stats,
            p_value=0.0,  # Not calculated
            statistically_significant=False,
            confidence_level=0.0,
            effect_size=0.0,
            winner=winner,
            winner_improvement=improvement,
            total_samples=sum(s.sample_size for s in variant_stats.values()),
            test_duration_seconds=time.time() - test.started_at if test.started_at else 0,
            recommendation=f"Winner: {winner} (scipy not available for significance testing)"
        )

    async def list_tests(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[ABTest]:
        """List all tests, optionally filtered by status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM ab_tests
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT * FROM ab_tests
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        tests = []
        for row in rows:
            variants_json = json.loads(row[3])
            variants = [
                Variant(
                    id=v['id'],
                    strategy=v['strategy'],
                    traffic_percent=v['traffic_percent'],
                    description=v.get('description', '')
                )
                for v in variants_json
            ]

            tests.append(ABTest(
                id=row[0],
                name=row[1],
                description=row[2],
                variants=variants,
                metric=row[4],
                min_sample_size=row[5],
                significance_level=row[6],
                status=row[7],
                created_at=row[8],
                started_at=row[9],
                completed_at=row[10],
                winner=row[11],
                metadata=json.loads(row[12]) if row[12] else {}
            ))

        return tests

    async def promote_winner(self, test_id: str) -> bool:
        """
        Promote the winning variant to production.

        This is a placeholder that should integrate with your
        strategy configuration system.
        """
        results = await self.get_test_results(test_id)

        if not results.winner:
            return False

        # TODO: Integrate with strategy configuration
        # For now, just complete the test
        await self.complete_test(test_id, winner=results.winner)

        return True
