from __future__ import annotations
"""
MRF Analytics Dashboard
=======================
Real-time performance monitoring for Metaprompting Refinement Framework.

**Phase 2 - November 2025**: Production-grade analytics dashboard for MRF usage tracking,
quality monitoring, and strategy effectiveness analysis with integrated Thompson Sampling
learning and A/B testing.

Features:
- Real-time quality tracking across all 7 integrated systems
- Strategy effectiveness monitoring (which strategies work best)
- Model adapter usage statistics
- Quality regression detection (>10% drop alerts)
- Thompson Sampling continuous learning (adaptive strategy selection)
- A/B testing framework (statistical validation before deployment)
- Tufte-style visualizations (HTML/Markdown/JSON reports)
- Prometheus metrics export (production monitoring)

Usage:
    from hololoom.prompting.analytics import MRFDashboard

    # Create dashboard with learning and A/B testing enabled
    dashboard = MRFDashboard(
        enable_learning=True,
        enable_ab_testing=True
    )

    # Log MRF usage (automatically updates learner)
    dashboard.log_enhancement(
        system="agentic",
        query="Explain Thompson Sampling",
        strategy="verify",
        quality_before=0.75,
        quality_after=0.92,
        execution_time_ms=450.0,
        metadata={"query_type": "factual"}  # For learner
    )

    # Get strategy recommendation from learner
    rec = dashboard.get_strategy_recommendation(
        query_type="factual",
        system="agentic"
    )
    print(f"Recommended: {rec['recommended_strategy']} (confidence: {rec['confidence']:.2f})")

    # Create A/B test
    dashboard.create_ab_test(
        name="mrf_verify_enhancement",
        control_description="Baseline verify mode",
        treatment_description="MRF-enhanced verify",
        traffic_split=0.5
    )

    # Log A/B test results
    dashboard.log_ab_test_result(
        test_name="mrf_verify_enhancement",
        user_id="user_123",
        quality_score=0.92,
        execution_time_ms=450.0
    )

    # Analyze A/B test (after sufficient data)
    results = dashboard.get_ab_test_results("mrf_verify_enhancement")
    if results["is_significant"]:
        print(f"Deploy treatment! Improvement: {results['statistics']['difference']['quality_improvement_percent']:.1f}%")

    # Get real-time statistics
    stats = dashboard.get_statistics()
    print(f"Total enhancements: {stats['total_enhancements']}")
    print(f"Avg quality improvement: {stats['avg_quality_improvement']:.1%}")

    # Export Prometheus metrics
    metrics = dashboard.export_prometheus_metrics()

    # Generate HTML report (includes learning and A/B testing sections)
    html = dashboard.generate_report()
    dashboard.save_report("mrf_dashboard.html")
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import learning and A/B testing modules (optional)
try:
    from hololoom.prompting.analytics.learning import ThompsonLearner
    LEARNING_AVAILABLE = True
except ImportError:
    ThompsonLearner = None
    LEARNING_AVAILABLE = False
    logger.warning("Thompson Sampling learner not available")

try:
    from hololoom.prompting.analytics.ab_testing import ABGroup, ABTest
    AB_TESTING_AVAILABLE = True
except ImportError:
    ABTest = None
    ABGroup = None
    AB_TESTING_AVAILABLE = False
    logger.warning("A/B testing framework not available")


@dataclass
class MRFEnhancementLog:
    """Log entry for MRF enhancement usage."""
    timestamp: float
    system: str  # "agentic", "rag", "alignment", "recursive", "memory"
    query: str
    strategy: str  # "verify", "refine", "critique", etc.
    quality_before: float  # 0.0-1.0
    quality_after: float  # 0.0-1.0
    execution_time_ms: float
    model_provider: str = "claude"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quality_improvement(self) -> float:
        """Quality improvement delta."""
        return self.quality_after - self.quality_before

    @property
    def timestamp_dt(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class SystemStatistics:
    """Statistics for a single system."""
    system_name: str
    total_enhancements: int = 0
    avg_quality_before: float = 0.0
    avg_quality_after: float = 0.0
    avg_quality_improvement: float = 0.0
    avg_execution_time_ms: float = 0.0
    strategy_usage: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, int] = field(default_factory=dict)
    recent_logs: list[MRFEnhancementLog] = field(default_factory=list)

    def update(self, log: MRFEnhancementLog):
        """Update statistics with new log entry."""
        self.total_enhancements += 1

        # Running averages
        n = self.total_enhancements
        self.avg_quality_before = ((self.avg_quality_before * (n - 1)) + log.quality_before) / n
        self.avg_quality_after = ((self.avg_quality_after * (n - 1)) + log.quality_after) / n
        self.avg_quality_improvement = ((self.avg_quality_improvement * (n - 1)) + log.quality_improvement) / n
        self.avg_execution_time_ms = ((self.avg_execution_time_ms * (n - 1)) + log.execution_time_ms) / n

        # Strategy usage
        self.strategy_usage[log.strategy] = self.strategy_usage.get(log.strategy, 0) + 1

        # Model usage
        self.model_usage[log.model_provider] = self.model_usage.get(log.model_provider, 0) + 1

        # Keep last 100 logs
        self.recent_logs.append(log)
        if len(self.recent_logs) > 100:
            self.recent_logs.pop(0)


class MRFDashboard:
    """
    Analytics dashboard for MRF performance monitoring.

    Tracks usage across all integrated systems, monitors quality improvements,
    and detects regressions.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        enable_learning: bool = False,
        enable_ab_testing: bool = False
    ):
        """
        Initialize MRF dashboard.

        Args:
            persist_path: Optional path to persist logs (defaults to ./mrf_logs)
            enable_learning: Enable Thompson Sampling learning
            enable_ab_testing: Enable A/B testing framework
        """
        self.persist_path = persist_path or Path("./mrf_logs")
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # System statistics
        self.systems: dict[str, SystemStatistics] = {}

        # Global logs
        self.all_logs: list[MRFEnhancementLog] = []

        # Regression tracking
        self.baseline_quality: dict[str, float] = {}  # system -> baseline quality
        self.regressions_detected: list[dict[str, Any]] = []

        # Learning module (optional)
        self.learner = None
        if enable_learning and LEARNING_AVAILABLE:
            self.learner = ThompsonLearner(
                persist_path=self.persist_path / "learning"
            )
            logger.info("Thompson Sampling learner enabled")
        elif enable_learning and not LEARNING_AVAILABLE:
            logger.warning("Learning requested but module not available")

        # A/B testing (optional)
        self.ab_tests: dict[str, ABTest] = {}
        self.enable_ab_testing = enable_ab_testing and AB_TESTING_AVAILABLE
        if enable_ab_testing and not AB_TESTING_AVAILABLE:
            logger.warning("A/B testing requested but module not available")

        # Load existing logs if available
        self._load_logs()

    def log_enhancement(
        self,
        system: str,
        query: str,
        strategy: str,
        quality_before: float,
        quality_after: float,
        execution_time_ms: float,
        model_provider: str = "claude",
        metadata: dict[str, Any] | None = None
    ):
        """
        Log an MRF enhancement usage.

        Args:
            system: System name ("agentic", "rag", "alignment", etc.)
            query: Original query
            strategy: Refinement strategy used
            quality_before: Quality before MRF
            quality_after: Quality after MRF
            execution_time_ms: Execution time in milliseconds
            model_provider: Model provider used
            metadata: Additional metadata
        """
        log = MRFEnhancementLog(
            timestamp=time.time(),
            system=system,
            query=query,
            strategy=strategy,
            quality_before=quality_before,
            quality_after=quality_after,
            execution_time_ms=execution_time_ms,
            model_provider=model_provider,
            metadata=metadata or {}
        )

        # Update system statistics
        if system not in self.systems:
            self.systems[system] = SystemStatistics(system_name=system)

        self.systems[system].update(log)
        self.all_logs.append(log)

        # Check for regression
        self._check_regression(system, quality_after)

        # Update Thompson Sampling learner (if enabled)
        if self.learner:
            query_type = metadata.get("query_type", "general") if metadata else "general"
            self.learner.update_from_outcome(
                query_type=query_type,
                system=system,
                strategy=strategy,
                quality_improvement=log.quality_improvement,
                execution_time_ms=execution_time_ms
            )

        # Persist log
        self._persist_log(log)

        logger.info(
            f"[MRF] {system}: {strategy} | "
            f"Quality: {quality_before:.2f} → {quality_after:.2f} (+{log.quality_improvement:.2f}) | "
            f"Time: {execution_time_ms:.1f}ms"
        )

    def _check_regression(self, system: str, quality: float, threshold: float = 0.1):
        """
        Check for quality regression.

        Args:
            system: System name
            quality: Current quality
            threshold: Regression threshold (default: 10% drop)
        """
        if system not in self.baseline_quality:
            # Establish baseline after 10 enhancements
            stats = self.systems.get(system)
            if stats and stats.total_enhancements >= 10:
                self.baseline_quality[system] = stats.avg_quality_after
            return

        baseline = self.baseline_quality[system]
        if quality < baseline * (1 - threshold):
            # Regression detected
            regression = {
                "timestamp": datetime.now().isoformat(),
                "system": system,
                "baseline_quality": baseline,
                "current_quality": quality,
                "drop_percent": (baseline - quality) / baseline * 100
            }
            self.regressions_detected.append(regression)

            logger.warning(
                f"[MRF REGRESSION] {system}: Quality dropped from {baseline:.2f} to {quality:.2f} "
                f"(-{regression['drop_percent']:.1f}%)"
            )

    def get_statistics(self, system: str | None = None) -> dict[str, Any]:
        """
        Get MRF usage statistics.

        Args:
            system: Optional system filter (returns all if None)

        Returns:
            Statistics dictionary
        """
        if system:
            # Single system statistics
            stats = self.systems.get(system)
            if not stats:
                return {"error": f"System '{system}' not found"}

            return {
                "system": system,
                "total_enhancements": stats.total_enhancements,
                "avg_quality_before": stats.avg_quality_before,
                "avg_quality_after": stats.avg_quality_after,
                "avg_quality_improvement": stats.avg_quality_improvement,
                "quality_improvement_percent": stats.avg_quality_improvement * 100,
                "avg_execution_time_ms": stats.avg_execution_time_ms,
                "strategy_usage": dict(stats.strategy_usage),
                "model_usage": dict(stats.model_usage),
                "recent_logs_count": len(stats.recent_logs)
            }

        # Global statistics
        total_enhancements = sum(s.total_enhancements for s in self.systems.values())
        if total_enhancements == 0:
            return {"error": "No enhancements logged yet"}

        # Aggregate across systems
        total_quality_before = sum(
            s.avg_quality_before * s.total_enhancements for s in self.systems.values()
        )
        total_quality_after = sum(
            s.avg_quality_after * s.total_enhancements for s in self.systems.values()
        )
        total_time = sum(
            s.avg_execution_time_ms * s.total_enhancements for s in self.systems.values()
        )

        avg_quality_before = total_quality_before / total_enhancements
        avg_quality_after = total_quality_after / total_enhancements
        avg_quality_improvement = avg_quality_after - avg_quality_before
        avg_execution_time = total_time / total_enhancements

        # Strategy usage across all systems
        strategy_usage = defaultdict(int)
        for stats in self.systems.values():
            for strategy, count in stats.strategy_usage.items():
                strategy_usage[strategy] += count

        # Model usage across all systems
        model_usage = defaultdict(int)
        for stats in self.systems.values():
            for model, count in stats.model_usage.items():
                model_usage[model] += count

        return {
            "total_enhancements": total_enhancements,
            "systems_active": len(self.systems),
            "avg_quality_before": avg_quality_before,
            "avg_quality_after": avg_quality_after,
            "avg_quality_improvement": avg_quality_improvement,
            "quality_improvement_percent": avg_quality_improvement * 100,
            "avg_execution_time_ms": avg_execution_time,
            "strategy_usage": dict(strategy_usage),
            "model_usage": dict(model_usage),
            "regressions_detected": len(self.regressions_detected),
            "per_system": {
                name: {
                    "total": stats.total_enhancements,
                    "avg_improvement": stats.avg_quality_improvement,
                    "avg_time_ms": stats.avg_execution_time_ms
                }
                for name, stats in self.systems.items()
            }
        }

    def generate_report(self, format: str = "html") -> str:
        """
        Generate analytics report.

        Args:
            format: Report format ("html", "markdown", "json")

        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(self.get_statistics(), indent=2)

        elif format == "markdown":
            return self._generate_markdown_report()

        elif format == "html":
            return self._generate_html_report()

        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        stats = self.get_statistics()

        lines = []
        lines.append("# MRF Analytics Dashboard")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Global statistics
        lines.append("## Global Statistics")
        lines.append("")
        lines.append(f"- **Total Enhancements**: {stats['total_enhancements']}")
        lines.append(f"- **Systems Active**: {stats['systems_active']}")
        lines.append(f"- **Avg Quality Improvement**: +{stats['quality_improvement_percent']:.1f}%")
        lines.append(f"- **Avg Execution Time**: {stats['avg_execution_time_ms']:.1f}ms")
        lines.append("")

        # Per-system breakdown
        lines.append("## Per-System Performance")
        lines.append("")
        lines.append("| System | Enhancements | Avg Improvement | Avg Time (ms) |")
        lines.append("|--------|--------------|-----------------|---------------|")
        for system, data in stats["per_system"].items():
            lines.append(
                f"| {system} | {data['total']} | "
                f"+{data['avg_improvement']*100:.1f}% | "
                f"{data['avg_time_ms']:.1f} |"
            )
        lines.append("")

        # Strategy usage
        lines.append("## Strategy Usage")
        lines.append("")
        for strategy, count in sorted(stats["strategy_usage"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{strategy}**: {count} uses")
        lines.append("")

        # Regressions
        if stats["regressions_detected"] > 0:
            lines.append(f"## ⚠️  Regressions Detected: {stats['regressions_detected']}")
            lines.append("")
            for reg in self.regressions_detected[-5:]:  # Last 5
                lines.append(
                    f"- [{reg['timestamp']}] {reg['system']}: "
                    f"{reg['baseline_quality']:.2f} → {reg['current_quality']:.2f} "
                    f"(-{reg['drop_percent']:.1f}%)"
                )
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self) -> str:
        """Generate HTML report with Tufte-style visualizations."""
        stats = self.get_statistics()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MRF Analytics Dashboard</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            font-weight: normal;
            color: #111;
        }}
        .stat-box {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px 25px;
            background: #f8f8f8;
            border-left: 4px solid #333;
        }}
        .stat-label {{
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            text-align: left;
            padding: 12px 8px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            font-weight: bold;
            background: #f8f8f8;
        }}
        .regression-box {{
            background: #fff5f5;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>MRF Analytics Dashboard</h1>
    <p><strong>Generated</strong>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Global Statistics</h2>

    <div class="stat-box">
        <div class="stat-label">Total Enhancements</div>
        <div class="stat-value">{stats['total_enhancements']}</div>
    </div>

    <div class="stat-box">
        <div class="stat-label">Systems Active</div>
        <div class="stat-value">{stats['systems_active']}</div>
    </div>

    <div class="stat-box">
        <div class="stat-label">Avg Quality Improvement</div>
        <div class="stat-value positive">+{stats['quality_improvement_percent']:.1f}%</div>
    </div>

    <div class="stat-box">
        <div class="stat-label">Avg Execution Time</div>
        <div class="stat-value">{stats['avg_execution_time_ms']:.1f}ms</div>
    </div>

    <h2>Per-System Performance</h2>
    <table>
        <thead>
            <tr>
                <th>System</th>
                <th>Enhancements</th>
                <th>Avg Improvement</th>
                <th>Avg Time (ms)</th>
            </tr>
        </thead>
        <tbody>
"""

        for system, data in stats["per_system"].items():
            html += f"""            <tr>
                <td><strong>{system}</strong></td>
                <td>{data['total']}</td>
                <td class="positive">+{data['avg_improvement']*100:.1f}%</td>
                <td>{data['avg_time_ms']:.1f}</td>
            </tr>
"""

        html += """        </tbody>
    </table>

    <h2>Strategy Usage</h2>
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Uses</th>
                <th>Percentage</th>
            </tr>
        </thead>
        <tbody>
"""

        total_uses = sum(stats["strategy_usage"].values())
        for strategy, count in sorted(stats["strategy_usage"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_uses * 100) if total_uses > 0 else 0
            html += f"""            <tr>
                <td><strong>{strategy}</strong></td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""

        html += """        </tbody>
    </table>
"""

        # Regressions
        if stats["regressions_detected"] > 0:
            html += f"""
    <h2>⚠️  Regressions Detected: {stats['regressions_detected']}</h2>
"""
            for reg in self.regressions_detected[-5:]:
                html += f"""
    <div class="regression-box">
        <strong>{reg['system']}</strong> - {reg['timestamp']}<br>
        Quality dropped: {reg['baseline_quality']:.2f} → {reg['current_quality']:.2f}
        <span class="negative">(-{reg['drop_percent']:.1f}%)</span>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def save_report(self, filename: str, format: str = "html"):
        """
        Save report to file.

        Args:
            filename: Output filename
            format: Report format ("html", "markdown", "json")
        """
        report = self.generate_report(format)
        Path(filename).write_text(report, encoding="utf-8")
        logger.info(f"Report saved to: {filename}")

    # ===== Learning Integration =====

    def get_strategy_recommendation(
        self,
        query_type: str,
        system: str
    ) -> dict[str, Any] | None:
        """
        Get strategy recommendation from Thompson Sampling learner.

        Args:
            query_type: Type of query ("factual", "analytical", etc.)
            system: System name

        Returns:
            Recommendation dict or None if learner disabled
        """
        if not self.learner:
            return None

        return self.learner.get_strategy_recommendation(query_type, system)

    def get_learning_statistics(self) -> dict[str, Any] | None:
        """
        Get Thompson Sampling learning statistics.

        Returns:
            Learning statistics or None if learner disabled
        """
        if not self.learner:
            return None

        return self.learner.get_learning_stats()

    # ===== A/B Testing Integration =====

    def create_ab_test(
        self,
        name: str,
        control_description: str,
        treatment_description: str,
        traffic_split: float = 0.5
    ) -> str | None:
        """
        Create a new A/B test.

        Args:
            name: Test name (unique identifier)
            control_description: Description of control group
            treatment_description: Description of treatment group
            traffic_split: Fraction of traffic to treatment (0.0-1.0)

        Returns:
            Test name if created, None if A/B testing disabled
        """
        if not self.enable_ab_testing:
            return None

        if name in self.ab_tests:
            logger.warning(f"A/B test '{name}' already exists")
            return None

        self.ab_tests[name] = ABTest(
            name=name,
            control_description=control_description,
            treatment_description=treatment_description,
            traffic_split=traffic_split,
            persist_path=self.persist_path / "ab_tests"
        )

        logger.info(f"Created A/B test: {name}")
        return name

    def log_ab_test_result(
        self,
        test_name: str,
        user_id: str,
        quality_score: float,
        execution_time_ms: float,
        metadata: dict[str, Any] | None = None
    ):
        """
        Log A/B test result.

        Args:
            test_name: Test name
            user_id: User identifier
            quality_score: Quality score (0.0-1.0)
            execution_time_ms: Execution time
            metadata: Optional metadata
        """
        if not self.enable_ab_testing or test_name not in self.ab_tests:
            return

        test = self.ab_tests[test_name]

        # Assign group (deterministic)
        group = test.assign_group(user_id)

        # Log result
        test.log_result(
            user_id=user_id,
            group=group,
            quality_score=quality_score,
            execution_time_ms=execution_time_ms,
            metadata=metadata
        )

    def get_ab_test_results(self, test_name: str) -> dict[str, Any] | None:
        """
        Get A/B test analysis results.

        Args:
            test_name: Test name

        Returns:
            Analysis results or None if test doesn't exist
        """
        if not self.enable_ab_testing or test_name not in self.ab_tests:
            return None

        test = self.ab_tests[test_name]
        return test.analyze()

    def list_ab_tests(self) -> dict[str, dict[str, Any]]:
        """
        List all A/B tests with summaries.

        Returns:
            Dict of test_name -> summary
        """
        if not self.enable_ab_testing:
            return {}

        return {
            name: test.get_summary()
            for name, test in self.ab_tests.items()
        }

    # ===== CoVe (Chain of Verification) Integration =====

    def log_verification(
        self,
        verification_result: dict[str, Any],
        context: dict[str, Any] | None = None
    ):
        """
        Log Chain of Verification (CoVe) usage.

        Tracks CoVe verification alongside MRF strategies, enabling
        comparison and analysis of verification effectiveness.

        Args:
            verification_result: Result from CoVe verification containing:
                - verified_response: The verified output
                - original_response: Original response (if available)
                - claims_extracted: Number of claims extracted
                - questions_generated: Number of verification questions
                - contradictions_found: Number of contradictions detected
                - verification_time_ms: Time taken for verification
                - confidence: Final confidence score (0.0-1.0)
                - metadata: Additional verification metadata
            context: Optional context including:
                - query: Original query text
                - system: System that requested verification
                - model_provider: LLM provider used

        Example:
            dashboard.log_verification(
                verification_result={
                    "claims_extracted": 5,
                    "questions_generated": 10,
                    "contradictions_found": 1,
                    "verification_time_ms": 450.0,
                    "confidence": 0.85
                },
                context={
                    "query": "What is Thompson Sampling?",
                    "system": "agentic"
                }
            )
        """
        context = context or {}

        # Extract verification metrics
        claims = verification_result.get("claims_extracted", 0)
        questions = verification_result.get("questions_generated", 0)
        contradictions = verification_result.get("contradictions_found", 0)
        time_ms = verification_result.get("verification_time_ms", 0.0)
        confidence = verification_result.get("confidence", 0.0)

        # Initialize CoVe statistics if not present
        if not hasattr(self, "_cove_stats"):
            self._cove_stats = {
                "total_verifications": 0,
                "total_claims_extracted": 0,
                "total_questions_generated": 0,
                "total_contradictions_found": 0,
                "total_time_ms": 0.0,
                "confidence_sum": 0.0,
                "per_system": {},
                "recent_logs": []
            }

        # Update global CoVe statistics
        self._cove_stats["total_verifications"] += 1
        self._cove_stats["total_claims_extracted"] += claims
        self._cove_stats["total_questions_generated"] += questions
        self._cove_stats["total_contradictions_found"] += contradictions
        self._cove_stats["total_time_ms"] += time_ms
        self._cove_stats["confidence_sum"] += confidence

        # Track per-system statistics
        system = context.get("system", "unknown")
        if system not in self._cove_stats["per_system"]:
            self._cove_stats["per_system"][system] = {
                "verifications": 0,
                "claims": 0,
                "questions": 0,
                "contradictions": 0,
                "time_ms": 0.0,
                "confidence_sum": 0.0
            }

        sys_stats = self._cove_stats["per_system"][system]
        sys_stats["verifications"] += 1
        sys_stats["claims"] += claims
        sys_stats["questions"] += questions
        sys_stats["contradictions"] += contradictions
        sys_stats["time_ms"] += time_ms
        sys_stats["confidence_sum"] += confidence

        # Store recent log (keep last 100)
        log_entry = {
            "timestamp": time.time(),
            "system": system,
            "claims_extracted": claims,
            "questions_generated": questions,
            "contradictions_found": contradictions,
            "verification_time_ms": time_ms,
            "confidence": confidence,
            "query": context.get("query", ""),
            "model_provider": context.get("model_provider", "unknown")
        }
        self._cove_stats["recent_logs"].append(log_entry)
        if len(self._cove_stats["recent_logs"]) > 100:
            self._cove_stats["recent_logs"] = self._cove_stats["recent_logs"][-100:]

        # Persist to disk
        self._persist_cove_log(log_entry)

        logger.info(
            f"[CoVe] {system}: {claims} claims, {questions} questions, "
            f"{contradictions} contradictions | Confidence: {confidence:.2f} | "
            f"Time: {time_ms:.1f}ms"
        )

    def get_cove_statistics(self) -> dict[str, Any]:
        """
        Get Chain of Verification (CoVe) usage statistics.

        Returns comprehensive statistics about CoVe usage across all systems,
        including averages, per-system breakdowns, and effectiveness metrics.

        Returns:
            Dictionary containing:
                - total_verifications: Total number of verifications
                - avg_claims_extracted: Average claims per verification
                - avg_questions_generated: Average questions per verification
                - avg_contradictions_found: Average contradictions per verification
                - contradiction_rate: Contradictions per claim (quality indicator)
                - avg_verification_time_ms: Average verification time
                - avg_confidence: Average final confidence
                - per_system: Per-system breakdown
                - recent_trend: Last 10 verification confidence scores

        Example:
            stats = dashboard.get_cove_statistics()
            print(f"Total verifications: {stats['total_verifications']}")
            print(f"Contradiction rate: {stats['contradiction_rate']:.1%}")
        """
        if not hasattr(self, "_cove_stats") or self._cove_stats["total_verifications"] == 0:
            return {
                "error": "No CoVe verifications logged yet",
                "total_verifications": 0
            }

        stats = self._cove_stats
        total = stats["total_verifications"]

        # Calculate averages
        avg_claims = stats["total_claims_extracted"] / total
        avg_questions = stats["total_questions_generated"] / total
        avg_contradictions = stats["total_contradictions_found"] / total
        avg_time = stats["total_time_ms"] / total
        avg_confidence = stats["confidence_sum"] / total

        # Calculate contradiction rate (contradictions per claim)
        contradiction_rate = (
            stats["total_contradictions_found"] / stats["total_claims_extracted"]
            if stats["total_claims_extracted"] > 0 else 0.0
        )

        # Per-system statistics
        per_system = {}
        for system, sys_stats in stats["per_system"].items():
            sys_total = sys_stats["verifications"]
            per_system[system] = {
                "verifications": sys_total,
                "avg_claims": sys_stats["claims"] / sys_total if sys_total > 0 else 0,
                "avg_questions": sys_stats["questions"] / sys_total if sys_total > 0 else 0,
                "avg_contradictions": sys_stats["contradictions"] / sys_total if sys_total > 0 else 0,
                "avg_time_ms": sys_stats["time_ms"] / sys_total if sys_total > 0 else 0,
                "avg_confidence": sys_stats["confidence_sum"] / sys_total if sys_total > 0 else 0
            }

        # Recent trend (last 10 confidence scores)
        recent_confidences = [
            log["confidence"] for log in stats["recent_logs"][-10:]
        ]

        return {
            "total_verifications": total,
            "total_claims_extracted": stats["total_claims_extracted"],
            "total_questions_generated": stats["total_questions_generated"],
            "total_contradictions_found": stats["total_contradictions_found"],
            "avg_claims_extracted": avg_claims,
            "avg_questions_generated": avg_questions,
            "avg_contradictions_found": avg_contradictions,
            "contradiction_rate": contradiction_rate,
            "avg_verification_time_ms": avg_time,
            "avg_confidence": avg_confidence,
            "per_system": per_system,
            "recent_trend": recent_confidences
        }

    def _persist_cove_log(self, log_entry: dict[str, Any]):
        """Persist CoVe log entry to disk."""
        log_file = self.persist_path / f"cove_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    # ===== Prometheus Metrics Export =====

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        stats = self.get_statistics()
        if "error" in stats:
            return ""

        lines = []

        # Help and type declarations
        lines.append("# HELP mrf_enhancements_total Total number of MRF enhancements")
        lines.append("# TYPE mrf_enhancements_total counter")
        lines.append(f"mrf_enhancements_total {stats['total_enhancements']}")
        lines.append("")

        lines.append("# HELP mrf_quality_improvement_avg Average quality improvement")
        lines.append("# TYPE mrf_quality_improvement_avg gauge")
        lines.append(f"mrf_quality_improvement_avg {stats['avg_quality_improvement']:.4f}")
        lines.append("")

        lines.append("# HELP mrf_execution_time_ms_avg Average execution time in milliseconds")
        lines.append("# TYPE mrf_execution_time_ms_avg gauge")
        lines.append(f"mrf_execution_time_ms_avg {stats['avg_execution_time_ms']:.2f}")
        lines.append("")

        # Per-system metrics
        lines.append("# HELP mrf_system_enhancements_total Enhancements per system")
        lines.append("# TYPE mrf_system_enhancements_total counter")
        for system, data in stats["per_system"].items():
            lines.append(f'mrf_system_enhancements_total{{system="{system}"}} {data["total"]}')
        lines.append("")

        lines.append("# HELP mrf_system_quality_improvement Average quality improvement per system")
        lines.append("# TYPE mrf_system_quality_improvement gauge")
        for system, data in stats["per_system"].items():
            lines.append(f'mrf_system_quality_improvement{{system="{system}"}} {data["avg_improvement"]:.4f}')
        lines.append("")

        # Strategy usage
        lines.append("# HELP mrf_strategy_usage_total Strategy usage count")
        lines.append("# TYPE mrf_strategy_usage_total counter")
        for strategy, count in stats["strategy_usage"].items():
            lines.append(f'mrf_strategy_usage_total{{strategy="{strategy}"}} {count}')
        lines.append("")

        # Regressions
        lines.append("# HELP mrf_regressions_detected_total Total regressions detected")
        lines.append("# TYPE mrf_regressions_detected_total counter")
        lines.append(f"mrf_regressions_detected_total {stats['regressions_detected']}")
        lines.append("")

        # Learning statistics (if available)
        if self.learner:
            learning_stats = self.learner.get_learning_stats()
            lines.append("# HELP mrf_learning_queries_total Total queries processed by learner")
            lines.append("# TYPE mrf_learning_queries_total counter")
            lines.append(f"mrf_learning_queries_total {learning_stats['total_queries']}")
            lines.append("")

        # A/B test statistics (if available)
        if self.enable_ab_testing:
            for test_name, test in self.ab_tests.items():
                summary = test.get_summary()
                if summary["ready_for_analysis"]:
                    analysis = test.analyze()
                    lines.append('# HELP mrf_ab_test_significant A/B test significance')
                    lines.append('# TYPE mrf_ab_test_significant gauge')
                    lines.append(f'mrf_ab_test_significant{{test="{test_name}"}} {1 if analysis["is_significant"] else 0}')
                    lines.append("")

        # CoVe (Chain of Verification) statistics (if available)
        cove_stats = self.get_cove_statistics()
        if "error" not in cove_stats:
            lines.append("# HELP cove_verifications_total Total CoVe verifications")
            lines.append("# TYPE cove_verifications_total counter")
            lines.append(f"cove_verifications_total {cove_stats['total_verifications']}")
            lines.append("")

            lines.append("# HELP cove_claims_extracted_total Total claims extracted")
            lines.append("# TYPE cove_claims_extracted_total counter")
            lines.append(f"cove_claims_extracted_total {cove_stats['total_claims_extracted']}")
            lines.append("")

            lines.append("# HELP cove_contradictions_found_total Total contradictions found")
            lines.append("# TYPE cove_contradictions_found_total counter")
            lines.append(f"cove_contradictions_found_total {cove_stats['total_contradictions_found']}")
            lines.append("")

            lines.append("# HELP cove_contradiction_rate Contradictions per claim (quality indicator)")
            lines.append("# TYPE cove_contradiction_rate gauge")
            lines.append(f"cove_contradiction_rate {cove_stats['contradiction_rate']:.4f}")
            lines.append("")

            lines.append("# HELP cove_avg_verification_time_ms Average verification time")
            lines.append("# TYPE cove_avg_verification_time_ms gauge")
            lines.append(f"cove_avg_verification_time_ms {cove_stats['avg_verification_time_ms']:.2f}")
            lines.append("")

            lines.append("# HELP cove_avg_confidence Average verification confidence")
            lines.append("# TYPE cove_avg_confidence gauge")
            lines.append(f"cove_avg_confidence {cove_stats['avg_confidence']:.4f}")
            lines.append("")

            # Per-system CoVe metrics
            lines.append("# HELP cove_system_verifications_total Verifications per system")
            lines.append("# TYPE cove_system_verifications_total counter")
            for system, sys_data in cove_stats["per_system"].items():
                lines.append(f'cove_system_verifications_total{{system="{system}"}} {sys_data["verifications"]}')
            lines.append("")

        return "\n".join(lines)

    def _persist_log(self, log: MRFEnhancementLog):
        """Persist log to disk."""
        log_file = self.persist_path / f"mrf_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log)) + "\n")

    def _load_logs(self):
        """Load existing logs from disk."""
        log_files = sorted(self.persist_path.glob("mrf_log_*.jsonl"))
        for log_file in log_files[-7:]:  # Last 7 days
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        log = MRFEnhancementLog(**data)
                        if log.system not in self.systems:
                            self.systems[log.system] = SystemStatistics(system_name=log.system)
                        self.systems[log.system].update(log)
                        self.all_logs.append(log)
            except Exception as e:
                logger.warning(f"Error loading logs from {log_file}: {e}")


# Convenience functions
def create_dashboard(
    persist_path: Path | None = None,
    enable_learning: bool = False,
    enable_ab_testing: bool = False
) -> MRFDashboard:
    """
    Create MRF analytics dashboard.

    Args:
        persist_path: Optional path to persist logs
        enable_learning: Enable Thompson Sampling learning
        enable_ab_testing: Enable A/B testing framework

    Returns:
        MRFDashboard instance
    """
    return MRFDashboard(
        persist_path=persist_path,
        enable_learning=enable_learning,
        enable_ab_testing=enable_ab_testing
    )


__all__ = [
    "MRFDashboard",
    "MRFEnhancementLog",
    "SystemStatistics",
    "create_dashboard"
]
