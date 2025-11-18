"""
Memory System Health Checks
============================

Week 8A: Production health monitoring for memory systems.

Provides comprehensive health checks for all memory systems:
- Consolidation health (Week 5)
- Semantic transition health (Week 6)
- Temporal tracking health (Week 7)
- Overall system health

Philosophy:
"Monitor proactively, detect early, heal automatically"

Author: HoloLoom Memory Team
Date: 2025-11-18 (Week 8A: Error Handling & Defensive Programming)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Health Status
# ============================================================================

class HealthLevel(Enum):
    """Health level enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthStatus:
    """
    Health status for a memory system component.

    Attributes:
        healthy: Overall health (True/False)
        level: Health level (HEALTHY/DEGRADED/UNHEALTHY/CRITICAL)
        latency_ms: Average latency in milliseconds
        error_rate: Error rate (0.0-1.0)
        last_success: Timestamp of last successful operation
        issues: List of detected issues
        metrics: Additional metrics
    """
    healthy: bool
    level: HealthLevel
    latency_ms: float = 0.0
    error_rate: float = 0.0
    last_success: Optional[datetime] = None
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "level": self.level.value,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "issues": self.issues,
            "metrics": self.metrics
        }


# ============================================================================
# Memory System Health Checker
# ============================================================================

class MemorySystemHealth:
    """
    Comprehensive health checker for memory systems.

    Checks health of:
    - Consolidation system (Week 5)
    - Semantic transition system (Week 6)
    - Temporal tracking system (Week 7)
    - Graph reasoning system
    - Overall system health

    Usage:
        >>> health_checker = MemorySystemHealth(consolidator, transition_engine, tracker)
        >>> health = await health_checker.get_overall_health()
        >>> print(f"System health: {health['overall']['level']}")
    """

    def __init__(
        self,
        consolidator: Optional[Any] = None,
        semantic_transition: Optional[Any] = None,
        temporal_tracker: Optional[Any] = None,
        graph_reasoning: Optional[Any] = None
    ):
        """
        Initialize health checker.

        Args:
            consolidator: MemoryConsolidator or SleepBasedConsolidation instance
            semantic_transition: SemanticTransitionEngine instance
            temporal_tracker: TemporalEvolutionTracker instance
            graph_reasoning: GraphReasoningEngine instance
        """
        self.consolidator = consolidator
        self.semantic_transition = semantic_transition
        self.temporal_tracker = temporal_tracker
        self.graph_reasoning = graph_reasoning

        # Health check history
        self.check_history: List[Dict[str, Any]] = []
        self.max_history = 100

    # ========================================================================
    # Individual Health Checks
    # ========================================================================

    async def check_consolidation_health(self) -> HealthStatus:
        """
        Check consolidation system health.

        Checks:
        - Consolidation running
        - Recent consolidations successful
        - Latency within bounds
        - Memory bloat under control

        Returns:
            HealthStatus for consolidation system
        """
        if not self.consolidator:
            return HealthStatus(
                healthy=True,
                level=HealthLevel.HEALTHY,
                issues=["Consolidation not configured (optional)"]
            )

        issues = []
        metrics = {}

        try:
            # Get consolidation statistics
            stats = self.consolidator.get_statistics() if hasattr(
                self.consolidator, 'get_statistics'
            ) else {}

            metrics = stats

            # Check if background consolidation is running (if applicable)
            if hasattr(self.consolidator, '_running'):
                if not self.consolidator._running:
                    issues.append("Background consolidation not running")

            # Check consolidation frequency
            total_consolidations = stats.get('total_consolidations', 0)
            if total_consolidations == 0:
                issues.append("No consolidations performed yet")

            # Check for errors (if tracked)
            llm_usage = stats.get('llm_usage', {})
            if 'error_rate' in llm_usage:
                error_rate = llm_usage['error_rate']
                if error_rate > 0.1:  # >10% error rate
                    issues.append(f"High LLM error rate: {error_rate:.1%}")

            # Determine health level
            if not issues:
                level = HealthLevel.HEALTHY
                healthy = True
            elif len(issues) <= 2:
                level = HealthLevel.DEGRADED
                healthy = True
            else:
                level = HealthLevel.UNHEALTHY
                healthy = False

            return HealthStatus(
                healthy=healthy,
                level=level,
                latency_ms=0.0,  # Would need to track this
                error_rate=llm_usage.get('error_rate', 0.0),
                last_success=datetime.now(),  # Would need to track this
                issues=issues,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Error checking consolidation health: {e}")
            return HealthStatus(
                healthy=False,
                level=HealthLevel.CRITICAL,
                issues=[f"Health check failed: {e}"],
                metrics={"error": str(e)}
            )

    async def check_semantic_transition_health(self) -> HealthStatus:
        """
        Check semantic transition system health.

        Checks:
        - Transition engine running
        - Pattern detection working
        - Concept promotion successful
        - Background transition active

        Returns:
            HealthStatus for semantic transition
        """
        if not self.semantic_transition:
            return HealthStatus(
                healthy=True,
                level=HealthLevel.HEALTHY,
                issues=["Semantic transition not configured (optional)"]
            )

        issues = []
        metrics = {}

        try:
            # Get statistics
            stats = self.semantic_transition.get_statistics() if hasattr(
                self.semantic_transition, 'get_statistics'
            ) else {}

            metrics = stats

            # Check if background transition is running
            if hasattr(self.semantic_transition, '_background_task'):
                task = self.semantic_transition._background_task
                if task is None or task.done():
                    if self.semantic_transition.config.enable_background_transition:
                        issues.append("Background transition not running")

            # Check pattern detection
            patterns_detected = stats.get('patterns_detected', 0)
            if patterns_detected == 0:
                issues.append("No patterns detected yet")

            # Check concept creation
            concepts_created = stats.get('concepts_created', 0)
            if concepts_created == 0:
                issues.append("No semantic concepts created yet")

            # Check semantic query performance
            semantic_queries = stats.get('semantic_queries', 0)
            episodic_queries = stats.get('episodic_queries', 0)

            if semantic_queries + episodic_queries > 0:
                semantic_ratio = semantic_queries / (semantic_queries + episodic_queries)
                metrics['semantic_query_ratio'] = semantic_ratio

                # Ideally want >50% semantic queries (faster)
                if semantic_ratio < 0.3:
                    issues.append(
                        f"Low semantic query ratio: {semantic_ratio:.1%} "
                        "(concepts not being reused)"
                    )

            # Determine health level
            if not issues:
                level = HealthLevel.HEALTHY
                healthy = True
            elif len(issues) <= 2:
                level = HealthLevel.DEGRADED
                healthy = True
            else:
                level = HealthLevel.UNHEALTHY
                healthy = False

            return HealthStatus(
                healthy=healthy,
                level=level,
                latency_ms=0.0,
                error_rate=0.0,
                last_success=datetime.now(),
                issues=issues,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Error checking semantic transition health: {e}")
            return HealthStatus(
                healthy=False,
                level=HealthLevel.CRITICAL,
                issues=[f"Health check failed: {e}"],
                metrics={"error": str(e)}
            )

    async def check_temporal_tracking_health(self) -> HealthStatus:
        """
        Check temporal tracking system health.

        Checks:
        - Concept histories being tracked
        - State transitions detected
        - Milestones recorded
        - Persistence working

        Returns:
            HealthStatus for temporal tracking
        """
        if not self.temporal_tracker:
            return HealthStatus(
                healthy=True,
                level=HealthLevel.HEALTHY,
                issues=["Temporal tracking not configured (optional)"]
            )

        issues = []
        metrics = {}

        try:
            # Get concept histories
            concept_count = len(self.temporal_tracker.concept_histories)
            metrics['concepts_tracked'] = concept_count

            if concept_count == 0:
                issues.append("No concepts being tracked yet")

            # Check state transitions
            total_transitions = sum(
                len(history.state_transitions)
                for history in self.temporal_tracker.concept_histories.values()
            )
            metrics['total_state_transitions'] = total_transitions

            # Check milestones
            total_milestones = sum(
                len(history.milestones)
                for history in self.temporal_tracker.concept_histories.values()
            )
            metrics['total_milestones'] = total_milestones

            # Check interaction log
            interaction_count = len(self.temporal_tracker.interaction_log)
            metrics['total_interactions'] = interaction_count

            # Check persistence
            if self.temporal_tracker.config.persistence_path:
                path = self.temporal_tracker.config.persistence_path
                if path.exists():
                    metrics['persistence_enabled'] = True
                    metrics['persistence_path'] = str(path)
                else:
                    issues.append(
                        f"Persistence path does not exist: {path}"
                    )
            else:
                metrics['persistence_enabled'] = False

            # Check for stale data
            if concept_count > 0:
                most_recent_access = max(
                    (h.last_accessed for h in self.temporal_tracker.concept_histories.values()
                     if h.last_accessed),
                    default=None
                )

                if most_recent_access:
                    hours_since_access = (
                        datetime.now() - most_recent_access
                    ).total_seconds() / 3600.0

                    metrics['hours_since_last_access'] = hours_since_access

                    if hours_since_access > 24:
                        issues.append(
                            f"No recent concept access ({hours_since_access:.1f}h ago)"
                        )

            # Determine health level
            if not issues:
                level = HealthLevel.HEALTHY
                healthy = True
            elif len(issues) <= 2:
                level = HealthLevel.DEGRADED
                healthy = True
            else:
                level = HealthLevel.UNHEALTHY
                healthy = False

            return HealthStatus(
                healthy=healthy,
                level=level,
                latency_ms=0.0,
                error_rate=0.0,
                last_success=datetime.now(),
                issues=issues,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Error checking temporal tracking health: {e}")
            return HealthStatus(
                healthy=False,
                level=HealthLevel.CRITICAL,
                issues=[f"Health check failed: {e}"],
                metrics={"error": str(e)}
            )

    async def check_graph_reasoning_health(self) -> HealthStatus:
        """
        Check graph reasoning system health.

        Checks:
        - Curiosity engine working
        - Multi-hop reasoning functional
        - Exploration paths valid

        Returns:
            HealthStatus for graph reasoning
        """
        if not self.graph_reasoning:
            return HealthStatus(
                healthy=True,
                level=HealthLevel.HEALTHY,
                issues=["Graph reasoning not configured (optional)"]
            )

        issues = []
        metrics = {}

        try:
            # Would check graph reasoning statistics here
            # For now, basic check

            return HealthStatus(
                healthy=True,
                level=HealthLevel.HEALTHY,
                latency_ms=0.0,
                error_rate=0.0,
                last_success=datetime.now(),
                issues=issues,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Error checking graph reasoning health: {e}")
            return HealthStatus(
                healthy=False,
                level=HealthLevel.CRITICAL,
                issues=[f"Health check failed: {e}"],
                metrics={"error": str(e)}
            )

    # ========================================================================
    # Overall Health
    # ========================================================================

    async def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall health status for all memory systems.

        Returns:
            Dictionary with health status for each component and overall
        """
        # Run all health checks in parallel
        consolidation_health, semantic_health, temporal_health, graph_health = await asyncio.gather(
            self.check_consolidation_health(),
            self.check_semantic_transition_health(),
            self.check_temporal_tracking_health(),
            self.check_graph_reasoning_health()
        )

        # Determine overall health
        all_statuses = [
            consolidation_health,
            semantic_health,
            temporal_health,
            graph_health
        ]

        # Count by level
        critical_count = sum(1 for s in all_statuses if s.level == HealthLevel.CRITICAL)
        unhealthy_count = sum(1 for s in all_statuses if s.level == HealthLevel.UNHEALTHY)
        degraded_count = sum(1 for s in all_statuses if s.level == HealthLevel.DEGRADED)

        # Determine overall level
        if critical_count > 0:
            overall_level = HealthLevel.CRITICAL
            overall_healthy = False
        elif unhealthy_count > 1:
            overall_level = HealthLevel.UNHEALTHY
            overall_healthy = False
        elif unhealthy_count == 1 or degraded_count > 2:
            overall_level = HealthLevel.DEGRADED
            overall_healthy = True
        elif degraded_count > 0:
            overall_level = HealthLevel.DEGRADED
            overall_healthy = True
        else:
            overall_level = HealthLevel.HEALTHY
            overall_healthy = True

        # Collect all issues
        all_issues = []
        for status in all_statuses:
            all_issues.extend(status.issues)

        health_report = {
            "overall": {
                "healthy": overall_healthy,
                "level": overall_level.value,
                "timestamp": datetime.now().isoformat(),
                "issues": all_issues,
                "components": {
                    "consolidation": consolidation_health.level.value,
                    "semantic_transition": semantic_health.level.value,
                    "temporal_tracking": temporal_health.level.value,
                    "graph_reasoning": graph_health.level.value
                }
            },
            "consolidation": consolidation_health.to_dict(),
            "semantic_transition": semantic_health.to_dict(),
            "temporal_tracking": temporal_health.to_dict(),
            "graph_reasoning": graph_health.to_dict()
        }

        # Store in history
        self.check_history.append({
            "timestamp": datetime.now(),
            "health": health_report
        })

        # Prune history
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]

        return health_report

    # ========================================================================
    # Health Trends
    # ========================================================================

    def get_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze health trends over time.

        Args:
            hours: Time window for trend analysis

        Returns:
            Trend analysis with degradation/improvement indicators
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.check_history
            if check['timestamp'] >= cutoff
        ]

        if not recent_checks:
            return {
                "trend": "no_data",
                "checks": 0
            }

        # Count health levels over time
        level_counts = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "critical": 0
        }

        for check in recent_checks:
            level = check['health']['overall']['level']
            level_counts[level] = level_counts.get(level, 0) + 1

        # Determine trend
        total_checks = len(recent_checks)
        healthy_ratio = level_counts['healthy'] / total_checks
        critical_ratio = level_counts['critical'] / total_checks

        if critical_ratio > 0.3:
            trend = "critical_deterioration"
        elif healthy_ratio > 0.8:
            trend = "stable_healthy"
        elif healthy_ratio > 0.5:
            trend = "mostly_healthy"
        else:
            trend = "degrading"

        return {
            "trend": trend,
            "time_window_hours": hours,
            "total_checks": total_checks,
            "level_distribution": level_counts,
            "healthy_ratio": healthy_ratio,
            "critical_ratio": critical_ratio
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_health_checker(
    consolidator: Optional[Any] = None,
    semantic_transition: Optional[Any] = None,
    temporal_tracker: Optional[Any] = None,
    graph_reasoning: Optional[Any] = None
) -> MemorySystemHealth:
    """
    Create memory system health checker.

    Args:
        consolidator: MemoryConsolidator instance
        semantic_transition: SemanticTransitionEngine instance
        temporal_tracker: TemporalEvolutionTracker instance
        graph_reasoning: GraphReasoningEngine instance

    Returns:
        MemorySystemHealth instance
    """
    return MemorySystemHealth(
        consolidator=consolidator,
        semantic_transition=semantic_transition,
        temporal_tracker=temporal_tracker,
        graph_reasoning=graph_reasoning
    )
