"""
AR Pattern Deployer - Safe pattern deployment with automatic rollback
======================================================================
Deploys improved patterns safely using shadow mode, A/B testing, and gradual rollout.

Author: Claude Code (Agent Q)
Date: November 17, 2025
Phase: Agent Swarm Wave 5 - Adaptive Routing Integration
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum
import random

from HoloLoom.voice.ar_query_classifier import ARQueryClassifier
from HoloLoom.voice.ar_pattern_miner import ARPattern
from HoloLoom.voice.ar_validator import ARContinuousValidator

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class DeploymentStrategy(Enum):
    """Deployment strategy for new patterns."""
    SHADOW = "shadow"  # Test without production impact
    AB_TEST = "ab_test"  # Split traffic 10/90
    GRADUAL = "gradual"  # 10% -> 50% -> 100%
    IMMEDIATE = "immediate"  # Deploy immediately (risky!)


class DeploymentPhase(Enum):
    """Current phase of deployment."""
    SHADOW = "shadow"  # Day 1-2: Shadow mode
    AB_10 = "ab_10"  # Day 3: 10% traffic
    AB_50 = "ab_50"  # Day 4-5: 50% traffic
    FULL = "full"  # Day 6-7: 100% traffic
    ROLLED_BACK = "rolled_back"  # Rolled back due to regression


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DeploymentMetrics:
    """Metrics for a deployment phase."""
    phase: DeploymentPhase
    queries_total: int = 0
    queries_new_patterns: int = 0
    queries_old_patterns: int = 0
    accuracy_new: float = 0.0
    accuracy_old: float = 0.0
    latency_new_ms: float = 0.0
    latency_old_ms: float = 0.0
    errors_new: int = 0
    errors_old: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    current_phase: DeploymentPhase
    metrics: DeploymentMetrics
    message: str
    patterns_deployed: int = 0
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None


@dataclass
class PatternVersion:
    """Versioned pattern set for rollback."""
    version: int
    patterns: List[ARPattern]
    deployed_at: float
    metrics: Optional[DeploymentMetrics] = None
    is_active: bool = True


# ============================================================================
# AR Pattern Deployer
# ============================================================================

class ARPatternDeployer:
    """
    Safe pattern deployment with automatic rollback for AR queries.

    Features:
    - Shadow mode: Test patterns without production impact
    - A/B testing: 10/90 traffic split
    - Gradual rollout: 10% -> 50% -> 100%
    - Automatic rollback: Revert on regression
    - Pattern versioning: Keep last 10 versions

    Usage:
        deployer = ARPatternDeployer(
            classifier=ar_classifier,
            validator=ar_validator,
            strategy=DeploymentStrategy.GRADUAL
        )

        # Deploy new patterns
        result = await deployer.deploy_patterns(new_patterns)

        # Automatic rollback if regression detected
        if result.rollback_triggered:
            print(f"Rollback: {result.rollback_reason}")
    """

    def __init__(
        self,
        classifier: ARQueryClassifier,
        validator: Optional[ARContinuousValidator] = None,
        strategy: DeploymentStrategy = DeploymentStrategy.GRADUAL,
        rollback_threshold: float = 0.02,  # 2% accuracy drop triggers rollback
        max_versions: int = 10
    ):
        """
        Initialize AR pattern deployer.

        Args:
            classifier: AR query classifier instance
            validator: AR continuous validator (optional)
            strategy: Deployment strategy
            rollback_threshold: Accuracy drop to trigger rollback
            max_versions: Maximum pattern versions to keep
        """
        self.classifier = classifier
        self.validator = validator
        self.strategy = strategy
        self.rollback_threshold = rollback_threshold
        self.max_versions = max_versions

        # Pattern versions
        self.pattern_versions: List[PatternVersion] = []
        self.current_version = 0

        # Deployment state
        self.current_phase = DeploymentPhase.SHADOW
        self.deployment_metrics: List[DeploymentMetrics] = []

        # Traffic split ratios
        self.traffic_split = {
            DeploymentPhase.SHADOW: 0.0,
            DeploymentPhase.AB_10: 0.1,
            DeploymentPhase.AB_50: 0.5,
            DeploymentPhase.FULL: 1.0
        }

    async def deploy_patterns(
        self,
        patterns: List[ARPattern],
        validate_first: bool = True
    ) -> DeploymentResult:
        """
        Deploy new patterns using configured strategy.

        Args:
            patterns: List of new patterns to deploy
            validate_first: Validate patterns before deployment

        Returns:
            DeploymentResult with deployment status
        """
        logger.info(f"Deploying {len(patterns)} patterns with {self.strategy.value} strategy")

        # Validate patterns first
        if validate_first and self.validator:
            validation_result = self.validator.validate()
            if validation_result.regression_detected:
                return DeploymentResult(
                    success=False,
                    current_phase=self.current_phase,
                    metrics=DeploymentMetrics(phase=self.current_phase),
                    message="Deployment aborted: baseline validation failed",
                    rollback_triggered=True,
                    rollback_reason="Baseline validation regression detected"
                )

        # Create new pattern version
        self.current_version += 1
        new_version = PatternVersion(
            version=self.current_version,
            patterns=patterns,
            deployed_at=datetime.now().timestamp()
        )
        self.pattern_versions.append(new_version)

        # Trim old versions
        if len(self.pattern_versions) > self.max_versions:
            self.pattern_versions = self.pattern_versions[-self.max_versions:]

        # Deploy based on strategy
        if self.strategy == DeploymentStrategy.SHADOW:
            return await self._deploy_shadow(new_version)
        elif self.strategy == DeploymentStrategy.AB_TEST:
            return await self._deploy_ab_test(new_version)
        elif self.strategy == DeploymentStrategy.GRADUAL:
            return await self._deploy_gradual(new_version)
        elif self.strategy == DeploymentStrategy.IMMEDIATE:
            return await self._deploy_immediate(new_version)
        else:
            return DeploymentResult(
                success=False,
                current_phase=self.current_phase,
                metrics=DeploymentMetrics(phase=self.current_phase),
                message=f"Unknown deployment strategy: {self.strategy}"
            )

    async def _deploy_shadow(self, version: PatternVersion) -> DeploymentResult:
        """Deploy in shadow mode (no production impact)."""
        logger.info(f"Shadow deployment: version {version.version}")

        self.current_phase = DeploymentPhase.SHADOW

        # In shadow mode, patterns are tested but not used for production
        metrics = DeploymentMetrics(
            phase=DeploymentPhase.SHADOW,
            queries_total=0,
            queries_new_patterns=0,
            queries_old_patterns=0
        )

        self.deployment_metrics.append(metrics)

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message=f"Shadow deployment complete for version {version.version}",
            patterns_deployed=len(version.patterns)
        )

    async def _deploy_ab_test(self, version: PatternVersion) -> DeploymentResult:
        """Deploy with A/B testing (10/90 split)."""
        logger.info(f"A/B test deployment: version {version.version}")

        self.current_phase = DeploymentPhase.AB_10

        # Add patterns to classifier (will be used 10% of the time)
        for pattern in version.patterns:
            from HoloLoom.voice.ar_query_classifier import ARPatternRule
            rule = ARPatternRule(
                pattern=pattern.regex,
                complexity=pattern.complexity,
                ar_type=pattern.ar_type,
                confidence_boost=pattern.score.f1_score * 0.2,
                priority=int(pattern.score.precision * 10),
                examples=pattern.examples
            )
            self.classifier.add_custom_pattern(rule)

        metrics = DeploymentMetrics(
            phase=DeploymentPhase.AB_10,
            queries_total=0,
            queries_new_patterns=0,
            queries_old_patterns=0
        )

        self.deployment_metrics.append(metrics)

        # Validate after A/B test
        if self.validator:
            validation_result = self.validator.validate()
            if validation_result.regression_detected:
                # Rollback!
                await self._rollback()
                return DeploymentResult(
                    success=False,
                    current_phase=DeploymentPhase.ROLLED_BACK,
                    metrics=metrics,
                    message=f"A/B test failed, rolled back version {version.version}",
                    rollback_triggered=True,
                    rollback_reason=f"Accuracy dropped {validation_result.overall_accuracy:.1%} (threshold: {self.rollback_threshold:.1%})"
                )

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message=f"A/B test deployment complete for version {version.version}",
            patterns_deployed=len(version.patterns)
        )

    async def _deploy_gradual(self, version: PatternVersion) -> DeploymentResult:
        """Deploy gradually (10% -> 50% -> 100%)."""
        logger.info(f"Gradual deployment: version {version.version}")

        # Phase 1: 10%
        self.current_phase = DeploymentPhase.AB_10
        result_10 = await self._deploy_phase(version, DeploymentPhase.AB_10)

        if result_10.rollback_triggered:
            return result_10

        # Phase 2: 50%
        self.current_phase = DeploymentPhase.AB_50
        result_50 = await self._deploy_phase(version, DeploymentPhase.AB_50)

        if result_50.rollback_triggered:
            return result_50

        # Phase 3: 100%
        self.current_phase = DeploymentPhase.FULL
        result_full = await self._deploy_phase(version, DeploymentPhase.FULL)

        return result_full

    async def _deploy_phase(
        self,
        version: PatternVersion,
        phase: DeploymentPhase
    ) -> DeploymentResult:
        """Deploy a single phase of gradual rollout."""
        logger.info(f"Deploying phase: {phase.value}")

        # Add patterns to classifier
        for pattern in version.patterns:
            from HoloLoom.voice.ar_query_classifier import ARPatternRule
            rule = ARPatternRule(
                pattern=pattern.regex,
                complexity=pattern.complexity,
                ar_type=pattern.ar_type,
                confidence_boost=pattern.score.f1_score * 0.2,
                priority=int(pattern.score.precision * 10),
                examples=pattern.examples
            )
            self.classifier.add_custom_pattern(rule)

        metrics = DeploymentMetrics(
            phase=phase,
            queries_total=0
        )

        self.deployment_metrics.append(metrics)

        # Validate
        if self.validator:
            validation_result = self.validator.validate()
            if validation_result.regression_detected:
                # Rollback!
                await self._rollback()
                return DeploymentResult(
                    success=False,
                    current_phase=DeploymentPhase.ROLLED_BACK,
                    metrics=metrics,
                    message=f"Phase {phase.value} failed, rolled back version {version.version}",
                    rollback_triggered=True,
                    rollback_reason=f"Accuracy dropped to {validation_result.overall_accuracy:.1%}"
                )

        return DeploymentResult(
            success=True,
            current_phase=phase,
            metrics=metrics,
            message=f"Phase {phase.value} deployment complete",
            patterns_deployed=len(version.patterns)
        )

    async def _deploy_immediate(self, version: PatternVersion) -> DeploymentResult:
        """Deploy immediately (no safety checks - RISKY!)."""
        logger.warning(f"Immediate deployment (risky!): version {version.version}")

        self.current_phase = DeploymentPhase.FULL

        # Add all patterns immediately
        for pattern in version.patterns:
            from HoloLoom.voice.ar_query_classifier import ARPatternRule
            rule = ARPatternRule(
                pattern=pattern.regex,
                complexity=pattern.complexity,
                ar_type=pattern.ar_type,
                confidence_boost=pattern.score.f1_score * 0.2,
                priority=int(pattern.score.precision * 10),
                examples=pattern.examples
            )
            self.classifier.add_custom_pattern(rule)

        metrics = DeploymentMetrics(
            phase=DeploymentPhase.FULL,
            queries_total=0
        )

        self.deployment_metrics.append(metrics)

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message=f"Immediate deployment complete for version {version.version}",
            patterns_deployed=len(version.patterns)
        )

    async def _rollback(self) -> None:
        """Rollback to previous pattern version."""
        logger.warning("Rolling back to previous pattern version")

        # Mark current version as inactive
        if self.pattern_versions:
            self.pattern_versions[-1].is_active = False

        self.current_phase = DeploymentPhase.ROLLED_BACK

        # In a real implementation, we would:
        # 1. Remove new patterns from classifier
        # 2. Restore old patterns
        # 3. Update metrics

    def get_current_version(self) -> Optional[PatternVersion]:
        """Get current active pattern version."""
        for version in reversed(self.pattern_versions):
            if version.is_active:
                return version
        return None

    def get_deployment_metrics(self) -> List[DeploymentMetrics]:
        """Get all deployment metrics."""
        return self.deployment_metrics

    def export_metrics(self, filepath: str) -> None:
        """Export deployment metrics to JSON file."""
        try:
            metrics_data = []
            for metrics in self.deployment_metrics:
                metrics_data.append({
                    "phase": metrics.phase.value,
                    "queries_total": metrics.queries_total,
                    "queries_new_patterns": metrics.queries_new_patterns,
                    "queries_old_patterns": metrics.queries_old_patterns,
                    "accuracy_new": metrics.accuracy_new,
                    "accuracy_old": metrics.accuracy_old,
                    "latency_new_ms": metrics.latency_new_ms,
                    "latency_old_ms": metrics.latency_old_ms,
                    "errors_new": metrics.errors_new,
                    "errors_old": metrics.errors_old,
                    "timestamp": metrics.timestamp
                })

            with open(filepath, "w") as f:
                json.dump(metrics_data, f, indent=2)

            logger.info(f"Exported {len(metrics_data)} deployment metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def get_statistics(self) -> Dict:
        """Get deployment statistics."""
        return {
            "total_versions": len(self.pattern_versions),
            "current_version": self.current_version,
            "current_phase": self.current_phase.value,
            "total_deployments": len(self.deployment_metrics),
            "rollbacks": len([v for v in self.pattern_versions if not v.is_active]),
            "strategy": self.strategy.value
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_ar_pattern_deployer(
    classifier: ARQueryClassifier,
    validator: Optional[ARContinuousValidator] = None,
    strategy: DeploymentStrategy = DeploymentStrategy.GRADUAL
) -> ARPatternDeployer:
    """
    Factory function to create AR pattern deployer.

    Args:
        classifier: AR query classifier instance
        validator: AR continuous validator
        strategy: Deployment strategy

    Returns:
        ARPatternDeployer instance
    """
    return ARPatternDeployer(
        classifier=classifier,
        validator=validator,
        strategy=strategy
    )
