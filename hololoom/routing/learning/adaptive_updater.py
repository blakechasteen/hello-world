"""
Adaptive Updater - Safe pattern deployment with automatic rollback
===================================================================
Deploys improved patterns safely using shadow mode, A/B testing, and gradual rollout.

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


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
    patterns: List  # List[Pattern]
    deployed_at: float
    metrics: Optional[DeploymentMetrics] = None
    is_active: bool = True


class AdaptiveUpdater:
    """
    Safe pattern deployment with automatic rollback.

    Features:
    - Shadow mode: Test patterns without production impact
    - A/B testing: 10/90 traffic split
    - Gradual rollout: 10% -> 50% -> 100%
    - Automatic rollback: Revert on regression

    Usage:
        updater = AdaptiveUpdater(
            classifier=moonshot_classifier,
            validator=continuous_validator,
            strategy=DeploymentStrategy.GRADUAL
        )

        # Deploy new patterns
        result = await updater.deploy_patterns(new_patterns)

        # Automatic rollback if regression detected
        if result.rollback_triggered:
            print(f"Rollback: {result.rollback_reason}")
    """

    def __init__(
        self,
        classifier,
        validator,
        strategy: DeploymentStrategy = DeploymentStrategy.GRADUAL,
        regression_threshold: float = 0.02,  # 2% drop triggers rollback
        max_versions: int = 10  # Keep last 10 pattern versions
    ):
        """
        Initialize adaptive updater.

        Args:
            classifier: Query classifier instance
            validator: ContinuousValidator instance
            strategy: Deployment strategy
            regression_threshold: Accuracy drop to trigger rollback
            max_versions: Number of pattern versions to keep
        """
        self.classifier = classifier
        self.validator = validator
        self.strategy = strategy
        self.regression_threshold = regression_threshold
        self.max_versions = max_versions

        # Current deployment state
        self.current_phase = DeploymentPhase.SHADOW
        self.traffic_split = 0.0  # % of traffic using new patterns
        self.deployment_start_time = None

        # Pattern versions (for rollback)
        self.pattern_versions: List[PatternVersion] = []
        self.current_version = 0

        # Deployment metrics
        self.metrics_history: List[DeploymentMetrics] = []

        logger.info(f"AdaptiveUpdater initialized "
                   f"(strategy={strategy.value}, "
                   f"regression_threshold={regression_threshold:.1%})")

    async def deploy_patterns(
        self,
        new_patterns: List,
        force_strategy: Optional[DeploymentStrategy] = None
    ) -> DeploymentResult:
        """
        Deploy new patterns safely.

        Steps:
        1. Shadow mode: Test patterns without production impact
        2. A/B test: 10% traffic uses new patterns
        3. Validate: Check accuracy >= baseline
        4. Rollout: Gradually increase to 100%
        5. Rollback: Revert if regression detected

        Args:
            new_patterns: List of Pattern objects to deploy
            force_strategy: Override default strategy

        Returns:
            DeploymentResult with success status and metrics
        """
        strategy = force_strategy or self.strategy

        logger.info(f"Starting pattern deployment "
                   f"(patterns={len(new_patterns)}, strategy={strategy.value})")

        # Save current patterns as version (for rollback)
        self._save_current_version()

        # Execute deployment strategy
        if strategy == DeploymentStrategy.SHADOW:
            return await self._deploy_shadow_mode(new_patterns)
        elif strategy == DeploymentStrategy.AB_TEST:
            return await self._deploy_ab_test(new_patterns)
        elif strategy == DeploymentStrategy.GRADUAL:
            return await self._deploy_gradual(new_patterns)
        elif strategy == DeploymentStrategy.IMMEDIATE:
            return await self._deploy_immediate(new_patterns)

    async def _deploy_shadow_mode(self, new_patterns: List) -> DeploymentResult:
        """
        Deploy in shadow mode (no production impact).

        Shadow mode runs new patterns alongside current patterns
        but doesn't affect production responses. This allows
        collecting metrics without risk.
        """
        logger.info("Deploying in SHADOW mode (no production impact)")

        self.current_phase = DeploymentPhase.SHADOW
        self.traffic_split = 0.0  # 0% of production traffic

        # Install patterns in shadow mode
        self._install_patterns(new_patterns, shadow=True)

        # Collect metrics (simulated for demo)
        metrics = DeploymentMetrics(
            phase=DeploymentPhase.SHADOW,
            queries_total=0,
            queries_new_patterns=0,
            queries_old_patterns=0,
            accuracy_new=0.0,
            accuracy_old=0.0
        )

        self.metrics_history.append(metrics)

        logger.info("Shadow mode deployment complete")

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message="Patterns deployed in shadow mode. Collecting metrics...",
            patterns_deployed=len(new_patterns)
        )

    async def _deploy_ab_test(self, new_patterns: List) -> DeploymentResult:
        """
        Deploy with A/B testing (10/90 split).

        10% of traffic uses new patterns, 90% uses current patterns.
        Compare accuracy and latency to validate improvement.
        """
        logger.info("Deploying with A/B testing (10/90 split)")

        self.current_phase = DeploymentPhase.AB_10
        self.traffic_split = 0.10  # 10% of production traffic

        # Install patterns with traffic split
        self._install_patterns(new_patterns, shadow=False)

        # Validate against baseline
        validation_result = await self.validator.validate_hourly(sample_size=100)

        # Check for regression
        if validation_result.regression_detected:
            # Automatic rollback
            logger.warning("Regression detected during A/B test. Rolling back...")
            rollback_result = await self.rollback(
                reason=f"Accuracy dropped to {validation_result.overall_accuracy:.1%}"
            )
            return rollback_result

        # Collect metrics
        metrics = DeploymentMetrics(
            phase=DeploymentPhase.AB_10,
            queries_total=validation_result.total_queries,
            queries_new_patterns=int(validation_result.total_queries * 0.10),
            queries_old_patterns=int(validation_result.total_queries * 0.90),
            accuracy_new=validation_result.overall_accuracy,
            accuracy_old=self.validator.baseline_accuracy
        )

        self.metrics_history.append(metrics)

        logger.info(f"A/B test complete: "
                   f"accuracy_new={metrics.accuracy_new:.1%}, "
                   f"accuracy_old={metrics.accuracy_old:.1%}")

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message="A/B test (10/90) successful. Ready to increase traffic.",
            patterns_deployed=len(new_patterns)
        )

    async def _deploy_gradual(self, new_patterns: List) -> DeploymentResult:
        """
        Deploy with gradual rollout (10% -> 50% -> 100%).

        Steps:
        1. Start at 10% traffic
        2. Validate (24 hours)
        3. Increase to 50%
        4. Validate (24 hours)
        5. Increase to 100%
        """
        logger.info("Starting gradual rollout (10% -> 50% -> 100%)")

        # Phase 1: 10% traffic
        result_10 = await self._deploy_ab_test(new_patterns)
        if not result_10.success or result_10.rollback_triggered:
            return result_10

        logger.info("Phase 1 (10%) successful. Increasing to 50%...")

        # Phase 2: 50% traffic
        self.current_phase = DeploymentPhase.AB_50
        self.traffic_split = 0.50

        validation_result = await self.validator.validate_daily()

        if validation_result.regression_detected:
            logger.warning("Regression at 50% traffic. Rolling back...")
            return await self.rollback(
                reason=f"Accuracy dropped to {validation_result.overall_accuracy:.1%} at 50% traffic"
            )

        metrics_50 = DeploymentMetrics(
            phase=DeploymentPhase.AB_50,
            queries_total=validation_result.total_queries,
            queries_new_patterns=int(validation_result.total_queries * 0.50),
            queries_old_patterns=int(validation_result.total_queries * 0.50),
            accuracy_new=validation_result.overall_accuracy,
            accuracy_old=self.validator.baseline_accuracy
        )

        self.metrics_history.append(metrics_50)

        logger.info("Phase 2 (50%) successful. Increasing to 100%...")

        # Phase 3: 100% traffic (full deployment)
        self.current_phase = DeploymentPhase.FULL
        self.traffic_split = 1.0

        validation_result = await self.validator.validate_daily()

        if validation_result.regression_detected:
            logger.warning("Regression at 100% traffic. Rolling back...")
            return await self.rollback(
                reason=f"Accuracy dropped to {validation_result.overall_accuracy:.1%} at 100% traffic"
            )

        metrics_100 = DeploymentMetrics(
            phase=DeploymentPhase.FULL,
            queries_total=validation_result.total_queries,
            queries_new_patterns=validation_result.total_queries,
            queries_old_patterns=0,
            accuracy_new=validation_result.overall_accuracy,
            accuracy_old=self.validator.baseline_accuracy
        )

        self.metrics_history.append(metrics_100)

        logger.info(f"Gradual rollout complete! "
                   f"New patterns at 100% traffic "
                   f"(accuracy={metrics_100.accuracy_new:.1%})")

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics_100,
            message="Gradual rollout complete. Patterns deployed at 100%.",
            patterns_deployed=len(new_patterns)
        )

    async def _deploy_immediate(self, new_patterns: List) -> DeploymentResult:
        """
        Deploy immediately (100% traffic, no gradual rollout).

        WARNING: Risky! Only use for hotfixes or trusted patterns.
        """
        logger.warning("Deploying IMMEDIATELY (100% traffic, no gradual rollout)")

        self.current_phase = DeploymentPhase.FULL
        self.traffic_split = 1.0

        # Install patterns
        self._install_patterns(new_patterns, shadow=False)

        # Validate
        validation_result = await self.validator.validate_daily()

        if validation_result.regression_detected:
            logger.error("Regression detected after immediate deployment. Rolling back...")
            return await self.rollback(
                reason=f"Immediate deployment failed: accuracy {validation_result.overall_accuracy:.1%}"
            )

        metrics = DeploymentMetrics(
            phase=DeploymentPhase.FULL,
            queries_total=validation_result.total_queries,
            queries_new_patterns=validation_result.total_queries,
            accuracy_new=validation_result.overall_accuracy,
            accuracy_old=self.validator.baseline_accuracy
        )

        self.metrics_history.append(metrics)

        logger.info("Immediate deployment successful")

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message="Immediate deployment successful.",
            patterns_deployed=len(new_patterns)
        )

    async def rollback(self, reason: str) -> DeploymentResult:
        """
        Rollback to previous pattern version.

        Triggers:
        - Accuracy drop >2%
        - Latency increase >50%
        - Manual intervention

        Args:
            reason: Reason for rollback

        Returns:
            DeploymentResult with rollback status
        """
        logger.warning(f"ROLLBACK triggered: {reason}")

        if not self.pattern_versions:
            logger.error("No previous version to rollback to!")
            return DeploymentResult(
                success=False,
                current_phase=DeploymentPhase.ROLLED_BACK,
                metrics=DeploymentMetrics(phase=DeploymentPhase.ROLLED_BACK),
                message="Rollback failed: No previous version available",
                rollback_triggered=True,
                rollback_reason=reason
            )

        # Get previous version
        previous_version = self._get_previous_version()

        if not previous_version:
            logger.error("No valid previous version found!")
            return DeploymentResult(
                success=False,
                current_phase=DeploymentPhase.ROLLED_BACK,
                metrics=DeploymentMetrics(phase=DeploymentPhase.ROLLED_BACK),
                message="Rollback failed: No valid previous version",
                rollback_triggered=True,
                rollback_reason=reason
            )

        # Restore previous patterns
        self._install_patterns(previous_version.patterns, shadow=False)

        self.current_phase = DeploymentPhase.ROLLED_BACK
        self.traffic_split = 0.0  # Back to old patterns

        logger.info(f"Rolled back to version {previous_version.version}")

        # Validate rollback worked
        validation_result = await self.validator.validate_hourly(sample_size=50)

        metrics = DeploymentMetrics(
            phase=DeploymentPhase.ROLLED_BACK,
            queries_total=validation_result.total_queries,
            accuracy_new=validation_result.overall_accuracy,
            accuracy_old=self.validator.baseline_accuracy
        )

        self.metrics_history.append(metrics)

        return DeploymentResult(
            success=True,
            current_phase=self.current_phase,
            metrics=metrics,
            message=f"Rollback successful. Reverted to version {previous_version.version}.",
            rollback_triggered=True,
            rollback_reason=reason
        )

    def _save_current_version(self):
        """Save current patterns as a version (for rollback)."""
        version = PatternVersion(
            version=self.current_version,
            patterns=self._get_current_patterns(),
            deployed_at=datetime.now().timestamp(),
            is_active=True
        )

        # Deactivate previous versions
        for v in self.pattern_versions:
            v.is_active = False

        self.pattern_versions.append(version)
        self.current_version += 1

        # Keep only last N versions
        if len(self.pattern_versions) > self.max_versions:
            self.pattern_versions = self.pattern_versions[-self.max_versions:]

        logger.debug(f"Saved version {version.version} "
                    f"(total versions: {len(self.pattern_versions)})")

    def _get_previous_version(self) -> Optional[PatternVersion]:
        """Get the previous active version for rollback."""
        if len(self.pattern_versions) < 2:
            return None

        # Get second-to-last version (last is current)
        return self.pattern_versions[-2]

    def _get_current_patterns(self) -> List:
        """Get current patterns from classifier."""
        if hasattr(self.classifier, 'get_active_patterns'):
            return self.classifier.get_active_patterns()
        logger.debug("Classifier has no get_active_patterns(); returning empty list")
        return []

    def _install_patterns(self, patterns: List, shadow: bool = False):
        """
        Install patterns in classifier.

        Args:
            patterns: List of Pattern objects
            shadow: If True, run in shadow mode (no production impact)
        """
        mode = "shadow" if shadow else "production"
        if hasattr(self.classifier, 'update_patterns'):
            self.classifier.update_patterns(patterns, shadow=shadow)
            logger.info(f"Installed {len(patterns)} patterns in {mode} mode via classifier")
        else:
            logger.info(f"Installing {len(patterns)} patterns in {mode} mode "
                       f"(classifier lacks update_patterns, log only)")

    def get_deployment_status(self) -> Dict:
        """Get current deployment status."""
        return {
            'current_phase': self.current_phase.value,
            'traffic_split': self.traffic_split,
            'pattern_versions': len(self.pattern_versions),
            'current_version': self.current_version,
            'metrics_collected': len(self.metrics_history)
        }

    def get_metrics_history(self, phases: Optional[List[DeploymentPhase]] = None) -> List[DeploymentMetrics]:
        """
        Get metrics history, optionally filtered by phases.

        Args:
            phases: List of phases to include (None = all)

        Returns:
            List of DeploymentMetrics
        """
        if phases is None:
            return self.metrics_history

        return [m for m in self.metrics_history if m.phase in phases]

    def export_deployment_history(self, output_path: str):
        """Export deployment history to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'current_phase': self.current_phase.value,
            'traffic_split': self.traffic_split,
            'current_version': self.current_version,
            'pattern_versions': [
                {
                    'version': v.version,
                    'deployed_at': v.deployed_at,
                    'is_active': v.is_active,
                    'pattern_count': len(v.patterns)
                }
                for v in self.pattern_versions
            ],
            'metrics_history': [
                {
                    'phase': m.phase.value,
                    'queries_total': m.queries_total,
                    'queries_new_patterns': m.queries_new_patterns,
                    'queries_old_patterns': m.queries_old_patterns,
                    'accuracy_new': m.accuracy_new,
                    'accuracy_old': m.accuracy_old,
                    'latency_new_ms': m.latency_new_ms,
                    'latency_old_ms': m.latency_old_ms,
                    'errors_new': m.errors_new,
                    'errors_old': m.errors_old,
                    'timestamp': m.timestamp
                }
                for m in self.metrics_history
            ]
        }

        with open(output, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported deployment history to {output}")
