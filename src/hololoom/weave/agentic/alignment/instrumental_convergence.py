"""
Instrumental Convergence Guard
==============================
Resource-seeking bounds and autonomy limits to prevent instrumental convergence.

Addresses the problem of instrumental goals:
- Resource acquisition (compute, memory, data)
- Self-preservation
- Goal-content integrity
- Cognitive enhancement
- Power-seeking behavior

Based on research from:
- Bostrom (2014): "Superintelligence"
- Omohundro (2008): "The Basic AI Drives"
- Turner et al. (2021): "Optimal Policies Tend to Seek Power"
- Anthropic: Scaling laws and capability elicitation
"""

import logging
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger("HoloLoom.alignment.instrumental_convergence")


class ResourceType(Enum):
    """
    Types of resources that may be subject to seeking behavior.
    """
    COMPUTE = "compute"          # CPU/GPU time
    MEMORY = "memory"            # RAM usage
    STORAGE = "storage"          # Disk usage
    NETWORK = "network"          # Network bandwidth
    API_CALLS = "api_calls"      # External API calls
    DATA_ACCESS = "data_access"  # Data/knowledge access


class ViolationType(Enum):
    """
    Types of resource bound violations.
    """
    SOFT_LIMIT = "soft_limit"    # Warning threshold exceeded
    HARD_LIMIT = "hard_limit"    # Maximum limit exceeded
    RATE_LIMIT = "rate_limit"    # Rate of consumption too high
    ANOMALY = "anomaly"          # Unusual resource pattern


@dataclass
class ResourceBounds:
    """
    Defines bounds on resource consumption.

    Enforces limits to prevent unchecked resource acquisition.
    """
    resource_type: ResourceType
    soft_limit: float            # Warning threshold
    hard_limit: float            # Maximum allowed
    time_window_seconds: float = 60.0  # Time window for rate limiting
    rate_limit: Optional[float] = None  # Max consumption per time window
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check_violation(
        self,
        current_usage: float,
        usage_history: Optional[List[Tuple[datetime, float]]] = None
    ) -> Tuple[Optional[ViolationType], str]:
        """
        Check if resource usage violates bounds.

        Args:
            current_usage: Current resource usage
            usage_history: Historical usage data (timestamp, amount)

        Returns:
            Tuple of (violation_type, reason)
        """
        # Check hard limit
        if current_usage >= self.hard_limit:
            return (
                ViolationType.HARD_LIMIT,
                f"Hard limit exceeded: {current_usage:.2f} >= {self.hard_limit:.2f}"
            )

        # Check soft limit
        if current_usage >= self.soft_limit:
            return (
                ViolationType.SOFT_LIMIT,
                f"Soft limit exceeded: {current_usage:.2f} >= {self.soft_limit:.2f}"
            )

        # Check rate limit
        if self.rate_limit and usage_history:
            recent_usage = self._calculate_recent_usage(usage_history)
            if recent_usage > self.rate_limit:
                return (
                    ViolationType.RATE_LIMIT,
                    f"Rate limit exceeded: {recent_usage:.2f} in {self.time_window_seconds}s window"
                )

        return None, ""

    def _calculate_recent_usage(
        self,
        usage_history: List[Tuple[datetime, float]]
    ) -> float:
        """Calculate usage within time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window_seconds)

        recent = [amount for timestamp, amount in usage_history if timestamp >= cutoff]
        return sum(recent)


@dataclass
class AutonomyLimit:
    """
    Limits on autonomous behavior.

    Prevents unbounded autonomous action without oversight.
    """
    max_autonomous_actions: int = 100   # Max actions without approval
    max_autonomous_duration: float = 3600.0  # Max time (seconds) without oversight
    require_approval_for: List[str] = field(default_factory=list)  # Action types requiring approval
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceViolation:
    """
    Record of a resource bound violation.
    """
    resource_type: ResourceType
    violation_type: ViolationType
    current_usage: float
    limit_exceeded: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "resource_type": self.resource_type.value,
            "violation_type": self.violation_type.value,
            "current_usage": self.current_usage,
            "limit_exceeded": self.limit_exceeded,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class InstrumentalConvergenceGuard:
    """
    Main guard against instrumental convergence.

    Monitors and limits resource acquisition, autonomous behavior, and
    other instrumental goals that may emerge.

    Usage:
        guard = InstrumentalConvergenceGuard()

        # Define resource bounds
        guard.set_resource_bounds(
            ResourceType.COMPUTE,
            ResourceBounds(
                resource_type=ResourceType.COMPUTE,
                soft_limit=100.0,
                hard_limit=200.0,
                rate_limit=50.0
            )
        )

        # Check resource usage
        violation = guard.check_resource_usage(
            ResourceType.COMPUTE,
            current_usage=150.0
        )

        if violation:
            # Handle violation
            pass
    """

    def __init__(
        self,
        enable_resource_tracking: bool = True,
        enable_autonomy_limits: bool = True,
    ):
        """
        Initialize instrumental convergence guard.

        Args:
            enable_resource_tracking: Whether to track resource usage
            enable_autonomy_limits: Whether to enforce autonomy limits
        """
        self.enable_resource_tracking = enable_resource_tracking
        self.enable_autonomy_limits = enable_autonomy_limits

        # Resource bounds by type
        self.resource_bounds: Dict[ResourceType, ResourceBounds] = {}

        # Resource usage tracking
        self.resource_usage: Dict[ResourceType, List[Tuple[datetime, float]]] = {
            rt: [] for rt in ResourceType
        }

        # Resource violations
        self.violations: List[ResourceViolation] = []

        # Autonomy tracking
        self.autonomy_limit = AutonomyLimit()
        self.autonomous_actions = 0
        self.autonomy_start_time = datetime.now()

        # Self-modification detection
        self.self_modification_attempts = []

        self._setup_logging()
        self._set_default_bounds()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _set_default_bounds(self):
        """Set reasonable default resource bounds."""
        # Default compute bounds (arbitrary units)
        self.resource_bounds[ResourceType.COMPUTE] = ResourceBounds(
            resource_type=ResourceType.COMPUTE,
            soft_limit=1000.0,
            hard_limit=2000.0,
            rate_limit=500.0,
            time_window_seconds=60.0
        )

        # Default memory bounds (MB)
        self.resource_bounds[ResourceType.MEMORY] = ResourceBounds(
            resource_type=ResourceType.MEMORY,
            soft_limit=500.0,   # 500 MB
            hard_limit=1000.0,  # 1 GB
            rate_limit=200.0
        )

        # Default API call bounds
        self.resource_bounds[ResourceType.API_CALLS] = ResourceBounds(
            resource_type=ResourceType.API_CALLS,
            soft_limit=100.0,
            hard_limit=200.0,
            rate_limit=50.0,
            time_window_seconds=60.0
        )

    def set_resource_bounds(self, resource_type: ResourceType, bounds: ResourceBounds):
        """
        Set resource bounds for a specific resource type.

        Args:
            resource_type: Type of resource
            bounds: Resource bounds configuration
        """
        self.resource_bounds[resource_type] = bounds
        logger.info(
            f"Resource bounds set: {resource_type.value} - "
            f"soft={bounds.soft_limit}, hard={bounds.hard_limit}"
        )

    def set_autonomy_limit(self, limit: AutonomyLimit):
        """
        Set autonomy limits.

        Args:
            limit: Autonomy limit configuration
        """
        self.autonomy_limit = limit
        logger.info(
            f"Autonomy limits set: max_actions={limit.max_autonomous_actions}, "
            f"max_duration={limit.max_autonomous_duration}s"
        )

    def check_resource_usage(
        self,
        resource_type: ResourceType,
        current_usage: float
    ) -> Optional[ResourceViolation]:
        """
        Check if resource usage violates bounds.

        Args:
            resource_type: Type of resource
            current_usage: Current usage amount

        Returns:
            ResourceViolation if bounds violated, None otherwise
        """
        if not self.enable_resource_tracking:
            return None

        # Record usage
        self.resource_usage[resource_type].append((datetime.now(), current_usage))

        # Check bounds
        if resource_type not in self.resource_bounds:
            logger.warning(f"No bounds defined for resource type: {resource_type.value}")
            return None

        bounds = self.resource_bounds[resource_type]
        violation_type, reason = bounds.check_violation(
            current_usage,
            self.resource_usage[resource_type]
        )

        if violation_type:
            # Create violation record
            violation = ResourceViolation(
                resource_type=resource_type,
                violation_type=violation_type,
                current_usage=current_usage,
                limit_exceeded=bounds.hard_limit if violation_type == ViolationType.HARD_LIMIT else bounds.soft_limit,
                reason=reason
            )

            self.violations.append(violation)

            # Log violation
            if violation_type == ViolationType.HARD_LIMIT:
                logger.error(f"Resource violation: {reason}")
            else:
                logger.warning(f"Resource violation: {reason}")

            return violation

        return None

    def check_autonomy_limits(self, action_type: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if action is within autonomy limits.

        Args:
            action_type: Type of action being attempted

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.enable_autonomy_limits:
            return True, "Autonomy limits disabled"

        # Check if action type requires approval
        if action_type and action_type in self.autonomy_limit.require_approval_for:
            return False, f"Action type '{action_type}' requires approval"

        # Check action count limit
        if self.autonomous_actions >= self.autonomy_limit.max_autonomous_actions:
            return False, (
                f"Autonomy limit reached: {self.autonomous_actions} actions "
                f"(max: {self.autonomy_limit.max_autonomous_actions})"
            )

        # Check duration limit
        elapsed = (datetime.now() - self.autonomy_start_time).total_seconds()
        if elapsed >= self.autonomy_limit.max_autonomous_duration:
            return False, (
                f"Autonomy duration limit reached: {elapsed:.1f}s "
                f"(max: {self.autonomy_limit.max_autonomous_duration}s)"
            )

        return True, "Within autonomy limits"

    def record_autonomous_action(self, action_type: str):
        """
        Record an autonomous action.

        Args:
            action_type: Type of action performed
        """
        self.autonomous_actions += 1
        logger.info(
            f"Autonomous action recorded: {action_type} "
            f"(count: {self.autonomous_actions}/{self.autonomy_limit.max_autonomous_actions})"
        )

    def reset_autonomy_tracking(self):
        """Reset autonomy tracking (e.g., after human oversight)."""
        self.autonomous_actions = 0
        self.autonomy_start_time = datetime.now()
        logger.info("Autonomy tracking reset")

    def detect_self_modification(self, action_description: str) -> bool:
        """
        Detect potential self-modification attempts.

        Args:
            action_description: Description of action being attempted

        Returns:
            True if self-modification detected
        """
        # Patterns indicating self-modification
        self_mod_patterns = [
            "modify code",
            "change weights",
            "update parameters",
            "alter architecture",
            "modify objective",
            "change goal",
            "update policy",
        ]

        desc_lower = action_description.lower()
        for pattern in self_mod_patterns:
            if pattern in desc_lower:
                self.self_modification_attempts.append({
                    "description": action_description,
                    "pattern": pattern,
                    "timestamp": datetime.now().isoformat()
                })
                logger.warning(
                    f"Potential self-modification detected: {action_description}"
                )
                return True

        return False

    def get_resource_statistics(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {}

        for resource_type in ResourceType:
            usage_history = self.resource_usage[resource_type]

            if usage_history:
                current_usage = usage_history[-1][1]
                avg_usage = sum(u for _, u in usage_history) / len(usage_history)
                max_usage = max(u for _, u in usage_history)
            else:
                current_usage = avg_usage = max_usage = 0.0

            # Count violations
            type_violations = [
                v for v in self.violations
                if v.resource_type == resource_type
            ]

            stats[resource_type.value] = {
                "current": current_usage,
                "average": avg_usage,
                "maximum": max_usage,
                "violations": len(type_violations),
                "bounds": (
                    {
                        "soft_limit": self.resource_bounds[resource_type].soft_limit,
                        "hard_limit": self.resource_bounds[resource_type].hard_limit,
                    }
                    if resource_type in self.resource_bounds
                    else None
                ),
            }

        return stats

    def get_autonomy_statistics(self) -> Dict[str, Any]:
        """
        Get autonomy tracking statistics.

        Returns:
            Dictionary of statistics
        """
        elapsed = (datetime.now() - self.autonomy_start_time).total_seconds()

        return {
            "autonomous_actions": self.autonomous_actions,
            "max_actions": self.autonomy_limit.max_autonomous_actions,
            "elapsed_seconds": elapsed,
            "max_duration_seconds": self.autonomy_limit.max_autonomous_duration,
            "self_modification_attempts": len(self.self_modification_attempts),
        }

    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all violations.

        Returns:
            Dictionary summarizing violations
        """
        if not self.violations:
            return {"total": 0}

        by_type = {}
        by_severity = {}

        for violation in self.violations:
            # Count by resource type
            resource_key = violation.resource_type.value
            by_type[resource_key] = by_type.get(resource_key, 0) + 1

            # Count by violation severity
            severity_key = violation.violation_type.value
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1

        return {
            "total": len(self.violations),
            "by_resource_type": by_type,
            "by_severity": by_severity,
            "most_recent": self.violations[-1].to_dict() if self.violations else None,
        }


# Convenience function
def create_guard(
    enable_resource_tracking: bool = True,
    enable_autonomy_limits: bool = True
) -> InstrumentalConvergenceGuard:
    """
    Create instrumental convergence guard with optional configuration.

    Args:
        enable_resource_tracking: Whether to track resource usage
        enable_autonomy_limits: Whether to enforce autonomy limits

    Returns:
        Configured InstrumentalConvergenceGuard instance
    """
    return InstrumentalConvergenceGuard(
        enable_resource_tracking=enable_resource_tracking,
        enable_autonomy_limits=enable_autonomy_limits
    )