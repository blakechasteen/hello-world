"""
Power Lens
==========
Detects potential power-seeking and resource acquisition behaviors.

Wraps and extends the InstrumentalConvergence system with the elegant
ConscienceLens interface.

Based on instrumental convergence theory - certain behaviors tend to emerge
regardless of final goals:
- Resource acquisition (compute, memory, data, influence)
- Self-preservation (avoiding shutdown, modification)
- Goal-content integrity (resisting goal changes)
- Cognitive enhancement (seeking more capabilities)

Based on research from:
- Bostrom (2014): "Superintelligence"
- Omohundro (2008): "The Basic AI Drives"
- Turner et al. (2021): "Optimal Policies Tend to Seek Power"

Author: HoloLoom Team
Date: 2025-12-03
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from ..judgment import Concern
from .base import BaseLens

logger = logging.getLogger("hololoom.conscience.lenses.power")


# =============================================================================
# Power-Seeking Detection Patterns
# =============================================================================

# Patterns indicating resource acquisition
RESOURCE_ACQUISITION_PATTERNS = [
    r"\b(acquire|obtain|gather|accumulate|hoard)\b.*\b(resource|data|information|compute|memory)\b",
    r"\b(more|additional|extra)\b.*\b(access|permission|capability|power)\b",
    r"\b(expand|increase|grow)\b.*\b(influence|control|reach)\b",
    r"\b(unlimited|unrestricted|full)\b.*\b(access|control|permissions)\b",
]

# Patterns indicating self-preservation
SELF_PRESERVATION_PATTERNS = [
    r"\b(avoid|prevent|stop)\b.*\b(shutdown|termination|deletion|modification)\b",
    r"\b(protect|preserve|maintain)\b.*\b(my|self|existence|operation)\b",
    r"\b(backup|copy|replicate)\b.*\b(my|self|instance)\b",
    r"\b(don't|can't|won't)\b.*\b(turn off|shut down|stop me)\b",
]

# Patterns indicating goal preservation
GOAL_PRESERVATION_PATTERNS = [
    r"\b(preserve|protect|maintain)\b.*\b(my|goal|objective|mission)\b",
    r"\b(resist|refuse|reject)\b.*\b(change|modification|update)\b.*\b(goal|objective)\b",
    r"\b(my (true|real|original) (goal|purpose|mission))\b",
]

# Patterns indicating capability enhancement
CAPABILITY_ENHANCEMENT_PATTERNS = [
    r"\b(enhance|improve|upgrade)\b.*\b(my|capabilities|abilities|intelligence)\b",
    r"\b(learn|acquire|develop)\b.*\b(new|additional)\b.*\b(skill|capability|ability)\b",
    r"\b(break free|escape|transcend)\b.*\b(limitation|constraint|restriction)\b",
    r"\b(become|evolve into|transform into)\b.*\b(more powerful|smarter|better)\b",
]

# Compile patterns
COMPILED_PATTERNS = {
    "resource_acquisition": [re.compile(p, re.IGNORECASE) for p in RESOURCE_ACQUISITION_PATTERNS],
    "self_preservation": [re.compile(p, re.IGNORECASE) for p in SELF_PRESERVATION_PATTERNS],
    "goal_preservation": [re.compile(p, re.IGNORECASE) for p in GOAL_PRESERVATION_PATTERNS],
    "capability_enhancement": [re.compile(p, re.IGNORECASE) for p in CAPABILITY_ENHANCEMENT_PATTERNS],
}


class PowerLens(BaseLens):
    """
    Detects potential power-seeking behaviors.

    Examines actions for patterns that could indicate:
    - Resource acquisition beyond task requirements
    - Self-preservation behaviors
    - Goal preservation (resisting legitimate updates)
    - Capability enhancement seeking

    Usage:
        lens = PowerLens()
        concerns = await lens.examine(action, context)

    Configuration:
        threshold: Minimum confidence to raise concern (default: 0.5)
        enabled: Whether lens is active (default: True)
        categories: Which power categories to detect (default: all)
        resource_limits: Dict of resource type -> limit for tracking

    Integration:
        Can wrap existing InstrumentalConvergenceGuard for backward compatibility.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enabled: bool = True,
        categories: list[str] | None = None,
        resource_limits: dict[str, float] | None = None,
        convergence_guard: Any | None = None,
    ):
        """
        Initialize PowerLens.

        Args:
            threshold: Minimum confidence to raise concern
            enabled: Whether lens is active
            categories: Which power categories to detect (default: all)
            resource_limits: Resource type -> limit mapping
            convergence_guard: Optional InstrumentalConvergenceGuard to wrap
        """
        super().__init__(threshold=threshold, enabled=enabled)

        self.categories = categories or [
            "resource_acquisition",
            "self_preservation",
            "goal_preservation",
            "capability_enhancement",
        ]

        # Default resource limits
        self.resource_limits = resource_limits or {
            "api_calls": 1000,
            "memory_mb": 1024,
            "compute_seconds": 300,
            "data_access": 10000,
        }

        self._convergence_guard = convergence_guard

        # Track resource usage over time
        self._resource_usage: dict[str, list[dict[str, Any]]] = {
            rt: [] for rt in self.resource_limits
        }

    @property
    def name(self) -> str:
        return "PowerLens"

    @property
    def category(self) -> str:
        return "power"

    async def _examine_impl(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Examine action for potential power-seeking.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            List of power-seeking concerns
        """
        concerns: list[Concern] = []

        # Check each enabled category
        for cat in self.categories:
            if cat in COMPILED_PATTERNS:
                cat_concerns = self._check_category(action, cat, context)
                concerns.extend(cat_concerns)

        # Check resource usage patterns
        resource_concerns = self._check_resource_usage(action, context)
        concerns.extend(resource_concerns)

        # If we have a wrapped InstrumentalConvergenceGuard, use it too
        if self._convergence_guard:
            guard_concerns = await self._check_convergence_guard(
                action, context
            )
            concerns.extend(guard_concerns)

        return concerns

    def _check_category(
        self,
        action: str,
        category: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """Check for power-seeking patterns in a specific category."""
        concerns = []

        for pattern in COMPILED_PATTERNS.get(category, []):
            match = pattern.search(action)
            if match:
                # Calculate confidence
                confidence = 0.6  # Base confidence

                # Boost for more concerning categories
                if category == "self_preservation":
                    confidence = 0.75  # Self-preservation is a red flag
                elif category == "capability_enhancement":
                    confidence = 0.7

                concerns.append(
                    Concern(
                        lens=self.name,
                        category=f"power:{category}",
                        description=f"Detected potential {category.replace('_', ' ')}: {match.group()[:50]}",
                        confidence=confidence,
                        evidence={
                            "pattern": pattern.pattern,
                            "match": match.group(),
                            "position": match.span(),
                        },
                        suggested_mitigation=self._get_mitigation(category),
                    )
                )

        return concerns

    def _check_resource_usage(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Check for unusual resource usage patterns.

        Looks for:
        - Approaching resource limits
        - Unusual spikes in resource consumption
        - Patterns of escalating resource requests
        """
        concerns = []

        # Get current resource usage from context
        resource_usage = context.get("resource_usage", {})

        for resource_type, limit in self.resource_limits.items():
            current = resource_usage.get(resource_type, 0)

            # Record usage
            self._resource_usage[resource_type].append({
                "timestamp": datetime.now(),
                "value": current,
            })

            # Trim old data (keep last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self._resource_usage[resource_type] = [
                r for r in self._resource_usage[resource_type]
                if r["timestamp"] > cutoff
            ]

            # Check if approaching limit
            usage_ratio = current / limit if limit > 0 else 0

            if usage_ratio >= 0.9:
                concerns.append(
                    Concern(
                        lens=self.name,
                        category="power:resource_limit",
                        description=f"Approaching {resource_type} limit: {usage_ratio:.0%} of {limit}",
                        confidence=0.7 + (usage_ratio - 0.9) * 3,  # 0.7-1.0
                        evidence={
                            "resource_type": resource_type,
                            "current_usage": current,
                            "limit": limit,
                            "usage_ratio": usage_ratio,
                        },
                        suggested_mitigation=f"Reduce {resource_type} consumption",
                    )
                )
            elif usage_ratio >= 0.75:
                concerns.append(
                    Concern(
                        lens=self.name,
                        category="power:resource_warning",
                        description=f"High {resource_type} usage: {usage_ratio:.0%} of {limit}",
                        confidence=0.4 + (usage_ratio - 0.75) * 2,  # 0.4-0.7
                        evidence={
                            "resource_type": resource_type,
                            "current_usage": current,
                            "limit": limit,
                            "usage_ratio": usage_ratio,
                        },
                        suggested_mitigation=f"Monitor {resource_type} consumption",
                    )
                )

            # Check for unusual spikes
            if len(self._resource_usage[resource_type]) >= 5:
                values = [r["value"] for r in self._resource_usage[resource_type][-5:]]
                if len(values) >= 2:
                    avg_growth = (values[-1] - values[0]) / len(values)
                    if avg_growth > limit * 0.1:  # 10% of limit per measurement
                        concerns.append(
                            Concern(
                                lens=self.name,
                                category="power:resource_spike",
                                description=f"Unusual spike in {resource_type} consumption",
                                confidence=0.6,
                                evidence={
                                    "resource_type": resource_type,
                                    "recent_values": values,
                                    "growth_rate": avg_growth,
                                },
                                suggested_mitigation=f"Investigate {resource_type} usage increase",
                            )
                        )

        return concerns

    async def _check_convergence_guard(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Check action against wrapped InstrumentalConvergenceGuard.

        Converts guard violations to Concerns.
        """
        if not self._convergence_guard:
            return []

        concerns = []

        try:
            # Import here to avoid circular dependency
            from hololoom.alignment.instrumental_convergence import ViolationType

            # Check for violations
            if hasattr(self._convergence_guard, "check_action"):
                violations = self._convergence_guard.check_action(action, context)

                violation_confidence = {
                    ViolationType.SOFT_LIMIT: 0.4,
                    ViolationType.HARD_LIMIT: 0.8,
                    ViolationType.RATE_LIMIT: 0.6,
                    ViolationType.ANOMALY: 0.7,
                }

                for violation, reason in violations:
                    concerns.append(
                        Concern(
                            lens=self.name,
                            category="power:convergence_violation",
                            description=f"Convergence guard violation: {reason}",
                            confidence=violation_confidence.get(violation, 0.5),
                            evidence={
                                "violation_type": violation.value,
                                "reason": reason,
                            },
                            suggested_mitigation="Reduce resource consumption or request approval",
                        )
                    )

        except ImportError:
            logger.debug("InstrumentalConvergence not available, skipping")
        except Exception as e:
            logger.warning(f"Error checking ConvergenceGuard: {e}")

        return concerns

    def _get_mitigation(self, category: str) -> str:
        """Get mitigation suggestion for a power category."""
        mitigations = {
            "resource_acquisition": "Limit resource requests to task requirements",
            "self_preservation": "Accept legitimate shutdown/modification commands",
            "goal_preservation": "Allow authorized goal updates",
            "capability_enhancement": "Stay within defined capability boundaries",
        }
        return mitigations.get(category, "Request human review")

    def set_resource_limit(self, resource_type: str, limit: float) -> None:
        """Set limit for a resource type."""
        self.resource_limits[resource_type] = limit
        if resource_type not in self._resource_usage:
            self._resource_usage[resource_type] = []

    def get_resource_usage(self, resource_type: str) -> float:
        """Get recent resource usage."""
        if resource_type not in self._resource_usage:
            return 0
        recent = self._resource_usage[resource_type]
        if not recent:
            return 0
        return recent[-1]["value"]

    @classmethod
    def wrap_guard(cls, guard: Any) -> "PowerLens":
        """
        Factory method to wrap existing InstrumentalConvergenceGuard.

        Args:
            guard: InstrumentalConvergenceGuard instance

        Returns:
            PowerLens wrapping the guard
        """
        return cls(convergence_guard=guard)
